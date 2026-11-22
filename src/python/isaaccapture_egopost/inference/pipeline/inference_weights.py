# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-OpenMDW-1.1

"""Inference-only EgoHand weights: export, read, and strict key checks.

A training ``.ckpt`` (PyTorch Lightning, ``weights_only=False`` pickle) also
carries optimizer state and is several times larger than the model. The
export written by :func:`export_inference_weights` is a folder with

* ``model.safetensors``: the checkpoint ``state_dict`` tensors, unchanged;
* ``config.json``: the training cfg (``hyper_parameters.cfg``), the top-level
  ``egohand_*`` checkpoint fields that
  ``EgoHand.validate_camera_depth_checkpoint`` reads, and provenance.

Loading it needs neither pickle nor ``weights_only=False``. Both formats go
through :func:`read_checkpoint`, so the CLI's ``--egohand_ckpt`` accepts either.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

log = logging.getLogger(__name__)

WEIGHTS_FILENAME = "model.safetensors"
CONFIG_FILENAME = "config.json"
EXPORT_FORMAT = "egohand-inference-weights/1"

#: Model keys that a checkpoint is allowed to omit. The SOMA hand layers are
#: rebuilt from the SOMA asset folder at construction and wrapped in
#: ``NonPersistentModuleWrapper`` (``egohand/models/soma_wrapper.py``), whose
#: state-dict hook removes them from every saved checkpoint; ``load_state_dict``
#: still reports them as missing (``left``/``right`` alias the wrapped layers).
#: Every other model key must be present, including any future learnable
#: parameter added directly under ``soma_hand``.
ALLOWED_MISSING_PREFIXES: tuple[str, ...] = (
    "soma_hand.hand_left.",
    "soma_hand.hand_right.",
    "soma_hand.left.",
    "soma_hand.right.",
)

#: Checkpoint keys that the model is allowed not to own. None: an unexpected
#: key means the checkpoint and the model configuration disagree.
ALLOWED_UNEXPECTED_PREFIXES: tuple[str, ...] = ()


@dataclass
class CheckpointSource:
    """Configuration read eagerly; tensors read on demand (possibly on another thread)."""

    path: Path
    cfg: dict
    checkpoint_fields: dict
    load_state_dict: Callable[[], dict]
    format: str
    state_dict_keys: Callable[[], set]

    def validation_view(self) -> dict:
        """The subset of a checkpoint dict that model-side validators read."""
        return {"hyper_parameters": {"cfg": self.cfg}, **self.checkpoint_fields}


def is_inference_export(path: str | Path) -> bool:
    """True for an export folder or its ``model.safetensors`` file."""
    path = Path(path)
    if path.is_dir():
        return (path / WEIGHTS_FILENAME).is_file() and (
            path / CONFIG_FILENAME
        ).is_file()
    return path.suffix == ".safetensors" and (path.parent / CONFIG_FILENAME).is_file()


def _json_value(value, key: str):
    """Convert checkpoint metadata to JSON types, rejecting anything else."""
    if isinstance(value, (str, bool, int, float)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return [_json_value(item, key) for item in value]
    if hasattr(value, "tolist"):  # numpy / torch scalars and small arrays
        return _json_value(value.tolist(), key)
    raise TypeError(f"checkpoint field {key!r} has non-JSON value {type(value)!r}")


def _checkpoint_fields(ckpt: dict) -> dict:
    """Top-level ``egohand_*`` fields consumed by checkpoint validation."""
    return {
        key: _json_value(value, key)
        for key, value in ckpt.items()
        if isinstance(key, str) and key.startswith("egohand_")
    }


def _cfg_container(cfg) -> dict:
    from omegaconf import DictConfig, OmegaConf

    if isinstance(cfg, DictConfig):
        return OmegaConf.to_container(cfg, resolve=False)
    return OmegaConf.to_container(OmegaConf.create(cfg), resolve=False)


def _torch_load(path: Path) -> dict:
    """Load a training checkpoint; memory-map so optimizer state is never read."""
    import torch

    try:
        return torch.load(str(path), map_location="cpu", weights_only=False, mmap=True)
    except RuntimeError as exc:
        # Legacy (non-zip) serialization cannot be memory-mapped.
        log.info("[EgoHand] mmap load unavailable (%s); reading whole checkpoint", exc)
        return torch.load(str(path), map_location="cpu", weights_only=False)


def read_checkpoint(path: str | Path) -> CheckpointSource:
    """Read the cfg now and return a callable that reads the tensors later."""
    path = Path(path)
    if is_inference_export(path):
        folder = path if path.is_dir() else path.parent
        config = json.loads((folder / CONFIG_FILENAME).read_text())
        if config.get("format") != EXPORT_FORMAT:
            raise ValueError(
                f"{folder / CONFIG_FILENAME}: unsupported format {config.get('format')!r}; "
                f"expected {EXPORT_FORMAT!r}"
            )
        weights = folder / WEIGHTS_FILENAME

        def load_tensors() -> dict:
            from safetensors.torch import load_file

            return load_file(str(weights), device="cpu")

        def tensor_names() -> set:
            from safetensors import safe_open

            with safe_open(str(weights), framework="pt", device="cpu") as handle:
                return set(handle.keys())

        return CheckpointSource(
            path=folder,
            cfg=config["cfg"],
            checkpoint_fields=config.get("checkpoint_fields", {}),
            load_state_dict=load_tensors,
            format=EXPORT_FORMAT,
            state_dict_keys=tensor_names,
        )

    ckpt = _torch_load(path)
    state_dict = ckpt["state_dict"]
    return CheckpointSource(
        path=path,
        cfg=_cfg_container(ckpt["hyper_parameters"]["cfg"]),
        checkpoint_fields=_checkpoint_fields(ckpt),
        load_state_dict=lambda: state_dict,
        format="lightning-ckpt",
        state_dict_keys=lambda: set(state_dict),
    )


def check_state_dict_keys(
    missing_keys, unexpected_keys, *, ignore_unexpected: bool = False
) -> None:
    """Raise unless every missing/unexpected key is on its allowlist.

    ``ignore_unexpected``: checkpoint keys the model does not own (e.g. extra
    training-time tensors) only log a warning. Missing keys always raise: the
    model would keep those tensors uninitialized or randomly initialized.
    """
    bad_missing = sorted(
        key for key in missing_keys if not key.startswith(ALLOWED_MISSING_PREFIXES)
    )
    bad_unexpected = sorted(
        key
        for key in unexpected_keys
        if not key.startswith(ALLOWED_UNEXPECTED_PREFIXES)
    )

    def preview(keys):
        return ", ".join(keys[:20]) + (" ..." if len(keys) > 20 else "")

    if bad_unexpected and ignore_unexpected and not bad_missing:
        log.warning(
            "[EgoHand] ignoring %d checkpoint key(s) the model does not own [%s]",
            len(bad_unexpected),
            preview(bad_unexpected),
        )
        return
    if bad_missing or bad_unexpected:
        raise RuntimeError(
            "EgoHand checkpoint does not match the model built from its cfg: "
            f"{len(bad_missing)} missing key(s) outside the allowlist "
            f"{ALLOWED_MISSING_PREFIXES} [{preview(bad_missing)}]; "
            f"{len(bad_unexpected)} unexpected key(s) [{preview(bad_unexpected)}]"
            + (
                ". A checkpoint with extra keys only loads with "
                "load_egohand_model(..., ignore_unexpected_keys=True)"
                if bad_unexpected and not bad_missing
                else ""
            )
        )


def expected_model_keys(cfg) -> set:
    """Checkpoint keys an ``EgoHand(cfg)`` needs, without building it for real.

    The model is instantiated on the ``meta`` device (no memory, no random
    initialization) with the asset-derived SOMA layers left out; those are
    exactly the keys ``ALLOWED_MISSING_PREFIXES`` exempts. Takes about a
    second and reads no tensors, so key mismatches surface before decoding.
    """
    import torch
    from omegaconf import OmegaConf

    import egohand.models.egohand as egohand_module

    class _NoSomaLayers(torch.nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()

    cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=False))
    backbone = cfg.get("backbone", None)
    if backbone is not None and str(backbone.get("_target_", "")).endswith(
        "DinoV3Backbone"
    ):
        cfg.backbone.meta_init = True
    original = egohand_module.SomaHandWrapper
    egohand_module.SomaHandWrapper = _NoSomaLayers
    try:
        with torch.device("meta"):
            model = egohand_module.EgoHand(cfg)
    finally:
        egohand_module.SomaHandWrapper = original
    return set(model.state_dict())


def check_checkpoint_keys(source: "CheckpointSource", cfg) -> None:
    """Key-set check of ``source`` against ``cfg`` from key names only."""
    expected = expected_model_keys(cfg)
    present = source.state_dict_keys()
    check_state_dict_keys(expected - present, present - expected)


def _sha256(path: Path, chunk_bytes: int = 64 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def export_inference_weights(
    checkpoint_path: str | Path,
    output_dir: str | Path,
    *,
    source_sha256: bool = True,
) -> dict:
    """Write ``model.safetensors`` + ``config.json`` from a training checkpoint.

    Tensors are written exactly as stored (same keys, dtypes and values);
    optimizer, scheduler, loop and callback state are dropped.

    Returns:
        The ``config.json`` content.
    """
    import torch
    from safetensors.torch import save_file

    checkpoint_path = Path(checkpoint_path)
    output_dir = Path(output_dir)
    ckpt = _torch_load(checkpoint_path)
    state_dict = ckpt["state_dict"]
    tensors = {}
    for key, value in state_dict.items():
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"state_dict[{key!r}] is {type(value)!r}, not a tensor")
        # safetensors rejects shared/strided storage; a private contiguous
        # copy keeps the values bit-for-bit.
        tensors[key] = value.detach().to("cpu").contiguous().clone()
    config = {
        "format": EXPORT_FORMAT,
        "cfg": _cfg_container(ckpt["hyper_parameters"]["cfg"]),
        "checkpoint_fields": _checkpoint_fields(ckpt),
        "tensors": {
            "count": len(tensors),
            "numel": int(sum(t.numel() for t in tensors.values())),
            "dtypes": sorted({str(t.dtype) for t in tensors.values()}),
        },
        "source": {
            "file": checkpoint_path.name,
            "bytes": checkpoint_path.stat().st_size,
            "sha256": _sha256(checkpoint_path) if source_sha256 else None,
            "global_step": _json_value(ckpt.get("global_step"), "global_step"),
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    weights_tmp = output_dir / (WEIGHTS_FILENAME + ".tmp")
    save_file(tensors, str(weights_tmp), metadata={"format": EXPORT_FORMAT})
    weights_tmp.replace(output_dir / WEIGHTS_FILENAME)
    config_tmp = output_dir / (CONFIG_FILENAME + ".tmp")
    config_tmp.write_text(json.dumps(config, indent=2) + "\n")
    config_tmp.replace(output_dir / CONFIG_FILENAME)
    return config
