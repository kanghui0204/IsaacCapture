# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-OpenMDW-1.1

"""MHR TorchScript forward without the per-call sort of its sparse layer.

The MHR identity model of py-soma-x (``MHRHandIdentityModel._mhr_model``, a
``NonPersistentModuleWrapper`` around ``MHR/mhr_model_lod1.pt``) has one
sparse layer in its pose correctives (``SparseLinear``: 3000 x 750, 53,136
non-zeros). Its TorchScript ``forward`` builds an uncoalesced COO tensor on
every call, so every ``torch.matmul`` first coalesces it. On CUDA the coalesce
is ``thrust::sort_by_key`` = ``cub::DeviceMergeSort`` (1 block-sort + 5
partition + 5 merge kernels, launched with programmatic dependent launch).
The CCCL 3.0.1 merge kernel compiled into PyTorch 2.13.0+cu130 reads the
partition table before ``cudaGridDependencySynchronize()`` (NVIDIA/cccl#5297,
fixed in CCCL 3.0.3 by #5456/#5461). With other kernels running in the same
CUDA context it sometimes reads stale offsets and faults (Xid 13 / Xid 31,
``merge_sort: failed to synchronize: cudaErrorIllegalAddress``).

The stored indices are already in row-major order without duplicates, so the
coalesced tensor equals the stored one. ``PresortedMHR`` runs the same
computation as the TorchScript ``forward`` (``apply_correctives=True`` path)
from Python, calling the TorchScript submodules for every step except the
sparse product, which uses a COO tensor built once with
``is_coalesced=True``: no sort kernels, same COO -> CSR + cuSPARSE product.
With the zero pose that ``MHRHandIdentityModel`` passes, the sparse layer and
the pose correctives are bitwise the same as before; with a non-zero pose
cuSPARSE itself differs by a few ulp from run to run, with or without this.

The rewrite is applied only when the TorchScript code of the reimplemented
methods is exactly the known code (``_KNOWN_CODE``) and the stored indices are
checked to be strictly increasing; otherwise the model is left unchanged and
the reason is returned. ``apply_mhr_presort`` (called by
``EgoHandPredictor.from_checkpoint``) then raises when this process is exposed
to the race (``presort_required``: PyTorch built with CUDA 13.0, which bundles
CCCL 3.0.x, on an sm_90 or newer GPU, where the merge kernels use programmatic
dependent launch) and only warns otherwise.

Inference only: the cached matrix holds detached parameters (no gradient), and
the sparse product runs in eager mode, so call it with autocast off (the
TorchScript forward ignores autocast; all SOMA callers disable it).
"""

from __future__ import annotations

import hashlib
import logging

import torch
from torch import nn

log = logging.getLogger(__name__)

# sha256[:16] of ``.code`` of the methods PresortedMHR reimplements
# (py-soma-x 0.3.0 asset MHR/mhr_model_lod1.pt, PyTorch 2.13.0).
_KNOWN_CODE = {
    "": "9b6e6a25f5875f04",  # MHRDemo.forward
    "pose_correctives_model": "3bb8b1f2fb155b90",  # MHRPoseCorrectivesModel.forward
    "pose_correctives_model.pose_dirs_predictor": "34e7d6ee713f13f2",  # Sequential
    "pose_correctives_model.pose_dirs_predictor.0": "7975ce70ffd1cd10",  # SparseLinear
    "pose_correctives_model.pose_dirs_predictor.1": "c406775c105bccef",  # ReLU
    "pose_correctives_model.pose_dirs_predictor.2": "5ed428953f0a8012",  # Linear
}


def _code_digest(module) -> str | None:
    try:
        return hashlib.sha256(module.code.encode()).hexdigest()[:16]
    except (AttributeError, RuntimeError):
        return None


def _get(module, dotted: str):
    for part in filter(None, dotted.split(".")):
        module = getattr(module, part)
    return module


def check_mhr_torchscript(ts_module) -> str | None:
    """Return why ``ts_module`` can not be rewritten, or None if it can."""
    if not isinstance(ts_module, torch.jit.ScriptModule):
        return "not a TorchScript module"
    for name, digest in _KNOWN_CODE.items():
        try:
            sub = _get(ts_module, name)
        except AttributeError:
            return f"no submodule {name!r}"
        got = _code_digest(sub)
        if got != digest:
            return f"code of {name or 'forward'!r} is {got}, expected {digest}"
    sparse = _get(ts_module, "pose_correctives_model.pose_dirs_predictor.0")
    idx, weight = sparse.sparse_indices, sparse.sparse_weight
    rows, cols = (int(v) for v in sparse.sparse_shape)
    if (
        idx.dim() != 2
        or idx.shape[0] != 2
        or weight.dim() != 1
        or idx.shape[1] != weight.shape[0]
    ):
        return f"unexpected sparse layout {tuple(idx.shape)} / {tuple(weight.shape)}"
    with torch.no_grad():
        flat = idx[0] * cols + idx[1]
        ok = (
            bool((idx >= 0).all())
            and bool((idx[0] < rows).all())
            and bool((idx[1] < cols).all())
            and bool((flat[1:] > flat[:-1]).all())
        )
    if not ok:
        return "sparse indices are not strictly increasing in row-major order"
    return None


class PresortedMHR(nn.Module):
    """``MHRDemo.forward`` with the pose-correctives sparse matrix built once, coalesced.

    Same signature and outputs as the TorchScript module: ``(verts, skel_state)``.
    """

    def __init__(self, ts_module: torch.jit.ScriptModule):
        super().__init__()
        self.ts = ts_module
        self._coalesced = None
        self._coalesced_key = None

    def _sparse_matrix(self) -> torch.Tensor:
        layer = _get(self.ts, "pose_correctives_model.pose_dirs_predictor.0")
        idx, weight = layer.sparse_indices, layer.sparse_weight
        # Rebuilt when the parameters move, change dtype or are replaced.
        key = (weight.device, weight.dtype, idx.data_ptr(), weight.data_ptr())
        cached = self._coalesced
        if cached is None or self._coalesced_key != key:
            rows, cols = (int(v) for v in layer.sparse_shape)
            with torch.no_grad():
                cached = torch.sparse_coo_tensor(
                    idx.detach(),
                    weight.detach(),
                    (rows, cols),
                    is_coalesced=True,
                    check_invariants=False,  # checked once in check_mhr_torchscript
                )
            self._coalesced, self._coalesced_key = cached, key
        return cached

    def forward(
        self,
        identity_coeffs: torch.Tensor,
        model_parameters: torch.Tensor,
        face_expr_coeffs: torch.Tensor,
        apply_correctives: bool = True,
    ):
        ts = self.ts
        if identity_coeffs.dim() != 2:
            raise AssertionError(
                f"Expected batched (n_rows >= 1) identity coeffs, got {tuple(identity_coeffs.shape)}"
            )
        if face_expr_coeffs.dim() != 2:
            raise AssertionError(
                "Expected batched (n_rows >= 1) face expressions coeffs, "
                f"got {tuple(face_expr_coeffs.shape)}"
            )
        # Same steps as the TorchScript forward. Each intermediate is dropped at
        # its last use, as the TorchScript interpreter does, so the peak memory
        # of the call stays the same (the B x 18439 x 3 tensors are ~110 MB at
        # B = 512).
        character = ts.character_torch
        identity_rest_pose = character.blend_shape(identity_coeffs)
        joint_parameters = character.model_parameters_to_joint_parameters(
            torch.cat([model_parameters, torch.zeros_like(identity_coeffs)], 1)
        )
        skel_state = character.joint_parameters_to_skeleton_state(joint_parameters)
        face_expressions = ts.face_expressions_model(face_expr_coeffs)
        linear_model_unposed = identity_rest_pose + face_expressions
        del identity_rest_pose, face_expressions
        if apply_correctives:
            correctives = ts.pose_correctives_model
            feats = correctives._pose_features_from_joint_params(joint_parameters)
            del joint_parameters
            rows = feats.shape[0]
            predictor = correctives.pose_dirs_predictor
            relu, linear = getattr(predictor, "1"), getattr(predictor, "2")
            hidden = torch.matmul(self._sparse_matrix(), feats.T).T  # SparseLinear
            del feats
            hidden = relu(hidden)
            corrective = linear(hidden).reshape(rows, -1, 3)
            del hidden
            linear_model_unposed = linear_model_unposed + corrective
            del corrective
        else:
            del joint_parameters
        verts = character.skin_points(skel_state, linear_model_unposed)
        return verts, skel_state


def presort_mhr_sparse(model: nn.Module) -> dict:
    """Swap every MHR TorchScript model wrapped under ``model`` for ``PresortedMHR``.

    Looks for modules with a ``module`` attribute holding the MHR TorchScript
    model (py-soma-x's ``NonPersistentModuleWrapper``, e.g.
    ``soma_hand.hand_left.module.identity_model._mhr_model``). Returns
    ``{"patched": [names], "skipped": {name: reason}}``.
    """
    patched, skipped = [], {}
    for name, wrapper in list(model.named_modules()):
        inner = getattr(wrapper, "module", None)
        if isinstance(inner, PresortedMHR) or not isinstance(
            inner, torch.jit.ScriptModule
        ):
            continue
        if getattr(inner, "original_name", None) != "MHRDemo":
            continue
        reason = check_mhr_torchscript(inner)
        if reason is not None:
            skipped[name] = reason
            log.warning("MHR sparse layer left as is (%s): %s", name, reason)
            continue
        wrapper.module = PresortedMHR(inner)
        wrapper.module._sparse_matrix()  # build now, not inside the first SOMA call
        patched.append(name)
    return {"patched": patched, "skipped": skipped}


def _uses_mhr(model: nn.Module) -> bool:
    """True if ``model`` holds an MHR identity model or MHR TorchScript model."""
    return any(
        type(module).__name__ == "MHRHandIdentityModel"
        or (
            isinstance(module, torch.jit.ScriptModule)
            and getattr(module, "original_name", None) == "MHRDemo"
        )
        for module in model.modules()
    )


def presort_required(device) -> str | None:
    """Why the per-call merge sort can crash in this process, or None.

    The race (NVIDIA/cccl#5297, fixed in CCCL 3.0.3) is in the CCCL 3.0.x merge
    sort bundled with PyTorch builds for CUDA 13.0; its kernels use
    programmatic dependent launch, which needs compute capability 9.0 or newer.
    """
    cuda = torch.version.cuda or ""
    device = torch.device(device)
    if not cuda.startswith("13.0") or device.type != "cuda":
        return None
    if not torch.cuda.is_available():
        return None
    major, minor = torch.cuda.get_device_capability(device)
    if major < 9:
        return None
    return (
        f"PyTorch {torch.__version__} (CUDA {cuda}, CCCL 3.0.x) on an "
        f"sm_{major}{minor} GPU"
    )


def apply_mhr_presort(model: nn.Module, device) -> dict:
    """``presort_mhr_sparse(model)``, failing if an exposed process stays unpatched.

    Returns the ``presort_mhr_sparse`` record. If ``model`` holds an MHR model
    that could not be patched (for example after a py-soma-x or PyTorch update
    changed its TorchScript code), raises ``RuntimeError`` when
    ``presort_required(device)`` and otherwise logs a warning and keeps the
    TorchScript forward.
    """
    info = presort_mhr_sparse(model)
    if not info["skipped"]:
        if info["patched"]:
            log.info("MHR sparse layer coalesced once at load: %s", info["patched"])
            return info
        if any(isinstance(m, PresortedMHR) for m in model.modules()):
            return info  # patched by an earlier call
        if not _uses_mhr(model):
            return info  # no MHR identity model, no sparse layer to sort
    problem = (
        "MHR sparse layer could not be coalesced at load "
        f"(patched {info['patched']}, skipped {info['skipped'] or 'none'})"
    )
    exposed = presort_required(device)
    if exposed is not None:
        raise RuntimeError(
            f"{problem}. With {exposed} the per-call merge sort of that layer can "
            "fault (NVIDIA/cccl#5297). Update _KNOWN_CODE in "
            "inference/pipeline/mhr_presort.py for this py-soma-x / PyTorch, or "
            "pass --no-mhr_presorted_sparse with a PyTorch built with CCCL >= 3.0.3."
        )
    log.warning("[EgoHand] %s; keeping the TorchScript forward.", problem)
    return info
