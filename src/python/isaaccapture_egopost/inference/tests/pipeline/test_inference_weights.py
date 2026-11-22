# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-OpenMDW-1.1

"""Inference-weight export, strict key checks, meta-device DINO and background load."""

from __future__ import annotations

import gc
import json
import sys
import threading
import weakref

import pytest
import torch

from inference.pipeline import inference_weights as iw
from inference.pipeline.predictor import EgoHandPredictor, PendingEgoHandPredictor


def _fake_checkpoint(path):
    cfg = {
        "_target_": "egohand.EgoHand",
        "backbone": {"_target_": "x.DinoV3Backbone", "name": "dinov3_vith16plus"},
        "camera_mode": "sam3db_perspective",
        "soma_hand": {"data_root": None},
        "loss_weights": {"keypoints_2d": 0.01},
    }
    generator = torch.Generator().manual_seed(0)
    state_dict = {
        "backbone.encoder.w": torch.randn(4, 3, generator=generator),
        "hand_head.b": torch.randn(5, generator=generator).to(torch.bfloat16),
        "hand_head.idx": torch.arange(6, dtype=torch.int64),
    }
    # A non-contiguous view must still round-trip bit-for-bit.
    state_dict["hand_head.t"] = torch.randn(3, 4, generator=generator).t()
    torch.save(
        {
            "hyper_parameters": {"cfg": cfg},
            "state_dict": state_dict,
            "optimizer_states": [{"state": {0: torch.ones(1000)}}],
            "egohand_camera_depth_mode": "s",
            "egohand_camera_log_value_bounds": (-10.0, 10.0),
            "global_step": 60000,
        },
        path,
    )
    return cfg, state_dict


def test_export_round_trip_is_bitwise_and_drops_training_state(tmp_path):
    ckpt = tmp_path / "model.ckpt"
    cfg, state_dict = _fake_checkpoint(ckpt)
    out = tmp_path / "export"

    config = iw.export_inference_weights(ckpt, out)

    assert sorted(p.name for p in out.iterdir()) == ["config.json", "model.safetensors"]
    assert config["format"] == iw.EXPORT_FORMAT
    assert config["source"]["global_step"] == 60000
    assert len(config["source"]["sha256"]) == 64
    assert iw.is_inference_export(out)
    assert iw.is_inference_export(out / iw.WEIGHTS_FILENAME)
    assert not iw.is_inference_export(ckpt)

    source = iw.read_checkpoint(out)
    assert source.cfg == cfg
    assert source.checkpoint_fields == {
        "egohand_camera_depth_mode": "s",
        "egohand_camera_log_value_bounds": [-10.0, 10.0],
    }
    tensors = source.load_state_dict()
    assert tensors.keys() == state_dict.keys()
    for key, value in state_dict.items():
        assert tensors[key].dtype == value.dtype
        assert torch.equal(tensors[key], value)
    view = source.validation_view()
    assert view["hyper_parameters"]["cfg"] == cfg
    assert view["egohand_camera_depth_mode"] == "s"
    # The training checkpoint reads the same cfg/fields through the same API.
    reference = iw.read_checkpoint(ckpt)
    assert reference.cfg == cfg
    assert reference.checkpoint_fields == source.checkpoint_fields


def test_export_rejects_unknown_format(tmp_path):
    ckpt = tmp_path / "model.ckpt"
    _fake_checkpoint(ckpt)
    out = tmp_path / "export"
    iw.export_inference_weights(ckpt, out, source_sha256=False)
    config = json.loads((out / iw.CONFIG_FILENAME).read_text())
    config["format"] = "something-else/9"
    (out / iw.CONFIG_FILENAME).write_text(json.dumps(config))
    with pytest.raises(ValueError, match="unsupported format"):
        iw.read_checkpoint(out)


def test_strict_key_check_allows_only_asset_derived_soma_layers():
    iw.check_state_dict_keys(
        ["soma_hand.left.identity_model._mhr_model.module.mesh.faces"], []
    )
    with pytest.raises(RuntimeError, match="1 missing key"):
        iw.check_state_dict_keys(["soma_hand.new_learned_scale"], [])
    with pytest.raises(RuntimeError, match="1 missing key"):
        iw.check_state_dict_keys(["hand_head.decpose.weight"], [])
    with pytest.raises(RuntimeError, match=r"1 unexpected key.*ignore_unexpected_keys"):
        iw.check_state_dict_keys([], ["kp2d_head.extra"])


def test_ignore_unexpected_keys_still_rejects_missing_keys(caplog):
    iw.check_state_dict_keys([], ["ema.hand_head.weight"], ignore_unexpected=True)
    assert "ignoring 1 checkpoint key(s)" in caplog.text
    with pytest.raises(RuntimeError, match="1 missing key"):
        iw.check_state_dict_keys(
            ["hand_head.decpose.weight"], ["ema.x"], ignore_unexpected=True
        )


def test_dino_meta_init_allocates_no_storage_and_materializes_on_demand():
    pytest.importorskip("dinov3")
    from egohand.models.backbones.dinov3 import DinoV3Backbone

    backbone = DinoV3Backbone(meta_init=True)
    tensors = list(backbone.encoder.state_dict().values())
    assert tensors and all(t.is_meta for t in tensors)
    backbone.encoder.to_empty(device="cpu")
    assert not any(t.is_meta for t in backbone.encoder.state_dict().values())


def test_pending_predictor_serves_settings_without_waiting_and_forwards_after_load():
    release = threading.Event()

    class _Loaded:
        model = "loaded-model"

        def run(self):
            return "ran"

    def load():
        release.wait(5)
        return _Loaded()

    pending = PendingEgoHandPredictor(
        load,
        batch_size=7,
        device="cpu",
        dtype=torch.float32,
        smooth=True,
        camera_mode="sam3db_perspective",
    )
    # Construction-time settings never block on the load.
    assert pending.batch_size == 7
    assert pending.smooth is True
    assert pending.camera_mode == "sam3db_perspective"
    assert pending.load_seconds is None
    release.set()
    assert pending.run() == "ran"
    assert pending.model == "loaded-model"
    assert pending.load_seconds is not None


def test_pending_predictor_reraises_load_errors_on_use():
    def load():
        raise RuntimeError("bad checkpoint")

    pending = PendingEgoHandPredictor(
        load,
        batch_size=1,
        device="cpu",
        dtype=torch.float32,
        smooth=False,
        camera_mode="weak_perspective",
    )
    with pytest.raises(RuntimeError, match="bad checkpoint"):
        pending.run()


def test_load_in_background_rejects_unsupported_target_immediately(tmp_path):
    ckpt = tmp_path / "model.ckpt"
    torch.save(
        {
            "hyper_parameters": {"cfg": {"_target_": "gep.gep_hand2.GEPHand2"}},
            "state_dict": {},
        },
        ckpt,
    )
    with pytest.raises(ValueError, match="not supported"):
        EgoHandPredictor.load_in_background(ckpt, tmp_path, batch_size=1, device="cpu")


def test_background_model_load_is_default_and_can_be_disabled(monkeypatch):
    from inference.scripts.run_egohand_inference_pipeline import parse_args

    argv = ["x.py", "--input", "in.mp4", "--output_dir", "out"]
    monkeypatch.setattr(sys, "argv", argv)
    assert parse_args().background_model_load is True
    monkeypatch.setattr(sys, "argv", argv + ["--no-background_model_load"])
    assert parse_args().background_model_load is False


def test_pending_predictor_reports_key_check_failure_before_loading():
    loaded = threading.Event()

    def validate():
        raise RuntimeError("checkpoint does not match: 3 missing key(s)")

    def load():  # pragma: no cover - must not run
        loaded.set()

    pending = PendingEgoHandPredictor(
        load,
        validate=validate,
        batch_size=1,
        device="cpu",
        dtype=torch.float32,
        smooth=False,
        camera_mode="weak_perspective",
    )
    with pytest.raises(RuntimeError, match="3 missing key"):
        pending.wait_until_validated()
    assert not loaded.is_set()


def test_expected_keys_match_a_real_model_without_soma():
    pytest.importorskip("dinov3")
    pytest.importorskip("soma")
    from omegaconf import OmegaConf

    cfg = OmegaConf.create(
        {
            "_target_": "egohand.EgoHand",
            "backbone": {
                "_target_": "egohand.models.backbones.dinov3.DinoV3Backbone",
                "name": "dinov3_vith16plus",
                "drop_path_rate": 0.1,
            },
            "backbone_input_crop_w": None,
            "soma_hand_head": {
                "_target_": "egohand.models.heads.soma_hand_head_v2.SOMAHandTransformerDecoderHeadV2",
                "num_finger_joints": 24,
                "num_identity_coeffs": 5,
                "num_scale_params": 27,
                "transformer_dim": 1024,
                "transformer_depth": 6,
                "transformer_heads": 8,
                "transformer_dim_head": 64,
                "transformer_mlp_dim": 1024,
                "transformer_context_dim": 1280,
            },
            "soma_hand": {"data_root": None, "identity_model_type": "mhr"},
            "camera_mode": "sam3db_perspective",
            "cliff_condition": True,
        }
    )
    keys = iw.expected_model_keys(cfg)
    assert keys and not any(k.startswith("soma_hand.") for k in keys)
    assert sum(k.startswith("backbone.") for k in keys) == 552


def test_export_verify_handles_scalar_and_strided_tensors(tmp_path):
    from inference.scripts.export_inference_weights import _verify

    ckpt = tmp_path / "model.ckpt"
    torch.save(
        {
            "hyper_parameters": {"cfg": {"_target_": "egohand.EgoHand"}},
            "state_dict": {
                "a.scale": torch.tensor(1.5),  # 0-d
                "a.t": torch.arange(12.0).reshape(3, 4).t(),  # last dim not contiguous
                "a.nan": torch.tensor([float("nan"), 1.0]),
            },
        },
        ckpt,
    )
    iw.export_inference_weights(ckpt, tmp_path / "export", source_sha256=False)
    _verify(ckpt, tmp_path / "export", None)


def test_load_in_background_imports_usd_before_starting_the_worker(
    tmp_path, monkeypatch
):
    """OpenUSD dlopen + GIL vs. loader lock: import it while only one thread runs."""
    from inference.pipeline import predictor as predictor_module

    ckpt = tmp_path / "model.ckpt"
    _fake_checkpoint(ckpt)
    events = []
    real_start = threading.Thread.start

    def start(thread):
        if thread.name == "egohand-model-load":
            events.append("worker start")
        return real_start(thread)

    monkeypatch.setattr(threading.Thread, "start", start)
    monkeypatch.setattr(
        predictor_module, "preload_usd_libraries", lambda: events.append("usd import")
    )
    monkeypatch.setattr(
        predictor_module, "check_checkpoint_keys", lambda source, cfg: None
    )
    monkeypatch.setattr(
        EgoHandPredictor, "from_checkpoint", classmethod(lambda cls, **kw: None)
    )
    pending = EgoHandPredictor.load_in_background(
        ckpt, tmp_path, batch_size=1, device="cpu"
    )
    pending.wait_until_validated()
    pending._done.wait()
    assert events == ["usd import", "worker start"]


def test_load_in_background_passes_on_every_from_checkpoint_argument(
    tmp_path, monkeypatch
):
    """Arguments added to from_checkpoint later reach it through the background
    path too, and a finished load is not kept alive by the module."""
    from inference.pipeline import predictor as predictor_module

    ckpt = tmp_path / "model.ckpt"
    _fake_checkpoint(ckpt)
    seen = {}

    class Loaded:
        pass

    def from_checkpoint(cls, **kwargs):
        seen.update(kwargs)
        return Loaded()

    monkeypatch.setattr(
        predictor_module, "check_checkpoint_keys", lambda source, cfg: None
    )
    monkeypatch.setattr(
        EgoHandPredictor, "from_checkpoint", classmethod(from_checkpoint)
    )
    pending = EgoHandPredictor.load_in_background(
        ckpt,
        tmp_path,
        batch_size=4,
        device="cpu",
        use_perspective_cropping=True,
        some_future_option=7,
    )
    loaded = pending.result()
    assert seen["batch_size"] == 4 and seen["device"] == "cpu"
    assert seen["use_perspective_cropping"] is True
    assert seen["some_future_option"] == 7
    pending._thread.join()
    assert pending not in predictor_module._PENDING_LOADS
    ref = weakref.ref(loaded)
    del pending, loaded
    gc.collect()
    assert ref() is None


def test_preload_usd_libraries_imports_what_soma_uses():
    pytest.importorskip("pxr")
    from inference.pipeline.predictor import _USD_MODULES, preload_usd_libraries

    preload_usd_libraries()
    assert all(name in sys.modules for name in _USD_MODULES)
