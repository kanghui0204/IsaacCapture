# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-OpenMDW-1.1

"""Inference-weight export, strict key checks and meta-device DINO."""

from __future__ import annotations

import json

import pytest
import torch

from inference.pipeline import inference_weights as iw


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
