# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-OpenMDW-1.1

"""--soma_batch_size bookkeeping: grouped SOMA calls give the per-batch results.

The fake SOMA layer and mirror are row-wise and exact, so any difference
between grouped and per-batch results is a grouping, padding or ordering bug.
"""

import numpy as np
import pytest
import torch

from inference.pipeline.predictor import EgoHandPredictor

_K = np.array([[100.0, 0.0, 32.0], [0.0, 100.0, 32.0], [0.0, 0.0, 1.0]], np.float32)


class _FakeSomaHand:
    def __init__(self):
        self.calls = []

    def __call__(
        self, poses, identity_coeffs, hand_type, pose2rot, global_scale, scale_params
    ):
        self.calls.append(poses.shape[0])
        joints = (
            poses[:, :, :, 0] * global_scale[:, None, None]
            + identity_coeffs[:, :1, None]
        )
        vertices = joints[:, :3] + scale_params[:, :1, None]
        return {"joints": joints, "vertices": vertices}


class _FakeSplitModel:
    camera_mode = "sam3db_perspective"
    kp2d_min_depth = 1e-4

    def __init__(self):
        self.soma_hand = _FakeSomaHand()

    def forward_step_head(self, batch, train=False):
        value = batch["img"][:, 0, 0, 0]
        batch_size = value.shape[0]
        poses = (
            torch.eye(3).expand(batch_size, 25, 3, 3)
            * (1.0 + value)[:, None, None, None]
        )
        scale_params = torch.stack(
            [1.0 + 0.1 * value, value, 2.0 * value], dim=1
        ).float()
        params = {
            "wrist_orient": poses[:, :1].clone(),
            "hand_pose": poses[:, 1:].clone(),
            "identity_coeffs": torch.stack([value] * 5, dim=1),
            "scale_params": scale_params,
        }
        return {
            "batch_size": batch_size,
            "soma_inputs": {
                "poses": poses.clone(),
                "identity_coeffs": params["identity_coeffs"].clone(),
                "global_scale": scale_params[:, 0].clone(),
                "scale_params": scale_params[:, 1:].clone(),
            },
            "params": params,
            "cam_t": torch.stack([value, -value, 2.0 + value], dim=1),
        }

    def forward_step(self, batch, train=False):
        head = self.forward_step_head(batch, train=train)
        out = self.soma_hand(hand_type="left", pose2rot=False, **head["soma_inputs"])
        return self.forward_step_decode(head, out["joints"], out["vertices"])

    def forward_step_decode(self, head, joints, vertices):
        batch_size = head["batch_size"]
        pred_keypoints_2d = joints[..., :2] * 0.01
        pred_keypoints_2d[:, 0] = float("nan")
        valid = torch.ones(batch_size, 25, dtype=torch.bool)
        valid[:, 0] = False
        return {
            "pred_vertices": vertices,
            "pred_keypoints_3d": joints,
            "pred_cam_t": head["cam_t"],
            "pred_keypoints_2d": pred_keypoints_2d,
            "pred_projection_valid": valid,
            "pred_hand_params": head["params"],
            "pred_kp2d_sigma": joints[..., :2] * 0.5,
        }


def _fake_mirror(
    model,
    poses,
    transl,
    identity_coeffs,
    global_scale,
    K_fullimg,
    bbx_xys,
    scale_params=None,
    pose_mirror=None,
):
    _fake_mirror.calls.append(poses.shape[0])
    joints = poses[:, :, :, 1] * global_scale[:, None, None] - transl[:, None]
    valid = torch.ones(poses.shape[0], 25, dtype=torch.bool)
    valid[:, 1] = False
    return {
        "pred_vertices": joints[:, :3] + scale_params[:, :1, None],
        "transl": transl * 2.0,
        "pred_keypoints_3d": joints,
        "pred_keypoints_2d": joints[..., :2] / bbx_xys[:, None, 2:]
        + K_fullimg[:, None, 0, :2],
        "pred_projection_valid": valid,
        "poses": poses + identity_coeffs[:, :1, None, None],
    }


def _inputs(hands):
    return [
        {
            "image_id": f"frame-{i // 2}",
            "real_hand_side": hand,
            "crop_tensor": torch.full((3, 8, 8), float(i + 1)),
            "model_bbx_xys": np.array([32.0, 32.0, 10.0], np.float32),
            "model_K": _K.copy(),
            "model_img_size": np.array([64.0, 64.0], np.float32),
            "output_bbx_xys": np.array([32.0 + i, 30.0, 10.0 + i], np.float32),
            "output_K": _K + i,
        }
        for i, hand in enumerate(hands)
    ]


def _predictor(soma_batch_size):
    predictor = EgoHandPredictor.__new__(EgoHandPredictor)
    predictor.batch_size = 3
    predictor.device = "cpu"
    predictor.dtype = torch.float32
    predictor.model = _FakeSplitModel()
    predictor._pose_mirror = None
    predictor.soma_batch_size = soma_batch_size
    return predictor


def _run(soma_batch_size, hands, monkeypatch):
    _fake_mirror.calls = []
    for module in ("predictor", "soma_batching"):
        monkeypatch.setattr(
            f"inference.pipeline.{module}.mirror_left_prediction_to_right",
            _fake_mirror,
        )
    predictor = _predictor(soma_batch_size)
    results = predictor._run_imagebased_predictions(_inputs(hands))
    return results, predictor.model.soma_hand.calls, list(_fake_mirror.calls)


@pytest.mark.parametrize("soma_batch_size", [3, 6, 7, 9])
def test_grouped_soma_matches_per_batch(soma_batch_size, monkeypatch):
    # Batches of 3; a 2-crop tail batch; a batch with no right hand.
    hands = ["left", "right", "right", "left", "left", "left", "right", "left"]
    hands += ["right", "right", "left"]
    reference, ref_left_calls, ref_mirror_calls = _run(0, hands, monkeypatch)
    grouped, left_calls, mirror_calls = _run(soma_batch_size, hands, monkeypatch)

    assert ref_left_calls == [3, 3, 3, 2]
    assert ref_mirror_calls == [2, 2, 1]
    assert set(left_calls) == {soma_batch_size}
    assert len(left_calls) == -(-4 // (soma_batch_size // 3))
    assert set(mirror_calls) == {soma_batch_size}
    assert len(grouped) == len(reference) == len(hands)
    for got, want in zip(grouped, reference):
        assert got.keys() == want.keys()
        for key, value in want.items():
            if isinstance(value, np.ndarray):
                np.testing.assert_array_equal(got[key], value, err_msg=key)
            else:
                assert got[key] == value, key


def test_soma_batch_size_below_batch_size_is_rejected():
    with pytest.raises(ValueError, match="soma_batch_size"):
        EgoHandPredictor(
            model=torch.nn.Identity(), batch_size=32, device="cpu", soma_batch_size=16
        )
