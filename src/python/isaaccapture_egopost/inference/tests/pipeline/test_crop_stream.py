# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-OpenMDW-1.1

"""Crops streamed to the forward pass in batches of valid crops.

The streamed path must give the same batches as building every crop first and
slicing the valid ones into ``batch_size`` chunks (failed crops are dropped
before batching), with bounded look-ahead and errors raised to the caller.
"""

from __future__ import annotations

import random
import threading
import time

import numpy as np
import pytest
import torch

from inference.camera.camera_models import load_camera_calibration
from inference.pipeline import predictor as predictor_module
from inference.pipeline.crop_stream import StreamedCropBatches
from inference.pipeline.predictor import EgoHandPredictor


def _jobs(n):
    return [(i, f"frame{i // 2:04d}", ("left", "right")[i % 2]) for i in range(n)]


def _builder(fail=(), raise_at=None, jitter=0.0, calls=None):
    """Crop stub: a (3, 4, 4) non-contiguous float32 view filled with the job index."""
    lock = threading.Lock()

    def build(job):
        i, image_id, hand = job
        if calls is not None:
            with lock:
                calls.append(i)
        if jitter:
            time.sleep(random.random() * jitter)
        if raise_at is not None and i == raise_at:
            raise RuntimeError(f"crop {i} exploded")
        if i in fail:
            return image_id, hand, None
        hwc = np.full((4, 4, 3), float(i), dtype=np.float32)
        return (
            image_id,
            hand,
            {
                "i": i,
                "image_id": image_id,
                "real_hand_side": hand,
                "crop_tensor": torch.from_numpy(hwc.transpose(2, 0, 1)),
            },
        )

    return build


@pytest.mark.parametrize("batch_size, workers", [(4, 3), (1, 8), (7, 2)])
def test_batches_are_valid_crops_in_job_order(batch_size, workers):
    fail = {0, 3, 4, 11, 22}
    seen = []
    stream = StreamedCropBatches(
        _jobs(25),
        _builder(fail=fail, jitter=0.002),
        batch_size=batch_size,
        device="cpu",
        workers=workers,
        on_prepared=lambda image_id, hand, d: seen.append((image_id, hand, d["i"])),
    )
    batches = []
    with stream:
        for staged in stream:
            ids = [d["i"] for d in staged.hand_data]
            assert all("crop_tensor" not in d for d in staged.hand_data)
            images = staged.images("cpu")
            assert images.shape == (len(ids), 3, 4, 4)
            assert images.dtype == torch.float32
            assert [float(x) for x in images[:, 0, 0, 0]] == [float(i) for i in ids]
            batches.append(ids)
            staged.release()
    valid = [i for i in range(25) if i not in fail]
    expected = [valid[s : s + batch_size] for s in range(0, len(valid), batch_size)]
    assert batches == expected
    assert [i for _, _, i in seen] == valid
    assert stream.num_valid == len(valid) and stream.num_failed == len(fail)


def test_all_crops_failed_gives_no_batch():
    stream = StreamedCropBatches(
        _jobs(5), _builder(fail=set(range(5))), batch_size=2, device="cpu", workers=2
    )
    with stream:
        assert list(stream) == []


def test_look_ahead_is_bounded():
    calls = []
    batch_size = 4
    stream = StreamedCropBatches(
        _jobs(200),
        _builder(calls=calls),
        batch_size=batch_size,
        device="cpu",
        workers=4,
        prefetch_batches=2,
    )
    with stream:
        iterator = iter(stream)
        first = next(iterator)
        time.sleep(0.5)  # the forward of the first batch, slots never released
        started = len(calls)
        first.release()
        rest = sum(len(s.hand_data) for s in _drain(iterator))
    # 2 staged slots + 1 filled batch waiting for a slot + 2 batches of jobs in flight
    assert 2 * batch_size < started <= 5 * batch_size
    assert len(first.hand_data) + rest == 200


def _drain(iterator):
    for staged in iterator:
        yield staged
        staged.release()


def test_crop_error_is_raised_in_the_consumer_and_threads_stop():
    stream = StreamedCropBatches(
        _jobs(40), _builder(raise_at=9), batch_size=4, device="cpu", workers=3
    )
    with pytest.raises(RuntimeError, match="crop 9 exploded"):
        with stream:
            for staged in stream:
                staged.release()
    assert not stream._thread.is_alive()


def test_early_exit_stops_threads():
    stream = StreamedCropBatches(
        _jobs(400), _builder(), batch_size=4, device="cpu", workers=4
    )
    with stream:
        for _ in stream:
            break  # e.g. the forward raised; the batch is never released
    assert not stream._thread.is_alive()


@pytest.mark.parametrize("value", [0, -1, True, 2.5])
def test_invalid_sizes_are_rejected(value):
    with pytest.raises(ValueError):
        StreamedCropBatches([], _builder(), batch_size=value, device="cpu", workers=1)


# ---------------------------------------------------------------------------
# Predictor: streamed path vs build-everything-first path
# ---------------------------------------------------------------------------


class _RecordingModel:
    """Outputs depend on each crop, so any misalignment changes the results."""

    camera_mode = "sam3db_perspective"
    kp2d_min_depth = 1e-4

    def __init__(self):
        self.batches = []

    def forward_step(self, batch, train=False):
        img = batch["img"]
        n = img.shape[0]
        signature = img.float().mean(dim=(1, 2, 3)).cpu()
        self.batches.append(signature.tolist())
        signature = signature.to(img.device)
        identity = torch.eye(3, device=img.device).reshape(1, 1, 3, 3)
        scale_params = torch.zeros(n, 27, device=img.device)
        scale_params[:, 0] = 1.0 + signature
        cam_t = torch.stack(
            [signature, -signature, 2.0 + signature.abs()], dim=-1
        ).float()
        return {
            "pred_vertices": signature.reshape(n, 1, 1).expand(n, 3, 3).contiguous(),
            "pred_cam_t": cam_t,
            "pred_cam": torch.tensor([1.0, 0.0, 0.0], device=img.device).expand(n, -1),
            "pred_keypoints_3d": 0.01
            * signature.reshape(n, 1, 1).expand(n, 25, 3).contiguous(),
            "pred_keypoints_2d": torch.zeros(n, 25, 2, device=img.device),
            "pred_projection_valid": torch.ones(
                n, 25, dtype=torch.bool, device=img.device
            ),
            "pred_hand_params": {
                "wrist_orient": identity.expand(n, 1, 3, 3),
                "hand_pose": identity.expand(n, 24, 3, 3),
                "identity_coeffs": signature.reshape(n, 1).expand(n, 5).contiguous(),
                "scale_params": scale_params,
            },
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
    n = poses.shape[0]
    return {
        "pred_vertices": -identity_coeffs[:, :1]
        .reshape(n, 1, 1)
        .expand(n, 3, 3)
        .contiguous(),
        "transl": transl,
        "pred_keypoints_3d": torch.zeros(n, 25, 3, device=poses.device),
        "pred_keypoints_2d": torch.zeros(n, 25, 2, device=poses.device),
        "pred_projection_valid": torch.ones(
            n, 25, dtype=torch.bool, device=poses.device
        ),
        "poses": poses,
    }


def _perspective_predictor(device, batch_size, stream_crops):
    predictor = EgoHandPredictor.__new__(EgoHandPredictor)
    predictor.batch_size = batch_size
    predictor.device = device
    predictor.dtype = torch.float32
    predictor.smooth = False
    predictor.perf_reporter = None
    predictor.model = _RecordingModel()
    predictor._pose_mirror = None
    predictor._use_perspective_cropping = True
    predictor._model_wrist_orients = {"left": np.eye(3), "right": np.eye(3)}
    predictor.stream_crops = stream_crops
    return predictor


# A 1408 x 1408 pinhole camera, and boxes that make perspective_crop
# raise: "ROI perimeter reaches behind the virtual pinhole camera" and
# "Cannot undistort this bounding box" (NaN centre).
_PINHOLE_CAMERA = {
    "model": "PINHOLE",
    "image_width": 1408,
    "image_height": 1408,
    "camera_matrix": {
        "matrix_3x3": [
            [608.8129272460938, 0.0, 704.0],
            [0.0, 608.8129272460938, 704.0],
            [0.0, 0.0, 1.0],
        ]
    },
    "distortion_coefficients": [],
}
_BEHIND_CAMERA_BOX = [1.0e5, 704.0, 2.0e5]
_NAN_BOX = [float("nan"), 704.0, 300.0]


def _devices():
    devices = ["cpu"]
    if torch.cuda.is_available():
        devices.append("cuda")
    return devices


@pytest.mark.parametrize("device", _devices())
@pytest.mark.parametrize("batch_size", [3, 32])
def test_run_streams_the_same_batches_and_skips_failed_crops(
    monkeypatch, device, batch_size
):
    monkeypatch.setattr(
        predictor_module, "mirror_left_prediction_to_right", _fake_mirror
    )
    camera = load_camera_calibration(_PINHOLE_CAMERA)
    rng = np.random.default_rng(0)
    image_ids = [f"frame{i:04d}" for i in range(9)]
    images = [rng.integers(0, 255, (1408, 1408, 3), dtype=np.uint8) for _ in image_ids]
    bboxes = {
        image_id: {
            "left": np.array([500.0 + 7 * i, 650.0, 260.0 + 3 * i], dtype=np.float32),
            "right": np.array([900.0 - 5 * i, 700.0, 240.0 + 2 * i], dtype=np.float32),
        }
        for i, image_id in enumerate(image_ids)
    }
    bboxes["frame0002"]["right"] = np.array(_BEHIND_CAMERA_BOX, dtype=np.float32)
    bboxes["frame0005"]["left"] = np.array(_NAN_BOX, dtype=np.float32)
    del bboxes["frame0007"]["left"]  # no box: never becomes a crop job
    intrinsics = dict.fromkeys(image_ids, camera)

    outputs = {}
    for stream_crops in (False, True):
        predictor = _perspective_predictor(device, batch_size, stream_crops)
        results = predictor.run(
            images,
            image_ids,
            {k: dict(v) for k, v in bboxes.items()},
            ["left", "right"],
            intrinsics_per_image=intrinsics,
        )
        outputs[stream_crops] = (results, predictor.model.batches)

    (ref, ref_batches), (got, got_batches) = outputs[False], outputs[True]
    assert got_batches == ref_batches  # same batch boundaries and crop images
    assert sum(len(b) for b in got_batches) == 15  # 18 jobs - 1 missing box - 2 failed
    assert "frame0002" not in got["right"] and "frame0005" not in got["left"]
    assert "frame0007" not in got["left"]
    for hand in ("left", "right"):
        assert list(got[hand]) == list(ref[hand])
        for image_id, result in ref[hand].items():
            for key, value in result.items():
                other = got[hand][image_id][key]
                if isinstance(value, np.ndarray):
                    assert value.dtype == other.dtype
                    assert value.tobytes() == other.tobytes(), (hand, image_id, key)
                else:
                    assert value == other or (value is None and other is None)


def test_streamed_forward_matches_list_forward_cpu(monkeypatch):
    monkeypatch.setattr(
        predictor_module, "mirror_left_prediction_to_right", _fake_mirror
    )
    build = _builder(fail={2, 5})

    def prepared(job):
        image_id, hand, d = build(job)
        if d is not None:
            d.update(
                model_bbx_xys=np.array([32.0, 32.0, 10.0], np.float32),
                model_K=np.eye(3, dtype=np.float32),
                model_img_size=np.array([64.0, 64.0], np.float32),
                output_bbx_xys=np.array([32.0, 32.0, 10.0], np.float32),
                output_K=np.eye(3, dtype=np.float32),
            )
        return image_id, hand, d

    jobs = _jobs(13)
    as_list = [d for d in (prepared(j)[2] for j in jobs) if d is not None]
    a = _perspective_predictor("cpu", 4, True)
    ref = a._run_imagebased_predictions(as_list, frame_count=7)
    b = _perspective_predictor("cpu", 4, True)
    with StreamedCropBatches(
        jobs, prepared, batch_size=4, device="cpu", workers=3
    ) as s:
        got = b._run_imagebased_predictions(s, frame_count=7)
    assert a.model.batches == b.model.batches
    assert len(ref) == len(got) == 11
    for r, g in zip(ref, got):
        assert r.keys() == g.keys()
        for key in r:
            if isinstance(r[key], np.ndarray):
                assert r[key].tobytes() == g[key].tobytes()


# ---------------------------------------------------------------------------
# CUDA: pinned slots, copy stream, slot reuse ordering
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_cuda_slots_deliver_exact_images_while_the_gpu_is_still_reading():
    rng = np.random.default_rng(1)
    crops = {
        i: torch.from_numpy(
            rng.standard_normal((256, 256, 3), dtype=np.float32).transpose(2, 0, 1)
        )
        for i in range(30)
    }
    expected = {
        i: c.clone() for i, c in crops.items()
    }  # crops are popped by the stream

    def build(job):
        i, image_id, hand = job
        return image_id, hand, {"i": i, "crop_tensor": crops[i]}

    copies, pointers = [], []
    stream = StreamedCropBatches(
        _jobs(30), build, batch_size=4, device="cuda", workers=4
    )
    with stream:
        for staged in stream:
            ids = [d["i"] for d in staged.hand_data]
            images = staged.images("cuda")
            pointers.append(images.data_ptr())
            # Keep the GPU busy so the next uploads are queued while this batch
            # is still unread; they must wait for this batch's "consumed" event.
            torch.cuda._sleep(20_000_000)
            copies.append((ids, images.clone()))
            staged.release()
    torch.cuda.synchronize()
    assert [ids for ids, _ in copies] == [
        list(range(s, min(s + 4, 30))) for s in range(0, 30, 4)
    ]
    for ids, copy in copies:
        assert torch.equal(copy.cpu(), torch.stack([expected[i] for i in ids]))
    assert len(set(pointers)) == 2  # two device slots, alternating
    assert pointers[0] != pointers[1] and pointers[0] == pointers[2]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_predictor_reuses_one_copy_stream_per_device():
    predictor = EgoHandPredictor.__new__(EgoHandPredictor)
    predictor.device = "cuda"
    first = predictor._crop_copy_stream()
    assert first is predictor._crop_copy_stream()
    predictor.device = "cpu"
    assert predictor._crop_copy_stream() is None
