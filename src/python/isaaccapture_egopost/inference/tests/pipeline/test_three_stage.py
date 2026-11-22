# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-OpenMDW-1.1

"""Three-stage pipeline hand-off: same crops, batches, order and results as
``EgoHandPredictor.run``; failed crops dropped as before; one video's failure
does not touch the others. Fake model and detector; CPU except the CUDA
stream hand-off test."""

import os
import threading

import numpy as np
import pytest
import torch

pytest.importorskip("cv2")

from inference.camera.intrinsics import resolve_K  # noqa: E402
from inference.camera.perspective_crop import perspective_crop  # noqa: E402
from inference.pipeline import three_stage as ts  # noqa: E402
from inference.pipeline.predictor import EgoHandPredictor  # noqa: E402
from inference.preprocessing.crop import (  # noqa: E402
    IMAGENET_MEAN,
    IMAGENET_STD,
    IMG_SIZE,
    crop_and_preprocess,
    normalize_crop,
)
from inference.preprocessing.mirroring import (  # noqa: E402
    mirror_full_image_camera_horizontal,
    prepare_right_hand_model_input,
)
from inference.tests.pipeline.test_soma_batching import (  # noqa: E402
    _fake_mirror,
    _FakeSplitModel,
)

H, W = 72, 96
_K = np.array([[80.0, 0.0, W / 2], [0.0, 80.0, H / 2], [0.0, 0.0, 1.0]], np.float32)
_PINHOLE_JSON = {
    "model": "PINHOLE",
    "image_width": W,
    "image_height": H,
    "camera_matrix": {"matrix_3x3": _K.tolist()},
    "distortion_coefficients": [],
}


def _camera():
    camera, _ = resolve_K(W, H, _PINHOLE_JSON, perspective_cropping=True)
    return camera


# ----------------------------------------------------------------- video input


def test_stage_one_decodes_through_the_video_backend(monkeypatch):
    """Stage 1 reads videos through inference.video_io (the backend of the
    runtime profile), like run_egohand_inference_pipeline.py."""
    import inference.video_io as video_io

    frames = [np.full((H, W, 3), i, np.uint8) for i in range(5)]
    calls = []

    def fake_read(path, batch_size, max_frames=None):
        calls.append((path, batch_size, max_frames))
        stop = len(frames) if max_frames is None else min(max_frames, len(frames))
        for start in range(0, stop, batch_size):
            yield [(i, frames[i]) for i in range(start, min(start + batch_size, stop))]

    monkeypatch.setattr(video_io, "read_video_frames_batched", fake_read)
    monkeypatch.setattr(
        video_io,
        "get_video_meta",
        lambda path: {"fps": 0.0, "width": W, "height": H, "num_frames": 5},
    )
    decoded = ts.decode_all_frames("clip.mp4")
    assert [int(f[0, 0, 0]) for f in decoded] == list(range(5))
    assert ts.video_meta("clip.mp4")["num_frames"] == 5
    ts.check_video_decodable("clip.mp4", {"width": W, "height": H})
    assert calls[-1] == ("clip.mp4", 1, 1)
    frames.clear()
    with pytest.raises(RuntimeError, match="no decodable video stream"):
        ts.check_video_decodable("clip.mp4", {"width": W, "height": H})


# ----------------------------------------------------------------- fakes


class _RecordingModel(_FakeSplitModel):
    """Records every batch the head sees; the fake outputs depend on many pixels."""

    def __init__(self):
        super().__init__()
        self.images = []
        self.inputs = []
        self.fail_head = None  # callable(batch) -> bool

    def forward_step_head(self, batch, train=False):
        img = batch["img"]
        if self.fail_head is not None and self.fail_head(batch):
            raise ValueError("injected head failure")
        if "bbx_xys" in batch:
            self.images.append(img.clone())
            self.inputs.append(
                {k: batch[k].clone() for k in ("bbx_xys", "K_fullimg", "img_size")}
            )
        probe = img.reshape(img.shape[0], -1)[:, ::101].sum(dim=1) * 1e-4
        fake = probe[:, None, None, None].contiguous()
        return super().forward_step_head({"img": fake}, train)


def _predictor(perspective, soma_batch_size=0, batch_size=3):
    predictor = EgoHandPredictor.__new__(EgoHandPredictor)
    predictor.batch_size = batch_size
    predictor.soma_batch_size = soma_batch_size
    predictor.device = "cpu"
    predictor.dtype = torch.float32
    predictor.smooth = False
    predictor.perf_reporter = None
    predictor.model = _RecordingModel()
    predictor._pose_mirror = None
    predictor._use_perspective_cropping = perspective
    predictor._model_wrist_orients = (
        {"left": np.eye(3), "right": np.eye(3)} if perspective else None
    )
    return predictor


def _frames(clip, count):
    rng = np.random.default_rng(1000 + clip)
    return [rng.integers(0, 256, (H, W, 3), dtype=np.uint8) for _ in range(count)]


def _boxes(clip, frame_index, frame, hands, perspective):
    """Deterministic boxes; some frames miss a hand; some perspective crops fail."""
    boxes = {}
    shift = float(frame[0, 0, 0] % 7)
    if "left" in hands and frame_index % 4 != 3:
        boxes["left"] = np.array([30.0 + shift, 35.0, 24.0 + clip], np.float32)
    if "right" in hands and frame_index % 3 != 1:
        if perspective and frame_index % 5 == 2:
            # Perimeter rays reach behind the virtual camera: perspective_crop fails.
            boxes["right"] = np.array([0.9 * W, 0.5 * H, 1.0e6], np.float32)
        else:
            boxes["right"] = np.array([60.0, 30.0 + shift, 20.0 + 2 * clip], np.float32)
    return boxes


class _FakeDetector:
    writes_bbox_json = False
    requires_full_clip = False

    def __init__(self, clip_of, hands, perspective, fail_names=()):
        self.clip_of = clip_of  # id(first frame) -> (clip number, name)
        self.hands = hands
        self.perspective = perspective
        self.fail_names = set(fail_names)
        self.calls = []

    def run(self, image_ids, images_bgr, hands, output_paths_per_hand=None, **_):
        clip, name = self.clip_of[id(images_bgr[0])]
        self.calls.append(name)
        if name in self.fail_names:
            raise RuntimeError("injected detector failure")
        return {
            image_id: _boxes(clip, i, frame, hands, self.perspective)
            for i, (image_id, frame) in enumerate(zip(image_ids, images_bgr))
        }


def _setup(tmp_path, perspective, clips, *, soma_batch_size=0, fail_names=()):
    hands = ["left", "right"]
    camera = _camera() if perspective else _K
    requests, frames_of, clip_of = [], {}, {}
    for index, (clip, count, name) in enumerate(clips):
        frames = _frames(clip, count)
        frames_of[name] = (clip, frames)
        clip_of[id(frames[0])] = (clip, name)
        requests.append(
            ts.ClipRequest(
                index=index,
                name=name,
                video=f"/nonexistent/{name}.mp4",
                intrinsics=None,
                out_root=str(tmp_path / "out"),
                frames=frames,
                camera=camera,
                fps=30.0,
            )
        )
    predictor = _predictor(perspective, soma_batch_size)
    config = ts.StageConfig(
        hands=hands,
        use_perspective_cropping=perspective,
        camera_mode=predictor.model.camera_mode,
        batch_size=predictor.batch_size,
        smooth=False,
        crop_workers=3,
        norm_workers=2,
    )
    detector = _FakeDetector(clip_of, hands, perspective, fail_names)
    results = {}

    def writer(clip, results_per_hand, hands_):
        results[clip.request.name] = results_per_hand
        return {h: len(results_per_hand[h]) for h in hands_}

    prepare = ts.PrepareStage(detector, config)
    head = ts.HeadStage(predictor, config)
    soma = ts.SomaStage(predictor, config, writer=writer)
    return requests, frames_of, camera, predictor, prepare, head, soma, results


def _reference(frames_of, camera, perspective, soma_batch_size=0):
    """``EgoHandPredictor.run`` (the single-video path) per clip."""
    out = {}
    for name, (clip, frames) in frames_of.items():
        predictor = _predictor(perspective, soma_batch_size)
        ids = [f"{i:06d}" for i in range(len(frames))]
        boxes = {
            image_id: _boxes(clip, i, frame, ["left", "right"], perspective)
            for i, (image_id, frame) in enumerate(zip(ids, frames))
        }
        results = predictor.run(
            images_bgr=frames,
            image_ids=ids,
            bboxes_per_image=boxes,
            hands=["left", "right"],
            intrinsics_per_image=dict.fromkeys(ids, camera),
        )
        out[name] = (results, predictor.model.images, predictor.model.inputs)
    return out


def _assert_results_equal(got, want):
    assert list(got) == list(want)
    for hand in want:
        assert list(got[hand]) == list(want[hand]), hand
        for image_id, result in want[hand].items():
            other = got[hand][image_id]
            assert list(other) == list(result), (hand, image_id)
            for key, value in result.items():
                if isinstance(value, np.ndarray):
                    assert other[key].dtype == value.dtype, key
                    np.testing.assert_array_equal(other[key], value, err_msg=key)
                else:
                    assert other[key] == value or (
                        other[key] is None and value is None
                    ), key


@pytest.fixture(autouse=True)
def _fake_mirror_patch(monkeypatch):
    _fake_mirror.calls = []
    for module in ("predictor", "soma_batching"):
        monkeypatch.setattr(
            f"inference.pipeline.{module}.mirror_left_prediction_to_right", _fake_mirror
        )


# ----------------------------------------------------------------- crops


@pytest.mark.parametrize("hand", ["left", "right"])
@pytest.mark.parametrize("camera_mode", ["sam3db_perspective", "weak_perspective"])
def test_uint8_pinhole_crop_normalizes_to_original_input(hand, camera_mode):
    frame = _frames(3, 1)[0]
    bbox = np.array([41.3, 30.7, 27.9], np.float32)
    _, _, record = ts.build_crop_record(
        ("000000", hand, frame, bbox, _K),
        use_perspective_cropping=False,
        camera_mode=camera_mode,
    )
    if hand == "left":
        want, want_bbox, want_K = crop_and_preprocess(frame, bbox)[0], bbox, _K
    else:
        want, want_bbox, want_K = prepare_right_hand_model_input(
            frame,
            bbox.copy(),
            _K.copy(),
            calibrated_full_image=camera_mode == "sam3db_perspective",
        )
    got = normalize_crop(record["crop_u8"])
    assert record["crop_u8"].dtype == np.uint8
    assert got.dtype == want.dtype and got.shape == want.shape
    assert torch.equal(got, want)
    np.testing.assert_array_equal(record["model_bbx_xys"], want_bbox)
    np.testing.assert_array_equal(record["model_K"], want_K)


@pytest.mark.parametrize("hand", ["left", "right"])
def test_uint8_perspective_crop_normalizes_to_original_input(hand):
    frame = _frames(4, 1)[0]
    camera = _camera()
    bbox = np.array([30.5, 40.25, 22.0], np.float32)
    _, _, record = ts.build_crop_record(
        ("000000", hand, frame, bbox, camera),
        use_perspective_cropping=True,
        camera_mode="sam3db_perspective",
    )
    # predictor.run's perspective branch, verbatim.
    crop_rgb, K_virtual, bbox_virtual, rotation = perspective_crop(
        frame, bbox, camera, IMG_SIZE
    )
    model_bbx_xys, model_K = bbox_virtual, K_virtual
    if hand == "right":
        crop_rgb = np.ascontiguousarray(crop_rgb[:, ::-1])
        model_bbx_xys, model_K = mirror_full_image_camera_horizontal(
            model_bbx_xys, model_K, crop_rgb.shape[1]
        )
    normalized = (crop_rgb - IMAGENET_MEAN) / IMAGENET_STD
    want = torch.from_numpy(normalized.transpose(2, 0, 1))
    assert torch.equal(normalize_crop(record["crop_u8"]), want)
    np.testing.assert_array_equal(record["model_bbx_xys"], model_bbx_xys)
    np.testing.assert_array_equal(record["model_K"], model_K)
    np.testing.assert_array_equal(record["crop_to_source_camera_rotation"], rotation)


def test_failed_perspective_crop_is_dropped_with_none():
    frame = _frames(5, 1)[0]
    bbox = np.array([0.9 * W, 0.5 * H, 1.0e6], np.float32)
    with pytest.raises(ValueError):
        perspective_crop(frame, bbox, _camera(), IMG_SIZE)
    assert ts.build_crop_record(
        ("000002", "right", frame, bbox, _camera()),
        use_perspective_cropping=True,
        camera_mode="sam3db_perspective",
    ) == ("000002", "right", None)


# ----------------------------------------------------------------- end to end

_CLIPS = [(0, 7, "clip_a"), (1, 1, "clip_b"), (2, 11, "clip_c"), (3, 4, "clip_d")]


@pytest.mark.parametrize("perspective", [False, True])
@pytest.mark.parametrize("soma_batch_size", [0, 6])
@pytest.mark.parametrize("mode", ["serial", "threaded"])
def test_three_stage_equals_predictor_run(tmp_path, perspective, soma_batch_size, mode):
    requests, frames_of, camera, predictor, prepare, head, soma, results = _setup(
        tmp_path, perspective, _CLIPS, soma_batch_size=soma_batch_size
    )
    records = []
    if mode == "serial":
        ts.run_serial(prepare, head, soma, requests, records.append)
    else:
        runner = ts.ThreadedThreeStage(prepare, head, soma, soma_videos=2)
        runner.run(iter(requests), records.append)
        assert runner.fatal is None
    reference = _reference(frames_of, camera, perspective, soma_batch_size)

    assert [r["name"] for r in records] == [name for _, _, name in _CLIPS]
    assert all(r["status"] == "ok" for r in records), records
    # Same forward batches, in the same order, with the same bytes.
    want_images = [img for name in results for img in reference[name][1]]
    want_inputs = [x for name in results for x in reference[name][2]]
    got_images = predictor.model.images
    assert [i.shape[0] for i in got_images] == [i.shape[0] for i in want_images]
    assert any(
        i.shape[0] < predictor.batch_size for i in got_images
    )  # partial last batches
    for got, want in zip(got_images, want_images):
        assert torch.equal(got, want)
    for got, want in zip(predictor.model.inputs, want_inputs):
        for key in want:
            assert torch.equal(got[key], want[key]), key
    for name in results:
        _assert_results_equal(results[name], reference[name][0])
    if perspective:
        failed = sum(r["failed_crops"] for r in records)
        assert failed > 0
        for record in records:
            assert record["crops"] + record["failed_crops"] == record["crop_jobs"]
    # perf.json per clip
    for _, _, name in _CLIPS:
        assert os.path.exists(tmp_path / "out" / name / "perf.json")


def test_failures_stay_with_their_video(tmp_path, monkeypatch):
    clips = [
        (0, 7, "ok_1"),
        (1, 5, "bad_detect"),
        (2, 8, "bad_head"),
        (3, 6, "bad_soma"),
        (4, 9, "ok_2"),
    ]
    requests, frames_of, camera, _, prepare, head, soma, results = _setup(
        tmp_path, False, clips, fail_names=("bad_detect",)
    )
    real_head_run = head.run

    def head_run(clip, emit, crops=None):
        if clip.request.name != "bad_head":
            return real_head_run(clip, emit, crops)
        calls = {"n": 0}

        def emit_then_fail(batch):
            emit(batch)  # stage 3 already has one batch of this clip
            calls["n"] += 1
            raise RuntimeError("injected head failure after one batch")

        return real_head_run(clip, emit_then_fail, crops)

    real_finish = soma.finish

    def finish(clip):
        if clip.request.name == "bad_soma":
            raise ValueError("injected smoothing failure")
        return real_finish(clip)

    monkeypatch.setattr(head, "run", head_run)
    monkeypatch.setattr(soma, "finish", finish)
    records = []
    runner = ts.ThreadedThreeStage(prepare, head, soma)
    runner.run(iter(requests), records.append)
    assert runner.fatal is None
    status = {r["name"]: (r["status"], r.get("error_stage")) for r in records}
    assert status == {
        "ok_1": ("ok", None),
        "bad_detect": ("error", "prepare"),
        "bad_head": ("error", "head"),
        "bad_soma": ("error", "soma"),
        "ok_2": ("ok", None),
    }
    reference = _reference(
        {k: v for k, v in frames_of.items() if k.startswith("ok_")}, camera, False
    )
    assert sorted(results) == ["ok_1", "ok_2"]
    for name in results:
        _assert_results_equal(results[name], reference[name][0])


def test_cuda_error_stops_the_run_and_reports_in_flight(tmp_path, monkeypatch):
    clips = [(0, 7, "first"), (1, 5, "broken"), (2, 8, "never")]
    requests, _, _, _, prepare, head, soma, results = _setup(tmp_path, False, clips)
    real_head_run = head.run

    def head_run(clip, emit, crops=None):
        if clip.request.name == "broken":
            raise RuntimeError("CUDA error: an illegal memory access was encountered")
        return real_head_run(clip, emit, crops)

    monkeypatch.setattr(head, "run", head_run)
    records = []
    runner = ts.ThreadedThreeStage(prepare, head, soma, poll_s=0.01)
    done = threading.Event()

    def target():
        runner.run(iter(requests), records.append)
        done.set()

    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    assert done.wait(60), "run did not stop after a fatal error"
    assert runner.fatal is not None
    by_name = {r["name"]: r["status"] for r in records}
    # The CUDA context is gone: whatever was still in flight is aborted, not
    # finished (stage 3 may or may not have finished "first" before the stop).
    assert by_name["first"] in ("ok", "aborted")
    assert (by_name["first"] == "ok") == ("first" in results)
    assert by_name["broken"] in ("error", "aborted")
    assert by_name.get("never", "aborted") == "aborted"
    assert "never" not in results
    assert len(records) == len({r["name"] for r in records})


class _HeadValues:
    """``forward_step_head`` returning values computed from the images; on CUDA
    it first spins the current stream (stage 2's), so the outputs are written
    late."""

    camera_mode = "sam3db_perspective"

    def __init__(self, spin_cycles=0):
        self.spin_cycles = spin_cycles

    def forward_step_head(self, batch, train=False):
        img = batch["img"]
        if self.spin_cycles:
            torch.cuda._sleep(self.spin_cycles)
        value = img.reshape(img.shape[0], -1)[:, ::101].sum(dim=1)
        return {"value": value, "scaled": value * 2.0}


class _ReadOnCallerStream:
    """Stands in for SOMA: reads the head outputs on the caller's stream,
    optionally after spinning that stream (so stage 2 is ahead of it)."""

    def __init__(self, delay_cycles=0):
        self.delay_cycles = delay_cycles
        self.values = []

    def add(self, batch, hand_data, head):
        if self.delay_cycles:
            torch.cuda._sleep(self.delay_cycles)
        self.values.append(head["value"].clone())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("reader_behind", [False, True], ids=["event", "record_stream"])
def test_cuda_handoff_from_stage_two_stream_to_stage_three(tmp_path, reader_behind):
    """thr mode: stage 2 runs on its own stream; stage 3 (another thread, its
    default stream) must wait for each batch's event before reading, and the
    allocator must not hand stage 2 the memory of a batch stage 3 has not read
    yet (record_stream). Without the wait, ``event`` reads values before the
    spinning head wrote them; without record_stream, ``record_stream`` reads
    blocks stage 2 already reused."""
    import queue as queue_module

    clips = [(0, 30, "cuda_a"), (1, 20, "cuda_b")]
    requests, _, _, predictor, prepare, head_cpu, _, _ = _setup(tmp_path, False, clips)
    config = head_cpu.config
    prepared = [prepare.run(request) for request in requests]
    assert all(clip.error is None for clip in prepared)

    predictor.model = _HeadValues()
    want = []
    for clip in prepared:
        ts.HeadStage(predictor, config).run(
            clip, lambda b: want.append(b.head["value"].clone())
        )

    predictor.device = "cuda"
    # GPU clock cycles: roughly 5 ms per head batch, 40 ms per delayed read.
    predictor.model = _HeadValues(spin_cycles=10_000_000)
    head = ts.HeadStage(predictor, config, own_stream=True)
    soma = ts.SomaStage(predictor, config)
    reader = _ReadOnCallerStream(delay_cycles=80_000_000 if reader_behind else 0)
    soma._batcher = reader
    handoff: queue_module.Queue = queue_module.Queue()

    def stage_two():
        try:
            for clip in prepared:
                head.run(clip, handoff.put)
        finally:
            handoff.put(None)

    worker = threading.Thread(target=stage_two, daemon=True)
    worker.start()
    while (head_batch := handoff.get(timeout=120)) is not None:
        soma.add(head_batch)
        del head_batch  # stage 3 drops its references right away
    worker.join()
    torch.cuda.synchronize()
    assert len(reader.values) == len(want) > 10
    for got, expected in zip(reader.values, want):
        torch.testing.assert_close(got.cpu(), expected, rtol=1e-4, atol=1e-3)


def test_claims_split_one_queue(tmp_path):
    requests = [
        ts.ClipRequest(index=i, name=f"v{i}", video="", intrinsics=None, out_root="")
        for i in range(9)
    ]
    claim_dir = str(tmp_path / "claims")
    os.makedirs(claim_dir)
    a = ts.claim_requests(requests, claim_dir)
    b = ts.claim_requests(requests, claim_dir)
    got_a, got_b = [], []
    for step in range(9):
        target, source = (got_a, a) if step % 3 else (got_b, b)
        item = next(source, None)
        if item is not None:
            target.append(item.index)
    got_a += [r.index for r in a]
    got_b += [r.index for r in b]
    assert sorted(got_a + got_b) == list(range(9))
    assert not set(got_a) & set(got_b)
    assert got_a == sorted(got_a) and got_b == sorted(got_b)
    assert [r.index for r in ts.claim_requests(requests, None)] == list(range(9))
