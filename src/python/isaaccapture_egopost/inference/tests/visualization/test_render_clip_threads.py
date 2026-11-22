# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-OpenMDW-1.1

"""render_clip writes bbox_viz on a worker thread: same bytes, errors surface, no thread left behind.

CPU only; no checkpoint. ``--no_mesh`` style (skeleton only), as in the documented example.
"""

import hashlib
import threading
from concurrent.futures import Future

import cv2
import numpy as np
import pytest

import inference.visualization.overlay as overlay

HANDS = ["left", "right"]


def _clip(n=12, w=96, h=72, seed=0):
    rng = np.random.default_rng(seed)
    frames = [rng.integers(0, 256, (h, w, 3), dtype=np.uint8) for _ in range(n)]
    names = [f"{i:06d}" for i in range(n)]
    results, boxes = {}, {}
    for k, hand in enumerate(HANDS):
        results[hand], boxes[hand] = [], []
        for i in range(n):
            if (i + k) % 5 == 4:  # some frames without this hand
                results[hand].append(None)
                boxes[hand].append(None)
                continue
            box = np.array(
                [30.0 + 2 * i + 10 * k, 36.0 - i, 40.0 + k], dtype=np.float32
            )
            kp = rng.uniform(-0.45, 0.45, (25, 2)).astype(np.float32)
            results[hand].append({"pred_keypoints_2d": kp})
            boxes[hand].append(box)
    K = np.array([[80.0, 0, w / 2], [0, 80.0, h / 2], [0, 0, 1]], dtype=np.float32)
    faces = {hand: np.zeros((1, 3), dtype=np.int64) for hand in HANDS}
    return frames, names, results, boxes, faces, K, w, h


def _render(out_dir, save_images=False):
    frames, names, results, boxes, faces, K, w, h = _clip()
    overlay.render_clip(
        out_dir,
        frames,
        names,
        results,
        {},
        boxes,
        faces,
        HANDS,
        w,
        h,
        K,
        30.0,
        (1.0, 1.0, 0.9),
        save_images,
        True,
        draw_skeleton=True,
        no_mesh=True,
    )


def _decoded_md5(path):
    cap = cv2.VideoCapture(str(path))
    out = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        out.append(hashlib.md5(frame.tobytes()).hexdigest())
    cap.release()
    return out


class _InlineExecutor:
    """Runs the submitted job immediately on the calling thread (the previous serial order)."""

    def __init__(self, *args, **kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def submit(self, fn, *args, **kwargs):
        future = Future()
        try:
            future.set_result(fn(*args, **kwargs))
        except BaseException as exc:  # noqa: BLE001 - mirrors ThreadPoolExecutor
            future.set_exception(exc)
        return future


def test_threaded_outputs_byte_identical_to_serial(tmp_path, monkeypatch):
    _render(tmp_path / "threaded", save_images=True)
    monkeypatch.setattr(overlay, "ThreadPoolExecutor", _InlineExecutor)
    _render(tmp_path / "serial", save_images=True)
    for name in ("bbox_viz.mp4", "fullimg_overlay.mp4"):
        a = (tmp_path / "threaded" / name).read_bytes()
        b = (tmp_path / "serial" / name).read_bytes()
        assert a == b, name
        assert len(_decoded_md5(tmp_path / "threaded" / name)) == 12
        assert _decoded_md5(tmp_path / "threaded" / name) == _decoded_md5(
            tmp_path / "serial" / name
        )
    for sub in ("bbox_viz", "fullimg_overlay"):
        a = sorted(p.name for p in (tmp_path / "threaded" / sub).iterdir())
        b = sorted(p.name for p in (tmp_path / "serial" / sub).iterdir())
        assert a == b and len(a) == 12
        for name in a:
            assert (tmp_path / "threaded" / sub / name).read_bytes() == (
                tmp_path / "serial" / sub / name
            ).read_bytes()


def test_bbox_writer_error_is_raised(tmp_path, monkeypatch):
    def boom(*args, **kwargs):
        raise RuntimeError("bbox drawing failed")

    monkeypatch.setattr(overlay, "draw_bbox_on_image", boom)
    with pytest.raises(RuntimeError, match="bbox drawing failed"):
        _render(tmp_path / "out")


def test_fullimg_error_joins_bbox_worker(tmp_path, monkeypatch):
    started = threading.Event()
    real_draw = overlay.draw_bbox_on_image

    def slow_draw(*args, **kwargs):
        started.set()
        return real_draw(*args, **kwargs)

    def boom(*args, **kwargs):
        started.wait(5)
        raise ValueError("skeleton drawing failed")

    monkeypatch.setattr(overlay, "draw_bbox_on_image", slow_draw)
    monkeypatch.setattr(overlay, "draw_hand_skeleton", boom)
    before = {t.name for t in threading.enumerate()}
    with pytest.raises(ValueError, match="skeleton drawing failed"):
        _render(tmp_path / "out")
    left = [
        t.name
        for t in threading.enumerate()
        if t.name.startswith("bbox-viz") and t.name not in before
    ]
    assert left == []
    # The worker finished its file before render_clip returned.
    assert len(_decoded_md5(tmp_path / "out" / "bbox_viz.mp4")) == 12
