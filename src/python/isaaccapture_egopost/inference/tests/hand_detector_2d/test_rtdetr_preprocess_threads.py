# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-OpenMDW-1.1

"""Threaded RT-DETR preprocessing feeds ORT exactly the batches of the sequential loop."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("cv2")

from inference.hand_detector_2d.rtdetr.detector import (  # noqa: E402
    RTDetrConfig,
    RTDetrHandBBoxDetector,
)
from inference.hand_detector_2d.rtdetr.onnx_model import HandDetector  # noqa: E402


class _RecordingDetector:
    """Stands in for HandDetector: records every ORT batch instead of running it."""

    def __init__(self):
        self.batches = []

    def _record(self, batch, sizes):
        self.batches.append(
            (batch.tobytes(), batch.shape, sizes.tobytes(), sizes.dtype)
        )
        return [[] for _ in range(len(batch))]

    def detect_batch(self, images):
        # HandDetector.detect_batch, minus session.run.
        batch = np.concatenate([HandDetector._preprocess(p) for p in images], axis=0)
        sizes = np.array([[p.size[0], p.size[1]] for p in images], dtype=np.int64)
        return self._record(batch, sizes)

    def detect_preprocessed(self, inputs, sizes):
        return self._record(np.concatenate(inputs, axis=0), sizes)


def _detector(threads, batch_size=8):
    det = object.__new__(RTDetrHandBBoxDetector)
    det.batch_size = batch_size
    det.preprocess_threads = threads
    det._ort_batch_sizes_run = set()
    det.detector = _RecordingDetector()
    return det


def _frames(count, height, width, seed=0):
    rng = np.random.default_rng(seed)
    return [
        rng.integers(0, 256, (height, width, 3), dtype=np.uint8) for _ in range(count)
    ]


@pytest.mark.parametrize(
    "height,width", [(96, 128), (1080, 1920), (641, 639), (200, 300)]
)
@pytest.mark.parametrize("count", [1, 8, 19])
def test_threaded_batches_equal_sequential(height, width, count):
    frames = _frames(count, height, width, seed=height + width + count)
    sequential = _detector(threads=1)
    threaded = _detector(threads=4)
    out_seq = sequential._detect_per_frame(frames)
    out_thr = threaded._detect_per_frame(frames)
    assert len(out_seq) == len(out_thr) == count
    assert len(threaded.detector.batches) == -(-count // 8)
    assert threaded.detector.batches == sequential.detector.batches


def test_mixed_sizes_in_one_chunk_keep_order():
    frames = (
        _frames(3, 120, 160, seed=1)
        + _frames(2, 90, 200, seed=2)
        + _frames(4, 64, 64, seed=3)
    )
    sequential, threaded = _detector(threads=1), _detector(threads=3)
    sequential._detect_per_frame(frames)
    threaded._detect_per_frame(frames)
    assert threaded.detector.batches == sequential.detector.batches


def test_second_clip_prefetches_from_the_start():
    det = _detector(threads=2)
    frames = _frames(17, 64, 96, seed=4)
    det._detect_per_frame(frames)
    assert det._ort_batch_sizes_run == {8, 1}
    det._detect_per_frame(frames)
    seq = _detector(threads=1)
    seq._detect_per_frame(frames)
    seq._detect_per_frame(frames)
    assert det.detector.batches == seq.detector.batches  # 3 + 3 batches, same bytes


def test_wrapper_without_detect_preprocessed_keeps_the_sequential_loop():
    """An ORT wrapper with only detect_batch (e.g. a test double) still works."""
    frames = _frames(19, 64, 96, seed=5)
    recording = _RecordingDetector()

    class BatchOnly:
        def detect_batch(self, images):
            return recording.detect_batch(images)

    det = _detector(threads=4)
    det.detector = BatchOnly()
    assert len(det._detect_per_frame(frames)) == 19
    seq = _detector(threads=1)
    seq._detect_per_frame(frames)
    assert recording.batches == seq.detector.batches


def test_empty_clip():
    assert _detector(threads=4)._detect_per_frame([]) == []


def test_config_preprocess_threads():
    assert RTDetrConfig().preprocess_threads is None
    assert RTDetrConfig.from_dict({"preprocess_threads": 4}).preprocess_threads == 4
    assert RTDetrConfig.from_spec("preprocess_threads=1").preprocess_threads == 1
    for bad in (0, -2, True, 2.5, "4"):
        with pytest.raises(ValueError):
            RTDetrConfig.from_dict({"preprocess_threads": bad})
