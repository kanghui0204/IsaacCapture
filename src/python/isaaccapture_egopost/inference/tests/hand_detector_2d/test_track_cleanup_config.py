# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-OpenMDW-1.1

"""track_cleanup config parsing, detector construction and the detector's run() (fake
ONNX model)."""

import json
import subprocess
import sys
from pathlib import Path
from typing import ClassVar

import numpy as np
import pytest

from inference.hand_detector_2d.configs import rtdetr as rtdetr_backend
from inference.hand_detector_2d.rtdetr import onnx_model
from inference.hand_detector_2d.rtdetr.detector import RTDetrHandBBoxDetector
from inference.hand_detector_2d.rtdetr.track_cleanup import (
    RTDetrTrackCleanupDetector,
    TrackCleanupConfig,
)

RULES = {
    "sides_by_position": True,
    "drop_off_screen": True,
    "score_hysteresis": True,
    "constant_velocity": True,
    "gap_bridge": True,
    "containment_nms": True,
    "sides_per_tracklet": True,
}
SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "hand_detector_2d"
    / "scripts"
    / "run_hand_detector.py"
)


class FakeHandDetector:
    """Stands in for the ONNX session: returns queued detections above ``score_thr``."""

    queue: ClassVar[list] = []

    def __init__(self, weights, score_thr=0.5, providers=None, **kwargs):
        self.weights = weights
        self.score_thr = score_thr
        self.providers = ["CPUExecutionProvider"]
        self.kwargs = kwargs

    def detect_batch(self, images):
        out = []
        for _ in images:
            dets = FakeHandDetector.queue.pop(0)
            out.append([d for d in dets if d.score > self.score_thr])
        return out


@pytest.fixture
def fake_onnx(monkeypatch):
    monkeypatch.setattr(onnx_model, "HandDetector", FakeHandDetector)
    FakeHandDetector.queue = []
    return FakeHandDetector


def _cfg(**extra):
    cfg = {
        "detector_backend": "rtdetr",
        "rtdetr_config": {"weights": "hand_detector.onnx", "bbox_smoothing": True},
        "tracking": True,
    }
    cfg.update(extra)
    return cfg


# --- config ---


@pytest.mark.parametrize("data", [None, {}, False])
def test_empty_config_is_off(data):
    cfg = TrackCleanupConfig.from_dict(data)
    assert not cfg.any_rule
    assert cfg.name == "off"
    assert cfg.min_score() == pytest.approx(0.15)


@pytest.mark.parametrize(
    ("data", "message"),
    [
        ({"sides_by_postion": True}, "unknown key"),
        ({"sides_by_position": "yes"}, "true or false"),
        ({"start_score": True}, "finite number"),
        ({"start_score": float("nan")}, "finite number"),
        ({"max_gap_s": -1}, "nonnegative"),
        ({"start_score": 1.5}, "between 0 and 1"),
        ({"bridge_share": 2}, "between 0 and 1"),
        ({"nms_iou": 0}, "> 0 and <= 1"),
        ({"max_area_frac": 0}, "> 0 and <= 1"),
        ({"tracklet_band": 0}, "must be > 0"),
        ({"r_pos": 0}, "must be > 0"),
        ({"continue_score": 0.6}, "must not exceed start_score"),
        ({"tracklet_sep_lo": 0.8}, "below tracklet_sep_hi"),
        (True, "must be a mapping"),
        ({"gap_bridge": True}, "gap_bridge requires drop_off_screen"),
        ({"containment_nms": True}, "requires sides_by_position"),
        ({"sides_per_tracklet": True}, "requires sides_by_position"),
        ([1, 2], "must be a mapping"),
    ],
)
def test_invalid_config_is_rejected(data, message):
    with pytest.raises(ValueError, match=message):
        TrackCleanupConfig.from_dict(data)


def test_score_thresholds():
    base = dict.fromkeys(
        (
            "sides_by_position",
            "drop_off_screen",
            "score_hysteresis",
            "constant_velocity",
        ),
        True,
    )
    assert TrackCleanupConfig.from_dict(base).min_score() == pytest.approx(0.15)
    bridge = TrackCleanupConfig.from_dict({**base, "gap_bridge": True})
    assert bridge.min_score() == pytest.approx(0.05), (
        "the gap bridge reads boxes down to bridge_min_score"
    )
    assert bridge.select_min_score() == pytest.approx(0.15), (
        "the per-frame choice does not change"
    )
    redetect = TrackCleanupConfig.from_dict({**base, "redetect_score": 0.05})
    assert redetect.min_score() == pytest.approx(0.05)
    # Every threshold a rule compares against lowers the ONNX threshold below
    # 0.15 when set lower; otherwise such a setting would do nothing.
    hysteresis = {"score_hysteresis": True, "continue_score": 0.05}
    assert TrackCleanupConfig.from_dict(hysteresis).min_score() == pytest.approx(0.05)
    assert TrackCleanupConfig.from_dict(
        {**base, "continue_score": 0.05}
    ).min_score() == pytest.approx(0.15), "with constant_velocity redetect_score rules"
    position = {"sides_by_position": True, "start_score": 0.1, "continue_score": 0.1}
    assert TrackCleanupConfig.from_dict(position).min_score() == pytest.approx(0.1)
    assert TrackCleanupConfig.from_dict(RULES).name == "+".join(RULES)


# --- detector construction ---


def test_no_track_cleanup_builds_the_stock_detector(fake_onnx):
    det = rtdetr_backend.build(_cfg(), "cpu", 1, True)
    assert type(det) is RTDetrHandBBoxDetector
    assert det.detector.score_thr == pytest.approx(0.15)


@pytest.mark.parametrize(
    "cleanup", [{}, {"sides_by_position": False}, dict.fromkeys(RULES, False)]
)
def test_every_switch_off_builds_the_stock_detector(fake_onnx, cleanup):
    det = rtdetr_backend.build(_cfg(track_cleanup=cleanup), "cpu", 1, True)
    assert type(det) is RTDetrHandBBoxDetector
    assert det.detector.score_thr == pytest.approx(0.15)


def test_switches_build_the_cleanup_detector(fake_onnx):
    det = rtdetr_backend.build(_cfg(track_cleanup=RULES), "cpu", 1, True)
    assert isinstance(det, RTDetrTrackCleanupDetector)
    assert det.requires_full_clip
    assert det.use_tracker and not det.drop_gap_filled
    assert det.detector.score_thr == pytest.approx(0.05)
    assert "cleanup[" in det.mode_name


def test_false_builds_the_stock_detector(fake_onnx):
    det = rtdetr_backend.build(_cfg(track_cleanup=False), "cpu", 1, True)
    assert type(det) is RTDetrHandBBoxDetector


def test_every_rtdetr_setting_reaches_the_cleanup_detector(fake_onnx, monkeypatch):
    """The cleanup detector is constructed from the same RTDetrConfig arguments
    as the stock detector (a setting added to RTDetrConfig.detector_kwargs
    reaches both)."""
    seen = {}
    real_init = RTDetrHandBBoxDetector.__init__

    def spy(self, **kwargs):
        seen[type(self).__name__] = dict(kwargs)
        real_init(self, **kwargs)

    monkeypatch.setattr(RTDetrHandBBoxDetector, "__init__", spy)
    settings = {
        "weights": "x.onnx",
        "batch_size": 3,
        "lr_order": False,
        "bbox_smoothing": True,
        "bbox_size_scale": 0.7,
    }
    rtdetr_backend.build(_cfg(rtdetr_config=settings), "cpu", 1, True)
    rtdetr_backend.build(
        _cfg(rtdetr_config=settings, track_cleanup=RULES), "cpu", 1, True
    )
    stock = seen["RTDetrHandBBoxDetector"]
    cleanup = seen["RTDetrTrackCleanupDetector"]
    assert cleanup.pop("providers") is None
    assert cleanup == stock
    assert stock["batch_size"] == 3 and stock["enforce_lr_order"] is False
    assert stock["smoothing"].target_size_scale == pytest.approx(0.7)


@pytest.mark.parametrize(
    ("cfg", "tracking", "message"),
    [
        (_cfg(track_cleanup={"sides_by_position": True}), False, "tracking: true"),
        (
            _cfg(
                rtdetr_config={"weights": "x.onnx", "drop_gap_filled": True},
                track_cleanup={"sides_by_position": True},
            ),
            True,
            "drop_gap_filled",
        ),
        (_cfg(track_cleanup={"bogus": 1}), True, "unknown key"),
    ],
)
def test_invalid_combinations_are_rejected(fake_onnx, cfg, tracking, message):
    with pytest.raises(ValueError, match=message):
        rtdetr_backend.build(cfg, "cpu", 1, tracking)


def test_pipeline_reports_an_invalid_track_cleanup_as_a_config_error(fake_onnx):
    from inference.scripts import run_egohand_inference_pipeline as script

    with pytest.raises(script.DetectorConfigError, match="unknown key"):
        script._build_detector_from_config(_cfg(track_cleanup={"bogus": 1}), "cpu", 1)
    det, backend = script._build_detector_from_config(
        _cfg(track_cleanup=RULES), "cpu", 1
    )
    assert backend == "rtdetr" and isinstance(det, RTDetrTrackCleanupDetector)
    with pytest.raises(SystemExit):
        script.check_frame_batching(det, 64)


# --- run() ---


def _det(cx, cy, s, score, label_id):
    return onnx_model.Detection(
        label=onnx_model.LABELS[label_id],
        label_id=label_id,
        score=score,
        xyxy=(cx - s / 2, cy - s / 2, cx + s / 2, cy + s / 2),
    )


def _clip(n=40):
    """Left hand on frames 0..39; right hand only on frames 0..19 (it leaves); one 0.3
    clutter box."""
    frames = []
    for t in range(n):
        dets = [_det(150 + t, 240, 80, 0.9, 1)]
        if t < 20:
            dets.append(_det(480, 240, 80, 0.9, 2))
        if t == 30:
            dets.append(_det(600, 60, 40, 0.3, 2))
        frames.append(dets)
    return frames


def test_run_matches_track_boxes_and_writes_json(fake_onnx, tmp_path):
    det = rtdetr_backend.build(_cfg(track_cleanup=RULES), "cpu", 4, True)
    n = 40
    ids = [f"{i:06d}" for i in range(n)]
    images = [np.zeros((480, 640, 3), np.uint8) for _ in range(n)]
    fake_onnx.queue = _clip(n)
    paths = {h: tmp_path / f"clip_{h}_bbox.json" for h in ("left", "right")}
    out = det.run(
        ids, images, ["left", "right"], paths, frame_times={i: int(i) / 30 for i in ids}
    )
    ref, info = det.track_boxes(
        ids,
        [[d for d in f if d.score > det.detector.score_thr] for f in _clip(n)],
        ["left", "right"],
        (640, 480),
        frame_times=[int(i) / 30 for i in ids],
    )
    assert info["per_hand"]["right"]["emitted"] == 20, (
        "no box after the right hand left"
    )
    for h in ("left", "right"):
        saved = json.loads(paths[h].read_text())
        assert saved.keys() == ref[h].keys()
        for k, box in ref[h].items():
            np.testing.assert_array_equal(out[k][h], box)
            assert saved[k] == [float(v) for v in box]


def test_run_needs_frame_times(fake_onnx):
    det = rtdetr_backend.build(_cfg(track_cleanup=RULES), "cpu", 4, True)
    with pytest.raises(ValueError, match="frame_times"):
        det.run(["000000"], [np.zeros((48, 64, 3), np.uint8)], ["left"])


def test_standalone_script_imports_in_its_own_package_context():
    # run_hand_detector.py rebinds its package to ``rtdetr``; the cleanup's relative
    # imports must resolve there
    res = subprocess.run(
        [sys.executable, str(SCRIPT), "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert res.returncode == 0, res.stderr
