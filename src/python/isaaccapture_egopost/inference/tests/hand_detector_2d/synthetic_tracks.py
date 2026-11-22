# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-OpenMDW-1.1

"""Synthetic RT-DETR candidate tracks for the track cleanup tests."""

import random

import numpy as np

from inference.hand_detector_2d.rtdetr.detector import RTDetrConfig, boxes_from_tracks
from inference.hand_detector_2d.rtdetr.tracking import THR_RESCUE, HandTracker
from inference.hand_detector_2d.rtdetr.track_cleanup.config import TrackCleanupConfig
from inference.hand_detector_2d.rtdetr.track_cleanup.track import track_clip

W, H, FPS = 1408, 1408, 30.0
SMOOTH = RTDetrConfig(bbox_smoothing=True).smoothing_config()
HANDS = ("left", "right")

# The four base switches; most tests add one switch to them.
BASE = (
    "sides_by_position",
    "drop_off_screen",
    "score_hysteresis",
    "constant_velocity",
)


def cand(cx, cy, s=200.0, score=0.9, label=1):
    """One detector candidate [cx, cy, w, h, score, label_id] (label 1 = left, 2 =
    right)."""
    return [float(cx), float(cy), float(s), float(s), float(score), int(label)]


def recs_from(tracks, n):
    """tracks: list of (label, {t: cand}) -> per-frame tracker records."""
    recs = [{"L": [], "R": []} for _ in range(n)]
    for lab, d in tracks:
        for t, c in d.items():
            recs[t]["L" if lab == 1 else "R"].append(c)
    for r in recs:
        for k in ("L", "R"):
            r[k].sort(key=lambda c: -c[4])
    return recs


def config(*switches, **params):
    return TrackCleanupConfig(**dict.fromkeys(switches, True), **params)


def run(recs, *switches, smoothing=SMOOTH, **params):
    """track_clip with the given switches on (and parameter overrides)."""
    return track_clip(
        recs,
        W,
        H,
        FPS,
        config(*switches, **params),
        smoothing=smoothing,
        frame_times=np.arange(len(recs)) / FPS,
    )


def stock(recs, smoothing=SMOOTH):
    """The stock tracker on the same records, in track_clip's layout (final boxes and
    real detections)."""
    n = len(recs)
    ids = [f"{i:06d}" for i in range(n)]
    # the stock detector returns candidates above THR_RESCUE only
    recs = [{k: [c for c in r[k] if c[4] > THR_RESCUE] for k in r} for r in recs]
    out, _ = boxes_from_tracks(
        ids,
        recs,
        list(HANDS),
        image_width=W,
        smoothing=smoothing,
        frame_times=np.arange(n) / FPS,
    )
    final = {}
    for h in HANDS:
        dense = np.full((n, 3), np.nan, np.float32)
        for k, v in out[h].items():
            dense[int(k)] = v
        final[h] = dense
    tracks = HandTracker(image_width=W).track(recs)
    return {"final": final, "present": tracks["raw_present"]}


def same_outputs(a, b):
    for k in ("final", "native", "present", "emit"):
        for h in HANDS:
            x, y = np.asarray(a[k][h], np.float64), np.asarray(b[k][h], np.float64)
            if not (
                np.array_equal(np.isnan(x), np.isnan(y))
                and np.array_equal(x[~np.isnan(x)], y[~np.isnan(y)])
            ):
                return False
    return True


def random_recs(seed, n=300):
    """Two noisy hands with dropouts and label flips plus random clutter, every score >
    0.15."""
    rng = random.Random(seed)
    recs = []
    xL, xR = 400.0, 1000.0
    for _ in range(n):
        xL += rng.gauss(0, 15)
        xR += rng.gauss(0, 15)
        rec = {"L": [], "R": []}
        for x, lab in ((xL, 1), (xR, 2)):
            if rng.random() < 0.85:
                rec["L" if (lab == 1) != (rng.random() < 0.1) else "R"].append(
                    cand(
                        x,
                        700 + rng.gauss(0, 20),
                        180 + rng.gauss(0, 10),
                        rng.uniform(0.16, 0.99),
                        lab,
                    )
                )
        for _ in range(rng.randint(0, 2)):
            rec["L" if rng.random() < 0.5 else "R"].append(
                cand(
                    rng.uniform(0, W),
                    rng.uniform(0, H),
                    rng.uniform(80, 300),
                    rng.uniform(0.16, 0.6),
                    rng.choice((1, 2)),
                )
            )
        for k in ("L", "R"):
            rec[k].sort(key=lambda c: -c[4])
        recs.append(rec)
    return recs
