# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-OpenMDW-1.1

"""Synthetic-track tests of the RT-DETR track cleanup switches (no model, no GPU)."""

import numpy as np
import pytest

from inference.hand_detector_2d.rtdetr import tracking as T
from inference.hand_detector_2d.rtdetr.detector import boxes_from_tracks
from inference.hand_detector_2d.rtdetr.track_cleanup import (
    TrackCleanupConfig,
    boxes_from_tracks_cleanup,
)
from inference.hand_detector_2d.rtdetr.track_cleanup.selection import (
    contain_cxcywh,
    select_continuity,
)
from inference.tests.hand_detector_2d.synthetic_tracks import (
    BASE,
    FPS,
    HANDS,
    SMOOTH,
    W,
    H,
    cand,
    random_recs,
    recs_from,
    run,
    same_outputs,
    stock,
)

POSITION = "sides_by_position"
OFF_SCREEN = "drop_off_screen"
HYSTERESIS = "score_hysteresis"
MOTION = "constant_velocity"
BRIDGE = "gap_bridge"
CONTAIN = "containment_nms"
TRACKLET = "sides_per_tracklet"


# --- every switch off: the stock tracker ---


@pytest.mark.parametrize("seed", range(20))
def test_continuity_engine_with_switches_off_equals_select_new(seed):
    recs = random_recs(seed)
    ref, _ = T.select_new(recs)
    out, _ = select_continuity(recs, TrackCleanupConfig(), W, H, FPS)
    for side in ("L", "R"):
        np.testing.assert_array_equal(np.array(out[side]["present"]), ref[side][1])
        sq = np.array(
            [[b[0], b[1], max(b[2], b[3]) * T.PAD] for b in out[side]["box"]],
            np.float32,
        )
        sq[~np.array(out[side]["present"])] = 0
        np.testing.assert_array_equal(sq, ref[side][0])


@pytest.mark.parametrize("smoothing", [SMOOTH, None])
@pytest.mark.parametrize("seed", range(5))
def test_switches_off_is_the_stock_tracker(seed, smoothing):
    recs = random_recs(100 + seed)
    n = len(recs)
    ids = [f"{i:06d}" for i in range(n)]
    times = np.arange(n) / FPS
    ref, ref_info = boxes_from_tracks(
        ids,
        recs,
        ["left", "right"],
        image_width=W,
        smoothing=smoothing,
        frame_times=times,
    )
    out, info = boxes_from_tracks_cleanup(
        ids,
        recs,
        ["left", "right"],
        (W, H),
        TrackCleanupConfig(),
        smoothing=smoothing,
        frame_times=times,
    )
    assert info == ref_info
    for h in ("left", "right"):
        assert out[h].keys() == ref[h].keys()
        for k in ref[h]:
            np.testing.assert_array_equal(out[h][k], ref[h][k])


def test_track_clip_needs_a_switch():
    with pytest.raises(ValueError, match="at least one"):
        run(random_recs(0, n=10))


# --- sides_by_position ---


def test_one_hand_with_two_labels_gets_one_side():
    # the same right hand reported as right (0.9) and as left (0.7, IoU ~0.6): only the
    # right side gets it
    n = 30
    recs = recs_from(
        [
            (2, {t: cand(1000, 700, 200, 0.9, 2) for t in range(n)}),
            (1, {t: cand(1030, 720, 200, 0.7, 1) for t in range(n)}),
        ],
        n,
    )
    r = run(recs, POSITION)
    assert r["present"]["right"].all()
    assert not r["present"]["left"].any()
    assert stock(recs)["present"]["left"].all(), (
        "the stock tracker puts the duplicate on the left"
    )


def test_two_boxes_by_x_order_and_a_lone_box_by_image_half():
    n = 10
    # labels swapped by the detector: x order decides
    recs = recs_from(
        [
            (2, {t: cand(300, 700) for t in range(n)}),
            (1, {t: cand(1100, 700) for t in range(n)}),
        ],
        n,
    )
    r = run(recs, POSITION)
    assert r["final"]["left"][0, 0] < r["final"]["right"][0, 0]
    # lone box, right half
    recs = recs_from([(1, {t: cand(1200, 700, label=1) for t in range(n)})], n)
    r = run(recs, POSITION)
    assert r["present"]["right"].all()
    assert not r["present"]["left"].any()


# --- drop_off_screen ---


def test_no_hold_at_clip_ends_and_short_gap_filled():
    n = 100
    d = {t: cand(700, 700) for t in range(20, 80) if not (40 <= t < 45)}
    recs = recs_from([(1, d)], n)
    e = run(recs, POSITION, OFF_SCREEN)["emit"]["left"]
    assert not (e[:20].any() or e[80:].any()), (
        "no box before the first / after the last detection"
    )
    assert e[40:45].all(), "5-frame gap in the middle is filled"
    assert run(recs, POSITION)["emit"]["left"].all(), (
        "the stock rules hold boxes to both clip ends"
    )


def test_long_gap_not_filled():
    n = 120
    d = {t: cand(700, 700) for t in [*range(30), *range(70, 120)]}
    r = run(recs_from([(1, d)], n), POSITION, OFF_SCREEN)
    assert not r["emit"]["left"][30:70].any()


def test_exit_through_border_not_filled():
    # hand moves left at 30 px/frame, touches the left border, is gone 10 frames, comes
    # back at the border
    n = 60
    d = {
        t: cand(100 + 30 * (19 - t), 700, 200) for t in range(20)
    }  # x from 670 to 100: left edge at 0
    d.update({t: cand(100 + 30 * (t - 30), 700, 200) for t in range(30, 60)})
    r = run(recs_from([(1, d)], n), POSITION, OFF_SCREEN)
    assert not r["emit"]["left"][20:30].any()


def test_far_apart_ends_not_filled():
    n = 40
    d = {t: cand(300, 700) for t in range(15)}
    d.update({t: cand(700, 700) for t in range(20, 40)})
    r = run(recs_from([(1, d)], n), POSITION, OFF_SCREEN)
    assert not r["emit"]["left"][15:20].any()


# --- score_hysteresis ---


def test_low_score_clutter_alone_is_not_a_hand():
    n = 30
    recs = recs_from([(1, {t: cand(300, 300, 150, 0.3, 1) for t in range(n)})], n)
    assert stock(recs)["present"]["left"].all(), (
        "the stock tracker uses anything above 0.15"
    )
    r = run(recs, HYSTERESIS)
    assert not (r["present"]["left"].any() or r["present"]["right"].any())


def test_low_score_continues_an_existing_hand():
    n = 30
    d = {t: cand(400 + 2 * t, 700, 200, 0.9 if t < 10 else 0.3, 1) for t in range(n)}
    r = run(recs_from([(1, d)], n), POSITION, HYSTERESIS)
    assert r["present"]["left"].all()
    assert (r["src"]["left"][10:] == 2).all()
    r1 = run(recs_from([(1, d)], n), POSITION)
    assert not r1["present"]["left"][10:].any(), (
        "the position rule alone drops the 0.3 boxes"
    )


def test_short_segment_removed_unless_confident():
    n = 40
    hand = {t: cand(400, 700) for t in range(n)}

    def right_present(score):
        blip = {20: cand(1100, 300, 150, score, 2), 21: cand(1100, 300, 150, score, 2)}
        return run(recs_from([(1, hand), (2, blip)], n), POSITION, HYSTERESIS)[
            "present"
        ]["right"]

    assert not right_present(0.6).any()
    assert right_present(0.9)[20:22].all()


# --- constant_velocity ---


def _fast_vertical(v=60.0, drop=()):
    n = 24
    d = {}
    for t in range(n):
        # dropped frames only have a 0.08 box (re-detection: redetect_score 0.05)
        sc = 0.08 if t in drop else 0.9
        d[t] = cand(300, 100 + v * t, 200, sc, 1)  # vertical motion on the left half
    return recs_from([(1, d)], n)


def test_redetection_near_the_prediction():
    recs = _fast_vertical(drop=(10, 11, 12))
    assert run(recs, POSITION, MOTION, redetect_score=0.05)["present"]["left"][
        10:13
    ].all()
    assert not run(recs, POSITION, MOTION)["present"]["left"][10:13].any(), (
        "default threshold 0.15"
    )
    assert not run(recs, POSITION, HYSTERESIS)["present"]["left"][10:13].any()


def test_no_90px_clamp_and_no_lag():
    n, v = 12, 110.0  # 110 px/frame vertical, above the stock 90 px/frame clamp
    recs = recs_from(
        [(1, {t: cand(300, 100 + v * t, 200, 0.9, 1) for t in range(n)})], n
    )
    truth = 100 + v * np.arange(n)
    err_stock = np.abs(stock(recs)["final"]["left"][:, 1] - truth)
    err_cv = np.abs(run(recs, POSITION, MOTION)["final"]["left"][:, 1] - truth)
    assert err_stock.max() > 100, (
        "clamp + causal EMA leave the stock crop behind the hand"
    )
    assert err_cv.max() < 5.0


def test_crop_not_inflated_when_moving():
    recs = _fast_vertical(v=60.0)
    side_stock = stock(recs)["final"]["left"][5:, 2]
    side_cv = run(recs, POSITION, MOTION)["final"]["left"][5:, 2]
    assert np.median(side_cv) < np.median(side_stock)
    assert float(np.median(side_cv)) == pytest.approx(200 * T.PAD * 0.8, abs=5.0)


# --- gap_bridge ---

# one left hand at x 400 for 0..39 and 80..119; frames 40..79 only weak rows (0.08) on
# the same spot
_HAND = {t: cand(400, 700, score=0.9) for t in [*range(40), *range(80, 120)]}
_WEAK = {t: cand(402, 701, score=0.08) for t in range(40, 80)}


def test_bridge_fills_a_gap_supported_by_low_score_rows():
    recs = recs_from([(1, {**_HAND, **_WEAK})], 120)
    base = run(recs, *BASE)
    new = run(recs, *BASE, BRIDGE)
    assert not np.isfinite(base["final"]["left"][40:80]).all(1).any(), (
        "40 frames: too long for drop_off_screen"
    )
    assert np.isfinite(new["final"]["left"][40:80]).all(1).all()
    assert new["stats"]["bridged_gaps"] == 1


def test_no_bridge_without_rows():
    new = run(recs_from([(1, _HAND)], 120), *BASE, BRIDGE)
    assert not np.isfinite(new["final"]["left"][40:80]).all(1).any()


def test_no_bridge_with_too_few_supported_frames():
    weak = {
        t: c for t, c in _WEAK.items() if t < 60
    }  # 20 of 40 gap frames < bridge_share 0.6
    new = run(recs_from([(1, {**_HAND, **weak})], 120), *BASE, BRIDGE)
    assert new["stats"]["bridged_gaps"] == 0


def test_rows_on_the_other_hand_are_not_support():
    # the left hand leaves at 40; the right hand walks along the straight path between
    # the left hand's two ends
    right = {t: cand(400, 700, score=0.9, label=2) for t in range(41, 79)}
    new = run(recs_from([(1, _HAND), (2, right)], 120), *BASE, BRIDGE)
    assert new["stats"].get("bridged_gaps", 0) == 0


# --- containment_nms ---


def test_containment_measure():
    assert contain_cxcywh([500, 700, 150, 150], [560, 760, 300, 300]) == pytest.approx(
        1.0
    )
    assert contain_cxcywh([0, 0, 100, 100], [50, 0, 100, 100]) == pytest.approx(0.5)


def test_hand_box_inside_a_forearm_box_is_one_hand():
    # a hand box (0.9) inside a forearm box (0.8), IoU 0.25 < NMS 0.5: without the
    # switch the hand gets two sides
    hand = {t: cand(500, 700, s=150, score=0.9) for t in range(30)}
    arm = {t: [560.0, 760.0, 300.0, 300.0, 0.8, 2] for t in range(30)}
    recs = recs_from([(1, hand), (2, arm)], 30)
    base = run(recs, POSITION)
    new = run(recs, POSITION, CONTAIN)
    assert base["present"]["left"].all() and base["present"]["right"].all()
    assert new["present"]["left"].all()
    assert not new["present"]["right"].any()


def test_continuation_inside_the_other_sides_box_is_not_taken():
    # frames 20..24: one big box over both hands goes to the left (image half); the
    # right hand only has a 0.3 box that lies inside it -> without the switch it
    # continues the right hand, with it it does not
    left = {t: cand(500, 700, score=0.9) for t in range(20)}
    right = {t: cand(760, 700, score=0.9, label=2) for t in range(20)}
    right.update({t: cand(760, 700, score=0.3, label=2) for t in range(20, 25)})
    big = {t: cand(640, 700, s=420, score=0.9) for t in range(20, 25)}
    recs = recs_from([(1, left), (2, right), (1, big)], 25)
    assert run(recs, POSITION, HYSTERESIS, MOTION)["present"]["right"][20:25].all()
    assert not run(recs, POSITION, HYSTERESIS, MOTION, CONTAIN)["present"]["right"][
        20:25
    ].any()


# --- sides_per_tracklet ---


def test_lone_hand_hovering_at_the_centre_keeps_its_side():
    # one left hand moves from x 450 to the image centre and hovers just right of it
    # (weak evidence)
    xs = [450 + 10 * t for t in range(27)] + [
        710 + 8 * ((t % 4) - 1.5) for t in range(33)
    ]
    recs = recs_from(
        [(1, {t: cand(x, 700, score=0.9) for t, x in enumerate(xs)})], len(xs)
    )
    right = np.array([x >= (W - 1) / 2 for x in xs])
    base = run(recs, POSITION)
    new = run(recs, POSITION, TRACKLET)
    assert np.array_equal(base["present"]["right"], right)
    assert new["present"]["left"].all()
    assert not new["present"]["right"].any()


def test_reaching_over_the_other_hand_keeps_the_side():
    # left hand still at x 500 (low in the image); the right hand reaches over it to x
    # 470, high in the image
    xr = (
        [900 - 30 * t for t in range(15)]
        + [470] * 10
        + [470 + 30 * t for t in range(1, 15)]
    )
    n = len(xr)
    left = {t: cand(500, 950, score=0.9) for t in range(n)}
    right = {t: cand(x, 450, score=0.9, label=2) for t, x in enumerate(xr)}
    recs = recs_from([(1, left), (2, right)], n)
    crossed = np.array([x < 500 for x in xr])
    assert (run(recs, POSITION)["cand"]["right"][:, 1][crossed] == 950).all(), (
        "x order swaps the hands"
    )
    assert (run(recs, POSITION, TRACKLET)["cand"]["right"][:, 1] == 450).all()


def test_strong_frames_are_not_changed():
    # two hands far apart for the whole clip: every frame is strong, nothing changes
    left = {t: cand(300, 700, score=0.9) for t in range(40)}
    right = {t: cand(1100, 700, score=0.9, label=2) for t in range(40)}
    recs = recs_from([(1, left), (2, right)], 40)
    assert same_outputs(run(recs, POSITION), run(recs, POSITION, TRACKLET))


# --- parameters of switches that are off ---


def test_parameters_of_switches_that_are_off_change_nothing():
    rng = np.random.default_rng(3)
    d1 = {
        t: cand(
            400 + 5 * t + rng.normal(0, 8), 700, score=float(rng.uniform(0.2, 0.95))
        )
        for t in range(80)
        if rng.random() < 0.8
    }
    d2 = {
        t: cand(
            900 - 4 * t,
            650 + rng.normal(0, 8),
            score=float(rng.uniform(0.2, 0.95)),
            label=2,
        )
        for t in range(80)
        if rng.random() < 0.7
    }
    recs = recs_from([(1, d1), (2, d2)], 80)
    a = run(recs, *BASE)
    b = run(
        recs,
        *BASE,
        bridge_share=0.1,
        bridge_iou=0.1,
        containment_frac=0.1,
        tracklet_band=0.5,
        tracklet_dist=3.0,
    )
    assert same_outputs(a, b)


# --- degenerate clips ---

RULES7 = (*BASE, "gap_bridge", "containment_nms", "sides_per_tracklet")


@pytest.mark.parametrize(
    "fill",
    [
        pytest.param(lambda n: [], id="no-hands"),
        pytest.param(lambda n: [(0, "L", cand(400, 700))], id="first-frame-only"),
        pytest.param(lambda n: [(n - 1, "R", cand(900, 700, label=2))], id="last-only"),
    ],
)
@pytest.mark.parametrize("n", [0, 1, 2, 100])
def test_degenerate_clips_do_not_fail(n, fill):
    recs = [{"L": [], "R": []} for _ in range(n)]
    for t, k, c in fill(n) if n else []:
        recs[t][k].append(c)
    ids = [f"{i:06d}" for i in range(n)]
    out, info = boxes_from_tracks_cleanup(
        ids,
        recs,
        list(HANDS),
        (W, H),
        TrackCleanupConfig(**dict.fromkeys(RULES7, True)),
        smoothing=SMOOTH,
        frame_times=np.arange(n) / FPS,
    )
    assert set(out) == set(HANDS)
    for hand in HANDS:
        assert set(out[hand]) <= set(ids)
        assert info["per_hand"][hand]["frames"] == n
