# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-OpenMDW-1.1

"""Whole-clip boxes with the track cleanup: selection -> gap filling / box rules ->
smoothing.

``track_clip`` returns per hand the native square track (``[cx, cy, side]``, NaN where
no box is emitted), the final (smoothed) crop track, the real-detection mask and the
chosen candidate. It needs at least one switch on; with every switch off
``detector.boxes_from_tracks_cleanup`` calls the stock ``boxes_from_tracks`` instead.
"""

import numpy as np

from .. import tracking as T
from ..bbox_smoothing import _alpha as _bs_alpha
from ..bbox_smoothing import _limit_size_changes, smooth_bbox_track
from .config import REF_FPS
from .fill import apply_rules_runs, bridge_gaps, fill_v2
from .motion import rts_smooth_square
from .selection import select_continuity, select_position
from .sides import relabel

HANDS = (("left", "L"), ("right", "R"))


def _filter_recs(recs, thr):
    return [{k: [c for c in rec[k] if c[4] > thr] for k in ("L", "R")} for rec in recs]


def _segment_filter(sel, cfg, fps):
    """``score_hysteresis``: remove short segments of real boxes whose best score stays
    below seg_keep_score. A segment = real boxes joined across gaps of at most 2
    frames at 30 fps."""
    min_len = max(1, round(cfg.min_seg_s * fps))
    max_join_gap = max(1, round(2 * fps / REF_FPS))
    removed = 0
    for s in ("L", "R"):
        pres = np.asarray(sel[s]["present"], bool)
        sc = np.asarray(sel[s]["score"], np.float64)
        idx = np.flatnonzero(pres)
        if len(idx) == 0:
            continue
        cut = np.flatnonzero(np.diff(idx) > max_join_gap + 1)
        starts = np.r_[idx[0], idx[cut + 1]]
        ends = np.r_[idx[cut], idx[-1]]
        for a, b in zip(starts, ends):
            if b - a + 1 < min_len and sc[a : b + 1].max() < cfg.seg_keep_score:
                for t in range(a, b + 1):
                    if pres[t]:
                        sel[s]["present"][t] = False
                        sel[s]["src"][t] = 0
                removed += 1
    return removed


def _crossing_fix(native, smoothed, present):
    """detector.boxes_from_tracks: undo left/right x-crossings introduced by independent
    smoothing."""
    if not all(present[h].any() for h in ("left", "right")):
        return
    left, right = smoothed["left"], smoothed["right"]
    nl, nr = native["left"][:, 0], native["right"][:, 0]
    with np.errstate(invalid="ignore"):
        crossed = (nl <= nr) & (left[:, 0] > right[:, 0])
    if not crossed.any():
        return
    boundary = np.clip(
        (left[crossed, 0] + right[crossed, 0]) / 2, nl[crossed], nr[crossed]
    )
    left[crossed, 0] = right[crossed, 0] = boundary


def _rts_final(native, observed, fps, cfg, smoothing):
    """``constant_velocity`` smoothing: RTS per run (split at missing frames and at
    jumps > reset_jump_fraction * side), then the production crop size path of
    ``bbox_smoothing.smooth_bbox_track`` (0.8 scale, EMA in log size with grow / shrink
    weights, coverage guard, size-rate limit) on the smoothed centres."""
    out = np.full_like(native, np.nan)
    ok = np.isfinite(native).all(1) & (native[:, 2] > 0)
    idx = np.flatnonzero(ok)
    if len(idx) == 0:
        return out
    scale = smoothing.target_size_scale if smoothing is not None else 1.0
    margin = smoothing.coverage_margin_fraction if smoothing is not None else 0.0
    jump = smoothing.reset_jump_fraction if smoothing is not None else 0.75
    brk = np.zeros(len(native), bool)
    brk[idx[0]] = True
    for a, b in zip(idx, idx[1:]):
        if (
            b != a + 1
            or np.hypot(*(native[b, :2] - native[a, :2])) > jump * native[b, 2]
        ):
            brk[b] = True
    starts = np.flatnonzero(brk)
    resets = []
    for s0 in starts:
        e0 = s0
        while e0 + 1 < len(native) and ok[e0 + 1] and not brk[e0 + 1]:
            e0 += 1
        run = native[s0 : e0 + 1]
        obs = observed[s0 : e0 + 1]
        if not obs.any():
            # filled-only run (cannot happen with drop_off_screen; keep the boxes)
            obs = np.ones(len(run), bool)
        sm = rts_smooth_square(run, obs, fps, cfg)
        if smoothing is None:
            out[s0 : e0 + 1] = sm
            continue
        req = (
            run[:, 2] / T.PAD
            + 2 * np.max(np.abs(sm[:, :2] - run[:, :2]), axis=1)
            + 2 * margin * run[:, 2]
        )
        # production size path of bbox_smoothing.smooth_bbox_track: EMA in log size
        # (grow / shrink weights) whose state is raised to the coverage guard every
        # frame
        size = np.empty(len(run))
        st = max(run[0, 2] * scale, req[0])
        size[0] = st
        for i in range(1, len(run)):
            tgt = run[i, 2] * scale
            a = _bs_alpha(
                smoothing.size_alpha_grow if tgt > st else smoothing.size_alpha_shrink,
                1.0 / fps,
            )
            st = max(float(np.exp((1 - a) * np.log(st) + a * np.log(tgt))), req[i])
            size[i] = st
        sm[:, 2] = size
        out[s0 : e0 + 1] = sm
        resets.append(s0)
    if smoothing is not None and resets:
        times = np.arange(len(out)) / fps
        _limit_size_changes(out, times, resets, smoothing.max_size_log_rate_per_second)
    return out


def track_clip(
    recs,
    W,
    H,
    fps,
    cfg,
    smoothing=None,
    frame_times=None,
    verify=None,
    enforce_lr=True,
):
    """recs: per frame {'L': [...], 'R': [...]} of [cx, cy, w, h, score, label_id]
    candidates (any threshold <= cfg.min_score(); filtered here). ``verify(t, cand)
    -> (presence, handedness)`` answers the MediaPipe check (``mediapipe_check``).
    Returns dict with 'native', 'final', 'present', 'emit', 'cand', 'score', 'src'
    (each {hand: array}) and 'stats'."""
    if not cfg.any_rule:
        raise ValueError(
            "track_clip needs at least one track_cleanup switch; "
            "use boxes_from_tracks otherwise"
        )
    cfg.validate()
    if cfg.mediapipe_check and verify is None:
        raise ValueError("mediapipe_check needs a verify callable")
    N = len(recs)
    min_size = T.MIN_SIZE_FRAC * W
    if frame_times is None:
        frame_times = np.arange(N) / fps

    # the gap bridge reads its own lower rows from ``recs`` below
    ms = cfg.select_min_score()
    recs_f = _filter_recs(recs, ms - 1e-12 if ms < T.THR_RESCUE else T.THR_RESCUE)
    stats = {}
    filled_mask = {}
    if cfg.sides_by_position:
        sel, st = select_position(recs_f, cfg, W, H, fps, verify)
        stats.update(st, engine="position")
        if cfg.sides_per_tracklet:
            sel, stats["sides_per_tracklet"] = relabel(sel, cfg, W, fps)
    else:
        sel, st = select_continuity(recs_f, cfg, W, H, fps, verify)
        stats.update(st, engine="continuity")
    if cfg.score_hysteresis:
        stats["segments_removed"] = _segment_filter(sel, cfg, fps)
    force = None
    if cfg.gap_bridge and cfg.drop_off_screen:
        rows = [
            sorted(
                [c for k in ("L", "R") for c in rec[k] if c[4] >= cfg.bridge_min_score],
                key=lambda c: -c[4],
            )
            for rec in recs
        ]
        force, st = bridge_gaps(sel, rows, W, H, fps, cfg, verify)
        stats.update(st)

    raw = {s: np.asarray(sel[s]["box"], np.float64).reshape(N, 4) for s in ("L", "R")}
    pres = {s: np.asarray(sel[s]["present"], bool) for s in ("L", "R")}
    sq = {}
    for s in ("L", "R"):
        q = np.zeros((N, 3))
        q[:, 0], q[:, 1] = raw[s][:, 0], raw[s][:, 1]
        q[:, 2] = np.maximum(raw[s][:, 2], raw[s][:, 3]) * T.PAD
        q[~pres[s]] = 0.0
        sq[s] = q
    if not cfg.sides_by_position and enforce_lr:
        # continuity engine: the stock per-frame left/right order rule
        # (tracking.enforce_lr_per_frame: swap where both hands have a box and the left
        # one is right of the right one), applied to every per-frame array
        swap = np.flatnonzero(pres["L"] & pres["R"] & (sq["L"][:, 0] > sq["R"][:, 0]))
        stats["lr_swapped"] = len(swap)
        for t in swap:
            sq["L"][t], sq["R"][t] = sq["R"][t].copy(), sq["L"][t].copy()
            raw["L"][t], raw["R"][t] = raw["R"][t].copy(), raw["L"][t].copy()
            for k in ("score", "src"):
                sel["L"][k][t], sel["R"][k][t] = sel["R"][k][t], sel["L"][k][t]
    native, emit = {}, {}
    center_speed = np.inf if cfg.constant_velocity else T.MAX_CENTER_SPEED
    for h, s in HANDS:
        if cfg.drop_off_screen:
            q, em, fl = fill_v2(
                sq[s],
                pres[s],
                raw[s],
                W,
                H,
                fps,
                cfg,
                None if force is None else force[s],
            )
            native[h] = apply_rules_runs(
                q, em, min_size, center_speed, T.MAX_SIZE_RATIO, T.CENTER_STILL
            )
            emit[h] = em
            filled_mask[h] = fl
        else:
            if pres[s].any():
                f, _ = T.apply_bbox_rules(
                    sq[s].astype(np.float32),
                    pres[s].copy(),
                    min_size,
                    center_speed,
                    T.MAX_SIZE_RATIO,
                    T.CENTER_STILL,
                )
                native[h] = f.astype(np.float64)
                emit[h] = np.ones(N, bool)
            else:
                native[h] = np.full((N, 3), np.nan)
                emit[h] = np.zeros(N, bool)
            filled_mask[h] = emit[h] & ~pres[s]
    present = {h: pres[s] for h, s in HANDS}
    final = {}
    for h, _ in HANDS:
        if not emit[h].any():
            final[h] = np.full((N, 3), np.nan)
        elif cfg.constant_velocity:
            final[h] = _rts_final(native[h], present[h], fps, cfg, smoothing)
        elif smoothing is not None:
            final[h] = smooth_bbox_track(
                np.where(emit[h][:, None], native[h], np.nan),
                frame_times,
                present[h],
                smoothing,
            ).astype(np.float64)
        else:
            final[h] = native[h].copy()
    if enforce_lr and smoothing is not None:
        _crossing_fix(native, final, present)
    cand = {h: raw[s] for h, s in HANDS}
    score = {h: np.asarray(sel[s]["score"], np.float64) for h, s in HANDS}
    src = {h: np.asarray(sel[s]["src"], np.int8) for h, s in HANDS}
    stats["filled"] = {h: int(filled_mask[h].sum()) for h in filled_mask}
    stats["present"] = {h: int(present[h].sum()) for h in present}
    stats["emitted"] = {h: int(emit[h].sum()) for h in emit}
    return {
        "native": {h: native[h].astype(np.float32) for h in native},
        "final": {h: final[h].astype(np.float32) for h in final},
        "present": present,
        "emit": emit,
        "cand": cand,
        "score": score,
        "src": src,
        "stats": stats,
    }
