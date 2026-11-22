# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-OpenMDW-1.1

"""``drop_off_screen`` gap filling (short gaps only, no holding at the clip ends, no
fill across an exit), the ``gap_bridge`` on detector evidence, and the stock box rules
applied per emitted run.

Square boxes ``[cx, cy, side]`` as in ``tracking.apply_bbox_rules``
(side = PAD * max(w, h)).
"""

import math

import numpy as np

from .config import REF_FPS
from .motion import iou_cxcywh, touched_borders, visible_fraction


def _velocity(raw, present, i, direction, fps=REF_FPS):
    """Mean centre velocity (px/frame) over up to 4 real boxes ending (direction -1) /
    starting (+1) at i, at most 4 frames away at 30 fps (both scale with the rate)."""
    span = max(1, round(4 * fps / REF_FPS))
    idx = [i]
    j = i + direction
    while len(idx) < span and 0 <= j < len(present) and abs(j - i) <= span:
        if present[j]:
            idx.append(j)
        j += direction
    if len(idx) < 2:
        return None
    a, b = (idx[-1], idx[0]) if direction < 0 else (idx[0], idx[-1])
    dt = b - a
    return ((raw[b][0] - raw[a][0]) / dt, (raw[b][1] - raw[a][1]) / dt)


def _crosses_border(raw, present, i, direction, W, H, cfg, fps=REF_FPS):
    """Leaving (direction -1, box i is the last before a gap) or entering (+1, first
    after a gap) through a border the box touches, moving toward it faster than
    ``exit_speed`` sides/frame (at 30 fps)."""
    c = raw[i]
    borders = touched_borders(c, W, H, cfg.edge_margin)
    if not borders:
        return False
    v = _velocity(raw, present, i, direction, fps)
    if v is None:
        return False
    # sides per frame at the clip rate -> sides per frame at 30 fps
    s = max(c[2], c[3], 1.0) * REF_FPS / fps
    for bx, by in borders:
        # leaving: velocity toward the border; entering: velocity away from it (coming
        # in from outside)
        toward = (v[0] * bx + v[1] * by) / s
        if direction < 0 and toward >= cfg.exit_speed:
            return True
        if direction > 0 and -toward >= cfg.exit_speed:
            return True
    return False


def _v2_accepts(sq, present, raw, a, b, W, H, fps, cfg):
    """The ``drop_off_screen`` rule for the gap between the real boxes a and b: short,
    ends close, no border crossing."""
    g = b - a - 1
    max_gap = max(1, round(cfg.max_gap_s * fps))
    if g < 1 or g > max_gap:
        return False
    # distance rule on the detection boxes (HaMeR: centre distance <= mean box width)
    da = max(raw[a][2], raw[a][3])
    db = max(raw[b][2], raw[b][3])
    if math.hypot(sq[b, 0] - sq[a, 0], sq[b, 1] - sq[a, 1]) > cfg.gap_max_dist * 0.5 * (
        da + db
    ):
        return False
    if _crosses_border(raw, present, a, -1, W, H, cfg, fps) or _crosses_border(
        raw, present, b, +1, W, H, cfg, fps
    ):
        return False
    return True


def bridge_gaps(sel, rows, W, H, fps, cfg, verify=None):
    """``gap_bridge``: fill gaps ``drop_off_screen`` would not fill when detector rows
    lie on the straight path between the two ends.

    sel: {'L'|'R': {'present', 'box', 'score', 'src'}} lists, modified in place (the
    supporting rows become selected detections, src 3). rows: per frame [cx, cy, w, h,
    score, ...] candidates of both classes (score-sorted). With a MediaPipe ``verify``
    (``mediapipe_check``) one supporting row must have presence >=
    ``bridge_min_presence``. Returns ({'L'|'R': bool (N,) frames inside a bridged
    gap}, stats)."""
    N = len(rows)
    max_g = max(1, round(cfg.bridge_max_s * fps))
    force = {}
    stats = {"bridged_gaps": 0, "bridged_rows": 0}
    for s, o in (("L", "R"), ("R", "L")):
        pres = np.asarray(sel[s]["present"], bool)
        raw = np.asarray(sel[s]["box"], np.float64).reshape(N, 4)
        opres = np.asarray(sel[o]["present"], bool)
        oraw = np.asarray(sel[o]["box"], np.float64).reshape(N, 4)
        sq = np.zeros((N, 3))
        sq[:, :2] = raw[:, :2]
        fm = np.zeros(N, bool)
        idx = np.flatnonzero(pres)
        for a, b in zip(idx, idx[1:]):
            g = b - a - 1
            if g < 1 or g > max_g or _v2_accepts(sq, pres, raw, a, b, W, H, fps, cfg):
                continue
            sup = []
            for t in range(a + 1, b):
                w = (t - a) / (b - a)
                p = (1 - w) * raw[a] + w * raw[b]
                best, bc = 0.0, None
                for c in rows[t]:
                    if c[4] < cfg.bridge_min_score:
                        continue
                    if opres[t] and iou_cxcywh(c, oraw[t]) >= cfg.bridge_other_iou:
                        # a row on the other hand's box is not evidence for this hand
                        continue
                    v = iou_cxcywh(c, p)
                    if v > best:
                        best, bc = v, c
                if best >= cfg.bridge_iou:
                    sup.append((t, bc))
            if len(sup) < cfg.bridge_share * g:
                continue
            if cfg.bridge_min_presence > 0 and verify is not None:
                ok = False
                for t, c in sup:
                    presence, _ = verify(t, c)
                    # NaN (not computed) passes, as in the per-frame check
                    if presence != presence or presence >= cfg.bridge_min_presence:
                        ok = True
                        break
                if not ok:
                    continue
            stats["bridged_gaps"] += 1
            fm[a + 1 : b] = True
            for t, c in sup:
                sel[s]["present"][t] = True
                sel[s]["box"][t] = list(c[:4])
                sel[s]["score"][t] = float(c[4])
                sel[s]["src"][t] = 3
                stats["bridged_rows"] += 1
        force[s] = fm
    return force, stats


def fill_v2(sq, present, raw, W, H, fps, cfg, force=None):
    """-> (sq_filled (N,3), emit (N,), filled (N,)). Gaps are filled linearly between
    the two real boxes. force: frames of bridged gaps (``gap_bridge``), filled without
    the length / distance / border rules."""
    sq = np.asarray(sq, np.float64).copy()
    present = np.asarray(present, bool)
    emit = present.copy()
    filled = np.zeros_like(present)
    idx = np.flatnonzero(present)
    for a, b in zip(idx, idx[1:]):
        if b - a - 1 < 1:
            continue
        if not (force is not None and force[a + 1 : b].all()) and not _v2_accepts(
            sq, present, raw, a, b, W, H, fps, cfg
        ):
            continue
        for t in range(a + 1, b):
            w = (t - a) / (b - a)
            box = (1 - w) * sq[a] + w * sq[b]
            rw = (1 - w) * np.asarray(raw[a][:4]) + w * np.asarray(raw[b][:4])
            if visible_fraction(rw[0], rw[1], rw[2], rw[3], W, H) < cfg.min_visible:
                continue
            sq[t] = box
            emit[t] = True
            filled[t] = True
    return sq, emit, filled


def apply_rules_runs(
    sq, emit, min_size, max_center_speed, max_size_ratio, center_still
):
    """tracking.apply_bbox_rules steps 2-5 (min size, size freeze, size rate, centre
    speed) on each run of emitted frames separately; no gap filling here (done before).
    Frames outside runs become NaN."""
    sq = np.asarray(sq, np.float64).copy()
    emit = np.asarray(emit, bool)
    out = np.full_like(sq, np.nan)
    idx = np.flatnonzero(emit)
    if len(idx) == 0:
        return out
    starts = np.r_[idx[0], idx[1:][np.diff(idx) > 1]]
    ends = np.r_[idx[:-1][np.diff(idx) > 1], idx[-1]]
    for s0, e0 in zip(starts, ends):
        b = sq[s0 : e0 + 1].copy()
        b[:, 2] = np.maximum(b[:, 2], min_size)
        anchor_c = b[0, :2].copy()
        for t in range(1, len(b)):
            prev_c = b[t - 1, :2].copy()
            s_prev = b[t - 1, 2]
            d = b[t, :2] - prev_c
            dist = float(np.hypot(d[0], d[1]))
            ad = b[t, :2] - anchor_c
            still = float(np.hypot(ad[0], ad[1])) <= center_still
            if not still:
                anchor_c = b[t, :2].copy()
            if still:
                if s_prev > 0 and abs(b[t, 2] - s_prev) > 1e-6:
                    b[t, 2] = s_prev
            elif s_prev > 0:
                hi, lo = s_prev * max_size_ratio, s_prev / max_size_ratio
                if b[t, 2] > hi:
                    b[t, 2] = hi
                elif b[t, 2] < lo:
                    b[t, 2] = lo
            if dist > max_center_speed:
                b[t, :2] = prev_c + d * (max_center_speed / dist)
        out[s0 : e0 + 1] = b
    return out
