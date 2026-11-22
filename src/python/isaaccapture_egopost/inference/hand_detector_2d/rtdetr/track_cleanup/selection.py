# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-OpenMDW-1.1

"""Frame-by-frame choice of the left and right box.

Input records are the tracker's ``{'L': [[cx, cy, w, h, score, label_id], ...], 'R':
[...]}`` per frame, holding every detector candidate above
``TrackCleanupConfig.min_score()``. Output per side: ``present`` (N,), ``box`` (N, 4)
the chosen candidate [cx, cy, w, h], ``score`` (N,), ``src`` (N,) 0 none / 1 used on its
own / 2 continuation of a hand below the start threshold.

Two engines:

* continuity (``sides_by_position`` off): the stock 2x2 assignment (score + IoU with
  the hand's previous box), generalised -- with every switch off and every candidate
  above 0.15 it reproduces ``tracking.select_new`` exactly (unit-tested);
* position (``sides_by_position`` on): the image-position rule plus the
  ``score_hysteresis`` / ``constant_velocity`` continuation rules,
  ``containment_nms`` and the MediaPipe skip rules.

Both engines apply ``mediapipe_check`` through ``verify(t, cand) -> (presence,
handedness)`` (``mediapipe_check.LazyVerifier``).
"""

import math

from .. import tracking as T
from .motion import CVKalman, LastBox, iou_cxcywh, side_of, visible_fraction

SIDES = ("L", "R")


def contain_cxcywh(a, b):
    """Intersection of two [cx, cy, w, h] boxes divided by the area of the smaller
    one."""
    ix = max(
        0.0,
        min(a[0] + a[2] / 2, b[0] + b[2] / 2) - max(a[0] - a[2] / 2, b[0] - b[2] / 2),
    )
    iy = max(
        0.0,
        min(a[1] + a[3] / 2, b[1] + b[3] / 2) - max(a[1] - a[3] / 2, b[1] - b[3] / 2),
    )
    return ix * iy / min(max(a[2] * a[3], 1e-6), max(b[2] * b[3], 1e-6))


def _contained(c, p, cfg):
    """``containment_nms``: c and p overlap by >= containment_frac of the smaller box
    (hand box inside a hand-plus-forearm box)."""
    return cfg.containment_nms and contain_cxcywh(c, p) >= cfg.containment_frac


def _frames_to_pools(recs, cfg, W, H):
    """Per frame candidate list (score-sorted) after de-duplication / NMS / size
    rule."""
    pools = []
    max_area = cfg.max_area_frac * W * H if cfg.score_hysteresis else None
    for rec in recs:
        if cfg.sides_by_position:
            allc = sorted(
                [list(c) for key in ("L", "R") for c in rec[key]], key=lambda c: -c[4]
            )
            pool = []
            for c in allc:
                if all(
                    iou_cxcywh(c, p) < cfg.nms_iou and not _contained(c, p, cfg)
                    for p in pool
                ):
                    pool.append(c)
        else:
            # tracking.pool_candidates de-duplication (IoU > 0.9 merge, keep the better
            # score and its label)
            pool = []
            for key in ("L", "R"):
                for c in rec[key]:
                    dup = None
                    for p in pool:
                        if T.iou_xywh(p, c) > 0.9:
                            dup = p
                            break
                    if dup is None:
                        pool.append(list(c))
                    elif c[4] > dup[4]:
                        dup[4] = c[4]
                        if len(dup) > 5 and len(c) > 5:
                            dup[5] = c[5]
            pool.sort(key=lambda c: -c[4])
        if max_area is not None:
            pool = [c for c in pool if c[2] * c[3] <= max_area]
        pools.append(pool)
    return pools


class _Hands:
    """The two per-hand predictors plus the activity / exit rules shared by both
    engines."""

    def __init__(self, cfg, fps, W, H):
        self.cfg = cfg
        self.W, self.H = W, H
        mk = (lambda: CVKalman(cfg, fps)) if cfg.constant_velocity else LastBox
        self.p = {s: mk() for s in SIDES}
        self.max_age = (
            max(1, round(cfg.max_age_s * fps))
            if (cfg.score_hysteresis or cfg.constant_velocity)
            else None
        )
        # mediapipe_skip_reacquired: real detections of the current run per hand (a
        # run restarts after a gap of more than max_age_s)
        self.run = dict.fromkeys(SIDES, 0)
        self.gap_reset = max(1, round(cfg.max_age_s * fps))

    def active(self, s, t):
        p = self.p[s]
        if not p.active(t, self.max_age):
            return False
        if self.cfg.drop_off_screen and self.max_age is not None:
            b = p.predict(t)
            if b is not None and visible_fraction(*b[:4], self.W, self.H) < 0.5:
                # predicted box mostly outside the image: the hand has left
                return False
        return True

    def gated(self, c, s, t):
        if not self.active(s, t):
            return False, math.inf
        return self.p[s].gate(c, t, self.cfg)

    def pred(self, s, t):
        return self.p[s].predict(t) if self.active(s, t) else None

    def update(self, s, t, c):
        p = self.p[s]
        new_run = p.t is None or t - p.t > self.gap_reset
        self.run[s] = 1 if new_run else self.run[s] + 1
        p.update(t, c)

    def recent(self, c, t, max_frames, dist, min_run):
        """``mediapipe_skip_reacquired``: c lies within ``dist`` box sides of the last
        box of a hand (either side) updated 1..max_frames ago whose run had at least
        ``min_run`` real detections."""
        for s in SIDES:
            p = self.p[s]
            if p.t is None or not (0 < t - p.t <= max_frames) or self.run[s] < min_run:
                continue
            last = [p.x[0, 0], p.x[1, 0], *p.wh] if isinstance(p, CVKalman) else p.box
            d = math.hypot(c[0] - last[0], c[1] - last[1])
            if d / max(side_of(c), side_of(last)) <= dist:
                return True
        return False

    def fast(self, s):
        """``mediapipe_skip_fast``: the hand's Kalman speed is >= fast_speed box sides
        per frame (30 fps)."""
        p = self.p[s]
        if not isinstance(p, CVKalman):
            return False
        vel = p.velocity()
        if vel is None:
            return False
        speed = math.hypot(*vel) / p.k / side_of([0, 0, *p.wh])
        return speed >= self.cfg.fast_speed


def _verified(verify, cfg, t, c):
    if not cfg.mediapipe_check or verify is None:
        return True
    presence, _ = verify(t, c)
    # NaN (a verify callable that did not compute it) passes: as if unchecked
    return not (presence == presence) or presence >= cfg.mediapipe_min_presence


def select_continuity(recs, cfg, W, H, fps, verify=None):
    pools = _frames_to_pools(recs, cfg, W, H)
    hands = _Hands(cfg, fps, W, H)
    thr_use = cfg.start_score if cfg.score_hysteresis else T.THR
    thr_rescue = (
        cfg.redetect_score
        if cfg.constant_velocity
        else (cfg.continue_score if cfg.score_hysteresis else T.THR_RESCUE)
    )
    valued = cfg.score_hysteresis
    out = {s: {"present": [], "box": [], "score": [], "src": []} for s in SIDES}
    stats = {"rescued": 0, "label_ignored": 0, "crossed": 0, "mediapipe_rejected": 0}
    for t, (rec, pool) in enumerate(zip(recs, pools)):
        gate = {}
        usable = []
        for c in pool:
            g = {s: hands.gated(c, s, t) for s in SIDES}
            gate[id(c)] = g
            any_g = g["L"][0] or g["R"][0]
            if c[4] > thr_use:
                pass
            elif c[4] >= thr_rescue and any_g:
                pass
            else:
                continue
            ask = not any_g or (
                cfg.mediapipe_check_continuations and c[4] <= cfg.start_score
            )
            if cfg.mediapipe_check and ask and not _verified(verify, cfg, t, c):
                stats["mediapipe_rejected"] += 1
                continue
            usable.append(c)
        usable.sort(key=lambda c: -c[4])
        usable = usable[:5]
        L = R = None
        if not usable:
            pass
        elif not hands.active("L", t) and not hands.active("R", t):
            if len(usable) == 1 and T.candidate_label(usable[0]) is not None:
                if T.candidate_label(usable[0]) == 1:
                    L = usable[0]
                else:
                    R = usable[0]
            else:
                s2 = sorted(usable[:2], key=lambda c: c[0])
                L = s2[0]
                R = s2[1] if len(s2) > 1 else None
            if valued:  # score_hysteresis: a hand starts only above start_score
                L = L if (L is not None and L[4] > cfg.start_score) else None
                R = R if (R is not None and R[4] > cfg.start_score) else None
        else:
            predL, predR = hands.pred("L", t), hands.pred("R", t)

            def val(c, s, pred, gate=gate):
                v = T.W_DET * c[4] + T.W_IOU * iou_cxcywh(c, pred)
                if valued:
                    tau = cfg.continue_score if gate[id(c)][s][0] else cfg.start_score
                    v -= T.W_DET * tau
                return v

            best = None
            opts = usable + [None]
            for a in opts:
                if valued and a is not None and val(a, "L", predL) <= 0:
                    continue
                for b in opts:
                    if a is not None and b is not None and a is b:
                        continue
                    if valued and b is not None and val(b, "R", predR) <= 0:
                        continue
                    v = 0.0
                    if a is not None:
                        v += val(a, "L", predL)
                    if b is not None:
                        v += val(b, "R", predR)
                    if a is None and b is None:
                        v = -1e9
                    if best is None or v > best[0]:
                        best = (v, a, b)
            L, R = best[1], best[2]
        topL = rec["L"][0] if rec["L"] else None
        if L is not None and topL is not None and iou_cxcywh(L, topL) < 0.5:
            stats["label_ignored"] += 1
        if L is not None and R is not None and L[0] > R[0]:
            stats["crossed"] += 1
        for s, c in (("L", L), ("R", R)):
            o = out[s]
            if c is None:
                o["present"].append(False)
                o["box"].append([0.0, 0.0, 0.0, 0.0])
                o["score"].append(0.0)
                o["src"].append(0)
            else:
                low = c[4] <= thr_use
                if low:
                    stats["rescued"] += 1
                o["present"].append(True)
                o["box"].append(list(c[:4]))
                o["score"].append(c[4])
                o["src"].append(2 if low else 1)
                hands.update(s, t, c)
    return out, stats


def select_position(recs, cfg, W, H, fps, verify=None):
    pools = _frames_to_pools(recs, cfg, W, H)
    hands = _Hands(cfg, fps, W, H)
    cont = cfg.score_hysteresis or cfg.constant_velocity
    thr_cont = cfg.redetect_score if cfg.constant_velocity else cfg.continue_score
    reacquire = (
        max(1, round(cfg.reacquire_s * fps)),
        cfg.reacquire_dist,
        max(1, round(cfg.reacquire_min_run_s * fps)),
    )
    out = {s: {"present": [], "box": [], "score": [], "src": []} for s in SIDES}
    stats = {
        "rescued": 0,
        "side_by_half": 0,
        "crossed": 0,
        "mediapipe_rejected": 0,
        "reacquired": 0,
        "fast_unchecked": 0,
    }
    for t, pool in enumerate(pools):
        high = []
        for c in pool:
            if c[4] <= cfg.start_score:
                continue
            if cfg.mediapipe_check:
                g_any = hands.gated(c, "L", t)[0] or hands.gated(c, "R", t)[0]
                if (
                    not g_any
                    and cfg.mediapipe_skip_reacquired
                    and hands.recent(c, t, *reacquire)
                ):
                    g_any = True  # a hand tracked here moments ago comes back
                    stats["reacquired"] += 1
                if not g_any and not _verified(verify, cfg, t, c):
                    stats["mediapipe_rejected"] += 1
                    continue
            high.append(c)
        pick = high[:2]
        A = {"L": None, "R": None}
        if len(pick) == 2:
            a, b = sorted(pick, key=lambda c: c[0])
            A["L"], A["R"] = a, b
        elif len(pick) == 1:
            c = pick[0]
            side = "R" if c[0] >= (W - 1) / 2 else "L"
            stats["side_by_half"] += 1
            A[side] = c
        if cont:
            for s in SIDES:
                if A[s] is not None or not hands.active(s, t):
                    continue
                other = A["R" if s == "L" else "L"]
                best = None
                fast = cfg.mediapipe_skip_fast and hands.fast(s)
                thr_c = min(cfg.fast_score, thr_cont) if fast else thr_cont
                for c in pool:
                    if c in pick or c[4] < thr_c:
                        continue
                    ok, cost = hands.gated(c, s, t)
                    if not ok:
                        continue
                    if other is not None and (
                        iou_cxcywh(c, other) >= cfg.nms_iou
                        or _contained(c, other, cfg)
                        or (s == "L" and c[0] > other[0])
                        or (s == "R" and c[0] < other[0])
                    ):
                        continue
                    check = cfg.mediapipe_check and cfg.mediapipe_check_continuations
                    if check and c[4] <= cfg.start_score:
                        if fast:
                            # a fast (blurred) hand is not asked
                            stats["fast_unchecked"] += 1
                        elif not _verified(verify, cfg, t, c):
                            stats["mediapipe_rejected"] += 1
                            continue
                    key = (c[4], -cost)
                    if best is None or key > best[0]:
                        best = (key, c)
                if best is not None:
                    A[s] = best[1]
                    stats["rescued"] += 1
        if A["L"] is not None and A["R"] is not None and A["L"][0] > A["R"][0]:
            stats["crossed"] += 1
        for s in SIDES:
            c = A[s]
            o = out[s]
            if c is None:
                o["present"].append(False)
                o["box"].append([0.0, 0.0, 0.0, 0.0])
                o["score"].append(0.0)
                o["src"].append(0)
            else:
                o["present"].append(True)
                o["box"].append(list(c[:4]))
                o["score"].append(c[4])
                o["src"].append(1 if c[4] > cfg.start_score else 2)
                hands.update(s, t, c)
    return out, stats
