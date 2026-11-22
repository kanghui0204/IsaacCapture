# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-OpenMDW-1.1

"""``sides_per_tracklet``: one side per tracklet, voted over its whole life.

The position engine (``select.select_position``) decides the side frame by frame: two
boxes -> x order, one box -> image half. A hand that hovers around the image centre
then flips sides from frame to frame, and two crossing hands swap. Here the real boxes
the engine chose (at most two per frame) are linked into tracklets without looking at
the side, and each tracklet gets one side from the sum of its per-frame evidence
(+ = right):

* two boxes on the frame: x order (+-1) scaled by
  ``clip((|dx| / mean side - tracklet_sep_lo) / (tracklet_sep_hi - tracklet_sep_lo),
  0, 1)``;
* one box: ``clip((cx - W/2) / (tracklet_band * W), -1, 1)``.

The RT-DETR class is not used. Tracklets are given sides in order of |vote|; one that
would take the side of a tracklet it shares a frame with takes the other side, or
keeps the per-frame sides when both are taken. With ``tracklet_weak_only`` > 0 only
frames whose own evidence (the scale factor above, or |lone-box vote|) is below it
take the tracklet's side; the others keep the side the engine gave them. A frame whose
two boxes would end up on one side keeps the per-frame sides.
"""

import math

from .motion import side_of


def link_tracklets(items_by_t, N, cfg, fps):
    """items_by_t[t]: list of item dicts with 'box' [cx, cy, w, h]. Sets item['tl'] and
    returns the tracklets.

    A box links to a live tracklet (last box at most tracklet_gap_s ago) when its centre
    is within tracklet_dist box sides of the tracklet's constant-velocity prediction and
    the side ratio is within e^+-tracklet_size. A box with two candidate tracklets (or a
    tracklet with two candidate boxes) whose distances differ by < tracklet_ambiguity
    gets no link and the tracklet ends. At most two boxes per frame: the one-to-one
    matching with the most links, then the lowest total distance, is found by
    enumeration."""
    max_gap = max(1, round(cfg.tracklet_gap_s * fps))
    tracks = []
    for t in range(N):
        items = items_by_t[t]
        if not items:
            continue
        act = [
            k
            for k, tr in enumerate(tracks)
            if tr["alive"] and t - tr["last_t"] <= max_gap
        ]
        for tr in tracks:
            if tr["alive"] and t - tr["last_t"] > max_gap:
                tr["alive"] = False
        D = {}
        for i, it in enumerate(items):
            b = it["box"]
            for k in act:
                tr = tracks[k]
                dt = t - tr["last_t"]
                px, py = tr["box"][0] + tr["v"][0] * dt, tr["box"][1] + tr["v"][1] * dt
                s = max(side_of(b), side_of(tr["box"]))
                d = math.hypot(b[0] - px, b[1] - py) / s
                if (
                    d <= cfg.tracklet_dist
                    and abs(math.log(side_of(b) / side_of(tr["box"])))
                    <= cfg.tracklet_size
                ):
                    D[(i, k)] = d
        amb_i, amb_k = set(), set()
        for i in range(len(items)):
            ds = sorted(d for (ii, k), d in D.items() if ii == i)
            if len(ds) >= 2 and ds[1] - ds[0] < cfg.tracklet_ambiguity:
                amb_i.add(i)
        for k in act:
            ds = sorted(d for (i, kk), d in D.items() if kk == k)
            if len(ds) >= 2 and ds[1] - ds[0] < cfg.tracklet_ambiguity:
                amb_k.add(k)
        cand = [(i, k) for (i, k) in D if i not in amb_i and k not in amb_k]
        best, best_key = {}, None

        def search(idx, used_k, cur, cost, items=items, cand=cand, D=D):
            nonlocal best, best_key
            if idx == len(items):
                key = (-len(cur), cost)
                if best_key is None or key < best_key:
                    best_key, best = key, dict(cur)
                return
            search(idx + 1, used_k, cur, cost)
            for i, k in cand:
                if i == idx and k not in used_k:
                    cur[i] = k
                    search(idx + 1, used_k | {k}, cur, cost + D[(i, k)])
                    del cur[i]

        search(0, frozenset(), {}, 0.0)
        for k in amb_k:
            tracks[k]["alive"] = False
        for i, it in enumerate(items):
            b = it["box"]
            if i in best:
                k = best[i]
                tr = tracks[k]
                dt = t - tr["last_t"]
                tr["v"] = ((b[0] - tr["box"][0]) / dt, (b[1] - tr["box"][1]) / dt)
                tr["box"], tr["last_t"] = b, t
                tr["items"].append(it)
                it["tl"] = k
            else:
                tracks.append(
                    {
                        "alive": True,
                        "last_t": t,
                        "box": b,
                        "v": (0.0, 0.0),
                        "items": [it],
                    }
                )
                it["tl"] = len(tracks) - 1
    return tracks


def relabel(sel, cfg, W, fps):
    """sel: select_position output {'L'/'R': {'present', 'box', 'score', 'src'}} -> (new
    sel, same format; stats)."""
    N = len(sel["L"]["present"])
    items_by_t = [[] for _ in range(N)]
    for s in ("L", "R"):
        o = sel[s]
        for t in range(N):
            if o["present"][t]:
                items_by_t[t].append(
                    {
                        "t": t,
                        "side0": s,
                        "box": list(o["box"][t]),
                        "score": o["score"][t],
                        "src": o["src"][t],
                    }
                )
    tracks = link_tracklets(items_by_t, N, cfg, fps)
    sep = cfg.tracklet_sep_hi > cfg.tracklet_sep_lo
    for tr in tracks:
        e = 0.0
        for it in tr["items"]:
            t, b = it["t"], it["box"]
            others = [x for x in items_by_t[t] if x is not it]
            if others:
                ob = others[0]["box"]
                w = 1.0
                if sep:
                    r = abs(b[0] - ob[0]) / (0.5 * (side_of(b) + side_of(ob)))
                    w = max(
                        0.0,
                        min(
                            1.0,
                            (r - cfg.tracklet_sep_lo)
                            / (cfg.tracklet_sep_hi - cfg.tracklet_sep_lo),
                        ),
                    )
                v = w * (1.0 if b[0] > ob[0] else -1.0)
                it["strength"] = w
            else:
                v = max(-1.0, min(1.0, (b[0] - W / 2) / (cfg.tracklet_band * W)))
                it["strength"] = abs(v)
            e += v
        tr["vote"] = e
        tr["frames"] = {it["t"] for it in tr["items"]}
    order = sorted(range(len(tracks)), key=lambda k: -abs(tracks[k]["vote"]))
    side_at = [dict() for _ in range(N)]
    stats = {
        "tracklets": len(tracks),
        "flipped_frames": 0,
        "collisions": 0,
        "kept_frame_sides": 0,
    }
    for k in order:
        tr = tracks[k]
        want = "R" if tr["vote"] > 0 else "L"
        if any(want in side_at[t] for t in tr["frames"]):
            stats["collisions"] += 1
            other = "L" if want == "R" else "R"
            if any(other in side_at[t] for t in tr["frames"]):
                # both sides taken on some shared frame: keep the per-frame sides
                tr["side"] = None
                continue
            want = other
        tr["side"] = want
        for t in tr["frames"]:
            side_at[t][want] = k
    out = {
        s: {
            "present": [False] * N,
            "box": [[0.0] * 4 for _ in range(N)],
            "score": [0.0] * N,
            "src": [0] * N,
        }
        for s in ("L", "R")
    }
    for t in range(N):
        its = items_by_t[t]
        sides = [
            it["side0"]
            if (cfg.tracklet_weak_only > 0 and it["strength"] >= cfg.tracklet_weak_only)
            else (tracks[it["tl"]]["side"] or it["side0"])
            for it in its
        ]
        if len(its) == 2 and sides[0] == sides[1]:
            sides = [it["side0"] for it in its]
            stats["kept_frame_sides"] += 1
        for it, s in zip(its, sides):
            if s != it["side0"]:
                stats["flipped_frames"] += 1
            o = out[s]
            o["present"][t] = True
            o["box"][t] = list(it["box"])
            o["score"][t] = it["score"]
            o["src"][t] = it["src"]
    return out, stats
