# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-OpenMDW-1.1

"""Geometry helpers, per-hand predictors (last box / constant-velocity Kalman) and the
RTS smoother.

Boxes are centre form ``[cx, cy, w, h]`` (detector candidates) or ``[cx, cy, side]``
(square tracker boxes). Motion noise is expressed in box sides so the same constants
work at any resolution; ``dt`` is in frames at ``REF_FPS`` (a 20 fps clip has dt = 1.5
per frame).
"""

import math

import numpy as np


def iou_cxcywh(a, b) -> float:
    if a is None or b is None:
        return 0.0
    ax0, ay0, ax1, ay1 = (
        a[0] - a[2] / 2,
        a[1] - a[3] / 2,
        a[0] + a[2] / 2,
        a[1] + a[3] / 2,
    )
    bx0, by0, bx1, by1 = (
        b[0] - b[2] / 2,
        b[1] - b[3] / 2,
        b[0] + b[2] / 2,
        b[1] + b[3] / 2,
    )
    ix = max(0.0, min(ax1, bx1) - max(ax0, bx0))
    iy = max(0.0, min(ay1, by1) - max(ay0, by0))
    inter = ix * iy
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 0 else 0.0


def side_of(c) -> float:
    """Characteristic size of a [cx, cy, w, h] box (geometric mean of w and h, at least
    1 px)."""
    return max(math.sqrt(max(c[2], 1e-6) * max(c[3], 1e-6)), 1.0)


def visible_fraction(cx, cy, w, h, W, H) -> float:
    x0, x1 = max(0.0, cx - w / 2), min(float(W), cx + w / 2)
    y0, y1 = max(0.0, cy - h / 2), min(float(H), cy + h / 2)
    a = max(w, 1e-6) * max(h, 1e-6)
    return max(0.0, x1 - x0) * max(0.0, y1 - y0) / a


def touched_borders(c, W, H, margin):
    """Borders (as outward unit vectors) a [cx,cy,w,h] box touches within ``margin``
    (fraction of image size)."""
    mx, my = margin * W, margin * H
    out = []
    if c[0] - c[2] / 2 <= mx:
        out.append((-1.0, 0.0))
    if c[0] + c[2] / 2 >= W - mx:
        out.append((1.0, 0.0))
    if c[1] - c[3] / 2 <= my:
        out.append((0.0, -1.0))
    if c[1] + c[3] / 2 >= H - my:
        out.append((0.0, 1.0))
    return out


class LastBox:
    """``constant_velocity`` off: the prediction is the last real box."""

    def __init__(self):
        self.box = None
        self.t = None

    def active(self, t, max_age):
        return self.box is not None and (max_age is None or t - self.t <= max_age)

    def predict(self, t):
        return self.box

    def gate(self, c, t, cfg):
        """(passes, cost) for continuing this hand with candidate c at frame t."""
        v = iou_cxcywh(c, self.box)
        return v >= cfg.gate_iou, 1.0 - v

    def update(self, t, c):
        self.box, self.t = list(c), t


class CVKalman:
    """``constant_velocity``: Kalman filter on the box centre (x and y independent),
    noise in box sides.

    state per axis [p, v]; Q = discrete white-noise acceleration (q_acc * side)^2; R =
    (r_pos * side)^2. The size is not filtered for gating; the last measured w, h are
    used for the predicted box.
    """

    def __init__(self, cfg, fps):
        self.cfg = cfg
        self.k = 30.0 / fps  # reference frames per clip frame
        self.x = None  # (2 axes, 2)
        self.P = None  # (2 axes, 2, 2)
        self.wh = None
        self.t = None

    def active(self, t, max_age):
        return self.x is not None and (max_age is None or t - self.t <= max_age)

    def _propagate(self, t):
        dt = (t - self.t) * self.k
        F = np.array([[1.0, dt], [0.0, 1.0]])
        s = side_of([0, 0, *self.wh])
        q = (self.cfg.q_acc * s) ** 2
        Q = q * np.array([[dt**4 / 4, dt**3 / 2], [dt**3 / 2, dt**2]])
        x = self.x @ F.T
        P = F @ self.P @ F.T + Q
        return x, P

    def predict(self, t):
        if self.x is None:
            return None
        if t == self.t:
            return [self.x[0, 0], self.x[1, 0], *self.wh]
        x, _ = self._propagate(t)
        return [x[0, 0], x[1, 0], *self.wh]

    def velocity(self):
        """Velocity in pixels per CLIP frame."""
        if self.x is None:
            return None
        return (self.x[0, 1] * self.k, self.x[1, 1] * self.k)

    def gate(self, c, t, cfg):
        x, P = (self.x, self.P) if t == self.t else self._propagate(t)
        s = side_of(c)
        R = (cfg.r_pos * s) ** 2
        nis = 0.0
        for ax in range(2):
            S = P[ax, 0, 0] + R
            nis += (c[ax] - x[ax, 0]) ** 2 / S
        sp = side_of([0, 0, *self.wh])
        dls = abs(math.log(side_of(c) / sp))
        # the CV covariance grows with the cube of the coasting time; cap the gate at
        # gate_max box sides
        dist = math.hypot(c[0] - x[0, 0], c[1] - x[1, 0]) / max(sp, s)
        return (
            nis <= cfg.gate_chi2 and dls <= cfg.gate_size and dist <= cfg.gate_max
        ), nis

    def update(self, t, c):
        s = side_of(c)
        if self.x is None:
            self.x = np.array([[c[0], 0.0], [c[1], 0.0]])
            v0 = (0.3 * s) ** 2  # initial velocity uncertainty: 0.3 sides / frame
            p0 = (self.cfg.r_pos * s) ** 2
            self.P = np.array([[[p0, 0.0], [0.0, v0]]] * 2)
        else:
            x, P = self._propagate(t)
            R = (self.cfg.r_pos * s) ** 2
            for ax in range(2):
                S = P[ax, 0, 0] + R
                K = P[ax, :, 0] / S
                y = c[ax] - x[ax, 0]
                x[ax] = x[ax] + K * y
                P[ax] = P[ax] - np.outer(K, P[ax, 0, :])
            self.x, self.P = x, P
        self.wh = (float(c[2]), float(c[3]))
        self.t = t


def rts_1d(z, obs, dt, q, r, v0):
    """RTS smoother of one constant-velocity axis.

    z (N,) measurements (ignored where obs False), q (N,) acceleration variance per
    frame, r (N,) measurement variance, dt reference frames per clip frame, v0 initial
    velocity variance. Returns smoothed positions (N,).
    """
    n = len(z)
    F = np.array([[1.0, dt], [0.0, 1.0]])
    G = np.array([[dt**4 / 4, dt**3 / 2], [dt**3 / 2, dt**2]])
    xf = np.zeros((n, 2))
    Pf = np.zeros((n, 2, 2))
    xp = np.zeros((n, 2))
    Pp = np.zeros((n, 2, 2))
    first = int(np.flatnonzero(obs)[0])
    x = np.array([z[first], 0.0])
    P = np.array([[r[first], 0.0], [0.0, v0]])
    for i in range(n):
        if i > 0:
            x = F @ x
            P = F @ P @ F.T + q[i] * G
        xp[i], Pp[i] = x, P
        if obs[i]:
            S = P[0, 0] + r[i]
            K = P[:, 0] / S
            x = x + K * (z[i] - x[0])
            P = P - np.outer(K, P[0, :])
        xf[i], Pf[i] = x, P
    xs = xf.copy()
    for i in range(n - 2, -1, -1):
        C = Pf[i] @ F.T @ np.linalg.inv(Pp[i + 1])
        xs[i] = xf[i] + C @ (xs[i + 1] - xp[i + 1])
    return xs[:, 0]


def rts_smooth_square(sq, obs, fps, cfg):
    """RTS-smooth one run of square boxes (N,3) [cx, cy, side]; centre in pixels, side
    in log space.

    ``obs`` marks frames with a real detection; other frames of the run (gap fills) are
    estimated by the model.
    """
    sq = np.asarray(sq, np.float64)
    obs = np.asarray(obs, bool)
    if not obs.any():
        return sq.copy()
    dt = 30.0 / fps
    side = sq[:, 2]
    # noise scales with the (observed) box side, carried into gaps
    s_ref = np.interp(np.arange(len(sq)), np.flatnonzero(obs), side[obs])
    s_box = s_ref / 1.4  # square side = PAD * detection side
    q = (cfg.q_acc * s_box) ** 2
    r = (cfg.r_pos * s_box) ** 2
    v0 = (0.3 * s_box[np.flatnonzero(obs)[0]]) ** 2
    out = sq.copy()
    out[:, 0] = rts_1d(sq[:, 0], obs, dt, q, r, v0)
    out[:, 1] = rts_1d(sq[:, 1], obs, dt, q, r, v0)
    ls = np.log(np.maximum(side, 1e-6))
    n = len(sq)
    out[:, 2] = np.exp(
        rts_1d(ls, obs, dt, np.full(n, cfg.q_ls**2), np.full(n, cfg.r_ls**2), 0.05**2)
    )
    # frames before the first / after the last real box of the run (the stock lead/trail
    # hold when drop_off_screen is off) keep the smoothed first / last box instead of a
    # constant-velocity extrapolation
    oi = np.flatnonzero(obs)
    out[: oi[0]] = out[oi[0]]
    out[oi[-1] + 1 :] = out[oi[-1]]
    return out
