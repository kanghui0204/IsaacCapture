# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-OpenMDW-1.1

"""Switches and constants of the RT-DETR hand track cleanup.

The ``track_cleanup:`` mapping of the rtdetr detector YAML. Every switch is off by
default, and a config with no switch on builds the stock detector
(``tracking.HandTracker`` + ``bbox_smoothing``) unchanged.

Times are in seconds and converted with the clip frame rate; motion constants are
per frame at 30 fps (``REF_FPS``) and rescaled for other rates; the defaults were
tuned and evaluated on 30 fps clips only. Scores are RT-DETR scores (0..1).
Processing order: per-frame choice -> sides per tracklet -> short segments -> gap
bridge -> gap filling -> smoothing.

sides_by_position
    Left/right from image position instead of the RT-DETR class: class-agnostic
    NMS (IoU ``nms_iou``), the two best boxes above ``start_score``; of two boxes
    the one with the smaller x is the left hand; one box takes the side of the
    image half it is on. The sides then never cross, so ``rtdetr_config.lr_order``
    has no further effect.
drop_off_screen
    No box for a hand that is not in the image: the first/last box is not held to
    the clip ends; a gap is filled only if it is short (``max_gap_s``), its two ends
    are close (``gap_max_dist`` box sides) and the hand was not leaving or entering
    through a border; a track whose predicted box is mostly outside the image
    ends; an emitted box must keep ``min_visible`` of its area inside the image.
score_hysteresis
    A hand starts only above ``start_score``; ``continue_score`` to
    ``start_score`` only continues a hand whose previous box (at most
    ``max_age_s`` ago) it overlaps. Segments shorter than ``min_seg_s`` whose best
    score stays below ``seg_keep_score`` are removed; boxes larger than
    ``max_area_frac`` of the image are dropped.
constant_velocity
    Constant-velocity Kalman filter per hand: continuation is gated on the
    predicted box (chi-square ``gate_chi2``, at most ``gate_max`` box sides),
    re-detection down to ``redetect_score`` inside the gate, no 90 px/frame centre
    clamp, and a forward-backward (RTS) smoother replaces the causal EMA centre
    smoothing, so the crop does not lag a fast hand. The crop size keeps the
    production rules (0.8 scale, coverage guard, EMA in log size, rate limit).
gap_bridge
    Fill a gap that ``drop_off_screen`` does not fill (up to ``bridge_max_s``)
    when detector boxes with score >= ``bridge_min_score`` (not on the other
    hand's box of that frame) lie on the straight path between the two ends (IoU
    >= ``bridge_iou`` with the interpolated box) on at least ``bridge_share`` of
    the gap frames. The supporting boxes become detections; the other gap frames
    are interpolated. Requires ``drop_off_screen``. Lowers the ONNX score
    threshold to ``bridge_min_score``; the per-frame choice still uses 0.15 and up.
containment_nms
    A candidate whose intersection with a higher-scored kept box covers >=
    ``containment_frac`` of the smaller box is dropped (a hand box inside a
    hand-plus-forearm box), and a continuation that lies that much inside the
    other hand's box is not taken. Requires ``sides_by_position``.
sides_per_tracklet
    The chosen boxes are linked into side-agnostic tracklets; each tracklet takes
    one side by a vote over its whole life (x order of two boxes, scaled by their
    separation; image half of a lone box); frames whose own evidence is weak
    (< ``tracklet_weak_only``) take the tracklet's side. Requires
    ``sides_by_position``.
"""

import dataclasses
import math
from dataclasses import dataclass

REF_FPS = 30.0
# tracking.THR_RESCUE: the ONNX score threshold of the stock tracker
STOCK_SCORE_THR = 0.15

# Switches in processing order. ``any_rule`` is true when one of them is on.
RULES = (
    "sides_by_position",
    "drop_off_screen",
    "score_hysteresis",
    "constant_velocity",
    "gap_bridge",
    "containment_nms",
    "sides_per_tracklet",
)
# Values in [0, 1]: scores, IoUs, presences, fractions and shares.
UNIT_INTERVAL = (
    "start_score",
    "continue_score",
    "redetect_score",
    "gate_iou",
    "seg_keep_score",
    "edge_margin",
    "min_visible",
    "mediapipe_min_presence",
    "bridge_iou",
    "bridge_share",
    "bridge_min_score",
    "bridge_other_iou",
    "bridge_min_presence",
    "fast_score",
    "containment_frac",
)
# Values in (0, 1]: 0 would drop every box (or keep one per frame for nms_iou).
POSITIVE_FRACTION = ("nms_iou", "max_area_frac")
# Values > 0: divisors, measurement noise, gates and matching distances.
POSITIVE = (
    "r_pos",
    "r_ls",
    "gate_chi2",
    "gate_size",
    "gate_max",
    "tracklet_band",
    "tracklet_dist",
    "tracklet_size",
)
# Switch -> switches it needs (a switch without them would silently do nothing).
REQUIRES = {
    "gap_bridge": ("drop_off_screen",),
    "containment_nms": ("sides_by_position",),
    "sides_per_tracklet": ("sides_by_position",),
}


@dataclass(frozen=True)
class TrackCleanupConfig:
    sides_by_position: bool = False
    drop_off_screen: bool = False
    score_hysteresis: bool = False
    constant_velocity: bool = False
    gap_bridge: bool = False
    containment_nms: bool = False
    sides_per_tracklet: bool = False
    # thresholds
    start_score: float = 0.5
    continue_score: float = 0.15
    redetect_score: float = 0.15
    nms_iou: float = 0.5
    gate_iou: float = 0.35
    max_age_s: float = 0.2
    # score_hysteresis
    min_seg_s: float = 0.1
    seg_keep_score: float = 0.8
    max_area_frac: float = 0.5
    # drop_off_screen
    max_gap_s: float = 25 / 30
    gap_max_dist: float = 1.0
    edge_margin: float = 0.01
    exit_speed: float = 0.02
    min_visible: float = 0.1
    # constant_velocity
    q_acc: float = 0.08
    r_pos: float = 0.06
    gate_chi2: float = 9.21
    gate_size: float = 0.7
    gate_max: float = 1.0
    q_ls: float = 0.03
    r_ls: float = 0.08
    # gap_bridge
    bridge_max_s: float = 2.0
    bridge_iou: float = 0.5
    bridge_share: float = 0.6
    bridge_min_score: float = 0.05
    bridge_other_iou: float = 0.2
    # containment_nms
    containment_frac: float = 0.7
    # sides_per_tracklet. Link a box to a tracklet seen at most tracklet_gap_s ago
    # when its centre is within tracklet_dist box sides of the predicted centre and
    # |log(side ratio)| <= tracklet_size; no link when the second-best match is
    # within tracklet_ambiguity. Votes: a lone box saturates at tracklet_band * W
    # from the image centre; two boxes vote by x order, scaled by
    # clip((|dx| / side - tracklet_sep_lo) / (tracklet_sep_hi - tracklet_sep_lo),
    # 0, 1). With tracklet_weak_only > 0 only frames whose own evidence is below it
    # take the tracklet's side.
    tracklet_gap_s: float = 0.2
    tracklet_dist: float = 0.6
    tracklet_size: float = 0.7
    tracklet_ambiguity: float = 0.1
    tracklet_band: float = 0.1
    tracklet_sep_lo: float = 0.25
    tracklet_sep_hi: float = 0.75
    tracklet_weak_only: float = 1.0

    @property
    def any_rule(self) -> bool:
        return any(getattr(self, name) for name in RULES)

    @property
    def name(self) -> str:
        """The enabled switches joined by '+', or 'off'; used in logs."""
        on = [name for name in RULES if getattr(self, name)]
        return "+".join(on) if on else "off"

    def min_score(self) -> float:
        """Lowest detector score any enabled rule uses (the ONNX score threshold)."""
        if not self.any_rule:
            return STOCK_SCORE_THR
        lo = self.select_min_score()
        if self.gap_bridge:
            lo = min(lo, self.bridge_min_score)
        return lo

    def select_min_score(self) -> float:
        """Lowest detector score the per-frame choice uses.

        The gap bridge reads its own, lower boxes (``min_score``).
        """
        lo = STOCK_SCORE_THR
        if self.sides_by_position or self.score_hysteresis:
            lo = min(lo, self.start_score)
        if self.score_hysteresis and not self.constant_velocity:
            lo = min(lo, self.continue_score)
        if self.constant_velocity:
            lo = min(lo, self.redetect_score)
        return lo

    @classmethod
    def from_dict(cls, data) -> "TrackCleanupConfig":
        """Build from the parsed ``track_cleanup:`` YAML mapping (``None`` or
        ``false`` = all off).

        Raises:
            ValueError: not a mapping, unknown key, wrongly typed, out-of-range
                or non-finite value, or a switch whose required switch is off.
        """
        if data is None or data is False:
            data = {}
        if not isinstance(data, dict):
            raise ValueError(
                f"track_cleanup must be a mapping, got {type(data).__name__}"
            )
        fields = {f.name: f for f in dataclasses.fields(cls)}
        unknown = set(data) - set(fields)
        if unknown:
            raise ValueError(
                f"track_cleanup has unknown key(s) {sorted(unknown)}; "
                f"valid keys: {', '.join(sorted(fields))}"
            )
        for key, value in data.items():
            default = fields[key].default
            if isinstance(default, bool):
                if not isinstance(value, bool):
                    raise ValueError(
                        f"track_cleanup {key} must be true or false, got {value!r}"
                    )
            elif (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
            ):
                raise ValueError(
                    f"track_cleanup {key} must be a finite number, got {value!r}"
                )
        config = cls(**data)
        config.validate()
        return config

    def validate(self) -> None:
        """Value ranges, relations between values, and switch dependencies."""
        for f in dataclasses.fields(self):
            value = getattr(self, f.name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                continue
            if value < 0:
                raise ValueError(
                    f"track_cleanup {f.name} must be nonnegative, got {value!r}"
                )
            if f.name in UNIT_INTERVAL and value > 1:
                raise ValueError(
                    f"track_cleanup {f.name} must be between 0 and 1, got {value!r}"
                )
            if f.name in POSITIVE_FRACTION and not 0 < value <= 1:
                raise ValueError(
                    f"track_cleanup {f.name} must be > 0 and <= 1, got {value!r}"
                )
            if f.name in POSITIVE and value == 0:
                raise ValueError(f"track_cleanup {f.name} must be > 0, got {value!r}")
        if self.continue_score > self.start_score:
            raise ValueError(
                f"track_cleanup continue_score ({self.continue_score}) must not "
                f"exceed start_score ({self.start_score})"
            )
        if self.tracklet_sep_lo >= self.tracklet_sep_hi:
            raise ValueError(
                f"track_cleanup tracklet_sep_lo ({self.tracklet_sep_lo}) must be "
                f"below tracklet_sep_hi ({self.tracklet_sep_hi})"
            )
        for name, needed in REQUIRES.items():
            if getattr(self, name):
                missing = [n for n in needed if not getattr(self, n)]
                if missing:
                    raise ValueError(
                        f"track_cleanup {name} requires {' and '.join(missing)}"
                    )
