# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-OpenMDW-1.1

"""RT-DETR hand detector with the track cleanup, a drop-in ``HandBBoxDetector``.

Same ONNX model, preprocessing and decoding as ``RTDetrHandBBoxDetector``; only the step
after detection changes: ``track.track_clip`` instead of ``detector.boxes_from_tracks``.
``build_detector`` returns the stock detector when no cleanup switch is on.
"""

import json
import logging
from pathlib import Path

import numpy as np

from ..bbox_smoothing import validate_frame_times
from ..detector import (
    RTDetrHandBBoxDetector,
    _is_valid_box,
    _providers_for_device,
    boxes_from_tracks,
)
from ..tracking import THR_RESCUE, HandTracker
from .config import TrackCleanupConfig
from .track import track_clip

log = logging.getLogger(__name__)


def boxes_from_tracks_cleanup(
    image_ids,
    detections,
    hands,
    image_size,
    cleanup: TrackCleanupConfig,
    *,
    smoothing=None,
    frame_times=None,
    verify=None,
    enforce_lr_order=True,
):
    """``detector.boxes_from_tracks`` with the track cleanup.

    Args:
        image_ids: Image IDs aligned with ``detections``, in temporal order (a
            complete clip).
        detections: Per-frame lists of ``onnx_model.Detection`` (or tracker
            records) holding every candidate above ``cleanup.min_score()``; lower
            ones are ignored.
        hands: Requested hand sides.
        image_size: ``(width, height)`` of the frames.
        cleanup: The switches. With none on, this is the stock detector:
            ``boxes_from_tracks`` (``drop_gap_filled`` off) on the candidates
            above ``tracking.THR_RESCUE``.
        smoothing: Optional ``BBoxSmoothingConfig``, as for ``boxes_from_tracks``.
        frame_times: Timestamps in seconds aligned with ``image_ids``; required.
        verify: ``mediapipe_check.LazyVerifier`` (or any ``fn(t, cand) ->
            (presence, handedness)``); required with ``mediapipe_check``.
        enforce_lr_order: As for ``boxes_from_tracks``.

    Returns:
        ``({hand: {image_id: np.float32([cx, cy, size])}}, info)``. ``info`` has
        the keys ``RTDetrHandBBoxDetector.log_tracking`` reads, plus ``cleanup``
        (the switch names) and ``cleanup_stats``.
    """
    width, height = image_size
    frame_times = validate_frame_times(frame_times, len(image_ids))
    recs = (
        detections
        if (detections and isinstance(detections[0], dict))
        else HandTracker.detections_to_records(detections)
    )
    if not cleanup.any_rule:
        # the stock detector: candidates above THR_RESCUE, then the stock tracker
        recs = [
            {k: [c for c in rec[k] if c[4] > THR_RESCUE] for k in ("L", "R")}
            for rec in recs
        ]
        return boxes_from_tracks(
            image_ids,
            recs,
            hands,
            image_width=width,
            enforce_lr_order=enforce_lr_order,
            smoothing=smoothing,
            frame_times=frame_times,
        )
    n = len(image_ids)
    d = np.diff(frame_times)
    fps = float(1.0 / np.median(d)) if len(d) else 30.0
    r = track_clip(
        recs,
        width,
        height,
        fps,
        cleanup,
        smoothing=smoothing,
        frame_times=frame_times,
        verify=verify,
        enforce_lr=enforce_lr_order,
    )
    out, per_hand = {}, {}
    for hand in hands:
        boxes, present = r["final"][hand], r["present"][hand]
        emitted = {
            image_id: boxes[i].astype(np.float32)
            for i, image_id in enumerate(image_ids)
            if np.isfinite(boxes[i]).all() and _is_valid_box(boxes[i])
        }
        out[hand] = emitted
        detected = int(present.sum())
        per_hand[hand] = {
            "frames": n,
            "detected": detected,
            "emitted": len(emitted),
            "gap_filled": max(0, len(emitted) - detected),
            "never_detected": detected == 0,
        }
    st = r["stats"]
    info = {
        "per_hand": per_hand,
        "rescued": int(st.get("rescued", 0)),
        "label_overridden": int(st.get("label_ignored", 0)),
        "crossed": int(st.get("crossed", 0)),
        "cleanup": cleanup.name,
        "cleanup_stats": st,
        "mediapipe_crops": getattr(verify, "computed", 0),
    }
    return out, info


class RTDetrTrackCleanupDetector(RTDetrHandBBoxDetector):
    """RT-DETR hand detector + track cleanup; needs whole clips and frame times.

    Args:
        weights: Path to the RT-DETRv2 ``.onnx`` file.
        cleanup: The switches; at least one must be on (``build_detector``
            returns the stock detector otherwise). ``mediapipe_check`` loads
            ``cleanup.mediapipe_weights`` on the detector's ONNX providers.
        device, providers, **detector_kwargs: Passed to
            ``RTDetrHandBBoxDetector`` (``RTDetrConfig.detector_kwargs``), which
            runs with ``use_tracker=True`` and ``drop_gap_filled=False``.
    """

    def __init__(
        self,
        weights: str,
        cleanup: TrackCleanupConfig,
        device: str = "cuda",
        providers: list | None = None,
        **detector_kwargs,
    ):
        if not cleanup.any_rule:
            raise ValueError(
                "RTDetrTrackCleanupDetector needs at least one track_cleanup switch"
            )
        cleanup.validate()
        # set before the parent constructor, which logs mode_name
        self.cleanup = cleanup
        detector_kwargs.update(use_tracker=True, drop_gap_filled=False)
        super().__init__(
            weights=weights, device=device, providers=providers, **detector_kwargs
        )
        # The cleanup may use candidates below the stock tracker threshold (gap_bridge,
        # redetect_score).
        if cleanup.min_score() < self.detector.score_thr:
            self.detector.score_thr = cleanup.min_score() - 1e-12
        self.verifier = None
        if cleanup.mediapipe_check:
            if not cleanup.mediapipe_weights:
                raise ValueError(
                    "mediapipe_check needs track_cleanup.mediapipe_weights"
                )
            from .mediapipe_check import MediaPipeHandVerifier

            self.verifier = MediaPipeHandVerifier(
                cleanup.mediapipe_weights,
                providers=providers
                if providers is not None
                else _providers_for_device(device),
            )
        log.info(
            "[RTDetrTrackCleanup] switches=%s detector score_thr=%.3f mediapipe=%s",
            cleanup.name,
            self.detector.score_thr,
            None if self.verifier is None else self.verifier.providers,
        )

    @property
    def requires_full_clip(self) -> bool:
        return True

    @property
    def mode_name(self) -> str:
        mode = f"native-tracker+cleanup[{self.cleanup.name}]"
        return mode + "+bbox-smoothing" if self.smoothing is not None else mode

    def track_boxes(
        self, image_ids, detections, hands, image_size, *, frame_times, images_bgr=None
    ):
        """Boxes of a complete clip from its ``detect_frames`` detections.

        ``mediapipe_check`` reads the candidate crops from ``images_bgr`` (aligned
        with ``image_ids``). Returns ``(bbox_per_hand, info)`` like
        ``boxes_from_tracks_cleanup``.
        """
        verify = None
        if self.verifier is not None:
            if images_bgr is None:
                raise ValueError("mediapipe_check needs the clip's frames (images_bgr)")
            from .mediapipe_check import LazyVerifier

            verify = LazyVerifier(images_bgr=images_bgr, verifier=self.verifier)
        return boxes_from_tracks_cleanup(
            image_ids,
            detections,
            hands,
            image_size,
            self.cleanup,
            smoothing=self.smoothing,
            frame_times=frame_times,
            verify=verify,
            enforce_lr_order=self.enforce_lr_order,
        )

    def log_cleanup(self, info) -> None:
        stats = {
            k: v
            for k, v in info["cleanup_stats"].items()
            if k not in ("present", "emitted")
        }
        log.info(
            "[RTDetrTrackCleanup] %s: %s, MediaPipe crops %d",
            info["cleanup"],
            stats,
            info["mediapipe_crops"],
        )

    def run(
        self,
        image_ids,
        images_bgr,
        hands,
        output_paths_per_hand=None,
        *,
        frame_times: dict[str, float] | None = None,
    ) -> dict:
        """As ``RTDetrHandBBoxDetector.run``; ``frame_times`` is required.

        ``frame_times`` maps every image ID to its timestamp in seconds.
        """
        image_ids = [str(image_id) for image_id in image_ids]
        if not image_ids:
            return {}
        if frame_times is None or any(i not in frame_times for i in image_ids):
            raise ValueError("track_cleanup requires frame_times for every image ID")
        times = [frame_times[i] for i in image_ids]
        log.info(
            "[RTDetr] %d image(s), hands=%s, mode=%s",
            len(images_bgr),
            ",".join(hands),
            self.mode_name,
        )
        detections = self._detect_per_frame(images_bgr)
        height, width = images_bgr[0].shape[:2]
        bbox_per_hand, info = self.track_boxes(
            image_ids,
            detections,
            hands,
            (width, height),
            frame_times=times,
            images_bgr=images_bgr,
        )
        self.log_tracking(info, hands)
        self.log_cleanup(info)

        bboxes_per_image: dict[str, dict[str, np.ndarray]] = {
            image_id: {} for image_id in image_ids
        }
        for hand in hands:
            bbox_dict = bbox_per_hand[hand]
            if output_paths_per_hand is not None:
                output_path = Path(output_paths_per_hand[hand])
                output_path.parent.mkdir(parents=True, exist_ok=True)
                merged = (
                    json.loads(output_path.read_text()) if output_path.exists() else {}
                )
                merged.update({k: [float(v) for v in b] for k, b in bbox_dict.items()})
                with open(output_path, "w") as f:
                    json.dump(merged, f, indent=2)
                log.info("[Saved] %s  (%d image(s))", output_path, len(bbox_dict))
            for image_id, bbox in bbox_dict.items():
                bboxes_per_image[image_id][hand] = np.asarray(bbox, dtype=np.float32)
        return bboxes_per_image


def build_detector(rtdetr_config, cleanup, *, device: str, tracking: bool):
    """The detector for an ``RTDetrConfig`` plus a ``track_cleanup`` mapping.

    ``cleanup`` may also be a ``TrackCleanupConfig``. Returns
    ``rtdetr_config.build(...)`` (the stock detector) when no switch is on.

    Raises:
        ValueError: invalid ``track_cleanup``, or switches on without
            ``tracking: true`` or with ``drop_gap_filled: true``.
    """
    if not isinstance(cleanup, TrackCleanupConfig):
        cleanup = TrackCleanupConfig.from_dict(cleanup)
    if not cleanup.any_rule:
        return rtdetr_config.build(device=device, tracking=tracking)
    if not tracking:
        raise ValueError("track_cleanup works on whole-clip tracks; set tracking: true")
    if rtdetr_config.drop_gap_filled:
        raise ValueError(
            "track_cleanup decides which gap frames get a box; "
            "set drop_gap_filled: false"
        )
    if cleanup.mediapipe_check and not cleanup.mediapipe_weights:
        raise ValueError(
            "mediapipe_check needs track_cleanup.mediapipe_weights "
            "(handpose_estimation_mediapipe_2023feb.onnx)"
        )
    rtdetr_config._validate(rtdetr_config, prefix="rtdetr_config")
    # Every setting the stock detector gets from RTDetrConfig (in tracking mode
    # the tracker owns score_thr, padding and clip, as in the stock detector).
    kwargs = rtdetr_config.detector_kwargs(device=device, tracking=True)
    return RTDetrTrackCleanupDetector(cleanup=cleanup, **kwargs)
