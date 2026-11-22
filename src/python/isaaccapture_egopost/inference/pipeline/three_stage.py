# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-OpenMDW-1.1

"""Model-resident EgoHand inference over a queue of videos, in three stages.

The per-video work of ``run_egohand_inference_pipeline.py`` split so that the
stages work on different videos at once:

1. prepare: whole-clip decode, detection + tracking, bbox JSONs, uint8 crops
   (failed crops dropped where ``predictor.run`` drops them);
2. head: batches of valid crops in job order, normalized as
   ``crop_and_preprocess`` does, ``forward_step_head`` under autocast;
3. soma: ``SomaBatcher`` (SOMA decode, mirror), then per clip the source-camera
   rotation, adaptive smoothing and the results npz. Every SOMA call stays in
   this stage, in single-video order.

``ThreadedThreeStage`` runs stages 1-2 on threads (stage 2 on its own CUDA
stream) and stage 3 on the caller's thread; ``run_serial`` runs them back to
back. A Python error fails only its video; a CUDA error aborts the run.
"""

from __future__ import annotations

import contextlib
import dataclasses
import json
import logging
import os
import queue
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator

import numpy as np
import torch

log = logging.getLogger(__name__)

FPS_FALLBACK = 30.0  # inference.video_io.FPS_FALLBACK
_END = "__end_of_queue__"


# --------------------------------------------------------------------------- data


@dataclasses.dataclass
class ClipRequest:
    """One queue item. ``frames`` / ``camera`` / ``fps`` replace decoding (tests)."""

    index: int
    name: str
    video: str
    intrinsics: str | None
    out_root: str
    frames: list | None = None
    camera: Any = None
    fps: float | None = None


@dataclasses.dataclass
class StageConfig:
    """Per-run settings shared by the three stages."""

    hands: list
    use_perspective_cropping: bool
    camera_mode: str
    batch_size: int
    smooth: bool
    crop_workers: int = 8
    norm_workers: int = 4
    rectify_pixel_aspect_ratio: bool = True
    perf_metadata: dict | None = None


@dataclasses.dataclass
class PreparedClip:
    """Stage 1 output: everything later stages need, without the frames."""

    request: ClipRequest
    stem: str = ""
    out_dir: Path | None = None
    fps: float = FPS_FALLBACK
    width: int = 0
    height: int = 0
    image_ids: list = dataclasses.field(default_factory=list)
    frame_times: dict | None = None
    bboxes_per_image: dict = dataclasses.field(default_factory=dict)
    intrinsics_per_image: dict = dataclasses.field(default_factory=dict)
    records: list = dataclasses.field(default_factory=list)
    crops: Any = None  # uint8 [N, 256, 256, 3] (list of arrays or one array)
    num_jobs: int = 0
    num_failed_crops: int = 0
    times: dict = dataclasses.field(default_factory=dict)
    error: str | None = None
    error_class: str | None = None
    error_stage: str | None = None
    error_traceback: str | None = None
    began: bool = False

    def fail(self, stage: str, exc: BaseException) -> None:
        if self.error is None:
            self.error = f"{type(exc).__name__}: {exc}"[:2000]
            self.error_class = type(exc).__name__
            self.error_stage = stage
            self.error_traceback = traceback.format_exc()[-4000:]


@dataclasses.dataclass
class HeadBatch:
    """Stage 2 output for one forward batch."""

    start: int  # index of the batch's first record in ``PreparedClip.records``
    hand_data: list
    batch: dict  # _assemble_batch output without "img"
    head: dict  # forward_step_head output
    event: Any = None  # CUDA event recorded after the head on stage 2's stream


class UnsupportedOption(ValueError):
    """A pipeline option the three-stage driver does not implement."""


def is_fatal_cuda_error(exc: BaseException) -> bool:
    """True for errors that leave the CUDA context unusable (not OOM)."""
    if isinstance(exc, torch.cuda.OutOfMemoryError):
        return False
    accelerator_error = getattr(torch, "AcceleratorError", None)
    if accelerator_error is not None and isinstance(exc, accelerator_error):
        return True
    text = f"{type(exc).__name__}: {exc}"
    return any(
        marker in text
        for marker in (
            "CUDA error",
            "cudaError",
            "illegal memory access",
            "CUBLAS_STATUS_EXECUTION_FAILED",
            "CUDNN_STATUS_EXECUTION_FAILED",
            "device-side assert",
            "unspecified launch failure",
        )
    )


# --------------------------------------------------------------------------- stage 1


def decode_all_frames(video_path: str) -> list:
    """Every frame of the video, decoded by the active video backend
    (``inference.video_io``, as ``run_egohand_inference_pipeline.py`` without
    ``--frame_batch_size``)."""
    from inference.video_io import read_video_frames_batched

    return [
        frame
        for batch in read_video_frames_batched(str(video_path), 10**9)
        for _, frame in batch
    ]


def video_meta(video_path: str) -> dict:
    """``inference.video_io.get_video_meta``."""
    from inference.video_io import get_video_meta

    return get_video_meta(str(video_path))


def check_video_decodable(video_path, meta: dict) -> None:
    """``run_egohand_inference_pipeline.check_video_decodable`` (raises instead of exiting).

    The first frame is read through the active video backend, which also
    rejects an unsupported codec (``UnsupportedVideoCodecError``).
    """
    from inference.video_io import read_video_frames_batched

    decodable = meta["width"] > 0 and meta["height"] > 0
    if decodable:
        frames = read_video_frames_batched(str(video_path), 1, 1)
        try:
            decodable = next(frames, None) is not None
        finally:
            frames.close()
    if not decodable:
        raise RuntimeError(
            f"Cannot read video {video_path}: no decodable video stream "
            "(empty, truncated, or unsupported file)"
        )


def resolve_clip_camera(clip: "PreparedClip", use_perspective_cropping: bool):
    """The camera stage 1 resolved for ``clip`` (same JSON, same ``resolve_K`` call)."""
    from inference.camera.intrinsics import resolve_K

    request = clip.request
    if request.camera is not None:
        return request.camera
    calibration = None
    if request.intrinsics:
        with open(request.intrinsics) as fh:
            calibration = json.load(fh)
    camera, _ = resolve_K(
        clip.width,
        clip.height,
        calibration,
        perspective_cropping=use_perspective_cropping,
        calibration_source=(
            f"calibration {request.intrinsics}" if request.intrinsics else None
        ),
    )
    return camera


def build_crop_record(job, *, use_perspective_cropping: bool, camera_mode: str):
    """``predictor.run``'s ``_build_input`` with the crop kept as uint8 RGB.

    Returns ``(image_id, hand, record or None)``; the record holds ``crop_u8``
    (HWC, right hands already mirrored) instead of ``crop_tensor``.
    ``normalize_crop(record["crop_u8"])`` has the values of ``crop_tensor``.
    ``None`` is a failed perspective crop, skipped with the same warning.
    """
    from inference.camera.perspective_crop import perspective_crop_uint8
    from inference.preprocessing.crop import IMG_SIZE, crop_uint8
    from inference.preprocessing.mirroring import (
        mirror_full_image_camera_horizontal,
        prepare_right_hand_model_input_uint8,
    )

    image_id, hand, image_bgr, bbx_array, source_camera = job
    if use_perspective_cropping:
        try:
            crop_u8, K_virtual, bbox_virtual, crop_to_source_rotation = (
                perspective_crop_uint8(image_bgr, bbx_array, source_camera, IMG_SIZE)
            )
        except (FloatingPointError, ValueError) as exc:
            log.warning(
                "[Skip] %s %s: perspective cropping failed: %s", image_id, hand, exc
            )
            return image_id, hand, None
        model_bbx_xys = bbox_virtual
        model_K = K_virtual
        if hand == "right":
            crop_u8 = np.ascontiguousarray(crop_u8[:, ::-1])
            model_bbx_xys, model_K = mirror_full_image_camera_horizontal(
                model_bbx_xys, model_K, crop_u8.shape[1]
            )
        return (
            image_id,
            hand,
            {
                "image_id": image_id,
                "real_hand_side": hand,
                "crop_u8": crop_u8,
                "model_bbx_xys": model_bbx_xys,
                "model_K": model_K,
                "model_img_size": np.array(
                    [crop_u8.shape[1], crop_u8.shape[0]], dtype=np.float32
                ),
                "output_bbx_xys": bbox_virtual,
                "output_K": K_virtual,
                "crop_to_source_camera_rotation": crop_to_source_rotation,
            },
        )
    K_fullimg = np.asarray(source_camera, dtype=np.float32)
    image_width = image_bgr.shape[1]
    image_height = image_bgr.shape[0]
    model_bbx_xys = bbx_array.copy()
    model_K = K_fullimg.copy()
    if hand == "right":
        crop_u8, model_bbx_xys, model_K = prepare_right_hand_model_input_uint8(
            image_bgr,
            model_bbx_xys,
            model_K,
            calibrated_full_image=camera_mode == "sam3db_perspective",
        )
    else:
        crop_u8 = crop_uint8(image_bgr, bbx_array)
    return (
        image_id,
        hand,
        {
            "image_id": image_id,
            "real_hand_side": hand,
            "crop_u8": crop_u8,
            "model_bbx_xys": model_bbx_xys,
            "model_K": model_K,
            "model_img_size": np.array([image_width, image_height], dtype=np.float32),
            "output_bbx_xys": bbx_array,
            "output_K": K_fullimg,
        },
    )


class PrepareStage:
    """Stage 1: decode, detect, crop one clip (CPU + the detector's GPU work)."""

    def __init__(self, bbox_detector, config: StageConfig):
        if getattr(bbox_detector, "reports_in_source_frame", False):
            raise UnsupportedOption(
                "detectors that report boxes in the source frame (saved_bboxes)"
            )
        self.bbox_detector = bbox_detector
        self.config = config

    def run(self, request: ClipRequest) -> PreparedClip:
        """Prepare one clip; errors are recorded on the returned clip."""
        clip = PreparedClip(request=request)
        clip.times["s1_start"] = time.time()
        try:
            self._run(clip)
        except Exception as exc:  # noqa: BLE001 - per-video isolation
            clip.fail("prepare", exc)
            clip.crops = None
            if is_fatal_cuda_error(exc):
                raise
        clip.times["s1_end"] = time.time()
        return clip

    def _run(self, clip: PreparedClip) -> None:
        from inference.camera.intrinsics import resolve_K
        from inference.camera.square_pixel import compute_square_pixel_resize_params
        from inference.postprocessing.adaptive_smoothing import validate_frame_times

        config = self.config
        request = clip.request
        detector = self.bbox_detector
        pc = config.use_perspective_cropping
        out_root = Path(request.out_root)
        out_root.mkdir(parents=True, exist_ok=True)
        if not os.access(out_root, os.W_OK | os.X_OK):
            raise PermissionError(f"output root {out_root} is not writable")
        video_path = Path(request.video)
        clip.stem = video_path.stem
        if request.frames is None:
            calibration = None
            if request.intrinsics:
                with open(request.intrinsics) as fh:
                    calibration = json.load(fh)
            meta = video_meta(str(video_path))
            check_video_decodable(video_path, meta)
            K_original, _ = resolve_K(
                meta["width"],
                meta["height"],
                calibration,
                perspective_cropping=pc,
                calibration_source=(
                    f"calibration {request.intrinsics}" if request.intrinsics else None
                ),
            )
            clip.width, clip.height = meta["width"], meta["height"]
            fps = meta["fps"] or FPS_FALLBACK
        else:
            K_original = request.camera
            fps = request.fps or FPS_FALLBACK
            clip.height, clip.width = request.frames[0].shape[:2]
        if not np.isfinite(fps) or fps <= 0:
            fps = FPS_FALLBACK
        clip.fps = fps
        if not pc and not isinstance(K_original, np.ndarray):
            raise UnsupportedOption("full-image undistortion (non-pinhole camera)")
        if not pc and config.rectify_pixel_aspect_ratio:
            params = compute_square_pixel_resize_params(
                K_original, clip.width, clip.height
            )
            if not params.is_identity:
                raise UnsupportedOption("pixel-aspect rectification")
        K_model = K_original
        if hasattr(detector, "prepare_clip"):
            detector.prepare_clip(video_path=video_path, K=K_original)

        out_dir = out_root / clip.stem
        clip.out_dir = out_dir
        (out_dir / "quality_control.json").unlink(missing_ok=True)
        if detector.writes_bbox_json:
            for hand in config.hands:
                (out_dir / f"{clip.stem}_{hand}_bbox.json").unlink(missing_ok=True)

        frames = (
            request.frames
            if request.frames is not None
            else decode_all_frames(str(video_path))
        )
        clip.times["decode_end"] = time.time()
        if not frames:
            raise RuntimeError(f"[infer_video] {video_path}: no frames decoded.")
        image_ids = [f"{idx:06d}" for idx in range(len(frames))]
        clip.image_ids = image_ids
        smooth_bboxes = getattr(detector, "requires_full_clip", False)
        frame_times = None
        if config.smooth or smooth_bboxes:
            frame_times = {f"{idx:06d}": idx / fps for idx in range(len(frames))}
        clip.frame_times = frame_times

        # EgoHandInferencePipeline.run
        out_dir.mkdir(parents=True, exist_ok=True)
        bbox_paths = {
            hand: out_dir / f"{out_dir.name}_{hand}_bbox.json" for hand in config.hands
        }
        bboxes_per_image = detector.run(
            image_ids=image_ids,
            images_bgr=frames,
            hands=config.hands,
            output_paths_per_hand=bbox_paths,
            **({"frame_times": frame_times} if smooth_bboxes else {}),
        )
        clip.bboxes_per_image = bboxes_per_image
        clip.times["detect_end"] = time.time()

        # EgoHandPredictor.run, up to the crops
        if config.smooth:
            if frame_times is None:
                raise ValueError("Adaptive smoothing requires frame_times in seconds")
            validate_frame_times([frame_times[i] for i in image_ids], len(image_ids))
        clip.intrinsics_per_image = dict.fromkeys(image_ids, K_model)
        crop_jobs = []
        for image_bgr, image_id in zip(frames, image_ids):
            bboxes_per_hand = bboxes_per_image.get(image_id, {})
            for hand in config.hands:
                bbox = bboxes_per_hand.get(hand)
                if bbox is None:
                    log.info("[Skip] %s %s: no reliable hand bbox", image_id, hand)
                    continue
                crop_jobs.append(
                    (
                        image_id,
                        hand,
                        image_bgr,
                        np.asarray(bbox, dtype=np.float32),
                        K_model,
                    )
                )
        clip.num_jobs = len(crop_jobs)

        def build(job):
            return build_crop_record(
                job,
                use_perspective_cropping=pc,
                camera_mode=config.camera_mode,
            )

        records, crops = [], []
        if crop_jobs:
            with ThreadPoolExecutor(max_workers=config.crop_workers) as pool:
                for _, _, record in pool.map(build, crop_jobs):
                    if record is None:
                        clip.num_failed_crops += 1
                        continue
                    crops.append(record.pop("crop_u8"))
                    records.append(record)
        del crop_jobs, frames
        clip.records = records
        clip.crops = crops
        clip.times["crop_end"] = time.time()


# --------------------------------------------------------------------------- stage 2


class HeadStage:
    """Stage 2: batch assembly, upload and ``forward_step_head``."""

    def __init__(self, predictor, config: StageConfig, *, own_stream: bool = False):
        self.predictor = predictor
        self.config = config
        self.device = torch.device(predictor.device)
        self.cuda = self.device.type == "cuda"
        if self.cuda and self.device.index is None:
            self.device = torch.device("cuda", torch.cuda.current_device())
        # One stream for the whole run (a new stream per clip makes the caching
        # allocator keep a separate block pool per stream).
        self.stream = (
            torch.cuda.Stream(device=self.device)
            if (self.cuda and own_stream)
            else None
        )

    @torch.no_grad()
    def run(
        self, clip: PreparedClip, emit: Callable[[HeadBatch], None], crops=None
    ) -> int:
        """Run every forward batch of ``clip``; ``emit`` each one. Returns the count."""
        from inference.pipeline.crop_stream import StreamedCropBatches
        from inference.preprocessing.crop import normalize_crop

        predictor = self.predictor
        crops = clip.crops if crops is None else crops
        records = clip.records
        if len(records) != (0 if crops is None else len(crops)):
            raise RuntimeError(
                f"{clip.stem}: {len(records)} crop records but "
                f"{0 if crops is None else len(crops)} crops"
            )
        dtype = predictor.dtype

        def build(index):
            record = dict(records[index])
            record["crop_tensor"] = normalize_crop(np.asarray(crops[index]))
            return record["image_id"], record["real_hand_side"], record

        stream_ctx = (
            torch.cuda.stream(self.stream)
            if self.stream is not None
            else contextlib.nullcontext()
        )
        count = 0
        start = 0
        with stream_ctx:
            with StreamedCropBatches(
                range(len(records)),
                build,
                batch_size=predictor.batch_size,
                device=predictor.device,
                workers=self.config.norm_workers,
                copy_stream=predictor._crop_copy_stream(),
            ) as crop_batches:
                for staged in crop_batches:
                    batch_hand_data = staged.hand_data
                    batch = predictor._assemble_batch(batch_hand_data, staged)
                    with torch.autocast(
                        "cuda", dtype=dtype, enabled=(dtype != torch.float32)
                    ):
                        head = predictor.model.forward_step_head(batch, train=False)
                    batch.pop("img")
                    # The head's reads of the images are queued; reuse the slot.
                    staged.release()
                    event = None
                    if self.cuda:
                        event = torch.cuda.Event()
                        event.record(torch.cuda.current_stream(self.device))
                    emit(HeadBatch(start, batch_hand_data, batch, head, event))
                    start += len(batch_hand_data)
                    count += 1
        if start != len(records):
            raise RuntimeError(
                f"{clip.stem}: forward batches covered {start} of {len(records)} crops"
            )
        return count


# --------------------------------------------------------------------------- stage 3


def _record_stream(obj, stream) -> None:
    if isinstance(obj, torch.Tensor):
        if obj.is_cuda:
            obj.record_stream(stream)
    elif isinstance(obj, dict):
        for value in obj.values():
            _record_stream(value, stream)
    elif isinstance(obj, (list, tuple)):
        for value in obj:
            _record_stream(value, stream)


def save_clip_outputs(clip: PreparedClip, results_per_hand: dict, hands: list) -> dict:
    """The export part of ``process_video`` (N config): results npz per hand."""
    from inference.camera.square_pixel import rewrite_bbox_json_in_original_frame
    from inference.pipeline.results_io import save_hand_results_clip

    out_dir = clip.out_dir
    total_frames = len(clip.image_ids)
    inference_results: dict[str, dict] = {"left": {}, "right": {}}
    for hand in hands:
        inference_results[hand].update(results_per_hand[hand])
    for hand in hands:
        n_inferred = len(inference_results[hand])
        print(
            f"[infer_video] {clip.stem} {hand}: {n_inferred}/{total_frames} frames "
            f"inferred ({total_frames - n_inferred} skipped — no prediction)"
        )
    if not any(inference_results[hand] for hand in hands):
        raise RuntimeError(
            f"[infer_video] {clip.request.video}: 0/{total_frames} frames inferred across "
            "the requested hands. Check bbox inputs, frame IDs, and detector confidence."
        )
    for hand, results in inference_results.items():
        remapped = [
            {**r, "hand_poses": r["poses"], "pred_cam_t": r["transl"]}
            for r in results.values()
        ]
        save_hand_results_clip(
            out_dir / f"{hand}_results.npz", list(results.keys()), remapped
        )
    rewrite_bbox_json_in_original_frame(out_dir, hands, None)
    return {hand: len(inference_results[hand]) for hand in hands}


#: What the perf.json latencies of a three-stage clip measure (written into its
#: metadata): the stages overlap other videos, so these are per-frame wall times
#: of this clip's stage spans, including waits, not the single-video meanings of
#: ``inference.perf.STAGE_DESCRIPTIONS``.
THREE_STAGE_PERF_MEANING = {
    "bbox_detection": "stage 1: detection + tracking of the decoded clip",
    "preprocessing": "stage 1: uint8 crops of every hand",
    "model_forward": "stage 2 wall time for the clip: waiting for crop slots, "
    "normalization, upload, forward_step_head",
    "post_forward": "stage 3 from its first head batch of the clip to the last, "
    "including waits for stage 2 (SOMA decode, mirror)",
    "adaptive_smoothing": "stage 3: results assembly and smoothing",
    "total_wall": "stage 1 start to stage 3 end of the clip (overlaps other clips)",
}


class SomaStage:
    """Stage 3: SOMA decode, mirror, results, smoothing, export."""

    def __init__(
        self,
        predictor,
        config: StageConfig,
        *,
        writer: Callable | None = None,
        model_load_ms: float | None = None,
    ):
        self.predictor = predictor
        self.config = config
        self.device = torch.device(predictor.device)
        self.cuda = self.device.type == "cuda"
        if self.cuda and self.device.index is None:
            self.device = torch.device("cuda", torch.cuda.current_device())
        self.writer = writer if writer is not None else save_clip_outputs
        self.model_load_ms = model_load_ms
        self._batcher = None
        self._clip = None
        self.results = None  # results_per_hand of the last finished clip (tests)

    def begin(self, clip: PreparedClip) -> None:
        from inference.pipeline.soma_batching import SomaBatcher

        self._clip = clip
        self._batcher = SomaBatcher(
            self.predictor, getattr(self.predictor, "soma_batch_size", 0)
        )
        clip.times.setdefault("s3_start", time.time())

    def discard(self) -> None:
        self._batcher = None
        self._clip = None

    @torch.no_grad()
    def add(self, head_batch: HeadBatch) -> None:
        if head_batch.event is not None:
            stream = torch.cuda.current_stream(self.device)
            stream.wait_event(head_batch.event)
            # Stage 2's tensors were allocated on its stream; keep the
            # allocator from reusing them while this stream may read them.
            _record_stream((head_batch.batch, head_batch.head), stream)
        self._batcher.add(head_batch.batch, head_batch.hand_data, head_batch.head)

    @torch.no_grad()
    def finish(self, clip: PreparedClip) -> dict:
        predictor = self.predictor
        hands = self.config.hands
        clip.times["s3_batches_end"] = time.time()
        predictions = self._batcher.finish()
        self._batcher = None
        prepared_inputs_by_image: dict = {}
        for record in clip.records:
            prepared_inputs_by_image.setdefault(record["image_id"], {})[
                record["real_hand_side"]
            ] = record
        results_per_hand = predictor._results_per_hand(
            predictions, hands, prepared_inputs_by_image
        )
        t_smooth = time.time()
        predictor.postprocess_results(
            results_per_hand,
            clip.image_ids,
            clip.bboxes_per_image,
            frame_times=clip.frame_times,
            intrinsics_per_image=clip.intrinsics_per_image,
            source_cameras_per_image=None,
        )
        clip.times["s3_smooth_end"] = time.time()
        clip.times["smooth_s"] = clip.times["s3_smooth_end"] - t_smooth
        self.results = results_per_hand
        counts = self.writer(clip, results_per_hand, hands)
        clip.times["s3_end"] = time.time()
        self._write_perf(clip)
        self._clip = None
        return counts

    def _write_perf(self, clip: PreparedClip) -> None:
        if clip.out_dir is None or not clip.image_ids:
            return
        from inference.perf import LocalPerfWriter, PerfStage

        n = len(clip.image_ids)
        t = clip.times
        writer = LocalPerfWriter(
            output_path=clip.out_dir / "perf.json",
            metadata={
                **(self.config.perf_metadata or {}),
                "three_stage": True,
                "three_stage_latencies": dict(THREE_STAGE_PERF_MEANING),
            },
        )
        writer.record_frames(n)
        if self.model_load_ms is not None:
            writer.record_latency(PerfStage.MODEL_LOAD, self.model_load_ms)

        def per_frame(a, b):
            return (t[b] - t[a]) * 1000.0 / n if a in t and b in t else None

        for stage, value in (
            (PerfStage.BBOX_DETECTION, per_frame("decode_end", "detect_end")),
            (PerfStage.PREPROCESSING, per_frame("detect_end", "crop_end")),
            (PerfStage.MODEL_FORWARD, per_frame("s2_start", "s2_end")),
            (PerfStage.POST_FORWARD, per_frame("s3_start", "s3_batches_end")),
            (
                PerfStage.ADAPTIVE_SMOOTHING,
                per_frame("s3_batches_end", "s3_smooth_end"),
            ),
            (PerfStage.TOTAL_WALL, per_frame("s1_start", "s3_end")),
        ):
            if value is not None:
                writer.record_latency(stage, value)
        writer.write()


# --------------------------------------------------------------------------- records


def clip_record(clip: PreparedClip, status: str | None = None) -> dict:
    """Per-video summary record (driver JSON)."""
    request = clip.request
    t = dict(clip.times)
    start = t.get("s1_start")
    end = t.get("s3_end") or t.get("failed_at") or time.time()
    record = {
        "i": request.index,
        "name": request.name,
        "video": request.video,
        "out_root": request.out_root,
        "status": status or ("ok" if clip.error is None else "error"),
        "frames": len(clip.image_ids),
        "crop_jobs": clip.num_jobs,
        "crops": len(clip.records),
        "failed_crops": clip.num_failed_crops,
        "t_start_epoch": start,
        "t_end_epoch": end,
        "wall_s": (end - start) if start else None,
        "times": t,
    }
    if clip.error is not None:
        record.update(
            error=clip.error,
            error_class=clip.error_class,
            error_stage=clip.error_stage,
            traceback=clip.error_traceback,
        )
    return record


# --------------------------------------------------------------------------- thr mode


class _Stopped(Exception):
    pass


class ThreadedThreeStage:
    """thr mode: stage 1 and 2 on worker threads, stage 3 on the calling thread.

    Args:
        prepare_ahead: 1 (default): stage 1 starts decoding the next clip
            only after stage 2 has taken the previous one (at most one prepared
            clip waits). 2: one more clip may be prepared while one waits.
        soma_videos: Clips stage 2 may have handed to stage 3 and stage 3 has
            not finished (bounds the GPU memory held by queued head batches).
        warp_mempool: Enable Warp's memory pool on the device while running.
            py-soma-x disables it; then every Warp temporary is freed with
            ``cudaFree``, which waits for the whole device (also for stage 2).
            Only the allocation method changes.
    """

    def __init__(
        self,
        prepare: PrepareStage,
        head: HeadStage,
        soma: SomaStage,
        *,
        prepare_ahead: int = 1,
        soma_videos: int = 2,
        warp_mempool: bool = False,
        poll_s: float = 0.1,
    ):
        if prepare_ahead < 1 or soma_videos < 1:
            raise ValueError("prepare_ahead and soma_videos must be >= 1")
        self.prepare = prepare
        self.head = head
        self.soma = soma
        self.prepare_ahead = prepare_ahead
        self.soma_videos = soma_videos
        self.warp_mempool = warp_mempool
        self.poll_s = poll_s
        self._stop = threading.Event()
        self.fatal: BaseException | None = None
        self.fatal_traceback: str | None = None
        self.in_flight: dict[int, PreparedClip] = {}
        self._lock = threading.Lock()

    # queue helpers that give up when the run stops
    def _put(self, q: queue.Queue, item) -> None:
        while True:
            if self._stop.is_set():
                raise _Stopped()
            try:
                q.put(item, timeout=self.poll_s)
                return
            except queue.Full:
                continue

    def _get(self, q: queue.Queue):
        while True:
            if self._stop.is_set():
                raise _Stopped()
            try:
                return q.get(timeout=self.poll_s)
            except queue.Empty:
                continue

    def _wait(self, waitable) -> None:
        while not waitable(self.poll_s):
            if self._stop.is_set():
                raise _Stopped()

    def _set_fatal(self, exc: BaseException) -> None:
        with self._lock:
            if self.fatal is None:
                self.fatal = exc
                self.fatal_traceback = traceback.format_exc()[-4000:]
        self._stop.set()

    def _prepare_loop(self, requests: Iterable[ClipRequest], q12: queue.Queue) -> None:
        try:
            for request in requests:
                if self._stop.is_set():
                    return
                try:
                    clip = self.prepare.run(request)
                except (
                    BaseException
                ) as exc:  # fatal (CUDA): report this clip as aborted
                    lost = PreparedClip(request=request)
                    lost.fail("prepare", exc)
                    with self._lock:
                        self.in_flight[request.index] = lost
                    raise
                with self._lock:
                    self.in_flight[request.index] = clip
                taken = threading.Event()
                self._put(q12, (clip, taken))
                if self.prepare_ahead <= 1:
                    self._wait(taken.wait)
            self._put(q12, _END)
        except _Stopped:
            return
        except BaseException as exc:  # noqa: BLE001
            self._set_fatal(exc)

    def _head_loop(
        self, q12: queue.Queue, q23: queue.Queue, slots: threading.Semaphore
    ) -> None:
        try:
            while True:
                item = self._get(q12)
                if item == _END:
                    self._put(q23, ("end_of_queue", None, None))
                    return
                clip, taken = item
                taken.set()
                if clip.error is not None:
                    self._put(q23, ("fail", clip, None))
                    continue
                self._wait(lambda timeout: slots.acquire(timeout=timeout))
                clip.began = True
                clip.times["s2_start"] = time.time()
                self._put(q23, ("begin", clip, None))
                try:
                    self.head.run(
                        clip, lambda hb, clip=clip: self._put(q23, ("batch", clip, hb))
                    )
                except _Stopped:
                    raise
                except Exception as exc:  # noqa: BLE001
                    clip.fail("head", exc)
                    clip.times["failed_at"] = time.time()
                    self._put(q23, ("fail", clip, None))
                    if is_fatal_cuda_error(exc):
                        self._set_fatal(exc)
                        return
                    if isinstance(exc, torch.cuda.OutOfMemoryError):
                        torch.cuda.empty_cache()
                    continue
                finally:
                    clip.crops = None  # the uint8 crops are no longer needed
                clip.times["s2_end"] = time.time()
                self._put(q23, ("end", clip, None))
        except _Stopped:
            return
        except BaseException as exc:  # noqa: BLE001
            self._set_fatal(exc)

    def run(
        self, requests: Iterable[ClipRequest], on_record: Callable[[dict], None]
    ) -> None:
        """Process ``requests``; ``on_record`` gets one record per finished clip."""
        q12: queue.Queue = queue.Queue(maxsize=1)
        q23: queue.Queue = queue.Queue()
        slots = threading.Semaphore(self.soma_videos)
        wp_restore = None
        if self.warp_mempool and self.soma.cuda:
            wp_restore = _enable_warp_mempool(self.soma.device)
        threads = [
            threading.Thread(
                target=self._prepare_loop,
                args=(requests, q12),
                name="egopost-t3-prepare",
                daemon=True,
            ),
            threading.Thread(
                target=self._head_loop,
                args=(q12, q23, slots),
                name="egopost-t3-head",
                daemon=True,
            ),
        ]
        for thread in threads:
            thread.start()
        current = None
        try:
            while True:
                kind, clip, payload = self._get(q23)
                if kind == "end_of_queue":
                    break
                if kind == "begin":
                    current = clip
                    try:
                        self.soma.begin(clip)
                    except Exception as exc:  # noqa: BLE001
                        clip.fail("soma", exc)
                elif kind == "batch":
                    if clip.error is None:
                        try:
                            self.soma.add(payload)
                        except Exception as exc:  # noqa: BLE001
                            clip.fail("soma", exc)
                            clip.times["failed_at"] = time.time()
                            self.soma.discard()
                            if is_fatal_cuda_error(exc):
                                raise
                    del payload
                elif kind in ("end", "fail"):
                    if kind == "end" and clip.error is None:
                        try:
                            self.soma.finish(clip)
                        except Exception as exc:  # noqa: BLE001
                            clip.fail("soma", exc)
                            clip.times["failed_at"] = time.time()
                            self.soma.discard()
                            if is_fatal_cuda_error(exc):
                                raise
                            if isinstance(exc, torch.cuda.OutOfMemoryError):
                                torch.cuda.empty_cache()
                    else:
                        clip.times.setdefault("failed_at", time.time())
                        self.soma.discard()
                    if clip.began:
                        slots.release()
                    current = None
                    with self._lock:
                        self.in_flight.pop(clip.request.index, None)
                    clip.crops = None
                    on_record(clip_record(clip))
        except _Stopped:
            pass
        except BaseException as exc:  # noqa: BLE001
            if current is not None:
                current.fail("soma", exc)
            self._set_fatal(exc)
        finally:
            for thread in threads:
                thread.join(timeout=None if self.fatal is None else 60)
            if wp_restore is not None:
                wp_restore()
        if self.fatal is not None:
            for clip in list(self.in_flight.values()):
                if clip.error is None:
                    clip.error = f"aborted: {type(self.fatal).__name__}: {self.fatal}"[
                        :2000
                    ]
                    clip.error_class = "Aborted"
                    clip.error_stage = "run"
                clip.times.setdefault("failed_at", time.time())
                on_record(clip_record(clip, status="aborted"))
            self.in_flight.clear()


def _enable_warp_mempool(device: torch.device):
    try:
        import warp as wp
    except ImportError:
        return None
    wp_device = wp.device_from_torch(device)
    was = wp.is_mempool_enabled(wp_device)
    wp.set_mempool_enabled(wp_device, True)

    def restore():
        wp.set_mempool_enabled(wp_device, was)

    return restore


def run_serial(
    prepare: PrepareStage,
    head: HeadStage,
    soma: SomaStage,
    requests: Iterable[ClipRequest],
    on_record: Callable[[dict], None],
) -> None:
    """The three stages one after the other per clip, in the calling thread."""
    for request in requests:
        clip = prepare.run(request)
        if clip.error is None:
            clip.began = True
            clip.times["s2_start"] = time.time()
            try:
                soma.begin(clip)
                head.run(clip, soma.add)
                clip.times["s2_end"] = time.time()
                soma.finish(clip)
            except Exception as exc:  # noqa: BLE001
                clip.fail("head/soma", exc)
                clip.times["failed_at"] = time.time()
                soma.discard()
                if is_fatal_cuda_error(exc):
                    on_record(clip_record(clip, status="aborted"))
                    raise
        clip.crops = None
        on_record(clip_record(clip))


# --------------------------------------------------------------------------- driver setup


def claim_requests(
    requests: list[ClipRequest], claim_dir: str | None
) -> Iterator[ClipRequest]:
    """Yield the requests this worker claims (all of them without ``claim_dir``).

    Several workers sharing ``claim_dir`` take the queue in order, each item
    once: a claim is an exclusively created file ``<index>.claim``.
    """
    for request in requests:
        if claim_dir:
            path = os.path.join(claim_dir, f"{request.index:05d}.claim")
            try:
                fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
            except FileExistsError:
                continue
            with os.fdopen(fd, "w") as fh:
                fh.write(f"{os.getpid()} {time.time():.6f}\n")
        yield request


def build_detector(detector_cfg: dict, device: str, batch_size: int):
    """The detector of ``detector_cfg`` by backend name, as
    ``run_egohand_inference_pipeline._build_detector_from_config`` builds it.

    For library callers; the driver builds through the CLI function, which
    also turns config errors into its one-line invalid-input exit.
    """
    import importlib

    backend = detector_cfg["detector_backend"]
    module = importlib.import_module(f"inference.hand_detector_2d.configs.{backend}")
    return module.build(
        detector_cfg, device, batch_size, detector_cfg.get("tracking", False)
    )


def warm_detector(bbox_detector) -> list:
    """Run ORT once per batch size 1..B on zeros (``--warm_detector``), so its
    cuDNN algorithm search by timing happens before the first video."""
    warm = []
    inner = getattr(bbox_detector, "detector", None)
    if inner is None or not hasattr(inner, "_run"):
        return warm
    for b in range(1, int(getattr(bbox_detector, "batch_size", 8)) + 1):
        t = time.perf_counter()
        inner._run(
            np.zeros((b, 3, 640, 640), np.float32), np.full((b, 2), 640, np.int64)
        )
        warm.append({"batch": b, "s": time.perf_counter() - t})
    return warm


_DTYPES = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


def load_predictor(settings: dict):
    """``EgoHandPredictor.from_checkpoint`` (+ DINO compile) from driver settings.

    ``settings["mhr_presorted_sparse"]`` (default True) is passed on:
    ``from_checkpoint`` coalesces the MHR sparse matrix once
    (``mhr_presort.py``); ``predictor.mhr_presort`` records what it patched.
    """
    from inference.pipeline.predictor import EgoHandPredictor

    t = time.perf_counter()
    predictor = EgoHandPredictor.from_checkpoint(
        egohand_ckpt_path=settings["egohand_ckpt"],
        soma_hand_data_folder=settings["soma_hand_assets"],
        batch_size=settings["batch_size"],
        device=settings["device"],
        dtype=_DTYPES[settings["dtype"]],
        smooth=settings["smooth"],
        use_perspective_cropping=settings["use_perspective_cropping"],
        soma_batch_size=settings["soma_batch_size"],
        mhr_presorted_sparse=settings.get("mhr_presorted_sparse", True),
    )
    if settings.get("dino_compile"):
        predictor.enable_dino_compile()
    return predictor, (time.perf_counter() - t) * 1000.0


def stage_config_from_settings(settings: dict, camera_mode: str) -> StageConfig:
    return StageConfig(
        hands=list(settings["hands"]),
        use_perspective_cropping=settings["use_perspective_cropping"],
        camera_mode=camera_mode,
        batch_size=settings["batch_size"],
        smooth=settings["smooth"],
        crop_workers=settings.get("crop_workers", 8),
        norm_workers=settings.get("norm_workers", 4),
        rectify_pixel_aspect_ratio=settings.get("rectify_pixel_aspect_ratio", True),
        perf_metadata=settings.get("perf_metadata"),
    )
