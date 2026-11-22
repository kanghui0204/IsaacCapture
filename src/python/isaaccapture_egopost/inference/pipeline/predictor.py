# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-OpenMDW-1.1

"""EgoHand checkpoint loading, forward pass, and SOMA NPZ writing.

Owns the full model lifecycle for inference:
- ``load_egohand_model``: loads a checkpoint (.ckpt or inference export) and
  returns an eval model with strict key checks.
- ``EgoHandPredictor``: owns one loaded model, runs batched crop-and-forward loops,
  and writes soma.io-compatible prediction files.
- ``EgoHandPredictor.from_checkpoint``: convenience factory that combines both steps.
- ``EgoHandPredictor.load_in_background``: the same on a worker thread
  (``PendingEgoHandPredictor``), so loading overlaps decoding and detection.

``run()`` returns the per-frame result dict schema consumed by the SOMA NPZ writer.
Optional perspective cropping maps each detected hand region to a virtual pinhole
view for inference. Predictions are restored to source-camera axes, and overlays
are projected through the source camera model after temporal smoothing.
Full-image undistortion reuses this source-camera projection for its outputs.
"""

import contextlib
import io
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from inference.camera.camera_models import CameraModel
from inference.camera.perspective_crop import perspective_crop
from egohand.utils.cam_utils import compute_transl_full_cam
from inference.perf import PerfReporter, PerfStage
from inference.pipeline.crop_stream import StreamedCropBatches
from inference.pipeline.inference_weights import (
    CheckpointSource,
    check_checkpoint_keys,
    check_state_dict_keys,
    read_checkpoint,
)
from inference.pipeline.results_io import KEYPOINT_RELIABILITY_FIELDS
from inference.preprocessing.crop import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    IMG_SIZE,
    crop_and_preprocess,
    project_camera_points_to_crop_with_validity,
)
from inference.preprocessing.mirroring import (
    mirror_full_image_camera_horizontal,
    mirror_left_prediction_to_right,
    prepare_right_hand_model_input,
)
from inference.postprocessing.adaptive_smoothing import (
    smooth_adaptive_hand_results,
    validate_frame_times,
)

log = logging.getLogger(__name__)

_VALID_TARGETS = ("egohand.EgoHand",)


def _validated_cfg(source: CheckpointSource, soma_hand_data_folder):
    """Training cfg with the inference-only overrides applied."""
    cfg = OmegaConf.create(source.cfg)
    target = str(cfg.get("_target_", ""))
    if not any(target.endswith(t) for t in _VALID_TARGETS):
        raise ValueError(
            f"Checkpoint _target_ {target!r} is not supported. "
            f"Expected one of: {_VALID_TARGETS}"
        )
    cfg.backbone_weights = None
    cfg.soma_hand.data_root = str(soma_hand_data_folder)
    return cfg, target


def load_egohand_model(
    checkpoint_path: str | Path | CheckpointSource,
    soma_hand_data_folder: str | Path,
    device: str | torch.device = "cpu",
    *,
    ignore_unexpected_keys: bool = False,
):
    """Load a EgoHand model from a checkpoint.

    Reads the training-time cfg stored in ``ckpt["hyper_parameters"]["cfg"]``,
    overrides the two inference-only knobs (skip pretrained backbone load,
    redirect SOMAHand asset folder), instantiates the right model class, and
    loads weights. Returns the model in eval mode on ``device`` (CPU by
    default; the caller may still move it).

    Accepts a training ``.ckpt`` (memory-mapped, so optimizer state is never
    read) or an export from ``inference/scripts/export_inference_weights.py``.
    The DINOv3 encoder is allocated without random initialization and filled
    from the checkpoint. Loading is strict except for the keys listed in
    ``inference_weights.ALLOWED_MISSING_PREFIXES`` (asset-derived SOMA layers).

    Args:
        checkpoint_path: ``.ckpt`` file, export folder, or a ``CheckpointSource``
            already returned by ``read_checkpoint``.
        soma_hand_data_folder: SOMAHand asset folder to redirect ``cfg.soma_hand.data_root``.
        device: Device for the returned model.
        ignore_unexpected_keys: Load a checkpoint that also holds tensors the
            model does not own (logged as a warning instead of an error).
            Missing keys are always an error.

    Returns:
        model: EgoHand instance in eval mode on ``device``.
        target: The ``_target_`` string from the checkpoint cfg (for logging).
    """
    source = (
        checkpoint_path
        if isinstance(checkpoint_path, CheckpointSource)
        else read_checkpoint(checkpoint_path)
    )
    cfg, target = _validated_cfg(source, soma_hand_data_folder)
    backbone_cfg = cfg.get("backbone", None)
    meta_backbone = backbone_cfg is not None and str(
        backbone_cfg.get("_target_", "")
    ).endswith("DinoV3Backbone")
    if meta_backbone:
        cfg.backbone.meta_init = True

    from egohand.models.egohand import EgoHand

    model = EgoHand(cfg)
    model.validate_camera_depth_checkpoint(source.validation_view())
    if meta_backbone:
        model.backbone.encoder.to_empty(device=device)
    model = model.to(device)

    state_dict = source.load_state_dict()
    if meta_backbone:
        # to_empty() leaves memory uninitialized: every encoder tensor must come
        # from the checkpoint, independently of the allowlist below.
        encoder_keys = {
            f"backbone.encoder.{key}" for key in model.backbone.encoder.state_dict()
        }
        absent = sorted(encoder_keys - state_dict.keys())
        if absent:
            raise RuntimeError(
                f"Checkpoint lacks {len(absent)} DINOv3 encoder tensor(s): {absent[:10]}"
            )
        # Tensors to_empty() materialized that no checkpoint can fill (buffers
        # registered with persistent=False) would keep uninitialized memory.
        stray = sorted(
            name
            for name, _ in model.backbone.encoder.named_buffers()
            if name not in model.backbone.encoder.state_dict()
        )
        if stray:
            raise RuntimeError(
                f"DINOv3 encoder has non-persistent buffer(s) {stray[:10]} that "
                "meta initialization would leave uninitialized"
            )
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    check_state_dict_keys(missing, unexpected, ignore_unexpected=ignore_unexpected_keys)
    log.info(
        "[EgoHand] Loaded %s checkpoint %s: %d tensors; %d allowlisted missing key(s)",
        source.format,
        source.path,
        len(state_dict),
        len(missing),
    )
    return model.eval(), target


#: OpenUSD modules that py-soma-x imports (``soma.io``) during model construction.
_USD_MODULES = ("pxr.Gf", "pxr.Sdf", "pxr.Usd", "pxr.UsdGeom", "pxr.UsdSkel", "pxr.Vt")


def preload_usd_libraries() -> None:
    """Import OpenUSD on the calling thread before any worker thread starts.

    OpenUSD's shared libraries take the GIL in their static initializers, which
    run inside ``dlopen`` while glibc's loader lock is held. A thread that holds
    the GIL and needs the same loader lock -- a thread's first use of a C++
    ``thread_local`` with a destructor (``__cxa_thread_atexit``), e.g. a decode
    thread's first ``torch.device`` -- then deadlocks with a background model
    load that is importing ``soma``. Importing these modules first, while only
    one thread runs, removes the window (about 0.2 s). Without OpenUSD
    installed this does nothing.
    """
    import importlib

    for name in _USD_MODULES:
        try:
            importlib.import_module(name)
        except ImportError:
            return


class EgoHandPredictor:
    """Run bbox-conditioned predictions with an already-loaded EgoHand model.

    Use ``from_checkpoint()`` when you have checkpoint paths; construct directly
    when you already have a loaded model (e.g. in tests). This class owns one
    model instance, the batched crop-and-forward loop, and SOMA NPZ writing.

    Args:
        model: EgoHand instance. The predictor calls
            ``.eval().to(device)`` — the caller does not need to.
        batch_size: Hand crops per forward pass.
        device: Torch device for inference.
        dtype: Inference dtype (``torch.float32``, ``torch.float16``, or
            ``torch.bfloat16``).
        smooth: Apply motion-adaptive SOMA parameter smoothing and rebuild
            geometry before predictions are written or returned. Requires
            frame timestamps in seconds. ``False`` (default) returns raw output.
        use_perspective_cropping: Use perspective cropping for each hand
            bbox with the source camera's ``project`` and ``unproject`` methods.
    """

    # Crop while earlier batches run (see ``inference/pipeline/crop_stream.py``).
    # Same crops, order and batches as building every crop first; False
    # restores the build-everything-first path.
    stream_crops = True

    def __init__(
        self,
        model,
        batch_size,
        device="cuda",
        dtype: torch.dtype = torch.float32,
        smooth: bool = False,
        use_perspective_cropping: bool = False,
        perf_reporter: PerfReporter | None = None,
    ):
        self.batch_size = batch_size
        self.device = device
        self.dtype = dtype
        self.smooth = bool(smooth)
        self.perf_reporter = perf_reporter
        self._use_perspective_cropping = use_perspective_cropping

        self.model = model.eval().to(device)

        self._model_wrist_orients = None
        if self._use_perspective_cropping:
            self._model_wrist_orients = {}
            for hand in ("left", "right"):
                soma_hand_layer = getattr(self.model.soma_hand, hand)
                bind_pose_world = (
                    soma_hand_layer.bind_pose_world[0]
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(np.float64)
                )
                t_pose_world = (
                    soma_hand_layer.t_pose_world[0]
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(np.float64)
                )
                self._model_wrist_orients[hand] = (
                    np.linalg.inv(bind_pose_world) @ t_pose_world
                )[:3, :3]

        # PoseMirror_SOMA's setup is deterministic for a given left/right joint
        # naming, so build it once at predictor init and reuse across every
        # batch's right-hand mirror call.
        from soma.geometry.rig_utils import PoseMirror_SOMA

        left_joint_names = list(self.model.soma_hand.left.rig_data["joint_names"])
        right_joint_names = list(self.model.soma_hand.right.rig_data["joint_names"])
        paired_names = ["Root"] + left_joint_names + right_joint_names
        self._pose_mirror = PoseMirror_SOMA(paired_names, root_name="Root")

    @property
    def camera_mode(self) -> str:
        """Camera parameterization of the loaded checkpoint."""
        return getattr(self.model, "camera_mode", "weak_perspective")

    @classmethod
    def load_in_background(
        cls,
        egohand_ckpt_path: str | Path,
        soma_hand_data_folder: str | Path,
        batch_size: int,
        device: str = "cuda",
        dtype: torch.dtype = torch.float32,
        smooth: bool = False,
        dino_compile: bool = False,
        **from_checkpoint_kwargs,
    ) -> "PendingEgoHandPredictor":
        """Start ``from_checkpoint`` on a worker thread and return immediately.

        The checkpoint cfg is read (and its ``_target_`` validated) on the
        calling thread so configuration errors still fail fast; building the
        model and reading the tensors overlap whatever the caller does next,
        typically video decoding and 2D detection. The worker first checks the
        checkpoint's key set (``wait_until_validated``).

        Takes the arguments of ``from_checkpoint`` (every keyword not listed
        here is passed on unchanged) plus ``dino_compile``.
        """
        source = read_checkpoint(egohand_ckpt_path)
        cfg, _ = _validated_cfg(source, soma_hand_data_folder)
        try:
            import warp as wp

            # warp.init() has no lock; initialize on this thread before the
            # worker's SOMA construction and the detector/crop kernels race.
            wp.init()
        except ImportError:
            pass
        preload_usd_libraries()

        def validate():
            check_checkpoint_keys(source, cfg)

        def load():
            predictor = cls.from_checkpoint(
                egohand_ckpt_path=source,
                soma_hand_data_folder=soma_hand_data_folder,
                batch_size=batch_size,
                device=device,
                dtype=dtype,
                smooth=smooth,
                **from_checkpoint_kwargs,
            )
            if dino_compile:
                predictor.enable_dino_compile()
            return predictor

        return PendingEgoHandPredictor(
            load,
            validate=validate,
            batch_size=batch_size,
            device=device,
            dtype=dtype,
            smooth=smooth,
            camera_mode=str(cfg.get("camera_mode", "weak_perspective")),
        )

    def enable_dino_compile(self) -> None:
        """Compile full-batch DINO blocks for inference on this predictor."""
        backbone = getattr(self.model, "backbone", None)
        enable_compile = getattr(backbone, "enable_full_batch_compile", None)
        if enable_compile is None:
            raise RuntimeError(
                "DINO compilation requires a DINOv3 backbone with compile support"
            )
        enable_compile(self.batch_size)

    @classmethod
    def from_checkpoint(
        cls,
        egohand_ckpt_path: str | Path,
        soma_hand_data_folder: str | Path,
        batch_size: int,
        device: str = "cuda",
        dtype: torch.dtype = torch.float32,
        smooth: bool = False,
        use_perspective_cropping: bool = False,
    ) -> "EgoHandPredictor":
        """Load a checkpoint and return a ready-to-use predictor.

        Combines ``load_egohand_model`` and ``EgoHandPredictor.__init__`` into one
        call. Use this when you have checkpoint paths; construct directly when
        you already have a loaded model (e.g. in tests).

        Args:
            egohand_ckpt_path: EgoHand ``.ckpt`` path, inference-weights
                export folder, or a ``CheckpointSource``.
            soma_hand_data_folder: SOMAHand asset folder.
            batch_size: Hand crops per forward pass.
            device: Torch device for inference.
            dtype: Inference dtype.
            smooth: Enable adaptive parameter smoothing (default off).
            use_perspective_cropping: Use perspective cropping for each
                hand bbox before inference (default off).

        Returns:
            Predictor owning the loaded checkpoint and inference configuration.
        """
        model, target = load_egohand_model(
            egohand_ckpt_path, soma_hand_data_folder, device=device
        )
        log.warning("[EgoHand] Model: %s (dtype=%s)", target, dtype)
        return cls(
            model=model,
            batch_size=batch_size,
            device=device,
            dtype=dtype,
            smooth=smooth,
            use_perspective_cropping=use_perspective_cropping,
        )

    @torch.no_grad()
    def run(
        self,
        images_bgr,
        image_ids,
        bboxes_per_image,
        hands,
        *,
        intrinsics_per_image: dict[str, np.ndarray | CameraModel],
        output_paths_per_hand=None,
        frame_times: dict[str, float] | None = None,
        defer_smoothing: bool = False,
        source_cameras_per_image: dict[str, CameraModel] | None = None,
    ) -> dict[str, dict[str, dict]]:
        """Prepare hand crops from image lists and produce SOMA predictions.

        The in-memory result dict is always returned; SOMA NPZs are
        written to disk only when ``output_paths_per_hand`` is supplied.
        Pass ``None`` (or a dict that's missing the requested key) to
        skip per-frame NPZ writes when a caller only needs the in-memory
        dict, so saving them is purely disk cost.

        Args:
            images_bgr: Input images loaded by OpenCV in BGR channel order.
            image_ids: Image IDs aligned with ``images_bgr``.
            bboxes_per_image: Padded bbox mapping shaped as
                ``{image_id: {hand: [center_x, center_y, size]}}``.
            hands: Hands requested for this run.
            intrinsics_per_image: Per-image full-image pinhole camera
                intrinsics keyed by image ID, in decoded-image pixel
                coordinates, or source camera objects exposing ``project`` and
                ``unproject`` for perspective cropping.
            output_paths_per_hand: Optional per-hand per-image NPZ output
                paths. When ``None``, the NPZ save loop is skipped entirely.
            frame_times: Image ID to timestamp in seconds. Required when
                smoothing a complete clip; timestamps must be strictly increasing.
            defer_smoothing: Return raw predictions for a frame batch. Collect
                all batches and call ``postprocess_results`` before export.
                Requires ``output_paths_per_hand=None``.
            source_cameras_per_image: Original cameras when input frames were
                undistorted. Returned 2D predictions and bboxes use source pixels.

        Returns:
            Mapping from hand to image ID to newly inferred result dict.
        """
        if defer_smoothing and output_paths_per_hand is not None:
            raise ValueError("Deferred smoothing requires output_paths_per_hand=None")
        if self.smooth and not defer_smoothing:
            if frame_times is None:
                raise ValueError("Adaptive smoothing requires frame_times in seconds")
            validate_frame_times([frame_times[i] for i in image_ids], len(image_ids))

        # Build the (image, hand) jobs first, then run crop_and_preprocess in a
        # thread pool. cv2.warpAffine and the numpy normalization release the
        # GIL, so this parallelizes across cores and removes the previously
        # single-threaded preprocessing latency before the GPU forward pass.
        crop_jobs = []
        for image_bgr, image_id in zip(images_bgr, image_ids):
            bboxes_per_hand = bboxes_per_image.get(image_id, {})
            for hand in hands:
                bbox = bboxes_per_hand.get(hand)
                if bbox is None:
                    log.info("[Skip] %s %s: no reliable hand bbox", image_id, hand)
                    continue
                bbx_array = np.asarray(bbox, dtype=np.float32)
                # Use the camera intrinsics provided by the dataset for this image.
                source_camera = intrinsics_per_image[image_id]
                crop_jobs.append((image_id, hand, image_bgr, bbx_array, source_camera))

        def _build_input(job):
            """Build one normalized crop and its model/output camera metadata.

            Args:
                job: Image id, hand side, BGR image, source bbox, and camera.

            Returns:
                Image id, hand side, and prepared input, or None for an invalid ROI.
            """
            image_id, hand, image_bgr, bbx_array, source_camera = job
            if self._use_perspective_cropping:
                try:
                    crop_rgb, K_virtual, bbox_virtual, crop_to_source_rotation = (
                        perspective_crop(image_bgr, bbx_array, source_camera, IMG_SIZE)
                    )
                except (FloatingPointError, ValueError) as exc:
                    log.warning(
                        "[Skip] %s %s: perspective cropping failed: %s",
                        image_id,
                        hand,
                        exc,
                    )
                    return image_id, hand, None
                model_bbx_xys = bbox_virtual
                model_K = K_virtual
                # Canonicalize right hands in the rectified view. The output
                # camera remains unmirrored for the existing SOMA mirror path.
                if hand == "right":
                    crop_rgb = np.ascontiguousarray(crop_rgb[:, ::-1])
                    model_bbx_xys, model_K = mirror_full_image_camera_horizontal(
                        model_bbx_xys, model_K, crop_rgb.shape[1]
                    )
                # This is already a model-sized crop; normalize directly to
                # avoid a second crop shifting the virtual camera's pixels.
                normalized_crop_rgb = (crop_rgb - IMAGENET_MEAN) / IMAGENET_STD
                return (
                    image_id,
                    hand,
                    {
                        "image_id": image_id,
                        "real_hand_side": hand,
                        "crop_tensor": torch.from_numpy(
                            normalized_crop_rgb.transpose(2, 0, 1)
                        ),
                        "model_bbx_xys": model_bbx_xys,
                        "model_K": model_K,
                        "model_img_size": np.array(
                            [crop_rgb.shape[1], crop_rgb.shape[0]], dtype=np.float32
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
                # Training canonicalizes right hands by flipping them into the left-hand frame.
                calibrated_full_image = (
                    getattr(
                        self.model,
                        "camera_mode",
                        "weak_perspective",
                    )
                    == "sam3db_perspective"
                )
                crop_tensor, model_bbx_xys, model_K = prepare_right_hand_model_input(
                    image_bgr,
                    model_bbx_xys,
                    model_K,
                    calibrated_full_image=calibrated_full_image,
                )
            else:
                crop_tensor, _ = crop_and_preprocess(
                    image_bgr,
                    bbx_array,
                )
            return (
                image_id,
                hand,
                {
                    "image_id": image_id,
                    "real_hand_side": hand,
                    "crop_tensor": crop_tensor,
                    "model_bbx_xys": model_bbx_xys,
                    "model_K": model_K,
                    "model_img_size": np.array(
                        [image_width, image_height], dtype=np.float32
                    ),
                    "output_bbx_xys": bbx_array,
                    "output_K": K_fullimg,
                },
            )

        perf_reporter = getattr(self, "perf_reporter", None)
        prepared_inputs_by_image = {}
        if (
            crop_jobs
            and getattr(self, "stream_crops", False)
            and len(set(image_ids)) == len(image_ids)
        ):
            # Crops of later batches are prepared while earlier batches run.
            # Batches are formed from the valid crops in job order, exactly as
            # the list path below slices ``prepared_inputs``.
            def _keep_metadata(image_id, hand, prepared_hand_data):
                prepared_inputs_by_image.setdefault(image_id, {})[hand] = (
                    prepared_hand_data
                )

            log.warning(
                "[EgoHand] Running inference on up to %d hand crop(s), cropped "
                "while earlier batches run...",
                len(crop_jobs),
            )
            with StreamedCropBatches(
                crop_jobs,
                _build_input,
                batch_size=self.batch_size,
                device=self.device,
                workers=min(8, (os.cpu_count() or 1)),
                copy_stream=self._crop_copy_stream(),
                on_prepared=_keep_metadata,
            ) as crop_batches:
                predictions = self._run_imagebased_predictions(
                    crop_batches, frame_count=len(image_ids)
                )
            if perf_reporter is not None:
                # Crop time left on the critical path: waits for the next batch.
                perf_reporter.record_latency(
                    PerfStage.PREPROCESSING,
                    crop_batches.wait_s * 1000 / len(image_ids),
                )
        else:
            predictions = self._prepare_all_then_predict(
                crop_jobs, _build_input, image_ids, prepared_inputs_by_image
            )

        results_per_hand = {hand: {} for hand in hands}
        for result in predictions:
            image_id = result["image_id"]
            hand = result["hand_type"]
            if self._use_perspective_cropping:
                prepared_hand_data = prepared_inputs_by_image[image_id][hand]
                crop_to_source_rotation = prepared_hand_data[
                    "crop_to_source_camera_rotation"
                ]
                # Restore geometry to source-camera axes, retaining output dtypes.
                for geometry_key in ("pred_vertices", "pred_keypoints_3d"):
                    crop_camera_points = result[geometry_key]
                    result[geometry_key] = (
                        crop_camera_points @ crop_to_source_rotation.T
                    ).astype(crop_camera_points.dtype, copy=False)
                crop_translation = result["transl"]
                result["transl"] = (crop_to_source_rotation @ crop_translation).astype(
                    crop_translation.dtype, copy=False
                )
                # Saved SOMA poses need the same camera rotation, expressed in
                # wrist axes. Finger rotations remain relative to their parents.
                joint_rotations = result["poses"].copy()
                wrist_orientation = self._model_wrist_orients[hand]
                joint_rotations[0] = (
                    wrist_orientation
                    @ crop_to_source_rotation
                    @ wrist_orientation.T
                    @ joint_rotations[0]
                )
                result["poses"] = joint_rotations
                # A pinhole K cannot project source pixels with lens distortion.
                result["K_fullimg"] = None
            results_per_hand[hand][image_id] = result

        if not defer_smoothing:
            self.postprocess_results(
                results_per_hand,
                image_ids,
                bboxes_per_image,
                frame_times=frame_times,
                intrinsics_per_image=intrinsics_per_image,
                source_cameras_per_image=source_cameras_per_image,
            )
        if output_paths_per_hand is not None:
            self.save_results(results_per_hand, output_paths_per_hand)
        return results_per_hand

    def _prepare_all_then_predict(
        self, crop_jobs, build_input, image_ids, prepared_inputs_by_image
    ):
        """Build every crop first, then run the forward batches (previous path).

        Used when ``stream_crops`` is off or ``image_ids`` repeats an ID.
        Fills ``prepared_inputs_by_image`` and returns the predictions.
        """
        perf_reporter = getattr(self, "perf_reporter", None)
        if crop_jobs:
            crop_workers = min(8, (os.cpu_count() or 1))
            preprocessing_start = time.perf_counter()
            with ThreadPoolExecutor(max_workers=crop_workers) as pool:
                for image_id, hand, prepared_hand_data in pool.map(
                    build_input, crop_jobs
                ):
                    if prepared_hand_data is not None:
                        prepared_inputs_by_image.setdefault(image_id, {})[hand] = (
                            prepared_hand_data
                        )
            if perf_reporter is not None:
                preprocessing_ms_per_frame = (
                    (time.perf_counter() - preprocessing_start) * 1000 / len(image_ids)
                )
                perf_reporter.record_latency(
                    PerfStage.PREPROCESSING, preprocessing_ms_per_frame
                )

        prepared_inputs = [
            prepared_hand_data
            for image_id in image_ids
            for prepared_hand_data in prepared_inputs_by_image.get(
                image_id, {}
            ).values()
        ]

        if prepared_inputs:
            log.warning(
                "[EgoHand] Running inference on %d hand crop(s)...",
                len(prepared_inputs),
            )
        return self._run_imagebased_predictions(
            prepared_inputs, frame_count=len(image_ids)
        )

    def _crop_copy_stream(self):
        """One upload stream per predictor and CUDA device, reused across clips."""
        if torch.device(self.device).type != "cuda":
            return None
        streams = self.__dict__.setdefault("_crop_copy_streams", {})
        device = torch.device(self.device)
        if device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        if device not in streams:
            streams[device] = torch.cuda.Stream(device=device)
        return streams[device]

    def postprocess_results(
        self,
        results_per_hand,
        image_ids,
        bboxes_per_image,
        *,
        frame_times=None,
        intrinsics_per_image=None,
        source_cameras_per_image=None,
    ):
        """Finalize a complete clip before QC, exports or rendering.

        Joining frame batches before this call preserves the filter's temporal
        context. Missing predictions still split each hand's track. Fisheye
        output requires the original cameras in ``intrinsics_per_image``.
        Undistorted inputs use ``source_cameras_per_image`` to restore source pixels.
        """
        if self._use_perspective_cropping and intrinsics_per_image is None:
            raise ValueError("Fisheye postprocessing requires intrinsics_per_image")
        perf_reporter = getattr(self, "perf_reporter", None)
        if self.smooth:
            if frame_times is None:
                raise ValueError("Adaptive smoothing requires frame_times in seconds")
            log.info(
                "[EgoHand] Adaptive SOMA parameter smoothing and geometry reconstruction"
            )
            smoothing_start = time.perf_counter()
            smooth_adaptive_hand_results(
                results_per_hand,
                image_ids,
                bboxes_per_image,
                frame_times=[frame_times[i] for i in image_ids],
                soma_hand=self.model.soma_hand,
                device=self.device,
                batch_size=self.batch_size,
                intrinsics_per_image=intrinsics_per_image,
                min_depth=float(getattr(self.model, "kp2d_min_depth", 1e-4)),
            )
            if perf_reporter is not None and image_ids:
                smoothing_ms_per_frame = (
                    (time.perf_counter() - smoothing_start) * 1000 / len(image_ids)
                )
                perf_reporter.record_latency(
                    PerfStage.ADAPTIVE_SMOOTHING, smoothing_ms_per_frame
                )

        if source_cameras_per_image is not None:
            # Convert the detected pinhole boxes before reusing source-camera projection.
            gx, gy = np.meshgrid(np.linspace(-0.5, 0.5, 17), np.linspace(-0.5, 0.5, 17))
            offsets = np.stack((gx, gy), axis=-1).reshape(-1, 2)
            for image_id, boxes in bboxes_per_image.items():
                K = intrinsics_per_image[image_id]
                camera = source_cameras_per_image[image_id]
                for hand, bbox in list(boxes.items()):
                    pixels = np.asarray(bbox[:2]) + offsets * bbox[2]
                    rays = np.c_[
                        (pixels - K[:2, 2]) / K.diagonal()[:2], np.ones(len(pixels))
                    ]
                    projected = camera.project(rays)
                    projected = projected[np.isfinite(projected).all(axis=-1)]
                    if not len(projected):
                        del boxes[hand]
                        results_per_hand[hand].pop(image_id, None)
                        continue
                    low, high = projected.min(axis=0), projected.max(axis=0)
                    boxes[hand] = np.r_[(low + high) / 2, np.max(high - low)].astype(
                        np.float32
                    )

        if self._use_perspective_cropping or source_cameras_per_image is not None:
            # Project smoothed joints onto the source distorted image, then normalize
            # by the detector bbox for overlays and saved predictions.
            projection_cameras = source_cameras_per_image or intrinsics_per_image
            min_joint_depth = float(getattr(self.model, "kp2d_min_depth", 1e-4))
            for hand, hand_results in results_per_hand.items():
                for image_id, result in hand_results.items():
                    source_bbox = np.asarray(bboxes_per_image[image_id][hand])
                    source_camera_joints = np.add(
                        result["pred_keypoints_3d"], result["transl"], dtype=np.float64
                    )
                    valid_joint_projections = np.isfinite(source_camera_joints).all(
                        axis=-1
                    ) & (source_camera_joints[:, 2] > min_joint_depth)
                    source_pixels = np.full(
                        (len(source_camera_joints), 2), np.nan, dtype=np.float64
                    )
                    source_pixels[valid_joint_projections] = projection_cameras[
                        image_id
                    ].project(source_camera_joints[valid_joint_projections])
                    bbox_normalized_joints = (
                        source_pixels - source_bbox[:2]
                    ) / source_bbox[2]
                    valid_joint_projections &= np.isfinite(bbox_normalized_joints).all(
                        axis=-1
                    )
                    bbox_normalized_joints[~valid_joint_projections] = np.nan
                    result["pred_keypoints_2d"] = bbox_normalized_joints.astype(
                        result["pred_keypoints_2d"].dtype, copy=False
                    )
                    result["pred_projection_valid"] = valid_joint_projections
                    result["K_fullimg"] = None

        return results_per_hand

    def save_results(self, results_per_hand, output_paths_per_hand):
        """Write finalized predictions with the existing SOMA NPZ schema."""
        from soma.io import save_soma_npz

        for hand, hand_results in results_per_hand.items():
            if not hand_results:
                continue

            soma_hand_layer = getattr(self.model.soma_hand, hand)
            joint_orient = (
                soma_hand_layer.batched_skinning.joint_orient.detach()
                .cpu()
                .numpy()
                .astype(np.float32)
            )
            joint_names = list(soma_hand_layer.rig_data["joint_names"])

            for name, result in hand_results.items():
                out_path = Path(output_paths_per_hand[hand][name])
                out_path.parent.mkdir(parents=True, exist_ok=True)
                identity_coeffs = result["identity_coeffs"][None].astype(np.float32)
                if identity_coeffs.shape[-1] != 20:
                    # save_soma_npz expects the fixed 20D SOMAHand identity
                    # schema. Checkpoints using a different-dimensional
                    # identity model (e.g. 5D MHR) don't populate it, so
                    # write a zero placeholder here to satisfy the file
                    # format -- the real coefficients are stored unpadded
                    # in mhr_identity_coeffs below.
                    identity_coeffs = np.zeros((1, 20), dtype=np.float32)
                extra_arrays = {
                    "vertices": result["pred_vertices"][None].astype(np.float32),
                    "joints": result["pred_keypoints_3d"][None].astype(np.float32),
                    "pred_keypoints_2d": result["pred_keypoints_2d"][None].astype(
                        np.float32
                    ),
                    "pred_projection_valid": result["pred_projection_valid"][
                        None
                    ].astype(np.bool_),
                    "decoded_frame": np.array("wrist_local"),
                    "mhr_identity_coeffs": result["identity_coeffs"][None].astype(
                        np.float32
                    ),
                }
                if result["scale_params"] is not None:
                    extra_arrays["scale_params"] = result["scale_params"][None].astype(
                        np.float32
                    )
                for key in KEYPOINT_RELIABILITY_FIELDS:
                    if result.get(key) is not None:
                        extra_arrays[key] = result[key][None].astype(np.float32)

                with contextlib.redirect_stdout(io.StringIO()):
                    save_soma_npz(
                        out_path,
                        poses=result["poses"][None].astype(np.float32),
                        transl=result["transl"][None].astype(np.float32),
                        joint_names=joint_names,
                        identity_model_type="SOMAHand",
                        identity_coeffs=identity_coeffs,
                        joint_orient=joint_orient,
                        global_scale=float(result["global_scale"]),
                        hand_type=hand,
                        unit="meters",
                        keep_root=True,
                        extra_arrays=extra_arrays,
                    )
            log.info("[EgoHand] Saved %d %s SOMA NPZ file(s)", len(hand_results), hand)

    @torch.no_grad()
    def _run_imagebased_predictions(
        self, prepared_inputs: list, frame_count: int | None = None
    ) -> list:
        """Run batched inference for EgoHand checkpoints.

        Processes all (image, hand) crops in one flat batch stream. Both hands
        are interleaved — right-hand crops were already flipped into left-hand
        canonical form during preprocessing; a single batched mirror call
        converts their outputs back to right-hand SOMA at the end of each batch.

        Args:
            prepared_inputs: Ordered list of per-crop dicts built by ``run()``'s
                preprocessing step. Each dict carries ``crop_tensor``,
                ``model_bbx_xys``, ``model_K``, ``real_hand_side``, etc.
                Alternatively a ``StreamedCropBatches`` that yields the same
                batches, with the images already uploaded.
            frame_count: Number of frames these crops were drawn from, for
                ``PerfStage.MODEL_FORWARD`` normalization. Required when
                ``self.perf_reporter`` is set.

        Returns:
            Flat list of per-crop result dicts ready for ``results_per_hand``
            assembly and SOMA NPZ writing.
        """
        perf_reporter = getattr(self, "perf_reporter", None)
        predictions = []
        # Contiguous, non-overlapping per-batch phases: assembly -> forward -> post-forward.
        batch_assembly_ms_total = 0.0
        forward_ms_total = 0.0
        post_forward_ms_total = 0.0
        if isinstance(prepared_inputs, StreamedCropBatches):
            # Batches of valid crops, already uploaded while earlier batches ran.
            batch_source = ((staged.hand_data, staged) for staged in prepared_inputs)
        else:
            batch_source = (
                (prepared_inputs[start : start + self.batch_size], None)
                for start in range(0, len(prepared_inputs), self.batch_size)
            )
        for batch_hand_data, staged in batch_source:
            batch_assembly_start = time.perf_counter()
            batch = {
                "img": staged.images(self.device)
                if staged is not None
                else torch.stack([d["crop_tensor"] for d in batch_hand_data]).to(
                    self.device
                ),
                "bbx_xys": torch.from_numpy(
                    np.stack([d["model_bbx_xys"] for d in batch_hand_data])
                )
                .float()
                .to(self.device),
                "K_fullimg": torch.from_numpy(
                    np.stack([d["model_K"] for d in batch_hand_data])
                )
                .float()
                .to(self.device),
                "img_size": torch.from_numpy(
                    np.stack([d["model_img_size"] for d in batch_hand_data])
                )
                .float()
                .to(self.device),
                # Right-hand crops were flipped into the left-hand canonical frame
                # during preprocessing; the model always sees left-hand inputs.
                "hand_type": ["left"] * len(batch_hand_data),
            }
            if perf_reporter is not None and self.device != "cpu":
                # Counted in batch assembly: waits for the host-to-device copies.
                torch.cuda.synchronize(self.device)
            forward_start = time.perf_counter()
            batch_assembly_ms_total += (forward_start - batch_assembly_start) * 1000
            with torch.autocast(
                "cuda", dtype=self.dtype, enabled=(self.dtype != torch.float32)
            ):
                output = self.model.forward_step(batch, train=False)
            if perf_reporter is not None and self.device != "cpu":
                torch.cuda.synchronize(self.device)
            post_forward_start = time.perf_counter()
            forward_ms_total += (post_forward_start - forward_start) * 1000

            params = output["pred_hand_params"]
            # EgoHand packs global_scale into scale_params[:, 0] and per-finger
            # scale into the tail. Split here so the mirror call and D2H loop
            # below see consistent shapes.
            scale_params_raw = params["scale_params"]
            global_scale_batch = scale_params_raw[:, 0]
            scale_params_batch = (
                scale_params_raw[:, 1:] if scale_params_raw.shape[1] > 1 else None
            )

            if (
                getattr(self.model, "camera_mode", "weak_perspective")
                == "weak_perspective"
            ):
                output["pred_cam_t"] = compute_transl_full_cam(
                    output["pred_cam"].float(),
                    batch["bbx_xys"].float(),
                    batch["K_fullimg"].float(),
                )
                (
                    output["pred_keypoints_2d"],
                    output["pred_projection_valid"],
                ) = project_camera_points_to_crop_with_validity(
                    output["pred_keypoints_3d"].float()
                    + output["pred_cam_t"].unsqueeze(1),
                    batch["K_fullimg"].float(),
                    batch["bbx_xys"].float(),
                    min_depth=float(getattr(self.model, "kp2d_min_depth", 1e-4)),
                )

            pred_camera_points = output["pred_keypoints_3d"].float() + output[
                "pred_cam_t"
            ].float().unsqueeze(1)
            fallback_projection_valid = torch.isfinite(pred_camera_points).all(
                dim=-1
            ) & (
                pred_camera_points[..., 2]
                > float(
                    getattr(
                        self.model,
                        "kp2d_min_depth",
                        1e-4,
                    )
                )
            )
            projection_valid = output.get(
                "pred_projection_valid",
                fallback_projection_valid,
            )
            projection_valid = projection_valid & torch.isfinite(
                output["pred_keypoints_2d"]
            ).all(dim=-1)
            output["pred_projection_valid"] = projection_valid
            output["pred_keypoints_2d"] = output["pred_keypoints_2d"].masked_fill(
                ~projection_valid.unsqueeze(-1),
                float("nan"),
            )

            # Mirror the right-hand sub-batch in one vectorised call. right_indices
            # is a subset of the mixed-hand batch; right_pos gives O(1) lookup back
            # into right_batched in the per-sample loop below.
            right_indices = [
                i
                for i, d in enumerate(batch_hand_data)
                if d["real_hand_side"] == "right"
            ]
            right_batched = None
            if right_indices:
                idx = torch.tensor(right_indices, device=self.device, dtype=torch.long)
                # Disable any inherited autocast (e.g. from a preceding detector
                # that uses bfloat16 internally). The MHR identity model uses sparse
                # CUDA matmul which does not support half precision.
                with torch.autocast("cuda", enabled=False):
                    right_batched = mirror_left_prediction_to_right(
                        self.model,
                        torch.cat(
                            [
                                params["wrist_orient"][idx].float(),
                                params["hand_pose"][idx].float(),
                            ],
                            dim=1,
                        ),
                        output["pred_cam_t"][idx].float(),
                        params["identity_coeffs"][idx].float(),
                        global_scale_batch[idx].float(),
                        torch.from_numpy(
                            np.stack(
                                [batch_hand_data[i]["output_K"] for i in right_indices]
                            )
                        )
                        .float()
                        .to(self.device),
                        torch.from_numpy(
                            np.stack(
                                [
                                    batch_hand_data[i]["output_bbx_xys"]
                                    for i in right_indices
                                ]
                            )
                        )
                        .float()
                        .to(self.device),
                        scale_params=scale_params_batch[idx].float()
                        if scale_params_batch is not None
                        else None,
                        pose_mirror=self._pose_mirror,
                    )

            # One D2H copy per tensor for the whole batch, then index per sample.
            reliability_cpu = {
                key: output[key].float().cpu().numpy()
                for key in KEYPOINT_RELIABILITY_FIELDS
                if output.get(key) is not None
            }
            verts_cpu = output["pred_vertices"].cpu().numpy()
            transl_cpu = output["pred_cam_t"].cpu().numpy()
            kp3d_cpu = output["pred_keypoints_3d"].cpu().numpy()
            kp2d_cpu = output["pred_keypoints_2d"].cpu().numpy()
            kp2d_valid_cpu = (
                output.get(
                    "pred_projection_valid",
                    torch.isfinite(output["pred_keypoints_2d"]).all(dim=-1),
                )
                .cpu()
                .numpy()
            )
            wrist_cpu = params["wrist_orient"].float().cpu().numpy()
            fingers_cpu = params["hand_pose"].float().cpu().numpy()
            id_cpu = params["identity_coeffs"].float().cpu().numpy()
            gs_cpu = global_scale_batch.float().cpu().numpy()
            sp_cpu = (
                scale_params_batch.float().cpu().numpy()
                if scale_params_batch is not None
                else None
            )
            rv_cpu = rt_cpu = rk3_cpu = rk2_cpu = rkv_cpu = rp_cpu = None
            if right_batched is not None:
                rv_cpu = right_batched["pred_vertices"].cpu().numpy()
                rt_cpu = right_batched["transl"].cpu().numpy()
                rk3_cpu = right_batched["pred_keypoints_3d"].cpu().numpy()
                rk2_cpu = right_batched["pred_keypoints_2d"].cpu().numpy()
                rkv_cpu = right_batched["pred_projection_valid"].cpu().numpy()
                rp_cpu = right_batched["poses"].cpu().numpy()
            right_pos: dict[int, int] = {i: r for r, i in enumerate(right_indices)}

            for i, d in enumerate(batch_hand_data):
                side = d["real_hand_side"]
                if side == "right":
                    r = right_pos[i]
                    predictions.append(
                        {
                            "pred_vertices": rv_cpu[r],
                            "transl": rt_cpu[r],
                            "pred_keypoints_3d": rk3_cpu[r],
                            "pred_keypoints_2d": rk2_cpu[r],
                            "pred_projection_valid": rkv_cpu[r],
                            "poses": rp_cpu[r],
                            "identity_coeffs": id_cpu[i],
                            "scale_params": sp_cpu[i] if sp_cpu is not None else None,
                            "global_scale": float(gs_cpu[i]),
                            "hand_type": side,
                            "image_id": d["image_id"],
                            "K_fullimg": d["output_K"],
                        }
                    )
                else:
                    predictions.append(
                        {
                            "pred_vertices": verts_cpu[i],
                            "transl": transl_cpu[i],
                            "pred_keypoints_3d": kp3d_cpu[i],
                            "pred_keypoints_2d": kp2d_cpu[i],
                            "pred_projection_valid": kp2d_valid_cpu[i],
                            "poses": np.concatenate(
                                [wrist_cpu[i], fingers_cpu[i]], axis=0
                            ),
                            "identity_coeffs": id_cpu[i],
                            "scale_params": sp_cpu[i] if sp_cpu is not None else None,
                            "global_scale": float(gs_cpu[i]),
                            "hand_type": side,
                            "image_id": d["image_id"],
                            "K_fullimg": d["output_K"],
                        }
                    )
                # Both hands retain anatomical joint order; these describe the model crop.
                predictions[-1].update(
                    {key: values[i] for key, values in reliability_cpu.items()}
                )
            post_forward_ms_total += (time.perf_counter() - post_forward_start) * 1000
            if staged is not None:
                # Every read of this batch's images is queued; reuse the slot.
                del batch
                staged.release()
        if perf_reporter is not None and frame_count:
            perf_reporter.record_latency(
                PerfStage.BATCH_ASSEMBLY, batch_assembly_ms_total / frame_count
            )
            perf_reporter.record_latency(
                PerfStage.MODEL_FORWARD, forward_ms_total / frame_count
            )
            perf_reporter.record_latency(
                PerfStage.POST_FORWARD, post_forward_ms_total / frame_count
            )
        return predictions


#: PendingEgoHandPredictors of this process whose load is still running (each
#: worker removes itself when done, so a finished load is not kept alive here).
_PENDING_LOADS: list = []


def wait_for_background_loads(timeout: float) -> bool:
    """Wait (up to ``timeout`` seconds in total) until every background model
    load of this process has finished; True when none is still running.

    The loads run on daemon threads. When the main thread ends with an
    exception while one is still inside native code (checkpoint read, CUDA
    copies), interpreter shutdown stops that thread there and the process
    aborts ("terminate called without an active exception", status 134, core
    dump). Waiting first lets the interpreter shut down normally.
    """
    deadline = time.monotonic() + max(0.0, timeout)
    for pending in _PENDING_LOADS:
        if not pending._done.wait(max(0.0, deadline - time.monotonic())):
            return False
    return True


class PendingEgoHandPredictor:
    """An ``EgoHandPredictor`` that is still loading on a worker thread.

    ``batch_size``, ``device``, ``dtype``, ``smooth`` and ``camera_mode`` are
    known up front and never block. Any other attribute (``run``, ``model``,
    ``postprocess_results`` ...) waits for the load to finish and is then
    served by the loaded predictor; a load failure is re-raised there.
    ``perf_reporter`` may be set before the load finishes; it is handed to the
    loaded predictor.
    """

    def __init__(
        self, load, *, batch_size, device, dtype, smooth, camera_mode, validate=None
    ):
        import threading

        self.batch_size = batch_size
        self.device = device
        self.dtype = dtype
        self.smooth = bool(smooth)
        self.camera_mode = camera_mode
        self.load_seconds: float | None = None
        self.wait_seconds = 0.0
        self._predictor = None
        self._error: BaseException | None = None
        self._done = threading.Event()
        self._validated = threading.Event()
        self._validation_error: BaseException | None = None
        self.validate_seconds: float | None = None

        def worker():
            start = time.perf_counter()
            try:
                try:
                    if validate is not None:
                        validate()
                except BaseException as exc:
                    self._validation_error = exc
                    raise
                finally:
                    self.validate_seconds = time.perf_counter() - start
                    self._validated.set()
                predictor = load()
                if torch.cuda.is_available() and torch.cuda.is_initialized():
                    # Count queued weight copies as load time, without
                    # waiting on other threads' streams.
                    torch.cuda.current_stream().synchronize()
                self._predictor = predictor
            except BaseException as exc:  # re-raised on first use
                self._error = exc
            finally:
                self.load_seconds = time.perf_counter() - start
                self._done.set()
                with contextlib.suppress(ValueError):
                    _PENDING_LOADS.remove(self)

        self._thread = threading.Thread(
            target=worker, name="egohand-model-load", daemon=True
        )
        _PENDING_LOADS.append(self)
        self._thread.start()

    def wait_until_validated(self) -> float:
        """Block until the checkpoint key check is done; re-raise its error.

        Callers do this before decoding so a mismatched checkpoint fails
        before any frame is processed or output written. Returns the seconds
        spent waiting.
        """
        start = time.perf_counter()
        self._validated.wait()
        waited = time.perf_counter() - start
        if self._validation_error is not None:
            raise self._validation_error
        return waited

    def result(self) -> EgoHandPredictor:
        """Block until the predictor is loaded and return it."""
        if not self._done.is_set():
            start = time.perf_counter()
            self._done.wait()
            self.wait_seconds += time.perf_counter() - start
        if self._error is not None:
            raise self._error
        if "_perf_reporter" in self.__dict__:
            self._predictor.perf_reporter = self._perf_reporter
        return self._predictor

    def __setattr__(self, name, value):
        if name == "perf_reporter":
            # The command line sets the per-video perf reporter before the
            # model may have finished loading: keep it and hand it to the
            # predictor when it is resolved (result()).
            self.__dict__["_perf_reporter"] = value
            if self.__dict__.get("_predictor") is not None:
                self._predictor.perf_reporter = value
            return
        super().__setattr__(name, value)

    def __getattr__(self, name):
        # Only called for attributes not set in __init__.
        if name.startswith("__") or name in (
            "_predictor",
            "_error",
            "_done",
            "_thread",
            "_validated",
            "_validation_error",
            "_perf_reporter",
        ):
            raise AttributeError(name)
        return getattr(self.result(), name)
