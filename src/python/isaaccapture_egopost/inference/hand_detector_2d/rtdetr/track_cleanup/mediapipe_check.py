# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-OpenMDW-1.1

"""``mediapipe_check``: is there a hand in a candidate box? (MediaPipe hand landmarks)

Model: ``handpose_estimation_mediapipe_2023feb.onnx`` from the OpenCV Zoo
(https://huggingface.co/opencv/handpose_estimation_mediapipe), the MediaPipe
hand-landmark model converted to ONNX; Apache-2.0, 4.1 MB, fp32. It is not bundled;
point ``track_cleanup.mediapipe_weights`` at a local copy. The file must match
``MEDIAPIPE_SHA256``: the thresholds were tuned on it. It runs on the existing
``onnxruntime`` (the detector's providers) and OpenCV; no ``mediapipe`` package is
needed. Input ``[1, 224, 224, 3]`` RGB / 255 (NHWC); outputs (graph order, checked
at load by shape): landmarks ``[1, 63]``, hand presence ``[1, 1]``, handedness
``[1, 1]``, world landmarks ``[1, 63]``. Only the presence is used.

Crop: a square of side ``crop_scale * max(w, h)`` centred on the box centre,
axis-aligned, zero-padded outside the image, resized to 224 with ``INTER_AREA``.
"""

import hashlib
from pathlib import Path

import numpy as np

SIZE = 224
CROP_SCALE = 1.5
MEDIAPIPE_SHA256 = "db0898ae717b76b075d9bf563af315b29562e11f8df5027a1ef07b02bef6d81c"
#: Output shapes in graph order: landmarks, presence, handedness, world landmarks.
OUTPUT_SHAPES = ([1, 63], [1, 1], [1, 1], [1, 63])
MEDIAPIPE_URL = (
    "https://huggingface.co/opencv/handpose_estimation_mediapipe/resolve/main/"
    "handpose_estimation_mediapipe_2023feb.onnx"
)


def crop_blob(rgb, cxcywh, crop_scale=CROP_SCALE):
    """(1, 224, 224, 3) float32 blob and (x0, y0, side) of the square crop in pixels."""
    import cv2

    cx, cy, w, h = (float(v) for v in cxcywh[:4])
    s = crop_scale * max(w, h, 1.0)
    X1, Y1 = round(cx - s / 2), round(cy - s / 2)
    S = max(1, round(s))
    H, W = rgb.shape[:2]
    canvas = np.zeros((S, S, 3), np.uint8)
    sx1, sy1, sx2, sy2 = max(0, X1), max(0, Y1), min(W, X1 + S), min(H, Y1 + S)
    if sx2 > sx1 and sy2 > sy1:
        canvas[sy1 - Y1 : sy2 - Y1, sx1 - X1 : sx2 - X1] = rgb[sy1:sy2, sx1:sx2]
    blob = cv2.resize(canvas, (SIZE, SIZE), interpolation=cv2.INTER_AREA)
    return blob.astype(np.float32)[None] / 255.0, (X1, Y1, S)


def check_weights(path) -> Path:
    """The model path, after checking that it exists and is the expected file."""
    if "${" in str(path):
        raise FileNotFoundError(
            f"MediaPipe hand-landmark model path {path} still holds an unexpanded "
            "environment variable (not set?). Download the model from "
            f"{MEDIAPIPE_URL} (sha256 {MEDIAPIPE_SHA256}) and set it"
        )
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"MediaPipe hand-landmark model not found: {path}. Download it from "
            f"{MEDIAPIPE_URL} (sha256 {MEDIAPIPE_SHA256})"
        )
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != MEDIAPIPE_SHA256:
        raise ValueError(
            f"{path} has sha256 {digest}; mediapipe_check was tuned on "
            f"handpose_estimation_mediapipe_2023feb.onnx (sha256 {MEDIAPIPE_SHA256})"
        )
    return path


class MediaPipeHandVerifier:
    """Runs the landmark model on one candidate crop at a time (fixed batch of 1)."""

    def __init__(self, weights, providers=None, crop_scale=CROP_SCALE):
        weights = check_weights(weights)
        import onnxruntime as ort

        providers = providers or ["CPUExecutionProvider"]
        if any(
            (p[0] if isinstance(p, tuple) else p) == "CUDAExecutionProvider"
            for p in providers
        ):
            # Same CUDA/cuDNN preloading as the RT-DETR session; without it ORT may not
            # find the pip-installed CUDA libraries and silently falls back to CPU.
            from ..onnx_model import _preload_cuda_libs

            _preload_cuda_libs()
        options = ort.SessionOptions()
        options.log_severity_level = 3
        self.session = ort.InferenceSession(
            str(weights), sess_options=options, providers=providers
        )
        self.providers = self.session.get_providers()
        self.inp = self.session.get_inputs()[0].name
        outputs = self.session.get_outputs()
        shapes = tuple(list(o.shape) for o in outputs)
        if shapes != OUTPUT_SHAPES:
            raise ValueError(
                f"{weights}: unexpected MediaPipe output shapes {shapes}; "
                f"expected {OUTPUT_SHAPES} (landmarks, presence, handedness, world)"
            )
        self.outs = [o.name for o in outputs]
        self.crop_scale = crop_scale

    def presence(self, rgb, cxcywh) -> tuple[float, float]:
        """(hand presence, handedness) of the box ``cxcywh`` in the RGB image."""
        blob, _ = crop_blob(rgb, cxcywh, self.crop_scale)
        _, conf, handedness, _ = self.session.run(self.outs, {self.inp: blob})
        return float(conf.reshape(-1)[0]), float(handedness.reshape(-1)[0])


class LazyVerifier:
    """Per-clip memoised presence lookup for the tracker: ``fn(t, cand) -> (p, h)``.

    The tracker asks only about the candidates whose decision depends on it (new
    hands, low-score continuations, bridge evidence), so the model runs on those
    crops only, once each. ``cache`` can supply precomputed answers (key:
    ``LazyVerifier.key``); a crop missing from it with no verifier is an error
    (``KeyError``), so a missing model never silently turns the check off.
    """

    def __init__(self, images_bgr=None, verifier=None, cache=None):
        self.images_bgr = images_bgr
        self.verifier = verifier
        self.cache = cache if cache is not None else {}
        self.computed = 0
        self.keys_asked = set()
        self._rgb_t = None
        self._rgb = None

    @staticmethod
    def key(t, cand):
        """Frame index and the box rounded to 1e-3 px."""
        return (int(t), *(round(float(v) * 1000) for v in cand[:4]))

    def __call__(self, t, cand):
        k = self.key(t, cand)
        self.keys_asked.add(k)
        hit = self.cache.get(k)
        if hit is not None:
            return hit
        if self.verifier is None:
            raise KeyError(
                f"no MediaPipe model and no cached presence for frame {k[0]} box "
                f"{list(cand[:4])}"
            )
        import cv2

        if self._rgb_t != t:
            frame = np.ascontiguousarray(self.images_bgr[t])
            self._rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self._rgb_t = t
        result = self.verifier.presence(self._rgb, cand)
        self.computed += 1
        self.cache[k] = result
        return result
