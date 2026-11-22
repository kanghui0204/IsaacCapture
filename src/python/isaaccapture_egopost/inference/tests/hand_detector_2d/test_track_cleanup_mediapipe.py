# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-OpenMDW-1.1

"""MediaPipe hand-landmark check: crop, memoised lookup, model file checks."""

import os

import numpy as np
import pytest

from inference.hand_detector_2d.rtdetr.track_cleanup import mediapipe_check as mp


def test_crop_is_a_padded_square_around_the_box():
    rgb = np.full((80, 100, 3), 255, np.uint8)
    # a box near the top-left corner
    blob, (x0, y0, side) = mp.crop_blob(rgb, [10, 10, 20, 10])
    assert blob.shape == (1, mp.SIZE, mp.SIZE, 3) and blob.dtype == np.float32
    assert (x0, y0, side) == (-5, -5, 30)  # 1.5 x the longer side, centred on the box
    assert blob[0, 0, 0, 0] == 0.0, "outside the image is zero-padded"
    assert blob[0, -1, -1, 0] == pytest.approx(1.0), (
        "inside the image is the image / 255"
    )


class CountingVerifier:
    def __init__(self):
        self.calls = []

    def presence(self, rgb, cxcywh):
        self.calls.append((int(rgb[0, 0, 0]), tuple(cxcywh[:4])))
        return 0.7, 0.2


def test_lazy_verifier_runs_each_crop_once_on_rgb_frames():
    frames = [np.full((8, 8, 3), (t, 0, 255), np.uint8) for t in range(3)]  # BGR
    fake = CountingVerifier()
    verify = mp.LazyVerifier(images_bgr=frames, verifier=fake)
    box = [4.0, 4.0, 2.0, 2.0]
    assert verify(1, box) == (0.7, 0.2)
    assert verify(1, [4.0004, 4.0, 2.0, 2.0]) == (0.7, 0.2), "same crop at 1e-3 px"
    assert verify(2, box) == (0.7, 0.2)
    assert verify.computed == 2
    assert [c[0] for c in fake.calls] == [255, 255], "the frame is converted to RGB"
    assert len(verify.keys_asked) == 2


def test_lazy_verifier_cache_and_no_model():
    key = mp.LazyVerifier.key(5, [1.0, 2.0, 3.0, 4.0])
    verify = mp.LazyVerifier(cache={key: (0.3, 0.6)})
    assert verify(5, [1.0, 2.0, 3.0, 4.0]) == (0.3, 0.6)
    with pytest.raises(KeyError, match="no MediaPipe model"):
        verify(6, [1.0, 2.0, 3.0, 4.0])
    assert verify.computed == 0


def test_model_file_is_checked(tmp_path):
    with pytest.raises(FileNotFoundError, match=r"huggingface\.co/opencv"):
        mp.check_weights(tmp_path / "missing.onnx")
    with pytest.raises(FileNotFoundError, match="environment variable"):
        mp.check_weights("${MEDIAPIPE_HAND_WEIGHTS}")
    other = tmp_path / "other.onnx"
    other.write_bytes(b"not the model")
    with pytest.raises(ValueError, match="tuned on"):
        mp.check_weights(other)


@pytest.mark.skipif(
    not os.environ.get("EGOHAND_TEST_MEDIAPIPE_WEIGHTS"),
    reason="EGOHAND_TEST_MEDIAPIPE_WEIGHTS (the MediaPipe ONNX path) is not set",
)
def test_real_model_returns_a_presence():
    pytest.importorskip("onnxruntime")
    verifier = mp.MediaPipeHandVerifier(os.environ["EGOHAND_TEST_MEDIAPIPE_WEIGHTS"])
    assert verifier.providers == ["CPUExecutionProvider"]
    rgb = np.zeros((480, 640, 3), np.uint8)
    presence, handedness = verifier.presence(rgb, [320, 240, 120, 120])
    assert 0.0 <= presence <= 1.0 and 0.0 <= handedness <= 1.0
    assert presence < 0.1, "a black image holds no hand"
    # Fixed inputs, CPU provider: presence and handedness are the 2nd and 3rd
    # outputs (values from onnxruntime 1.28 on the CPU).
    blobs = {
        "zeros": (np.zeros((1, 224, 224, 3), np.float32), 0.014290, 0.085961),
        "ones": (np.ones((1, 224, 224, 3), np.float32), 0.003205, 0.214149),
    }
    for name, (blob, want_presence, want_handedness) in blobs.items():
        out = verifier.session.run(verifier.outs, {verifier.inp: blob})
        assert [list(o.shape) for o in out] == [list(s) for s in mp.OUTPUT_SHAPES]
        assert out[1].item() == pytest.approx(want_presence, abs=1e-4), name
        assert out[2].item() == pytest.approx(want_handedness, abs=1e-4), name
