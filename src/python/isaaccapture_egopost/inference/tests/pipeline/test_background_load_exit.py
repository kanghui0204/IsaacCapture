# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-OpenMDW-1.1

"""The command line fails while the model still loads in the background:
the error ends the process normally (traceback, non-zero status), after the
load has finished, instead of aborting in native code at interpreter shutdown
(``run_main`` / ``wait_for_background_loads``)."""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path


ROOT = (
    Path(__file__).resolve().parents[3]
)  # the egopost source root (inference/, egohand/)

SCRIPT = r"""
import sys, time
sys.path.insert(0, sys.argv[1])
from inference.pipeline.predictor import PendingEgoHandPredictor
from inference.scripts import run_egohand_inference_pipeline as cli

flag, mode = sys.argv[2], sys.argv[3]

def load():
    time.sleep(30 if mode == "timeout" else 1.5)
    open(flag, "w").write("loaded")
    return object()

PendingEgoHandPredictor(load, batch_size=1, device="cpu", dtype=None, smooth=False, camera_mode=None)

def main():
    if mode == "exit":
        raise SystemExit("[infer_video] bad configuration")
    raise RuntimeError("boom while detecting")

cli.run_main(main, wait_s=0.5 if mode == "timeout" else None)
"""


def _run(tmp_path, mode):
    flag = tmp_path / "loaded"
    start = time.monotonic()
    r = subprocess.run(
        [sys.executable, "-c", SCRIPT, str(ROOT), str(flag), mode],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    return r, flag.exists(), time.monotonic() - start


def test_error_waits_for_the_load_then_exits_normally(tmp_path):
    r, loaded, _ = _run(tmp_path, "raise")
    assert r.returncode == 1, r.stderr
    assert "Traceback (most recent call last)" in r.stderr
    assert "RuntimeError: boom while detecting" in r.stderr
    assert "terminate called" not in r.stderr
    assert loaded  # the load finished before the interpreter shut down


def test_exit_message_and_status_are_kept(tmp_path):
    r, loaded, _ = _run(tmp_path, "exit")
    assert r.returncode == 1, r.stderr
    assert "[infer_video] bad configuration" in r.stderr
    assert loaded


def test_a_load_that_does_not_finish_does_not_hold_the_exit(tmp_path):
    r, loaded, elapsed = _run(tmp_path, "timeout")
    assert r.returncode == 1, r.stderr
    assert (
        "RuntimeError: boom while detecting" in r.stderr
    )  # traceback printed before os._exit
    assert "the background model load did not finish" in r.stderr
    assert not loaded
    assert elapsed < 25


def test_success_is_unchanged():
    from inference.pipeline.predictor import wait_for_background_loads
    from inference.scripts.run_egohand_inference_pipeline import run_main

    calls = []
    run_main(lambda: calls.append(1))
    assert calls == [1]
    assert wait_for_background_loads(0.0)
