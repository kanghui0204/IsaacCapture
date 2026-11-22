# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-OpenMDW-1.1

"""Command line of run_egohand_three_stage.py: rejected options, exit status,
summary file names."""

from __future__ import annotations

import os
import sys

import pytest

pytest.importorskip("cv2")


@pytest.fixture(scope="module")
def driver():
    from inference.scripts import run_egohand_three_stage as drv

    return drv


def _cli_args(monkeypatch, drv, *extra):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_egohand_inference_pipeline.py",
            "--input",
            "in.mp4",
            "--egohand_ckpt",
            "m.ckpt",
            "--output_dir",
            "out",
            "--no_save_videos",
            *extra,
        ],
    )
    return drv.cli.parse_args()


def test_supported_arguments_pass(monkeypatch, driver):
    driver.check_supported(_cli_args(monkeypatch, driver))


@pytest.mark.parametrize(
    "extra",
    [["--dino_compile"], ["--save_npzs"], ["--frame_batch_size", "64"]],
)
def test_unsupported_arguments_are_invalid_input(monkeypatch, capsys, driver, extra):
    args = _cli_args(monkeypatch, driver, *extra)
    with pytest.raises(SystemExit) as exc:
        driver.check_supported(args)
    assert exc.value.code == driver.cli.EXIT_INVALID_INPUT
    assert extra[0] in capsys.readouterr().err


def test_exit_status(driver):
    ok, failed = {"status": "ok"}, {"status": "error"}
    assert driver.exit_status("done", []) == 0
    assert driver.exit_status("done", [ok, ok]) == 0
    assert driver.exit_status("done", [ok, failed]) == driver.EXIT_SOME_ITEMS_FAILED
    assert driver.exit_status("aborted", [ok]) == driver.cli.EXIT_RUNTIME_ERROR


def test_workers_sharing_a_queue_write_separate_summaries(driver):
    from inference.pipeline import three_stage as ts

    requests = [
        ts.ClipRequest(
            index=0, name="a", video="a.mp4", intrinsics=None, out_root="/r/out"
        )
    ]
    assert driver.default_summary_path(requests, None) == os.path.join(
        "/r", "three_stage_summary.json"
    )
    assert driver.default_summary_path(requests, "/r/claims") == os.path.join(
        "/r", f"three_stage_summary_{os.getpid()}.json"
    )
