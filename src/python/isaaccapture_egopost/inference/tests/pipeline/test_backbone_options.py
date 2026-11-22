# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-OpenMDW-1.1

"""Backbone options of the three-stage driver (--backbone_compile / _bf16_weights,
warm-up, per-code-version Inductor cache, deterministic Inductor). A
two-block DINOv3 with random weights; compiled paths use the ``eager``
torch.compile backend (graph capture and dispatch), except one CUDA test that
checks the settings in the kernels Inductor generates."""

from __future__ import annotations

import argparse
import os
import sys
from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("dinov3")

from egohand.models.backbones.dinov3 import DinoV3Backbone  # noqa: E402
from inference.pipeline import three_stage as ts  # noqa: E402
from inference.pipeline.predictor import EgoHandPredictor  # noqa: E402

_TINY = dict(
    img_size=32,
    patch_size=16,
    in_chans=3,
    pos_embed_rope_base=100,
    pos_embed_rope_normalize_coords="separate",
    pos_embed_rope_rescale_coords=2,
    pos_embed_rope_dtype="fp32",
    embed_dim=64,
    depth=2,
    num_heads=4,
    ffn_ratio=2.0,
    qkv_bias=True,
    layerscale_init=1.0e-05,
    norm_layer="layernormbf16",
    ffn_layer="swiglu",
    ffn_bias=True,
    proj_bias=True,
    n_storage_tokens=4,
    mask_k_bias=True,
)


def _build_tiny(monkeypatch, **arch):
    monkeypatch.setitem(DinoV3Backbone._ARCH_KWARGS, "tiny", {**_TINY, **arch})
    torch.manual_seed(0)
    backbone = DinoV3Backbone(name="tiny", drop_path_rate=0.0).eval()
    # Non-trivial layerscale so every block changes the tokens visibly.
    with torch.no_grad():
        for module in backbone.encoder.modules():
            if hasattr(module, "gamma"):
                module.gamma.fill_(0.5)
    return backbone


@pytest.fixture
def tiny(monkeypatch):
    return _build_tiny(monkeypatch)


def _x(batch, h=32, w=48, seed=1):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(batch, 3, h, w, generator=g)


def _fresh_dynamo():
    torch._dynamo.reset()


# ----------------------------------------------------------------- compile scopes


@pytest.mark.parametrize("scope", ["block", "whole"])
def test_compiled_full_batch_equals_eager_and_tail_stays_eager(tiny, scope):
    _fresh_dynamo()
    x = _x(4)
    with torch.no_grad():
        ref_full = tiny(x)
        ref_tail = tiny(x[:3])
    tiny.enable_full_batch_compile(4, scope=scope, backend="eager")
    with torch.no_grad():
        got_full = tiny(x)
        got_tail = tiny(x[:3])
    assert torch.equal(got_full, ref_full)
    assert torch.equal(got_tail, ref_tail)
    if scope == "whole":
        assert tiny._compiled_whole is not None and tiny._compiled_blocks is None
    else:
        assert tiny._compiled_blocks is not None and tiny._compiled_whole is None
    # Grad-enabled calls never take the compiled path.
    assert torch.equal(tiny(x).detach(), ref_full)


def test_compile_kwargs_options_replace_mode(tiny, monkeypatch):
    calls = []

    def fake_compile(fn, **kwargs):
        calls.append(kwargs)
        return fn

    monkeypatch.setattr(torch, "compile", fake_compile)
    tiny.enable_full_batch_compile(2)
    with torch.no_grad():
        tiny(_x(2))
    assert calls and all(
        c == dict(backend="inductor", fullgraph=True, dynamic=False, mode="default")
        for c in calls
    )
    assert len(calls) == len(tiny.encoder.blocks)

    calls.clear()
    tiny._compiled_blocks = None
    tiny.enable_full_batch_compile(
        2, scope="whole", options={"dynamic_scale_rblock": False}
    )
    with torch.no_grad():
        tiny(_x(2))
    assert calls == [
        dict(
            backend="inductor",
            fullgraph=True,
            dynamic=False,
            options={"dynamic_scale_rblock": False},
        )
    ]


def test_compile_settings_are_fixed_once_built(tiny):
    _fresh_dynamo()
    tiny.enable_full_batch_compile(2, scope="whole", backend="eager")
    with torch.no_grad():
        tiny(_x(2))
    tiny.enable_full_batch_compile(2, scope="whole", backend="eager")  # same: ok
    with pytest.raises(RuntimeError, match="cannot change"):
        tiny.enable_full_batch_compile(2, scope="block", backend="eager")
    with pytest.raises(ValueError, match="scope"):
        DinoV3Backbone.enable_full_batch_compile(tiny, 2, scope="graph")


# ----------------------------------------------------------------- bf16 weights


def test_bf16_weights_give_the_autocast_result_bitwise(tiny):
    x = _x(3)
    with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
        ref = tiny(x)
    converted = tiny.cast_matmul_weights_bf16()
    with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
        got = tiny(x)
    assert torch.equal(got, ref)
    assert ref.dtype == got.dtype
    # 2 blocks x (qkv, proj, w1, w2, w3) + the patch embedding conv.
    assert converted == 11
    enc = tiny.encoder
    assert enc.blocks[0].attn.qkv.weight.dtype == torch.bfloat16
    assert enc.blocks[0].attn.qkv.bias.dtype == torch.bfloat16
    assert enc.patch_embed.proj.weight.dtype == torch.bfloat16
    assert enc.blocks[0].norm1.weight.dtype == torch.float32
    assert enc.norm.weight.dtype == torch.float32
    assert enc.rope_embed.periods.dtype == torch.float32


# ----------------------------------------------------------------- settings / configure


def test_compile_scope_from_settings():
    assert ts.backbone_compile_scope({}) == "off"
    assert ts.backbone_compile_scope({"dino_compile": True}) == "block"
    assert ts.backbone_compile_scope({"backbone_compile": "whole"}) == "whole"
    assert (
        ts.backbone_compile_scope({"dino_compile": True, "backbone_compile": "off"})
        == "off"
    )
    with pytest.raises(ValueError):
        ts.backbone_compile_scope({"backbone_compile": "graph"})
    stripped = ts.without_backbone_options(
        {
            "dino_compile": True,
            "backbone_compile": "whole",
            "backbone_bf16_weights": True,
            "batch_size": 32,
        }
    )
    assert not ts.backbone_options_active(stripped)
    assert stripped["batch_size"] == 32


class _RecordingBackbone:
    def __init__(self):
        self.calls = []

    def enable_full_batch_compile(self, batch, **kwargs):
        self.calls.append(("compile", batch, kwargs))

    def cast_matmul_weights_bf16(self):
        self.calls.append(("bf16",))
        return 11


def _fake_predictor(backbone, batch_size=32, dtype=torch.bfloat16):
    predictor = object.__new__(EgoHandPredictor)
    predictor.model = SimpleNamespace(backbone=backbone)
    predictor.batch_size = batch_size
    predictor.dtype = dtype
    return predictor


def test_configure_backbone_all_off_touches_nothing():
    backbone = _RecordingBackbone()
    info = ts.configure_backbone(_fake_predictor(backbone), {})
    assert backbone.calls == []
    assert info == {"compile": "off", "bf16_weights": False}


_DET = {"dynamic_scale_rblock": False, "deterministic": True}


@pytest.mark.parametrize(
    "settings, calls",
    [
        (
            {"dino_compile": True},
            [("compile", 32, {"scope": "block", "options": _DET})],
        ),
        (
            {"backbone_compile": "whole", "backbone_bf16_weights": True},
            [
                ("bf16",),
                ("compile", 32, {"scope": "whole", "options": _DET}),
            ],
        ),
        ({"backbone_bf16_weights": True}, [("bf16",)]),
    ],
)
def test_configure_backbone_order(settings, calls):
    backbone = _RecordingBackbone()
    info = ts.configure_backbone(_fake_predictor(backbone), settings)
    assert backbone.calls == calls
    assert info["compile"] == ts.backbone_compile_scope(settings)
    if info["compile"] != "off":
        assert info["compile_options"] == _DET


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_bf16_weights_need_the_bfloat16_dtype(dtype):
    backbone = _RecordingBackbone()
    with pytest.raises(ValueError, match="bfloat16"):
        ts.configure_backbone(
            _fake_predictor(backbone, dtype=dtype), {"backbone_bf16_weights": True}
        )
    assert backbone.calls == []


def test_driver_rejects_bf16_weights_without_bfloat16(monkeypatch, capsys, driver):
    opt, _ = driver.parse_driver_args(
        ["--queue", "q.json", "--backbone_bf16_weights", "--"]
    )
    for dtype in ("float32", "float16"):
        args = _cli_args(monkeypatch, driver, "--dtype", dtype)
        with pytest.raises(SystemExit) as exc:
            driver.check_bf16_weights_dtype(opt, args)
        assert exc.value.code == driver.cli.EXIT_INVALID_INPUT
        assert "--dtype bfloat16" in capsys.readouterr().err
    driver.check_bf16_weights_dtype(
        opt, _cli_args(monkeypatch, driver, "--dtype", "bfloat16")
    )


# ----------------------------------------------------------------- warm-up


def test_warm_backbone_skips_uncompiled_backbones(tiny):
    predictor = SimpleNamespace(
        model=SimpleNamespace(backbone=tiny, _backbone_crop_w=(32, -32)),
        device="cpu",
        dtype=torch.float32,
    )
    assert ts.warm_backbone(predictor) is None
    assert ts.warm_backbone(SimpleNamespace(model=SimpleNamespace())) is None


def test_warm_backbone_compiles_once_for_the_head_call(tiny):
    _fresh_dynamo()
    tiny.enable_full_batch_compile(2, scope="whole", backend="eager")
    predictor = SimpleNamespace(
        model=SimpleNamespace(backbone=tiny, _backbone_crop_w=(32, -32)),
        device="cpu",
        dtype=torch.float32,
    )
    rng_before = torch.random.get_rng_state()
    warm = ts.warm_backbone(predictor)
    assert torch.equal(torch.random.get_rng_state(), rng_before)
    assert warm["batch"] == 2
    assert warm["input_shape"] == [2, 3, 256, 192]
    assert len(warm["later_calls_ms"]) == 2
    graphs = warm["dynamo_after"]["unique_graphs"]
    assert graphs >= 1
    # A head-like call: a fresh [B, 3, 256, 256] batch cropped the same way.
    img = torch.randn(2, 3, 256, 256)
    with torch.no_grad():
        tiny(img[:, :, :, 32:-32])
    assert ts.dynamo_counters()["unique_graphs"] == graphs


# ----------------------------------------------------------------- driver


@pytest.fixture
def driver(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["run_egohand_three_stage.py"])
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
            *extra,
        ],
    )
    return drv.cli.parse_args()


def test_driver_flags_and_dino_compile_alias(monkeypatch, driver):
    opt, rest = driver.parse_driver_args(["--queue", "q.json", "--", "--dino_compile"])
    assert opt.backbone_compile is None
    assert opt.backbone_bf16_weights is False
    assert opt.backbone_warmup is True
    args = _cli_args(monkeypatch, driver, *rest)
    assert driver.resolve_backbone_compile(opt, args) == "block"

    opt, rest = driver.parse_driver_args(
        [
            "--queue",
            "q.json",
            "--backbone_compile",
            "whole",
            "--backbone_bf16_weights",
            "--no-backbone_warmup",
            "--",
        ]
    )
    args = _cli_args(monkeypatch, driver, *rest)
    assert driver.resolve_backbone_compile(opt, args) == "whole"
    assert (opt.backbone_bf16_weights, opt.backbone_warmup) == (True, False)
    args = _cli_args(monkeypatch, driver, "--dino_compile")
    with pytest.raises(SystemExit):
        driver.resolve_backbone_compile(opt, args)

    opt, rest = driver.parse_driver_args(["--queue", "q.json", "--"])
    args = _cli_args(monkeypatch, driver, *rest)
    assert driver.resolve_backbone_compile(opt, args) == "off"


def test_release_profile_rejects_a_compiled_backbone(monkeypatch, driver):
    monkeypatch.setenv("EGOHAND_RUNTIME_PROFILE", "release")
    driver.check_backbone_compile_supported("off")
    for scope in ("block", "whole"):
        with pytest.raises(SystemExit):
            driver.check_backbone_compile_supported(scope)
    monkeypatch.setenv("EGOHAND_RUNTIME_PROFILE", "full")
    driver.check_backbone_compile_supported("whole")


def test_code_version_follows_the_sources(monkeypatch, tmp_path, driver):
    (tmp_path / "egohand").mkdir()
    (tmp_path / "inference" / "sub").mkdir(parents=True)
    (tmp_path / "egohand" / "a.py").write_text("A = 1\n")
    (tmp_path / "inference" / "sub" / "b.py").write_text("B = 2\n")
    (tmp_path / "inference" / "notes.txt").write_text("not code\n")
    monkeypatch.setattr(driver, "PROJECT_ROOT", tmp_path)
    first = driver.code_version()
    assert first["files"] == 2 and first["torch"] == torch.__version__
    assert driver.code_version()["hash"] == first["hash"]
    (tmp_path / "inference" / "notes.txt").write_text("changed, still not code\n")
    assert driver.code_version()["hash"] == first["hash"]
    (tmp_path / "egohand" / "a.py").write_text("A = 3\n")
    assert driver.code_version()["hash"] != first["hash"]


def test_setup_compile_env(monkeypatch, tmp_path, driver):
    for key in (*driver.COMPILE_ENV, "TORCHINDUCTOR_CACHE_DIR"):
        monkeypatch.setenv(key, "placeholder")  # records the original for teardown
        monkeypatch.delenv(key)
    import torch._inductor.config as config

    # config.patch restores the attributes; monkeypatch.setattr would delattr
    # them on undo (not in the module __dict__), hiding them from Inductor.
    attrs = {attr: getattr(config, attr) for _, attr, _ in driver.COMPILE_ENV.values()}
    opt = argparse.Namespace(inductor_cache_root=str(tmp_path / "ind"))
    with config.patch(attrs):
        env = driver.setup_compile_env(opt, "whole")
        applied = {attr: getattr(config, attr) for attr in attrs}
    assert os.environ["TORCHINDUCTOR_DYNAMIC_SCALE_RBLOCK"] == "0"
    assert os.environ["TORCHINDUCTOR_DETERMINISTIC"] == "1"
    assert env["TORCHINDUCTOR_DETERMINISTIC"] == "1"
    assert env["TORCHINDUCTOR_DETERMINISTIC_before"] is None
    assert env["inductor_config_imported_before"] is True
    cache_dir = os.environ["TORCHINDUCTOR_CACHE_DIR"]
    assert cache_dir == env["inductor_cache_dir"]
    assert os.path.dirname(cache_dir) == str(tmp_path / "ind")
    assert os.path.basename(cache_dir) == "egopost_" + env["code_version"]["hash"][:16]
    assert os.path.exists(os.path.join(cache_dir, "egopost_code_version.json"))
    # Applied to the already imported config too.
    assert applied == {"dynamic_scale_rblock": False, "deterministic": True}
    assert {attr: getattr(config, attr) for attr in attrs} == attrs


def test_compile_env_maps_to_inductor_config(driver):
    """Each COMPILE_ENV variable sets its config attribute in a fresh process,
    and the compile options are Inductor config keys of this torch."""
    import subprocess

    import torch._inductor.config as config

    known = config.get_config_copy()
    assert set(ts.DETERMINISTIC_INDUCTOR_OPTIONS) <= set(known)
    for var, (value, attr, attr_value) in driver.COMPILE_ENV.items():
        assert ts.DETERMINISTIC_INDUCTOR_OPTIONS[attr] == attr_value
        out = subprocess.run(
            [
                sys.executable,
                "-c",
                f"import torch._inductor.config as c; print(c.{attr})",
            ],
            env={**os.environ, var: value},
            capture_output=True,
            text=True,
            check=True,
        )
        assert out.stdout.strip() == str(attr_value)


def test_deterministic_mode_takes_one_reduction_config():
    """With deterministic on, Inductor keeps one config per reduction kernel,
    so nothing is left to pick by timing on the first call."""
    triton = pytest.importorskip("triton")
    from torch._inductor.runtime.triton_heuristics import (
        filter_reduction_configs_for_determinism,
    )

    configs = [
        triton.Config({"XBLOCK": 16, "R0_BLOCK": 32}, num_warps=4),
        triton.Config({"XBLOCK": 64, "R0_BLOCK": 64}, num_warps=16),
        triton.Config({"XBLOCK": 8, "R0_BLOCK": 128}, num_warps=8),
    ]
    assert len(filter_reduction_configs_for_determinism({}, list(configs))) == 3
    picked = filter_reduction_configs_for_determinism(
        {"deterministic": True}, list(configs)
    )
    again = filter_reduction_configs_for_determinism(
        {"deterministic": True}, configs[::-1]
    )
    assert len(picked) == 1
    assert (picked[0].kwargs, picked[0].num_warps) == (
        again[0].kwargs,
        again[0].num_warps,
    )


def _row_sums(y):
    """An unrelated function with a reduction kernel (compiled first)."""
    return (y * y).sum(dim=-1)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("scope", ["whole", "block"])
def test_compiled_backbone_kernels_are_deterministic(monkeypatch, scope):
    """The compile options reach every generated backbone kernel when the
    process-wide flag is False, which is what torch 2.13 Dynamo leaves after any
    earlier compile in the process (convert_frame restores it to
    are_deterministic_algorithms_enabled()). Another function is compiled
    first, without the options; its kernels show the test can see a
    non-deterministic kernel. Block scope: the blocks share one graph."""
    from torch._inductor.utils import run_and_get_code

    _fresh_dynamo()
    backbone = _build_tiny(monkeypatch).cuda()
    backbone.enable_full_batch_compile(
        2, scope=scope, options=dict(ts.DETERMINISTIC_INDUCTOR_OPTIONS)
    )
    x = _x(2).cuda()
    y = torch.randn(64, 256, device="cuda")
    with (
        torch.no_grad(),
        torch._inductor.config.patch(deterministic=False, force_disable_caches=True),
    ):
        _, other = run_and_get_code(torch.compile(_row_sums), y)
        before = ts.dynamo_counters()["unique_graphs"]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            _, code = run_and_get_code(backbone, x)
        graphs = ts.dynamo_counters()["unique_graphs"] - before
    other, code = "\n".join(other), "\n".join(code)
    assert "'deterministic': False" in other
    assert graphs == 1
    assert (
        "triton_heuristics.reduction" in code
        or "triton_heuristics.persistent_reduction" in code
    )
    assert "'deterministic': True" in code
    assert "'deterministic': False" not in code
    assert "'dynamic_scale_rblock': False" in code
