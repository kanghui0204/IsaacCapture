# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-OpenMDW-1.1

"""PresortedMHR: same outputs as the MHR TorchScript forward, no merge sort on CUDA.

Also: when an unpatched MHR model is an error (apply_mhr_presort), and that
EgoHandPredictor.from_checkpoint patches both hand models.

The real-model tests need py-soma-x with its MHR asset (skipped otherwise);
the CUDA kernel count needs a GPU.
"""

import pytest
import torch
from torch import nn

from inference.pipeline import mhr_presort
from inference.pipeline import predictor as predictor_module
from inference.pipeline.mhr_presort import (
    PresortedMHR,
    apply_mhr_presort,
    check_mhr_torchscript,
    presort_mhr_sparse,
    presort_required,
)


class _Wrapper(nn.Module):
    """Same shape as py-soma-x's NonPersistentModuleWrapper."""

    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)


class MHRDemo(nn.Module):  # same original_name as the real model, other code
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + 1


def test_unknown_code_is_left_alone():
    holder = nn.Module()
    holder.mhr = _Wrapper(torch.jit.script(MHRDemo()))
    holder.other = _Wrapper(torch.jit.script(nn.ReLU()))
    before_mhr, before_other = holder.mhr.module, holder.other.module
    info = presort_mhr_sparse(holder)
    assert info["patched"] == []
    assert list(info["skipped"]) == ["mhr"]
    assert holder.mhr.module is before_mhr
    assert holder.other.module is before_other


def _load_mhr(device):
    soma = pytest.importorskip("soma")
    try:
        path = soma.get_assets_dir() / "MHR" / "mhr_model_lod1.pt"
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"no soma assets: {exc}")
    if not path.exists():
        pytest.skip(f"no MHR asset at {path}")
    return torch.jit.load(str(path), map_location=device)


def _inputs(batch, device, pose_scale):
    """MHRHandIdentityModel passes a zero pose (pose_scale=0); a pose makes the
    sparse product non-zero, where cuSPARSE itself is not bitwise repeatable."""
    g = torch.Generator().manual_seed(5)
    identity = torch.randn(batch, 45, generator=g) * 0.5
    params = torch.zeros(batch, 204)
    params[:, 136:] = torch.randn(batch, 68, generator=g) * 0.1  # scales
    params[:, 130:136] = torch.randn(batch, 6, generator=g) * 0.1
    params[:, :130] = torch.randn(batch, 130, generator=g) * pose_scale
    face = torch.zeros(batch, 72)
    return identity.to(device), params.to(device), face.to(device)


def _max_diff(a, b):
    return float((a - b).abs().max())


def _check_against_torchscript(device):
    ts = _load_mhr(device)
    assert check_mhr_torchscript(ts) is None
    holder = nn.Module()
    holder.mhr = _Wrapper(ts)
    info = presort_mhr_sparse(holder)
    assert info == {"patched": ["mhr"], "skipped": {}}
    assert isinstance(holder.mhr.module, PresortedMHR)
    assert presort_mhr_sparse(holder)["patched"] == []  # idempotent
    pc = ts.pose_correctives_model
    sparse_ts = getattr(pc.pose_dirs_predictor, "0")
    for pose_scale in (0.0, 0.2):
        identity, params, face = _inputs(8, device, pose_scale)
        with torch.no_grad():
            jp = ts.character_torch.model_parameters_to_joint_parameters(
                torch.cat([params, torch.zeros_like(identity)], 1)
            )
            feats = pc._pose_features_from_joint_params(jp)
            a, a2 = sparse_ts(feats), sparse_ts(feats)
            b = torch.matmul(holder.mhr.module._sparse_matrix(), feats.T).T
            verts_ts, skel_ts = ts(identity, params, face)
            verts_new, skel_new = holder.mhr(identity, params, face)
            verts_ts2, _ = ts(identity, params, face)
        assert torch.equal(skel_ts, skel_new)
        if pose_scale == 0.0 or device == "cpu":
            assert torch.equal(a, b)
        else:  # cuSPARSE SpMM differs run to run by a few ulp
            assert _max_diff(a, b) <= max(_max_diff(a, a2), 1e-6)
        # skin_points itself is not bitwise repeatable on CUDA (3e-5 cm run to run)
        noise = _max_diff(verts_ts, verts_ts2)
        assert _max_diff(verts_ts, verts_new) <= max(noise, 1e-4)
    return ts, holder


def test_same_outputs_cpu():
    _check_against_torchscript("cpu")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_no_merge_sort_kernels_cuda():
    ts, holder = _check_against_torchscript("cuda")
    identity, params, face = _inputs(512, "cuda", 0.0)

    def merge_sort_kernels(fn):
        with torch.no_grad():
            fn(identity, params, face)  # warm up
            torch.cuda.synchronize()
            with torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CUDA]
            ) as prof:
                fn(identity, params, face)
                torch.cuda.synchronize()
        return sum(e.count for e in prof.key_averages() if "DeviceMergeSort" in e.key)

    assert merge_sort_kernels(ts) > 0  # the TorchScript forward sorts every call
    assert merge_sort_kernels(holder.mhr) == 0


def test_cache_follows_parameters_cpu():
    ts = _load_mhr("cpu")
    holder = nn.Module()
    holder.mhr = _Wrapper(ts)
    assert presort_mhr_sparse(holder)["patched"] == ["mhr"]
    first = holder.mhr.module._sparse_matrix()
    assert holder.mhr.module._sparse_matrix() is first  # reused while unchanged
    holder.double()  # parameters change dtype (new storage)
    second = holder.mhr.module._sparse_matrix()
    assert second is not first and second.dtype == torch.float64
    weight = getattr(ts.pose_correctives_model.pose_dirs_predictor, "0").sparse_weight
    assert torch.equal(second.values(), weight.detach())


def _unknown_mhr_holder():
    holder = nn.Module()
    holder.mhr = _Wrapper(torch.jit.script(MHRDemo()))
    return holder


def test_unpatched_mhr_raises_where_the_sort_can_fault(monkeypatch):
    monkeypatch.setattr(mhr_presort, "presort_required", lambda device: "exposed")
    with pytest.raises(RuntimeError, match="no-mhr_presorted_sparse"):
        apply_mhr_presort(_unknown_mhr_holder(), "cuda")


def test_unpatched_mhr_only_warns_elsewhere(monkeypatch, caplog):
    monkeypatch.setattr(mhr_presort, "presort_required", lambda device: None)
    holder = _unknown_mhr_holder()
    before = holder.mhr.module
    info = apply_mhr_presort(holder, "cpu")
    assert list(info["skipped"]) == ["mhr"]
    assert holder.mhr.module is before
    assert "could not be coalesced" in caplog.text


def test_model_without_mhr_is_fine(monkeypatch):
    monkeypatch.setattr(mhr_presort, "presort_required", lambda device: "exposed")
    holder = nn.Module()
    holder.other = _Wrapper(torch.jit.script(nn.ReLU()))
    assert apply_mhr_presort(holder, "cuda") == {"patched": [], "skipped": {}}


@pytest.mark.parametrize(
    "cuda,device,capability,exposed",
    [
        ("13.0", "cuda", (9, 0), True),
        ("13.0", "cuda:0", (12, 0), True),
        ("13.0", "cuda", (8, 9), False),
        ("12.8", "cuda", (12, 0), False),
        (None, "cuda", (12, 0), False),
        ("13.0", "cpu", (12, 0), False),
    ],
)
def test_presort_required(monkeypatch, cuda, device, capability, exposed):
    monkeypatch.setattr(torch.version, "cuda", cuda)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda d=None: capability)
    assert (presort_required(device) is not None) is exposed


def test_from_checkpoint_presorts_both_hand_models(monkeypatch, caplog):
    model = nn.Module()
    model.hand_left = _Wrapper(_load_mhr("cpu"))
    model.hand_right = _Wrapper(_load_mhr("cpu"))
    monkeypatch.setattr(
        predictor_module,
        "load_egohand_model",
        lambda *args, **kwargs: (model, "egohand.EgoHand"),
    )

    def init(self, model, batch_size, device="cuda", **kwargs):
        self.model = model
        self.mhr_presort = None

    monkeypatch.setattr(predictor_module.EgoHandPredictor, "__init__", init)

    off = predictor_module.EgoHandPredictor.from_checkpoint(
        "x.ckpt", "assets", batch_size=1, device="cpu", mhr_presorted_sparse=False
    )
    assert off.mhr_presort is None
    assert not isinstance(model.hand_left.module, PresortedMHR)

    caplog.clear()
    predictor = predictor_module.EgoHandPredictor.from_checkpoint(
        "x.ckpt", "assets", batch_size=1, device="cpu"
    )
    # Logged by from_checkpoint, so callers need not read predictor.mhr_presort.
    assert "MHR sparse layer coalesced at load: 2 model(s)" in caplog.text
    assert predictor.mhr_presort == {
        "patched": ["hand_left", "hand_right"],
        "skipped": {},
    }
    left, right = model.hand_left.module, model.hand_right.module
    assert isinstance(left, PresortedMHR) and isinstance(right, PresortedMHR)
    # A second call (e.g. a caller that also patches) leaves them as they are.
    assert apply_mhr_presort(model, "cpu") == {"patched": [], "skipped": {}}
    assert model.hand_left.module is left and model.hand_right.module is right
