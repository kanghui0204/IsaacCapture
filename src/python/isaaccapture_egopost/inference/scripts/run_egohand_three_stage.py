# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-OpenMDW-1.1

"""Run EgoHand inference over a queue of videos with the models loaded once,
as a three-stage pipeline (see ``inference/pipeline/three_stage.py``).

Usage:
    python inference/scripts/run_egohand_three_stage.py --queue queue.json \\
        [--mode thr|proc|serial] [--summary summary.json] [--claim_dir DIR] \\
        [--soma_videos 2] [--prepare_ahead 1] \\
        [--crop_workers 8] [--norm_workers 4] [--warp_mempool] \\
        [--switch_interval S] [--no-warm_detector] [--no-trim_models] \\
        [--backbone_compile off|block|whole] [--backbone_bf16_weights] \\
        [--no-backbone_warmup] \\
        [--inductor_cache_root DIR] \\
        -- <run_egohand_inference_pipeline.py arguments>

The arguments after ``--`` are parsed by ``run_egohand_inference_pipeline.py``
(same defaults). Its ``--input`` / ``--intrinsics_json`` / ``--output_dir`` are
placeholders: each queue item brings its own. Outputs of an item go to
``<out_root>/<video stem>/`` exactly as with ``--output_dir <out_root>``.

queue.json: ``{"items": [{"name", "video", "intrinsics", "out_root"}, ...]}``.

--mode thr     one process; stage 1 and 2 on worker threads, stage 3 on the
               main thread (default).
--mode proc    three processes (spawn); crops through /dev/shm.
--mode serial  the stages one after the other (no overlap).
--claim_dir    several workers share one queue: each item is processed by
               the first worker that claims it (exclusive file create).
               Without --summary each worker then writes
               three_stage_summary_<pid>.json.
--soma_videos  videos stage 2 may have handed to stage 3 before stage 3
               finishes them (bounds the GPU memory of queued head batches).
--prepare_ahead  1: stage 1 decodes the next video only after stage 2 took
               the previous one; 2: one more may be prepared meanwhile.
--crop_workers / --norm_workers  threads for crops (stage 1) and for
               normalizing crops (stage 2).
--warp_mempool  enable Warp's CUDA memory pool on the device while the
               pipeline runs (py-soma-x disables it, so every Warp temporary
               is freed with cudaFree, which waits for the whole device).
--no-warm_detector  skip the ORT warm-up over batch sizes 1..B at start.
--no-trim_models  proc mode: keep the modules a process does not use.

Not supported (rejected): overlay videos (pass --no_save_videos),
--save_images, --save_npzs, --per_frame_results, --postprocessing_qc,
--frame_batch_size, --max_seconds, --perf_output, a CPU --device; in
stage 1, full-image undistortion of non-pinhole cameras and pixel-aspect
rectification fail that video. RT-DETR preprocessing threads
come from the detector config (rtdetr_config preprocess_threads).

DINOv3 backbone options (all off by default: the eager backbone, as in
run_egohand_inference_pipeline.py; compiling needs the full runtime profile):
--backbone_compile  block = each of the 32 blocks compiled separately (what
               --dino_compile does; --dino_compile after -- means block);
               whole = one compiled graph from patch embedding to final norm.
               Only full batches (--batch_size crops) take the compiled path.
               Compiling sets TORCHINDUCTOR_DYNAMIC_SCALE_RBLOCK=0 and
               TORCHINDUCTOR_DETERMINISTIC=1 and passes the same two options to
               torch.compile: Inductor otherwise picks some reduction block sizes
               by timing on the first call, and processes cold-compiling at the
               same time on one GPU picked different ones (different outputs).
               Cost: none measurable. Outputs differ from builds without these
               settings (deterministic mode fixes one config per reduction
               kernel and also changes some fusion / padding choices; as far
               from eager as before). Applies to this driver (also to
               --dino_compile here); run_egohand_inference_pipeline.py
               --dino_compile still compiles with mode="default".
--backbone_bf16_weights  store the backbone Linear / Conv2d weights in bf16
               once at load. Needs --dtype bfloat16: the backbone then runs
               under bf16 autocast, which casts exactly these weights to bf16
               on every call, so the values are the same; saves the per-call
               casts and half the weight memory.
--backbone_warmup   (default on) compile before "ready": a few full batches of
               random crops run through the backbone in each process that
               runs it, so the queue timing excludes compilation; the summary
               reports the first-call time separately (backbone.warmup).
--inductor_cache_root  TORCHINDUCTOR_CACHE_DIR becomes
               <root>/egopost_<code version>, the code version being a hash of
               the mounted egohand/ + inference/ sources and the torch / triton
               versions; root defaults to $TORCHINDUCTOR_CACHE_DIR,
               else Inductor's default. Set only when compiling.

The summary JSON has one record per item (status, frames, crops, epoch start /
end, per-stage times) plus model load and memory figures; it is rewritten
after every item. Each item's perf.json lists what its latencies mean under
this driver (metadata three_stage_latencies).

Exit status: 0 when every item is ok; 3 when the queue ran to the end but
some items failed (Python errors fail only their item; see the summary);
1 when a CUDA error aborted the run; 2 for invalid input (as the
single-video command line).
"""

# ruff: noqa: E402
import argparse
import hashlib
import importlib.util
import json
import os
import sys
import tempfile
import time
from pathlib import Path

T_START = time.time()
T0 = time.perf_counter()

_HERE = Path(__file__).resolve()
PROJECT_ROOT = _HERE.parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Same module-level setup as `python run_egohand_inference_pipeline.py`
# (torch.load patch, sys.path); also runs in the spawned proc-mode children,
# which import this file as their main module.
_CLI_PATH = _HERE.parent / "run_egohand_inference_pipeline.py"
_spec = importlib.util.spec_from_file_location("egopost_cli", _CLI_PATH)
cli = importlib.util.module_from_spec(_spec)
sys.modules.setdefault("egopost_cli", cli)
_spec.loader.exec_module(cli)

import torch

from inference.media.runtime_profile import get_runtime_profile
from inference.pipeline import three_stage as ts

#: Exit status when the queue ran to the end but some items failed.
EXIT_SOME_ITEMS_FAILED = 3


def parse_driver_args(argv):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--queue", required=True)
    ap.add_argument("--summary", default=None)
    ap.add_argument("--mode", choices=("thr", "proc", "serial"), default="thr")
    ap.add_argument("--claim_dir", default=None)
    ap.add_argument("--soma_videos", type=int, default=2)
    ap.add_argument("--prepare_ahead", type=int, default=1)
    ap.add_argument("--crop_workers", type=int, default=8)
    ap.add_argument("--norm_workers", type=int, default=4)
    ap.add_argument("--warp_mempool", action="store_true")
    ap.add_argument("--switch_interval", type=float, default=None)
    ap.add_argument(
        "--warm_detector", action=argparse.BooleanOptionalAction, default=True
    )
    ap.add_argument(
        "--trim_models", action=argparse.BooleanOptionalAction, default=True
    )
    ap.add_argument("--shm_root", default="/dev/shm")
    ap.add_argument(
        "--backbone_compile", choices=ts.BACKBONE_COMPILE_SCOPES, default=None
    )
    ap.add_argument("--backbone_bf16_weights", action="store_true")
    ap.add_argument(
        "--backbone_warmup", action=argparse.BooleanOptionalAction, default=True
    )
    ap.add_argument("--inductor_cache_root", default=None)
    if "--" not in argv:
        ap.error("missing -- before the pipeline arguments")
    cut = argv.index("--")
    return ap.parse_args(argv[:cut]), argv[cut + 1 :]


def check_supported(args) -> None:
    """Reject the pipeline options the three-stage driver does not implement."""
    unsupported = []
    if not args.no_save_videos:
        unsupported.append("overlay videos (pass --no_save_videos)")
    if args.save_images:
        unsupported.append("--save_images")
    if args.save_npzs:
        unsupported.append("--save_npzs")
    if args.per_frame_results:
        unsupported.append("--per_frame_results")
    if args.postprocessing_qc:
        unsupported.append("--postprocessing_qc")
    if args.frame_batch_size is not None:
        unsupported.append("--frame_batch_size")
    if args.max_seconds is not None:
        unsupported.append("--max_seconds")
    if args.perf_output:
        unsupported.append("--perf_output")
    if not args.device.startswith("cuda"):
        unsupported.append(f"--device {args.device}")
    if unsupported:
        cli.reject_input(
            "run_egohand_three_stage.py does not support: " + ", ".join(unsupported)
        )


def default_summary_path(requests, claim_dir) -> str:
    """Next to the first item's out_root; per process when workers share a queue
    (``--claim_dir``), so they do not overwrite each other's summary."""
    name = (
        f"three_stage_summary_{os.getpid()}.json"
        if claim_dir
        else "three_stage_summary.json"
    )
    return os.path.join(
        os.path.dirname(requests[0].out_root) if requests else ".", name
    )


def exit_status(run_status: str, records: list) -> int:
    """0: every item ok; 3: the queue ran to the end, some items failed;
    1: the run was aborted (CUDA error)."""
    if run_status != "done":
        return cli.EXIT_RUNTIME_ERROR
    if any(record["status"] != "ok" for record in records):
        return EXIT_SOME_ITEMS_FAILED
    return 0


def resolve_backbone_compile(opt, args) -> str:
    """--backbone_compile, or block for the pipeline's --dino_compile."""
    if args.dino_compile and opt.backbone_compile not in (None, "block"):
        cli.reject_input(
            f"--dino_compile (= --backbone_compile block) conflicts with "
            f"--backbone_compile {opt.backbone_compile}"
        )
    if opt.backbone_compile is not None:
        return opt.backbone_compile
    return "block" if args.dino_compile else "off"


def check_bf16_weights_dtype(opt, args) -> None:
    """--backbone_bf16_weights gives the same values only under bf16 autocast."""
    if opt.backbone_bf16_weights and args.dtype != "bfloat16":
        cli.reject_input(
            "--backbone_bf16_weights needs --dtype bfloat16 (got --dtype "
            f"{args.dtype}): the weights only keep their values when the "
            "backbone runs under bf16 autocast"
        )


def check_backbone_compile_supported(scope: str) -> None:
    """Reject a compiled backbone where the runtime profile has no torch.compile
    (the release profile; ``--dino_compile`` is rejected there the same way)."""
    profile = get_runtime_profile()
    if scope != "off" and not profile.compilation:
        cli.reject_input(
            f"--backbone_compile {scope} is unsupported in the {profile.name} "
            "runtime profile"
        )


def code_version() -> dict:
    """Hash of the running egohand/ + inference/ sources and the compiler stack."""
    digest = hashlib.sha256()
    files = 0
    for top in ("egohand", "inference"):
        root = PROJECT_ROOT / top
        for path in sorted(root.rglob("*.py")):
            rel = path.relative_to(PROJECT_ROOT).as_posix()
            digest.update(rel.encode() + b"\0")
            digest.update(path.read_bytes() + b"\0")
            files += 1
    versions = {"torch": torch.__version__}
    try:
        import triton

        versions["triton"] = triton.__version__
    except ImportError:
        versions["triton"] = None
    digest.update(json.dumps(versions, sort_keys=True).encode())
    return {"hash": digest.hexdigest(), "files": files, **versions}


# Environment variable -> (value, torch._inductor.config attribute, attribute
# value) for a compiled backbone; why: three_stage.DETERMINISTIC_INDUCTOR_OPTIONS.
COMPILE_ENV = {
    "TORCHINDUCTOR_DYNAMIC_SCALE_RBLOCK": ("0", "dynamic_scale_rblock", False),
    "TORCHINDUCTOR_DETERMINISTIC": ("1", "deterministic", True),
}


def setup_compile_env(opt, scope: str) -> dict:
    """Process environment for a compiled backbone (inherited by proc-mode children).

    COMPILE_ENV (also applied to an already imported Inductor config) and a
    TORCHINDUCTOR_CACHE_DIR per code version. Dynamo resets the global
    ``deterministic`` after each traced frame; the backbone compiles get it from
    their options (three_stage.DETERMINISTIC_INDUCTOR_OPTIONS).
    """
    config = sys.modules.get("torch._inductor.config")
    config_was_imported = config is not None
    previous = {}
    for var, (value, attr, attr_value) in COMPILE_ENV.items():
        previous[var] = os.environ.get(var)
        os.environ[var] = value
        if config is not None:
            setattr(config, attr, attr_value)
    version = code_version()
    root = (
        opt.inductor_cache_root
        or os.environ.get("TORCHINDUCTOR_CACHE_DIR")
        or os.path.join(
            tempfile.gettempdir(), "torchinductor_" + os.environ.get("USER", "user")
        )
    )
    cache_dir = os.path.join(root, f"egopost_{version['hash'][:16]}")
    os.makedirs(cache_dir, exist_ok=True)
    stamp = os.path.join(cache_dir, "egopost_code_version.json")
    if not os.path.exists(stamp):
        tmp = f"{stamp}.{os.getpid()}.tmp"
        with open(tmp, "w") as fh:
            json.dump({**version, "created_epoch": time.time()}, fh, indent=1)
        os.replace(tmp, stamp)
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = cache_dir
    info = {
        "scope": scope,
        "code_version": version,
        "inductor_cache_root": root,
        "inductor_cache_dir": cache_dir,
        "triton_cache_dir": os.environ.get("TRITON_CACHE_DIR")
        or os.path.join(cache_dir, "triton"),
    }
    for var, (value, _, _) in COMPILE_ENV.items():
        info[var] = value
        info[f"{var}_before"] = previous[var]
    info["inductor_config_imported_before"] = config_was_imported
    return info


def proc_mem() -> dict:
    out = {}
    try:
        for line in open("/proc/self/status"):
            if line.startswith(("VmRSS:", "VmHWM:")):
                key, value = line.split(":", 1)
                out[key.lower() + "_kb"] = int(value.split()[0])
    except OSError:
        pass
    return out


def cuda_mem() -> dict:
    if not torch.cuda.is_available():
        return {}
    return {
        "allocated_mib": torch.cuda.memory_allocated() / 2**20,
        "reserved_mib": torch.cuda.memory_reserved() / 2**20,
        "max_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
        "max_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
    }


def main():
    opt, pipeline_cli = parse_driver_args(sys.argv[1:])
    sys.argv = [str(_CLI_PATH)] + pipeline_cli
    args = cli.parse_args()
    check_supported(args)
    compile_scope = resolve_backbone_compile(opt, args)
    check_backbone_compile_supported(compile_scope)
    check_bf16_weights_dtype(opt, args)
    backbone_env = (
        setup_compile_env(opt, compile_scope) if compile_scope != "off" else None
    )
    if backbone_env:
        print(
            f"[three_stage] backbone compile {compile_scope}: "
            + " ".join(f"{var}={backbone_env[var]}" for var in COMPILE_ENV)
            + f" TORCHINDUCTOR_CACHE_DIR={backbone_env['inductor_cache_dir']}",
            flush=True,
        )
    cli.check_gpu_visible(args.device)
    if opt.switch_interval:
        sys.setswitchinterval(opt.switch_interval)
    hands = ["left", "right"] if args.hand == "both" else [args.hand]
    if not args.bbox_detector_config:
        cli.reject_input("--bbox_detector_config is required.")
    try:
        detector_cfg = cli._load_detector_config(args.bbox_detector_config)
    except cli.DetectorConfigError as exc:
        cli.reject_input(str(exc))
    if args.egohand_ckpt is None:
        args.egohand_ckpt = cli.hf_download(args.egohand_ckpt_hf_path)
    if args.soma_hand_assets is None:
        from soma import get_assets_dir

        args.soma_hand_assets = str(get_assets_dir())

    queue_items = json.load(open(opt.queue))["items"]
    requests = [
        ts.ClipRequest(
            index=i,
            name=item["name"],
            video=item["video"],
            intrinsics=item.get("intrinsics"),
            out_root=item["out_root"],
        )
        for i, item in enumerate(queue_items)
    ]
    summary_path = opt.summary or default_summary_path(requests, opt.claim_dir)
    if opt.claim_dir:
        os.makedirs(opt.claim_dir, exist_ok=True)
    for root in sorted({r.out_root for r in requests}):
        cli.prepare_output_dir(Path(root))

    settings = {
        "egohand_ckpt": args.egohand_ckpt,
        "soma_hand_assets": args.soma_hand_assets,
        "batch_size": args.batch_size,
        "device": args.device,
        "dtype": args.dtype,
        "smooth": args.postprocessing_smooth,
        "use_perspective_cropping": args.perspective_cropping,
        "soma_batch_size": args.soma_batch_size,
        "mhr_presorted_sparse": args.mhr_presorted_sparse,
        "dino_compile": args.dino_compile,
        "backbone_compile": compile_scope,
        "backbone_bf16_weights": opt.backbone_bf16_weights,
        "backbone_warmup": opt.backbone_warmup,
        "detector_cfg": detector_cfg,
        "hands": hands,
        "crop_workers": opt.crop_workers,
        "norm_workers": opt.norm_workers,
        "prepare_ahead": opt.prepare_ahead,
        "soma_videos": opt.soma_videos,
        "warp_mempool": opt.warp_mempool,
        "trim_models": opt.trim_models,
        "warm_detector": opt.warm_detector,
        "switch_interval": opt.switch_interval,
        "rectify_pixel_aspect_ratio": args.rectify_pixel_aspect_ratio,
        "claim_dir": opt.claim_dir,
        "shm_dir": os.path.join(
            opt.shm_root, f"egopost_three_stage_{os.getpid()}_{int(T_START)}"
        ),
        "sys_path": [str(PROJECT_ROOT)],
        "perf_metadata": {
            "checkpoint": Path(args.egohand_ckpt).stem,
            "batch_size": args.batch_size,
            "soma_batch_size": args.soma_batch_size,
            "dtype": args.dtype,
            "dino_compile": args.dino_compile,
            "backbone_compile": compile_scope,
            "backbone_bf16_weights": opt.backbone_bf16_weights,
            "perspective_cropping": args.perspective_cropping,
            "three_stage_mode": opt.mode,
        },
    }
    records = []
    summary = {
        "driver": "run_egohand_three_stage.py",
        "mode": opt.mode,
        "options": vars(opt),
        "pipeline_cli": pipeline_cli,
        "settings": {k: v for k, v in settings.items() if k != "detector_cfg"},
        "detector_cfg": detector_cfg,
        "driver_start_epoch": T_START,
        "pid": os.getpid(),
        "backbone": {"env": backbone_env},
        "records": records,
    }

    def flush():
        tmp = summary_path + ".tmp"
        with open(tmp, "w") as fh:
            json.dump(summary, fh, indent=1, default=str)
        os.replace(tmp, summary_path)

    def on_record(record):
        record = dict(record)
        record["t_start_s"] = (
            (record["t_start_epoch"] - T_START) if record.get("t_start_epoch") else None
        )
        record["t_end_s"] = (
            (record["t_end_epoch"] - T_START) if record.get("t_end_epoch") else None
        )
        if opt.mode != "proc":
            record["cuda"] = cuda_mem()
            record["proc"] = proc_mem()
        records.append(record)
        print(
            f"[three_stage] {record['i']} {record['name']} {record['status']} "
            f"wall={record.get('wall_s') or 0:.2f}s frames={record.get('frames')} "
            f"crops={record.get('crops')}"
            + (f" error={record.get('error')}" if record["status"] != "ok" else ""),
            flush=True,
        )
        flush()

    if opt.mode == "proc":
        # The detector is built in the stage-1 process; check the backend
        # name here so a typo is still an invalid-input exit.
        backend = detector_cfg["detector_backend"]
        if (
            importlib.util.find_spec(f"inference.hand_detector_2d.configs.{backend}")
            is None
        ):
            cli.reject_input(
                f"Unknown detector_backend {backend!r} in the detector config; see "
                "inference/hand_detector_2d/configs/ for the available backends"
            )
        summary["ready_at_s"] = None
        print(
            "[three_stage] proc mode: starting P3 (SOMA) and P2 (head), then P1 (prepare)",
            flush=True,
        )

        def on_event(message):
            if message[0] == "ready":
                warm = (message[6] or {}).get("warmup") if len(message) > 6 else None
                print(
                    f"[three_stage] ready: {message[1]} pid={message[2]} load {message[3] / 1000:.2f}s"
                    + (
                        f" backbone warm-up first call {warm['first_call_s']:.2f}s"
                        if warm
                        else ""
                    ),
                    flush=True,
                )

        info = ts.run_process_pipeline(settings, requests, on_record, on_event)
        summary["processes"] = info
        head_ready = info.get("ready", {}).get("head", {}).get("backbone") or {}
        summary["backbone"].update(head_ready)
        head_done = info.get("done", {}).get("head", {}).get("detail") or []
        if len(head_done) > 1:
            summary["backbone"]["dynamo_end"] = head_done[1]
        ready = [
            v.get("epoch") for v in info.get("ready", {}).values() if v.get("epoch")
        ]
        if ready:
            summary["ready_epoch"] = max(ready)
            summary["ready_at_s"] = max(ready) - T_START
        summary["status"] = "aborted" if "abort" in info else "done"
    else:
        try:
            import warp as wp

            wp.init()  # once, on this thread, before any stage thread uses Warp
        except ImportError:
            pass
        t = time.perf_counter()
        try:
            bbox_detector, _ = cli._build_detector_from_config(
                detector_cfg, args.device, args.batch_size
            )
        except cli.DetectorConfigError as exc:
            cli.reject_input(str(exc))
        cli.check_frame_batching(bbox_detector, args.frame_batch_size)
        predictor, load_ms = ts.load_predictor(settings)
        camera_mode = getattr(predictor.model, "camera_mode", "weak_perspective")
        config = ts.stage_config_from_settings(settings, camera_mode)
        try:
            prepare = ts.PrepareStage(bbox_detector, config)
        except ts.UnsupportedOption as exc:
            cli.reject_input(f"run_egohand_three_stage.py does not support: {exc}")
        head = ts.HeadStage(predictor, config, own_stream=(opt.mode == "thr"))
        soma = ts.SomaStage(predictor, config, model_load_ms=load_ms)
        warm = ts.warm_detector(bbox_detector) if opt.warm_detector else []
        backbone_warm = ts.warm_backbone(predictor) if opt.backbone_warmup else None
        summary["backbone"].update(
            setup=getattr(predictor, "backbone_setup", None), warmup=backbone_warm
        )
        summary.update(
            mhr_presort=predictor.mhr_presort,
            model_load_ms=load_ms,
            load_total_s=time.perf_counter() - t,
            camera_mode=camera_mode,
            detector_warmup=warm,
            ready_epoch=time.time(),
            ready_at_s=time.perf_counter() - T0,
            cuda_ready=cuda_mem(),
        )
        print(
            f"[three_stage] mode={opt.mode} ready at {summary['ready_at_s']:.2f}s "
            f"(model load {load_ms / 1000:.2f}s, detector warm-up "
            f"{sum(w['s'] for w in warm):.2f}s"
            + (
                f", backbone {compile_scope} warm-up first call "
                f"{backbone_warm['first_call_s']:.2f}s then "
                f"{', '.join(f'{v:.1f}' for v in backbone_warm['later_calls_ms'])} ms"
                if backbone_warm
                else ""
            )
            + ")",
            flush=True,
        )
        flush()
        claimed = ts.claim_requests(requests, opt.claim_dir)
        if opt.mode == "serial":
            ts.run_serial(prepare, head, soma, claimed, on_record)
            summary["status"] = "done"
        else:
            runner = ts.ThreadedThreeStage(
                prepare,
                head,
                soma,
                prepare_ahead=opt.prepare_ahead,
                soma_videos=opt.soma_videos,
                warp_mempool=opt.warp_mempool,
            )
            runner.run(claimed, on_record)
            summary["status"] = "done" if runner.fatal is None else "aborted"
            if runner.fatal is not None:
                summary["fatal"] = f"{type(runner.fatal).__name__}: {runner.fatal}"
                summary["fatal_traceback"] = runner.fatal_traceback
        summary["cuda_end"] = cuda_mem()
        summary["proc_end"] = proc_mem()
        if backbone_warm:
            summary["backbone"]["dynamo_end"] = ts.dynamo_counters()
    summary["done_epoch"] = time.time()
    summary["t_done_s"] = time.perf_counter() - T0
    flush()
    ok = sum(r["status"] == "ok" for r in records)
    print(
        f"[three_stage] {summary['status']}: {ok}/{len(records)} ok, "
        f"{summary['t_done_s']:.1f}s since driver start; summary {summary_path}",
        flush=True,
    )
    status = exit_status(summary["status"], records)
    if status:
        sys.exit(status)


if __name__ == "__main__":
    main()
