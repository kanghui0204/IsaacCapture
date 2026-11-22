# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-OpenMDW-1.1

"""Export an EgoHand training checkpoint to inference weights.

Writes ``<output_dir>/model.safetensors`` (the checkpoint ``state_dict``,
values unchanged) and ``<output_dir>/config.json`` (training cfg, checkpoint
validation fields, provenance). Pass the folder to
``run_egohand_inference_pipeline.py --egohand_ckpt``.

With ``--verify``, the export is loaded back and compared tensor-by-tensor
against the checkpoint (bitwise), and a model is built from it with the
strict key check used at inference time.

Usage:
    python inference/scripts/export_inference_weights.py \\
        --checkpoint egohand_s060000.ckpt --output_dir weights/egohand_s060000 --verify
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from inference.pipeline.inference_weights import (  # noqa: E402
    export_inference_weights,
    read_checkpoint,
)


def _verify(checkpoint: Path, output_dir: Path, soma_hand_assets: str | None) -> None:
    import torch

    reference = read_checkpoint(checkpoint)
    exported = read_checkpoint(output_dir)
    ref_tensors = reference.load_state_dict()
    out_tensors = exported.load_state_dict()
    if ref_tensors.keys() != out_tensors.keys():
        raise SystemExit("export keys differ from the checkpoint state_dict")
    for key, value in ref_tensors.items():
        other = out_tensors[key]
        if value.dtype != other.dtype or value.shape != other.shape:
            raise SystemExit(f"{key}: dtype/shape changed")

        # Byte-level comparison (NaN payloads included); reshape(-1) also
        # handles 0-d and non-contiguous tensors.
        def as_bytes(tensor):
            return tensor.contiguous().reshape(-1).view(torch.uint8)

        if not torch.equal(as_bytes(value), as_bytes(other)):
            raise SystemExit(f"{key}: values changed")
    if exported.cfg != reference.cfg:
        raise SystemExit("exported cfg differs from the checkpoint cfg")
    print(f"[export] verified {len(out_tensors)} tensors bitwise and cfg equality")
    if soma_hand_assets is not None:
        from inference.pipeline.predictor import load_egohand_model

        load_egohand_model(exported, soma_hand_assets)
        print("[export] model built from the export with the strict key check")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--checkpoint", required=True, help="Training .ckpt")
    parser.add_argument("--output_dir", required=True, help="Destination folder")
    parser.add_argument(
        "--no_sha256",
        action="store_true",
        help="Skip hashing the source checkpoint for config.json provenance.",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Reload the export and compare it bitwise with the checkpoint.",
    )
    parser.add_argument(
        "--soma_hand_assets",
        default=None,
        help="With --verify, also build the model from the export (strict keys).",
    )
    args = parser.parse_args()
    config = export_inference_weights(
        args.checkpoint, args.output_dir, source_sha256=not args.no_sha256
    )
    print(
        f"[export] {config['tensors']['count']} tensors "
        f"({config['tensors']['numel']:,} values, {config['tensors']['dtypes']}) "
        f"-> {args.output_dir}"
    )
    if args.verify:
        _verify(Path(args.checkpoint), Path(args.output_dir), args.soma_hand_assets)


if __name__ == "__main__":
    main()
