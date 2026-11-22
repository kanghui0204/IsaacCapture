# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-OpenMDW-1.1

"""SOMA hand layer for several EgoHand head batches per call.

The SOMA layer costs about the same per call for 16 or 512 rows (launch and
host-sync overhead), so ``SomaBatcher`` calls it once per group of head
batches.
"""

import time

import torch

from inference.preprocessing.mirroring import mirror_left_prediction_to_right

_SOMA_INPUT_KEYS = ("poses", "identity_coeffs", "global_scale", "scale_params")


def _pad_rows(tensor: torch.Tensor, rows: int) -> torch.Tensor:
    """Pad to ``rows`` rows by repeating the last row.

    Repeat a real row: all-zero identity rows give a degenerate skeleton fit
    that takes py-soma-x's slow per-row fallback.
    """
    n = tensor.shape[0]
    if n == rows:
        return tensor
    return torch.cat([tensor, tensor[n - 1 :].expand(rows - n, *tensor.shape[1:])])


class SomaBatcher:
    """Left SOMA layer, camera solve and right-hand mirror for a stream of head batches.

    ``group_rows == 0`` makes one left call and one mirror call per head batch,
    the same calls as ``EgoHand.forward_step`` and the per-batch mirror.
    ``group_rows > 0`` calls the left layer once per ``group_rows // batch_size``
    head batches and the mirror once per at most ``group_rows`` right-hand rows,
    each call padded to ``group_rows`` rows so the MHR TorchScript model keeps
    one input shape. SOMA rows are independent, so only the float32 rounding of
    batched products changes. The camera solve and everything after it still
    run per head batch, and results come back in input order.
    """

    def __init__(self, predictor, group_rows: int = 0, sync_timing: bool = False):
        self.predictor = predictor
        self.model = predictor.model
        self.group_rows = int(group_rows)
        if self.group_rows and self.group_rows < predictor.batch_size:
            raise ValueError("soma_batch_size must be 0 or at least batch_size")
        self.left_batches = (
            max(1, self.group_rows // predictor.batch_size) if self.group_rows else 1
        )
        self.sync_timing = sync_timing
        self.predictions = []
        self.left_calls = 0
        self.mirror_calls = 0
        self.left_ms = 0.0
        self.right_ms = 0.0
        self._left = []
        self._right = []
        self._right_rows = 0

    def add(self, batch: dict, batch_hand_data: list, head: dict) -> None:
        """Queue one head batch (``batch`` without ``img``)."""
        self._left.append((batch, batch_hand_data, head))
        if len(self._left) >= self.left_batches:
            self._flush_left()

    def finish(self) -> list:
        if self._left:
            self._flush_left()
        if self._right:
            self._flush_right()
        return self.predictions

    def _flush_left(self) -> None:
        start = time.perf_counter()
        pending, self._left = self._left, []
        if self.group_rows:
            soma_inputs = {
                key: _pad_rows(
                    torch.cat([head["soma_inputs"][key] for _, _, head in pending]),
                    self.group_rows,
                )
                for key in _SOMA_INPUT_KEYS
            }
        else:
            soma_inputs = pending[0][2]["soma_inputs"]
        with torch.autocast("cuda", enabled=False):
            hand_out = self.model.soma_hand(
                hand_type="left", pose2rot=False, **soma_inputs
            )
        self.left_calls += 1
        decoded = []
        offset = 0
        for batch, batch_hand_data, head in pending:
            joints, vertices = hand_out["joints"], hand_out["vertices"]
            if self.group_rows:
                rows = head["batch_size"]
                # Row slices keep the per-batch call's strides and 16-byte
                # alignment (a joint row block is 1600 bytes).
                joints = joints[offset : offset + rows]
                vertices = vertices[offset : offset + rows]
                offset += rows
            output = self.model.forward_step_decode(head, joints, vertices)
            decoded.append((batch, batch_hand_data, output))
        if self.sync_timing:
            torch.cuda.synchronize(self.predictor.device)
        self.left_ms += (time.perf_counter() - start) * 1000
        for batch, batch_hand_data, output in decoded:
            self._queue_right(batch, batch_hand_data, output)

    def _queue_right(self, batch: dict, batch_hand_data: list, output: dict) -> None:
        params, global_scale, scale_params = self.predictor._split_batch_params(output)
        self.predictor._finalize_projection(output, batch)
        right_indices = [
            i for i, d in enumerate(batch_hand_data) if d["real_hand_side"] == "right"
        ]
        record = (
            batch_hand_data,
            output,
            params,
            global_scale,
            scale_params,
            right_indices,
        )
        if (
            self.group_rows
            and self._right
            and self._right_rows + len(right_indices) > self.group_rows
        ):
            self._flush_right()
        self._right.append(record)
        self._right_rows += len(right_indices)
        if not self.group_rows:
            self._flush_right()

    def _flush_right(self) -> None:
        start = time.perf_counter()
        records, self._right, self._right_rows = self._right, [], 0
        right_out = None
        # See the per-batch path in predictor.py: no inherited autocast for SOMA.
        with torch.autocast("cuda", enabled=False):
            inputs = [
                self.predictor._right_mirror_inputs(
                    batch_hand_data,
                    right_indices,
                    params,
                    output,
                    global_scale,
                    scale_params,
                )
                if right_indices
                else None
                for (
                    batch_hand_data,
                    output,
                    params,
                    global_scale,
                    scale_params,
                    right_indices,
                ) in records
            ]
            present = [args for args in inputs if args is not None]
            if present:
                if self.group_rows:
                    mirror_args = [
                        None
                        if present[0][j] is None
                        else _pad_rows(
                            torch.cat([args[j] for args in present]), self.group_rows
                        )
                        for j in range(len(present[0]))
                    ]
                else:
                    mirror_args = present[0]
                right_out = mirror_left_prediction_to_right(
                    self.model,
                    *mirror_args[:6],
                    scale_params=mirror_args[6],
                    pose_mirror=self.predictor._pose_mirror,
                )
                self.mirror_calls += 1
        offset = 0
        for record, args in zip(records, inputs):
            (
                batch_hand_data,
                output,
                params,
                global_scale,
                scale_params,
                right_indices,
            ) = record
            right_batched = None
            if args is not None:
                right_batched = right_out
                if self.group_rows:
                    rows = len(right_indices)
                    right_batched = {
                        key: value[offset : offset + rows]
                        for key, value in right_out.items()
                    }
                    offset += rows
            self.predictions.extend(
                self.predictor._collect_batch_predictions(
                    batch_hand_data,
                    output,
                    params,
                    global_scale,
                    scale_params,
                    right_batched,
                    right_indices,
                )
            )
        self.right_ms += (time.perf_counter() - start) * 1000
