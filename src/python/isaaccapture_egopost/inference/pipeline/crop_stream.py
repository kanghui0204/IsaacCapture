# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-OpenMDW-1.1

"""Crop hands while earlier EgoHand batches run, and hand them over in forward batches.

``EgoHandPredictor.run`` used to build every (image, hand) crop of a clip before
the first forward batch (a 1838-frame clip: 3676 float32 crops, 2.9 GB of host
memory, about 3.5 s before the GPU starts). ``StreamedCropBatches`` keeps the
same crop function, the same job order and the same batch boundaries -- batches
of ``batch_size`` crops taken in job order after dropping the jobs whose crop
failed -- but produces them while earlier batches are on the GPU::

    crop pool (``workers`` threads), at most ``prefetch_batches * batch_size``
    jobs ahead of the batch being filled
      -> stager thread: drops failed crops (``None``), fills a batch, copies it
         into one of two pinned host slots and uploads it on one reused copy
         stream into the matching device slot (event ``ready``)
      -> forward (caller's thread): its current stream waits for ``ready``, the
         model reads a view of the device slot, and ``release()`` records
         ``consumed`` on that stream and returns the slot for the batch after next.

A pinned slot is refilled only after its previous upload has finished, and a
device slot only after the forward that read it (``consumed``), so at most two
batches are staged. Crop tensors are dropped from the prepared records once
copied; the records keep their camera metadata. CPU devices, and batches whose
crops differ in shape or dtype, get the original ``torch.stack(...).to(device)``.
"""

from __future__ import annotations

import queue
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Iterable, Iterator

import numpy as np
import torch

_DONE = object()
# Crop dtypes copied through numpy into the pinned slot; others use torch.stack.
_NUMPY_DTYPES = (torch.float32, torch.float64, torch.float16, torch.uint8)


class _Slot:
    __slots__ = ("consumed", "device", "host", "ready", "uploaded", "used")

    def __init__(self) -> None:
        self.host: torch.Tensor | None = None
        self.device: torch.Tensor | None = None
        self.ready = torch.cuda.Event()
        self.consumed = torch.cuda.Event()
        self.uploaded = False
        self.used = False


class StagedCropBatch:
    """One forward batch: prepared records (without crops) and their images."""

    def __init__(self, hand_data, images, slot_index, owner, on_device):
        self.hand_data = hand_data
        self._images = images
        self._slot_index = slot_index
        self._owner = owner
        self._on_device = on_device
        self._released = False

    def images(self, device) -> torch.Tensor:
        """Return the batch images on ``device``, ordered on its current stream."""
        if not self._on_device:
            return self._images.to(device)
        slot = self._owner._slots[self._slot_index]
        stream = torch.cuda.current_stream(self._images.device)
        stream.wait_event(slot.ready)
        # The device slot was allocated on the copy stream; keep the allocator
        # from reusing it while this stream may still read it.
        slot.device.record_stream(stream)
        return self._images

    def release(self) -> None:
        """Mark the images consumed; their slot may then be refilled."""
        if self._released:
            return
        self._released = True
        self._images = None
        self._owner._release(self._slot_index, self._on_device)


class StreamedCropBatches:
    """Iterate forward batches of valid crops while later crops are prepared.

    Use as a context manager around the forward loop. Each yielded
    ``StagedCropBatch`` must be ``release()``-d after its forward pass, before the
    next one is requested; exiting the context stops the threads and waits for
    pending copies, also on early exit or error. Errors raised by
    ``build_input`` are re-raised in the consuming thread.

    Args:
        jobs: Crop jobs in model order.
        build_input: ``job -> (image_id, hand, prepared or None)``; ``None``
            drops the job (a failed crop), exactly as before.
        batch_size: Valid crops per forward batch (the last one may be short).
        device: Forward device. CUDA devices use pinned slots and a copy stream.
        workers: Crop threads.
        prefetch_batches: Crop jobs in flight, in units of ``batch_size``.
        copy_stream: Stream for the uploads. Reuse one per device across clips
            so the allocator does not keep a separate block pool per clip.
        on_prepared: Called in job order with ``(image_id, hand, prepared)`` for
            every valid crop, before it is batched.
    """

    def __init__(
        self,
        jobs: Iterable[Any],
        build_input: Callable[[Any], tuple],
        *,
        batch_size: int,
        device,
        workers: int,
        prefetch_batches: int = 2,
        copy_stream: torch.cuda.Stream | None = None,
        on_prepared: Callable[[Any, str, dict], None] | None = None,
    ):
        for name, value in (
            ("batch_size", batch_size),
            ("workers", workers),
            ("prefetch_batches", prefetch_batches),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        self._jobs = list(jobs)
        self._build_input = build_input
        self._batch_size = batch_size
        self._device = torch.device(device)
        self._cuda = self._device.type == "cuda"
        if self._cuda and self._device.index is None:
            self._device = torch.device("cuda", torch.cuda.current_device())
        self._workers = workers
        self._window = prefetch_batches * batch_size
        self._copy_stream = copy_stream
        self._on_prepared = on_prepared
        self._slots = [_Slot() for _ in range(2)] if self._cuda else [None, None]
        self._free: queue.Queue = queue.Queue()
        self._ready: queue.Queue = queue.Queue()
        self._stop = threading.Event()
        self._pool: ThreadPoolExecutor | None = None
        self._thread: threading.Thread | None = None
        self._iterated = False
        # Main-thread seconds spent waiting for the next batch (crop not hidden).
        self.wait_s = 0.0
        self.num_valid = 0
        self.num_failed = 0

    # ------------------------------------------------------------------ lifecycle
    def __enter__(self) -> "StreamedCropBatches":
        if self._thread is not None:
            raise RuntimeError("StreamedCropBatches can only be entered once")
        if self._cuda and self._copy_stream is None:
            self._copy_stream = torch.cuda.Stream(device=self._device)
        for index in range(2):
            self._free.put(index)
        self._pool = ThreadPoolExecutor(max_workers=self._workers)
        self._thread = threading.Thread(
            target=self._stage_loop, name="egohand-crop-stager", daemon=True
        )
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    def close(self) -> None:
        """Stop cropping, then wait for pending uploads and forward reads."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
        if self._pool is not None:
            self._pool.shutdown(wait=True, cancel_futures=True)
            self._pool = None
        if self._cuda:
            if self._copy_stream is not None:
                self._copy_stream.synchronize()
            for slot in self._slots:
                if slot.used:
                    slot.consumed.synchronize()
                slot.host = slot.device = None
        while not self._ready.empty():
            self._ready.get_nowait()

    # ------------------------------------------------------------------ consumer
    def __iter__(self) -> Iterator[StagedCropBatch]:
        if self._thread is None:
            raise RuntimeError("use StreamedCropBatches as a context manager")
        if self._iterated:
            raise RuntimeError("StreamedCropBatches can only be iterated once")
        self._iterated = True
        while True:
            start = time.perf_counter()
            item = self._ready.get()
            self.wait_s += time.perf_counter() - start
            if item is _DONE:
                return
            if isinstance(item, BaseException):
                raise item
            yield item

    def _release(self, slot_index: int, on_device: bool) -> None:
        if on_device:
            slot = self._slots[slot_index]
            slot.consumed.record(torch.cuda.current_stream(self._device))
            slot.used = True
        self._free.put(slot_index)

    # ------------------------------------------------------------------ producer
    def _stage_loop(self) -> None:
        try:
            pending: deque = deque()
            next_job = 0

            def top_up() -> None:
                nonlocal next_job
                while next_job < len(self._jobs) and len(pending) < self._window:
                    pending.append(
                        self._pool.submit(self._build_input, self._jobs[next_job])
                    )
                    next_job += 1

            top_up()
            batch: list[dict] = []
            while pending:
                if self._stop.is_set():
                    return
                image_id, hand, prepared = pending.popleft().result()
                top_up()
                if prepared is None:
                    self.num_failed += 1
                    continue
                self.num_valid += 1
                if self._on_prepared is not None:
                    self._on_prepared(image_id, hand, prepared)
                batch.append(prepared)
                if len(batch) == self._batch_size:
                    if not self._stage(batch):
                        return
                    batch = []
            if batch and not self._stage(batch):
                return
            self._ready.put(_DONE)
        except BaseException as exc:  # noqa: BLE001 - re-raised by the consumer
            self._ready.put(exc)

    def _stage(self, batch: list[dict]) -> bool:
        while True:
            if self._stop.is_set():
                return False
            try:
                slot_index = self._free.get(timeout=0.05)
                break
            except queue.Empty:
                continue
        images = [prepared.pop("crop_tensor") for prepared in batch]
        first = images[0]
        uniform = first.dtype in _NUMPY_DTYPES and all(
            image.device.type == "cpu"
            and image.shape == first.shape
            and image.dtype == first.dtype
            for image in images
        )
        if not (self._cuda and uniform):
            # Original assembly; torch.stack raises the same errors as before.
            self._ready.put(
                StagedCropBatch(batch, torch.stack(images), slot_index, self, False)
            )
            return True
        slot = self._slots[slot_index]
        count = len(images)
        if (
            slot.host is None
            or slot.host.shape[1:] != first.shape
            or slot.host.dtype != first.dtype
        ):
            if slot.uploaded:
                slot.ready.synchronize()
            slot.host = torch.empty(
                (self._batch_size, *first.shape), dtype=first.dtype, pin_memory=True
            )
            with torch.cuda.device(self._device), torch.cuda.stream(self._copy_stream):
                if slot.used:
                    self._copy_stream.wait_event(slot.consumed)
                slot.device = torch.empty(
                    (self._batch_size, *first.shape),
                    dtype=first.dtype,
                    device=self._device,
                )
        elif slot.uploaded:
            # The previous upload must have finished reading this pinned buffer.
            slot.ready.synchronize()
        host = slot.host[:count]
        host_array = host.numpy()
        for index, image in enumerate(images):
            # Single-threaded copy: no OpenMP team in this thread.
            np.copyto(host_array[index], image.numpy())
        del images, first
        with torch.cuda.device(self._device), torch.cuda.stream(self._copy_stream):
            if slot.used:
                # The forward that read this device slot must have finished.
                self._copy_stream.wait_event(slot.consumed)
            device_images = slot.device[:count]
            device_images.copy_(host, non_blocking=True)
            slot.ready.record(self._copy_stream)
        slot.uploaded = True
        self._ready.put(StagedCropBatch(batch, device_images, slot_index, self, True))
        return True
