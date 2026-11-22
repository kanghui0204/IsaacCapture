# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-OpenMDW-1.1

"""Optional cleanup of the RT-DETR whole-clip hand tracks (``track_cleanup:`` in the
rtdetr detector YAML).

Off unless a switch is on; see ``config.TrackCleanupConfig`` for what each switch does
and ``hand_detector_2d/README.md`` for how to enable it.
"""

from .config import TrackCleanupConfig
from .detector import (
    RTDetrTrackCleanupDetector,
    boxes_from_tracks_cleanup,
    build_detector,
)

__all__ = [
    "RTDetrTrackCleanupDetector",
    "TrackCleanupConfig",
    "boxes_from_tracks_cleanup",
    "build_detector",
]
