# SPDX-License-Identifier: Apache-2.0
"""Portable exact-shape bucket identifiers for preprocessed video data."""

from __future__ import annotations

import re
from dataclasses import dataclass

_VIDEO_SHAPE_BUCKET_PATTERN = re.compile(
    r"^bucket=(?P<width>[1-9][0-9]*)x(?P<height>[1-9][0-9]*)-(?P<num_frames>[1-9][0-9]*)f$"
)


@dataclass(frozen=True, slots=True)
class ExactVideoShapeBucket:
    """Pixel geometry and frame count encoded by one bucket directory."""

    width: int
    height: int
    num_frames: int

    @property
    def bucket_id(self) -> str:
        return f"bucket={self.width}x{self.height}-{self.num_frames}f"


def parse_video_shape_bucket_id(bucket_id: str) -> ExactVideoShapeBucket:
    """Parse ``bucket=<width>x<height>-<num_frames>f`` exactly.

    Width is deliberately first so the identifier matches common media
    geometry notation. The strict spelling makes independently generated data
    roots portable and prevents ranks from silently assigning the same shape
    two different names.
    """
    match = _VIDEO_SHAPE_BUCKET_PATTERN.fullmatch(str(bucket_id))
    if match is None:
        raise ValueError(
            "An exact video-shape bucket must be named "
            "'bucket=<width>x<height>-<num_frames>f' with positive decimal "
            f"integers (for example 'bucket=1344x768-124f'), got {bucket_id!r}"
        )
    return ExactVideoShapeBucket(
        width=int(match.group("width")),
        height=int(match.group("height")),
        num_frames=int(match.group("num_frames")),
    )


__all__ = ["ExactVideoShapeBucket", "parse_video_shape_bucket_id"]
