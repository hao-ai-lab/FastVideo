# SPDX-License-Identifier: Apache-2.0
"""Request model for cutting a job's video and color-grading it section by section."""

from pydantic import BaseModel, Field


class GradeSegment(BaseModel):
    """One razor-cut section of the kept range, and its own color grade.

    ``end_seconds`` is seconds into the *kept* (already start/end-trimmed)
    range -- the same timeline the preview video plays -- not the original
    file's. Only the last section in a list may leave it ``None``, meaning
    "to the end"; sections must be given in increasing order and cover the
    whole kept range, with no gaps or reordering.
    """
    end_seconds: float | None = None
    #: Added directly to 0-255 pixel values. 0 = unchanged.
    brightness: float = 0.0
    #: 1.0 = unchanged, 0 = flat gray, >1 = more contrast.
    contrast: float = 1.0
    #: 1.0 = unchanged, 0 = grayscale, >1 = more saturated.
    saturation: float = 1.0


class TrimRequest(BaseModel):
    start_seconds: float = 0.0
    #: None keeps to the end of the video.
    end_seconds: float | None = None
    #: Defaults to one neutral section covering the whole kept range, i.e. a plain trim with no grading.
    segments: list[GradeSegment] = Field(default_factory=lambda: [GradeSegment()])
