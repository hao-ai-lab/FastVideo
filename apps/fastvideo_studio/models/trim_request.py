# SPDX-License-Identifier: Apache-2.0
"""Request model for cutting and color-grading a job's video."""

from pydantic import BaseModel


class TrimRequest(BaseModel):
    start_seconds: float = 0.0
    #: None keeps to the end of the video.
    end_seconds: float | None = None
    #: Added directly to 0-255 pixel values. 0 = unchanged.
    brightness: float = 0.0
    #: 1.0 = unchanged, 0 = flat gray, >1 = more contrast.
    contrast: float = 1.0
    #: 1.0 = unchanged, 0 = grayscale, >1 = more saturated.
    saturation: float = 1.0
