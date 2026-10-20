# SPDX-License-Identifier: Apache-2.0
"""Per-request options owned by Kandinsky6 SR, carried in ``ForwardBatch.extra``."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, fields
from typing import Any


@dataclass(frozen=True)
class Kandinsky6SROptions:
    sr_resolution_scale: float = 2.25
    sr_tiles_batch_size: int = 1
    sr_tile_min_overlap: float = 0.20
    sr_target_resolution: str | None = None
    sr_target_resize_mode: str = "fit"

    @classmethod
    def from_extra(cls, extra: Mapping[str, Any]) -> Kandinsky6SROptions:
        """Keep omitted options at their defaults; preserve explicit values, including None."""
        return cls(**{key: extra[key] for key in SR_OPTION_FIELDS if key in extra})


SR_OPTION_FIELDS = tuple(field.name for field in fields(Kandinsky6SROptions))
SR_REQUEST_FIELDS = (*SR_OPTION_FIELDS, "sr_lr_latent", "sr_audio", "sr_audio_sample_rate")
