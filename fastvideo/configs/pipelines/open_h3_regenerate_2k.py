# SPDX-License-Identifier: Apache-2.0
"""Pipeline configuration for Open-H3-Regenerate-2K (768P H3 draft -> 2560x1440)."""

from __future__ import annotations

from dataclasses import dataclass

from fastvideo.configs.pipelines.minimax_h3 import MiniMaxH3PipelineConfig

DRAFT_SIZE_CHOICES = ("768p", "576p")


@dataclass
class OpenH3Regenerate2KPipelineConfig(MiniMaxH3PipelineConfig):
    """MiniMax-H3 policy plus the refine-stage knobs.

    ``draft_size`` is the size the draft video is VAE-encoded at: 768p (1344x768) for MiniMax-H3 drafts, 576p
    (1024x576) for drafts from other video models. ``refine_start_sigma`` re-noises the upscaled draft to another
    level; the later steps then follow the shift-6 rule (None keeps the released grid).
    """

    draft_size: str = "768p"
    refine_start_sigma: float | None = None

    def check_pipeline_config(self) -> None:
        super().check_pipeline_config()
        if self.draft_size not in DRAFT_SIZE_CHOICES:
            raise ValueError(f"draft_size must be one of {DRAFT_SIZE_CHOICES}, got {self.draft_size!r}.")
        if self.refine_start_sigma is not None and not 0.0 < self.refine_start_sigma < 1.0:
            raise ValueError(f"refine_start_sigma must be in (0, 1), got {self.refine_start_sigma}.")
