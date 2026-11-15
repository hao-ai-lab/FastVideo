# SPDX-License-Identifier: Apache-2.0
"""Pipeline configuration for Pika H3 Flex.

Pika H3 Flex is a post-trained MiniMax-H3: the transformer, Qwen3-VL
conditioner, and video/audio VAEs are the released H3 components, and a rank-128
adapter is merged at load time.  What changes is the sampler and the geometry
contract:

* six transformer forwards over the fixed clean-time knots
  ``1000, 750, 500, 250, 100, 40``, with a fresh-noise re-noise step between
  forwards (the terminal ``0`` knot is the output, never a forward);
* video and audio clock shifts of 12 and 3;
* 480p and 768p canvas buckets instead of a free-form canvas;
* center-cropped keyframes and tier-dependent reference-image sizing.

Everything else, including CFG being off and the 24 fps frame grid, is inherited
from MiniMax-H3.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from fastvideo.configs.models.dits.pika_h3_flex import PikaH3FlexDiTConfig
from fastvideo.configs.pipelines.minimax_h3 import MiniMaxH3PipelineConfig

if TYPE_CHECKING:
    from fastvideo.api.sampling_param import SamplingParam
    from fastvideo.api.schema import GenerationRequest

# The released re-noise ladder. The terminal 0 is a sigma-grid point, not a
# transformer forward: six forwards, seven sigmas.
PIKA_DENOISING_STEPS: tuple[int, ...] = (1000, 750, 500, 250, 100, 40)
PIKA_VIDEO_SCHEDULER_SHIFT = 12.0
PIKA_AUDIO_SCHEDULER_SHIFT = 3.0
PIKA_RESOLUTION_TIERS: tuple[str, ...] = ("p480", "p768")
PIKA_DEFAULT_RESOLUTION_TIER = "p768"


@dataclass
class PikaH3FlexPipelineConfig(MiniMaxH3PipelineConfig):
    """MiniMax-H3 components under Pika's adapter, sampler, and bucket contract."""

    dit_config: PikaH3FlexDiTConfig = field(default_factory=PikaH3FlexDiTConfig)
    # Pika's CLI calls this ``--resolution``; the API has no dedicated tier
    # field, so it is a pipeline config knob rather than a per-request one.
    pika_resolution_tier: str = PIKA_DEFAULT_RESOLUTION_TIER
    pika_denoising_steps: tuple[int, ...] = PIKA_DENOISING_STEPS

    def check_pipeline_config(self) -> None:
        super().check_pipeline_config()
        if self.pika_resolution_tier not in PIKA_RESOLUTION_TIERS:
            raise ValueError(f"Pika H3 Flex supports resolution tiers {list(PIKA_RESOLUTION_TIERS)}; got "
                             f"{self.pika_resolution_tier!r}.")
        if tuple(self.pika_denoising_steps) != PIKA_DENOISING_STEPS:
            raise ValueError(f"Pika H3 Flex's adapter contract fixes the denoising ladder at "
                             f"{list(PIKA_DENOISING_STEPS)}; got {list(self.pika_denoising_steps)}.")
        if self.dmd_denoising_steps is not None:
            raise ValueError("Pika H3 Flex uses its own six-forward re-noise schedule; dmd_denoising_steps must "
                             "remain unset.")
        if self.pdd_step_indices is not None:
            raise ValueError("Pika H3 Flex is not a FastH3 PDD student; pdd_step_indices must remain unset.")

    def apply_request_constraints(self, request: GenerationRequest, sampling_param: SamplingParam) -> SamplingParam:
        """Fix the six-forward ladder and fill an omitted canvas from ``pika_resolution_tier``.

        ``PikaH3FlexSamplingParam`` and the presets carry the 768p canvas, so by
        input preparation an omitted size is indistinguishable from an explicit
        1344x768. The request's explicit-field tracking is the last place the two
        differ: an omitted size takes the configured tier's 16:9 bucket, while an
        explicit size still wins and picks its own tier.
        """
        from fastvideo.api.compat import explicit_request_updates

        explicit = explicit_request_updates(request)
        steps = len(PIKA_DENOISING_STEPS)
        requested = explicit.get("num_inference_steps", steps)
        if requested != steps:
            raise ValueError(f"Pika H3 Flex runs exactly {steps} transformer forwards; the request sets "
                             f"num_inference_steps={requested}. The six-step ladder is part of the adapter "
                             f"contract, not a quality/speed knob.")
        sampling_param.num_inference_steps = steps

        height, width = explicit.get("height"), explicit.get("width")
        if (height is None) != (width is None):
            raise ValueError("Pika H3 Flex `height` and `width` must be passed together, or neither; the request "
                             f"sets height={height}, width={width}.")
        if height is None:
            from fastvideo.pipelines.basic.pika_h3_flex.buckets import DEFAULT_ASPECT, canvas_for_aspect

            sampling_param.height, sampling_param.width = canvas_for_aspect(DEFAULT_ASPECT, self.pika_resolution_tier)
        return sampling_param


__all__ = [
    "PIKA_AUDIO_SCHEDULER_SHIFT",
    "PIKA_DEFAULT_RESOLUTION_TIER",
    "PIKA_DENOISING_STEPS",
    "PIKA_RESOLUTION_TIERS",
    "PIKA_VIDEO_SCHEDULER_SHIFT",
    "PikaH3FlexPipelineConfig",
]
