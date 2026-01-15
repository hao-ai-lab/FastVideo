# SPDX-License-Identifier: Apache-2.0
"""
LongCat I2V Denoising Stage with conditioning support.

This stage implements Tier 3 I2V denoising:
1. Per-frame timestep masking (timestep[:, :num_cond_latents] = 0)
2. Passes num_cond_latents to transformer (for RoPE skipping)
3. Selective denoising (only updates non-conditioned frames)
4. CFG-zero optimized guidance
"""

from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.pipelines.pipeline_batch_info import ForwardBatch
from fastvideo.pipelines.stages.longcat_denoising import LongCatDenoisingStage


class LongCatI2VDenoisingStage(LongCatDenoisingStage):
    """
    LongCat denoising with I2V conditioning support.
    
    Key modifications from base LongCat denoising:
    1. Sets timestep=0 for conditioning frames
    2. Passes num_cond_latents to transformer
    3. Only applies scheduler step to non-conditioned frames
    """

    def forward(
        self,
        batch: ForwardBatch,
        fastvideo_args: FastVideoArgs,
    ) -> ForwardBatch:
        """Run denoising loop with I2V conditioning."""
        from fastvideo.pipelines.stages.denoising_engine import DenoisingEngine
        from fastvideo.pipelines.stages.denoising_longcat_strategy import (
            LongCatI2VStrategy)

        engine = DenoisingEngine(LongCatI2VStrategy(self))
        return engine.run(batch, fastvideo_args)
