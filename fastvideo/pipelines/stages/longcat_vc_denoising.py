# SPDX-License-Identifier: Apache-2.0
"""
LongCat VC Denoising Stage with KV cache support.

This stage extends the I2V denoising stage to support:
1. KV cache for conditioning frames
2. Video continuation with multiple conditioning frames
"""

from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.pipelines.pipeline_batch_info import ForwardBatch
from fastvideo.pipelines.stages.longcat_denoising import LongCatDenoisingStage


class LongCatVCDenoisingStage(LongCatDenoisingStage):
    """
    LongCat denoising with Video Continuation and KV cache support.
    
    Key differences from I2V denoising:
    - Supports KV cache (reuses cached K/V from conditioning frames)
    - Handles larger num_cond_latents
    - Concatenates conditioning latents back after denoising
    
    When use_kv_cache=True:
    - batch.latents contains ONLY noise frames (cond removed by KV cache init)
    - batch.kv_cache_dict contains cached K/V
    - batch.cond_latents contains conditioning latents for post-concat
    
    When use_kv_cache=False:
    - batch.latents contains ALL frames (cond + noise)
    - Timestep masking: timestep[:, :num_cond_latents] = 0
    - Selective denoising: only update noise frames
    """

    def forward(
        self,
        batch: ForwardBatch,
        fastvideo_args: FastVideoArgs,
    ) -> ForwardBatch:
        """Run denoising loop with VC conditioning and optional KV cache."""
        from fastvideo.pipelines.stages.denoising_engine import DenoisingEngine
        from fastvideo.pipelines.stages.denoising_longcat_strategy import (
            LongCatVCStrategy)

        engine = DenoisingEngine(LongCatVCStrategy(self))
        return engine.run(batch, fastvideo_args)
