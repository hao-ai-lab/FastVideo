# SPDX-License-Identifier: Apache-2.0
"""
LongCat-specific denoising stage implementing CFG-zero optimized guidance.
"""

import torch

from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.pipelines.pipeline_batch_info import ForwardBatch
from fastvideo.pipelines.stages.denoising import DenoisingStage


class LongCatDenoisingStage(DenoisingStage):
    """
    LongCat denoising stage with CFG-zero optimized guidance scale.
    
    Implements:
    1. Optimized CFG scale from CFG-zero paper
    2. Negation of noise prediction before scheduler step (flow matching convention)
    3. Batched CFG computation (unlike standard FastVideo separate passes)
    """

    def optimized_scale(self, positive_flat, negative_flat) -> torch.Tensor:
        """
        Calculate optimized scale from CFG-zero paper.
        
        st_star = (v_cond^T * v_uncond) / ||v_uncond||^2
        
        Args:
            positive_flat: Conditional prediction, flattened [B, -1]
            negative_flat: Unconditional prediction, flattened [B, -1]
        
        Returns:
            st_star: Optimized scale [B, 1]
        """
        # Calculate dot product
        dot_product = torch.sum(positive_flat * negative_flat,
                                dim=1,
                                keepdim=True)
        # Squared norm of uncondition
        squared_norm = torch.sum(negative_flat**2, dim=1, keepdim=True) + 1e-8
        # st_star = v_cond^T * v_uncond / ||v_uncond||^2
        st_star = dot_product / squared_norm
        return st_star

    def forward(
        self,
        batch: ForwardBatch,
        fastvideo_args: FastVideoArgs,
    ) -> ForwardBatch:
        """
        Run LongCat denoising loop with optimized CFG.
        """
        from fastvideo.pipelines.stages.denoising_engine import DenoisingEngine
        from fastvideo.pipelines.stages.denoising_longcat_strategy import (
            LongCatStrategy)

        engine = DenoisingEngine(LongCatStrategy(self))
        return engine.run(batch, fastvideo_args)
