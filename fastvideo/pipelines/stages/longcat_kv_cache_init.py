# SPDX-License-Identifier: Apache-2.0
"""
LongCat KV Cache Initialization Stage for Video Continuation (VC).

This stage pre-computes K/V cache for conditioning frames.

"""

import torch

from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.forward_context import set_forward_context
from fastvideo.logger import init_logger
from fastvideo.pipelines.pipeline_batch_info import ForwardBatch
from fastvideo.pipelines.stages.base import PipelineStage

logger = init_logger(__name__)


class LongCatKVCacheInitStage(PipelineStage):
    """
    Pre-compute KV cache for conditioning frames.

    After this stage:
    - batch.kv_cache_dict contains {block_idx: (k, v)}
    - batch.cond_latents contains the conditioning latents
    - batch.latents contains ONLY noise latents
    """

    def __init__(self, transformer):
        super().__init__()
        self.transformer = transformer

    def forward(
        self,
        batch: ForwardBatch,
        fastvideo_args: FastVideoArgs,
    ) -> ForwardBatch:
        """Initialize KV cache from conditioning latents."""

        # Check if KV cache is enabled
        use_kv_cache = getattr(fastvideo_args.pipeline_config, 'use_kv_cache',
                               True)
        if not use_kv_cache:
            batch.kv_cache_dict = {}
            batch.use_kv_cache = False
            logger.info("KV cache disabled, skipping initialization")
            return batch

        batch.use_kv_cache = True
        offload_kv_cache = getattr(fastvideo_args.pipeline_config,
                                   'offload_kv_cache', False)

        # Get conditioning latents
        num_cond_latents = batch.num_cond_latents
        if num_cond_latents <= 0:
            batch.kv_cache_dict = {}
            logger.warning("num_cond_latents <= 0, skipping KV cache init")
            return batch

        # Extract conditioning latents
        cond_latents = batch.latents[:, :, :num_cond_latents].clone()

        logger.info(
            "Initializing KV cache for %d conditioning latents, shape: %s",
            num_cond_latents, cond_latents.shape)

        # Timestep = 0 for conditioning (they are "clean")
        B = cond_latents.shape[0]
        T_cond = cond_latents.shape[2]
        timestep = torch.zeros(B,
                               T_cond,
                               device=cond_latents.device,
                               dtype=cond_latents.dtype)

        # Empty prompt embeddings (cross-attn will be skipped)
        max_seq_len = 512
        # Get caption dimension from transformer config
        caption_dim = self.transformer.config.caption_channels
        empty_embeds = torch.zeros(B,
                                   max_seq_len,
                                   caption_dim,
                                   device=cond_latents.device,
                                   dtype=cond_latents.dtype)

        # Get transformer dtype
        if hasattr(self.transformer, 'module'):
            transformer_dtype = next(self.transformer.module.parameters()).dtype
        else:
            transformer_dtype = next(self.transformer.parameters()).dtype

        # Run transformer with return_kv=True, skip_crs_attn=True
        with (
                torch.no_grad(),
                set_forward_context(
                    current_timestep=0,
                    attn_metadata=None,
                    forward_batch=batch,
                ),
                torch.autocast(device_type='cuda', dtype=transformer_dtype),
        ):
            _, kv_cache_dict = self.transformer(
                hidden_states=cond_latents.to(transformer_dtype),
                encoder_hidden_states=empty_embeds.to(transformer_dtype),
                timestep=timestep.to(transformer_dtype),
                return_kv=True,
                skip_crs_attn=True,
                offload_kv_cache=offload_kv_cache,
            )

        # Store cache and save cond_latents for later concatenation
        batch.kv_cache_dict = kv_cache_dict
        batch.cond_latents = cond_latents

        # Remove conditioning latents from main latents
        # After this, batch.latents contains ONLY noise frames
        batch.latents = batch.latents[:, :, num_cond_latents:]

        logger.info(
            "KV cache initialized: %d blocks, offload=%s, remaining latents shape: %s",
            len(kv_cache_dict), offload_kv_cache, batch.latents.shape)

        return batch
