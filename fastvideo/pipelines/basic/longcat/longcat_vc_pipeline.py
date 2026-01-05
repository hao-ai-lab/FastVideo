# SPDX-License-Identifier: Apache-2.0
"""
LongCat Video Continuation (VC) pipeline implementation.

This module implements VC (Video Continuation) generation for LongCat with
KV cache optimization for 2-3x speedup.

Supports:
- Basic VC (50 steps, guidance_scale=4.0)
- Distilled VC with LoRA (16 steps, guidance_scale=1.0)
- KV cache for conditioning frames
"""

from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.logger import init_logger
from fastvideo.pipelines import ComposedPipelineBase, LoRAPipeline
from fastvideo.pipelines.stages import (
    DecodingStage,
    InputValidationStage,
    TextEncodingStage,
    TimestepPreparationStage,
)
from fastvideo.pipelines.stages.longcat_video_vae_encoding import LongCatVideoVAEEncodingStage
from fastvideo.pipelines.stages.longcat_i2v_latent_preparation import LongCatI2VLatentPreparationStage
from fastvideo.pipelines.stages.longcat_kv_cache_init import LongCatKVCacheInitStage
from fastvideo.pipelines.stages.longcat_vc_denoising import LongCatVCDenoisingStage

logger = init_logger(__name__)


class LongCatVideoContinuationPipeline(LoRAPipeline, ComposedPipelineBase):
    """
    LongCat Video Continuation pipeline.
    
    Generates video continuation from multiple conditioning frames using
    optional KV cache for 2-3x speedup.
    
    Key features:
    - Takes video input (13+ frames typically)
    - Encodes conditioning frames via VAE
    - Optionally pre-computes KV cache for conditioning
    - Uses cached K/V during denoising for speedup
    - Concatenates conditioning back after denoising
    """

    _required_config_modules = [
        "text_encoder", "tokenizer", "vae", "transformer", "scheduler"
    ]

    def initialize_pipeline(self, fastvideo_args: FastVideoArgs):
        """Initialize LongCat-specific components."""
        pipeline_config = fastvideo_args.pipeline_config
        transformer = self.get_module("transformer", None)
        if transformer is None:
            return

        # Enable BSA if configured (for VC, BSA may not be needed)
        if getattr(pipeline_config, 'enable_bsa', False):
            bsa_params_cfg = getattr(pipeline_config, 'bsa_params', None) or {}
            sparsity = getattr(pipeline_config, 'bsa_sparsity', None)
            cdf_threshold = getattr(pipeline_config, 'bsa_cdf_threshold', None)
            chunk_q = getattr(pipeline_config, 'bsa_chunk_q', None)
            chunk_k = getattr(pipeline_config, 'bsa_chunk_k', None)

            effective_bsa_params = dict(bsa_params_cfg) if isinstance(
                bsa_params_cfg, dict) else {}
            if sparsity is not None:
                effective_bsa_params['sparsity'] = sparsity
            if cdf_threshold is not None:
                effective_bsa_params['cdf_threshold'] = cdf_threshold
            if chunk_q is not None:
                effective_bsa_params['chunk_3d_shape_q'] = chunk_q
            if chunk_k is not None:
                effective_bsa_params['chunk_3d_shape_k'] = chunk_k

            # Provide defaults
            effective_bsa_params.setdefault('sparsity', 0.9375)
            effective_bsa_params.setdefault('chunk_3d_shape_q', [4, 4, 4])
            effective_bsa_params.setdefault('chunk_3d_shape_k', [4, 4, 4])

            if hasattr(transformer, 'enable_bsa'):
                logger.info("Enabling BSA for LongCat VC transformer")
                transformer.enable_bsa()
                if hasattr(transformer, 'blocks'):
                    try:
                        for blk in transformer.blocks:
                            if hasattr(blk, 'self_attn'):
                                blk.self_attn.bsa_params = effective_bsa_params
                    except Exception as e:
                        logger.warning("Failed to set BSA params: %s", e)
                logger.info("BSA parameters: %s", effective_bsa_params)
        else:
            if hasattr(transformer, 'disable_bsa'):
                transformer.disable_bsa()

    def create_pipeline_stages(self, fastvideo_args: FastVideoArgs):
        """Set up VC-specific pipeline stages."""

        # 1. Input validation
        self.add_stage(stage_name="input_validation_stage",
                       stage=InputValidationStage())

        # 2. Text encoding
        self.add_stage(stage_name="prompt_encoding_stage",
                       stage=TextEncodingStage(
                           text_encoders=[self.get_module("text_encoder")],
                           tokenizers=[self.get_module("tokenizer")],
                       ))

        # 3. Video VAE encoding (encodes conditioning frames)
        self.add_stage(
            stage_name="video_vae_encoding_stage",
            stage=LongCatVideoVAEEncodingStage(vae=self.get_module("vae")))

        # 4. Timestep preparation
        self.add_stage(stage_name="timestep_preparation_stage",
                       stage=TimestepPreparationStage(
                           scheduler=self.get_module("scheduler")))

        # 5. Latent preparation (reuse I2V stage - it handles video_latent too)
        self.add_stage(stage_name="latent_preparation_stage",
                       stage=LongCatVCLatentPreparationStage(
                           scheduler=self.get_module("scheduler"),
                           transformer=self.get_module("transformer")))

        # 6. KV cache initialization (optional, based on config)
        # This is always added but will skip if use_kv_cache=False
        self.add_stage(stage_name="kv_cache_init_stage",
                       stage=LongCatKVCacheInitStage(
                           transformer=self.get_module("transformer")))

        # 7. Denoising with VC and KV cache support
        self.add_stage(stage_name="denoising_stage",
                       stage=LongCatVCDenoisingStage(
                           transformer=self.get_module("transformer"),
                           transformer_2=self.get_module("transformer_2", None),
                           scheduler=self.get_module("scheduler"),
                           vae=self.get_module("vae"),
                           pipeline=self))

        # 8. Decoding
        self.add_stage(stage_name="decoding_stage",
                       stage=DecodingStage(vae=self.get_module("vae"),
                                           pipeline=self))


class LongCatVCLatentPreparationStage(LongCatI2VLatentPreparationStage):
    """
    Prepare latents with video conditioning for first N frames.
    
    Extends I2V latent preparation to handle video_latent (multiple frames)
    instead of image_latent (single frame).
    """

    def forward(self, batch, fastvideo_args):
        """Prepare latents with VC conditioning."""

        # Check if we have video_latent (from VC encoding stage)
        video_latent = getattr(batch, 'video_latent', None)
        if video_latent is not None:
            # Set image_latent to video_latent for parent class compatibility
            batch.image_latent = video_latent

        # Call parent class forward
        return super().forward(batch, fastvideo_args)


EntryClass = LongCatVideoContinuationPipeline
