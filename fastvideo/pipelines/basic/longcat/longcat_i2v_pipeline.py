# SPDX-License-Identifier: Apache-2.0
"""
LongCat Image-to-Video pipeline implementation.

This module implements I2V (Image-to-Video) generation for LongCat using Tier 3
conditioning with timestep masking, num_cond_latents support, and RoPE skipping.
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
from fastvideo.pipelines.stages.longcat_image_vae_encoding import LongCatImageVAEEncodingStage
from fastvideo.pipelines.stages.longcat_i2v_latent_preparation import LongCatI2VLatentPreparationStage
from fastvideo.pipelines.stages.longcat_i2v_denoising import LongCatI2VDenoisingStage

logger = init_logger(__name__)


class LongCatImageToVideoPipeline(LoRAPipeline, ComposedPipelineBase):
    """
    LongCat Image-to-Video pipeline.
    
    Generates video from a single input image using Tier 3 I2V conditioning:
    - Per-frame timestep masking (timestep[:, 0] = 0)
    - num_cond_latents parameter to transformer
    - RoPE skipping for conditioning frames
    - Selective denoising (skip first frame in scheduler)
    """

    _required_config_modules = [
        "text_encoder", "tokenizer", "vae", "transformer", "scheduler"
    ]

    def initialize_pipeline(self, fastvideo_args: FastVideoArgs):
        """Initialize LongCat-specific components."""
        # Same BSA initialization as base LongCat pipeline
        pipeline_config = fastvideo_args.pipeline_config
        transformer = self.get_module("transformer", None)
        if transformer is None:
            return

        # Enable BSA if configured
        if pipeline_config.enable_bsa:
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
                logger.info("Enabling BSA for LongCat I2V transformer")
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
        """Set up I2V-specific pipeline stages."""

        # 1. Input validation
        self.add_stage(stage_name="input_validation_stage",
                       stage=InputValidationStage())

        # 2. Text encoding (same as T2V)
        self.add_stage(stage_name="prompt_encoding_stage",
                       stage=TextEncodingStage(
                           text_encoders=[self.get_module("text_encoder")],
                           tokenizers=[self.get_module("tokenizer")],
                       ))

        # 3. Image VAE encoding (NEW for I2V)
        self.add_stage(
            stage_name="image_vae_encoding_stage",
            stage=LongCatImageVAEEncodingStage(vae=self.get_module("vae")))

        # 4. Timestep preparation
        self.add_stage(stage_name="timestep_preparation_stage",
                       stage=TimestepPreparationStage(
                           scheduler=self.get_module("scheduler")))

        # 5. Latent preparation with I2V conditioning
        self.add_stage(stage_name="latent_preparation_stage",
                       stage=LongCatI2VLatentPreparationStage(
                           scheduler=self.get_module("scheduler"),
                           transformer=self.get_module("transformer")))

        # 6. Denoising with I2V support
        self.add_stage(stage_name="denoising_stage",
                       stage=LongCatI2VDenoisingStage(
                           transformer=self.get_module("transformer"),
                           transformer_2=self.get_module("transformer_2", None),
                           scheduler=self.get_module("scheduler"),
                           vae=self.get_module("vae"),
                           pipeline=self))

        # 7. Decoding
        self.add_stage(stage_name="decoding_stage",
                       stage=DecodingStage(vae=self.get_module("vae"),
                                           pipeline=self))


EntryClass = LongCatImageToVideoPipeline
