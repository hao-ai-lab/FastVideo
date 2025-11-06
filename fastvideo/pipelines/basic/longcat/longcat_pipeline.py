# SPDX-License-Identifier: Apache-2.0
"""
LongCat video diffusion pipeline implementation (Phase 1: Wrapper).

This module contains a wrapper implementation of the LongCat video diffusion pipeline
using FastVideo's modular pipeline architecture with the original LongCat modules.
"""

from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.logger import init_logger
from fastvideo.pipelines import ComposedPipelineBase, LoRAPipeline
from fastvideo.pipelines.stages import (
    DecodingStage,
    InputValidationStage,
    LatentPreparationStage,
    TextEncodingStage,
    TimestepPreparationStage,
)
from fastvideo.pipelines.stages.longcat_denoising import LongCatDenoisingStage

logger = init_logger(__name__)


class LongCatPipeline(LoRAPipeline, ComposedPipelineBase):
    """
    LongCat video diffusion pipeline with LoRA support.
    
    Phase 1 implementation using wrapper modules from third_party/longcat_video.
    This validates the pipeline infrastructure before full FastVideo integration.
    """

    _required_config_modules = [
        "text_encoder",
        "tokenizer", 
        "vae",
        "transformer",
        "scheduler"
    ]

    def initialize_pipeline(self, fastvideo_args: FastVideoArgs):
        """Initialize LongCat-specific components."""
        # LongCat uses FlowMatchEulerDiscreteScheduler which is already loaded
        # from the model_index.json, so no need to override
        
        # Enable BSA (Block Sparse Attention) if configured
        pipeline_config = fastvideo_args.pipeline_config
        if hasattr(pipeline_config, 'enable_bsa') and pipeline_config.enable_bsa:
            transformer = self.get_module("transformer", None)
            if transformer is not None and hasattr(transformer, 'enable_bsa'):
                logger.info("Enabling Block Sparse Attention (BSA) for LongCat transformer")
                transformer.enable_bsa()
                
                # Log BSA parameters if available
                if hasattr(transformer, 'blocks') and len(transformer.blocks) > 0:
                    bsa_params = transformer.blocks[0].attn.bsa_params
                    if bsa_params:
                        logger.info(f"BSA parameters: {bsa_params}")
            else:
                logger.warning("BSA is enabled in config but transformer does not support it")

    def create_pipeline_stages(self, fastvideo_args: FastVideoArgs) -> None:
        """Set up pipeline stages with proper dependency injection."""

        self.add_stage(
            stage_name="input_validation_stage",
            stage=InputValidationStage()
        )

        self.add_stage(
            stage_name="prompt_encoding_stage",
            stage=TextEncodingStage(
                text_encoders=[self.get_module("text_encoder")],
                tokenizers=[self.get_module("tokenizer")],
            )
        )

        self.add_stage(
            stage_name="timestep_preparation_stage",
            stage=TimestepPreparationStage(
                scheduler=self.get_module("scheduler")
            )
        )

        self.add_stage(
            stage_name="latent_preparation_stage",
            stage=LatentPreparationStage(
                scheduler=self.get_module("scheduler"),
                transformer=self.get_module("transformer", None)
            )
        )

        self.add_stage(
            stage_name="denoising_stage",
            stage=LongCatDenoisingStage(
                transformer=self.get_module("transformer"),
                transformer_2=self.get_module("transformer_2", None),
                scheduler=self.get_module("scheduler"),
                vae=self.get_module("vae"),
                pipeline=self
            )
        )

        self.add_stage(
            stage_name="decoding_stage",
            stage=DecodingStage(
                vae=self.get_module("vae"),
                pipeline=self
            )
        )


EntryClass = LongCatPipeline