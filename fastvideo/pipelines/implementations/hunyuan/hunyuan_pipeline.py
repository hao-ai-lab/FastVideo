"""
HunYuan video diffusion pipeline implementation.

This module contains an implementation of the HunYuan video diffusion pipeline
using the modular pipeline architecture.
"""

from typing import Any, Dict, List, Optional, Union, Tuple
import torch

from fastvideo.pipelines.composed.text_to_video import TextToVideoPipeline
from fastvideo.pipelines.stages import (
    InputValidationStage,
    PromptEncodingStage,
    TimestepPreparationStage,
    LatentPreparationStage,
    ConditioningStage,
    DenoisingStage,
    DecodingStage,
    PostProcessingStage,
)
from fastvideo.pipelines import register_pipeline
from fastvideo.pipelines.stages.prompt_encoding import DualEncoderPromptEncodingStage
from fastvideo.pipelines.stages.timestep_preparation import FlowMatchingTimestepPreparationStage


@register_pipeline("hunyuan-video")
class HunYuanVideoPipeline(TextToVideoPipeline):
    """
    HunYuan video diffusion pipeline.
    
    This pipeline implements the HunYuan video diffusion process using the
    modular pipeline architecture.
    """
    
    def __init__(
        self,
        unet,
        vae,
        text_encoder,
        text_encoder_2,
        tokenizer,
        tokenizer_2,
        scheduler,
    ):
        """
        Initialize the HunYuan video pipeline.
        
        Args:
            unet: The UNet model.
            vae: The VAE model.
            text_encoder: The primary text encoder.
            text_encoder_2: The secondary text encoder.
            tokenizer: The primary tokenizer.
            tokenizer_2: The secondary tokenizer.
            scheduler: The scheduler.
        """
        # Create the stages
        input_validation_stage = HunYuanInputValidationStage()
        prompt_encoding_stage = DualEncoderPromptEncodingStage(
            max_length=77,
            max_length_2=77,
        )
        timestep_preparation_stage = FlowMatchingTimestepPreparationStage(
            flow_shift=7,
            flow_reverse=False,
        )
        latent_preparation_stage = HunYuanLatentPreparationStage()
        conditioning_stage = HunYuanConditioningStage()
        denoising_stage = HunYuanDenoisingStage()
        decoding_stage = HunYuanDecodingStage()
        
        # Initialize the pipeline
        super().__init__(
            input_validation_stage=input_validation_stage,
            prompt_encoding_stage=prompt_encoding_stage,
            timestep_preparation_stage=timestep_preparation_stage,
            latent_preparation_stage=latent_preparation_stage,
            conditioning_stage=conditioning_stage,
            denoising_stage=denoising_stage,
            decoding_stage=decoding_stage,
        )
        
        # Register the modules
        self.register_modules(
            unet=unet,
            vae=vae,
            text_encoder=text_encoder,
            text_encoder_2=text_encoder_2,
            tokenizer=tokenizer,
            tokenizer_2=tokenizer_2,
            scheduler=scheduler,
        ) 