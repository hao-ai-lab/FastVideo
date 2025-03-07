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

# class HunyuanLatentPreparationStage(LatentPreparationStage):
#     def _call_implementation(self, batch: ForwardBatch, inference_args: InferenceArgs) -> ForwardBatch:
#         "custom logic for HunYuan latent preparation"
#         pass


# class hunyuanloader(PipelineLoader):
#     def load_components(self, inference_args: InferenceArgs):
#         pass



@register_pipeline("hunyuan-video")
class HunyuanVideoPipeline(TextToVideoPipeline):
    """
    HunYuan video diffusion pipeline.
    
    This pipeline implements the HunYuan video diffusion process using the
    modular pipeline architecture.
    """

    # def load_components(self, hf_config):
    #     # if we use a different dit class
    #     # or different weight 
    #     pass

    # "text_encoder_2" "encoder2"
    # "transformer" : "dit"

    
    def __init__(
        self,
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
        super().__init__()
        # Create the stages
        input_validation_stage = InputValidationStage()
        prompt_encoding_stage = DualEncoderPromptEncodingStage()
        timestep_preparation_stage = FlowMatchingTimestepPreparationStage()
        latent_preparation_stage = LatentPreparationStage()
        conditioning_stage = ConditioningStage()
        denoising_stage = DenoisingStage()
        decoding_stage = DecodingStage()
        
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
        
    
    # def __call__(self, batch: ForwardBatch, inference_args: InferenceArgs) -> ForwardBatch:
    #     # for stage in self._stages:

    #     batch = self.input_validation_stage(batch, inference_args)
    #     batch = self.prompt_encoding_stage(batch, inference_args)
    #     # add
    #     batch = self.timestep_preparation_stage(batch, inference_args)
    #     batch = self.latent_preparation_stage(batch, inference_args)
    #     # batch.latents = torch.randn(batch.num_videos, 4, 8, 64, 64)
    #     self.scheduler
    #     self.encoder1
    #     self.encoder2
    #     self.vae
    #     self.dit

    #     batch = self.conditioning_stage(batch, inference_args)
    #     batch = self.denoising_stage(batch, inference_args)
    #     batch = self.decoding_stage(batch, inference_args)