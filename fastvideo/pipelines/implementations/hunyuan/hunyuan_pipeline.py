"""
HunYuan video diffusion pipeline implementation.

This module contains an implementation of the HunYuan video diffusion pipeline
using the modular pipeline architecture.
"""

from typing import Any, Dict, List, Optional, Union, Tuple
import torch
from diffusers.image_processor import VaeImageProcessor

from fastvideo.pipelines.composed import ComposedPipelineBase
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
from fastvideo.pipelines.stages.prompt_encoding import PromptEncodingStage
# from fastvideo.pipelines.stages.timestep_preparation import FlowMatchingTimestepPreparationStage
from fastvideo.inference_args import InferenceArgs
# from fastvideo.pipelines.composed.composed_pipeline_base import DiffusionPipelineOutput
from fastvideo.pipelines.pipeline_batch_info import ForwardBatch
from diffusers.utils import BaseOutput
from typing import Union, Optional, Tuple, List
import numpy as np
from dataclasses import dataclass

from fastvideo.logger import init_logger
logger = init_logger(__name__)

# class HunyuanLatentPreparationStage(LatentPreparationStage):
#     def _call_implementation(self, batch: ForwardBatch, inference_args: InferenceArgs) -> ForwardBatch:
#         "custom logic for HunYuan latent preparation"
#         pass


# class hunyuanloader(PipelineLoader):
#     def load_components(self, inference_args: InferenceArgs):
#         pass

@dataclass
class DiffusionPipelineOutput(BaseOutput):
    videos: Union[torch.Tensor, np.ndarray]


@register_pipeline("hunyuan-video")
class HunyuanVideoPipeline(ComposedPipelineBase):

    def setup_pipeline(self, inference_args: InferenceArgs):
        self._stages = []
        self.add_stage("input_validation_stage", 
                        InputValidationStage())
        self.add_stage("prompt_encoding_stage_primary", 
                       PromptEncodingStage(is_secondary=False))
        self.add_stage("prompt_encoding_stage_secondary", 
                       PromptEncodingStage(is_secondary=True))
        self.add_stage("conditioning_stage", 
                       ConditioningStage())
        self.add_stage("timestep_preparation_stage", 
                       TimestepPreparationStage())
        self.add_stage("latent_preparation_stage", 
                       LatentPreparationStage())
        self.add_stage("denoising_stage", 
                       DenoisingStage())
        self.add_stage("decoding_stage", 
                       DecodingStage())
    
    def initialize_pipeline(self, inference_args: InferenceArgs):
        assert len(self._stages) > 0, "Pipeline stages are not set"
        assert len(self._modules) > 0, "Pipeline modules are not set"


        vae_scale_factor = 2**(len(self.vae.config.block_out_channels) - 1)
        inference_args.vae_scale_factor = vae_scale_factor

        self.image_processor = VaeImageProcessor(vae_scale_factor=vae_scale_factor)
        self.register_modules({"image_processor": self.image_processor})


        num_channels_latents = self.transformer.in_channels
        inference_args.num_channels_latents = num_channels_latents



    def adjust_video_length(self, batch: ForwardBatch, inference_args: InferenceArgs):
        """Adjust video length based on VAE version"""
        video_length = batch.num_frames
        vae_ver = inference_args.vae
        logger.info(f"Adjusting video length for VAE version: {vae_ver}")
        if "884" in vae_ver:
            batch.num_frames = (video_length - 1) // 4 + 1
        elif "888" in vae_ver:
            batch.num_frames = (video_length - 1) // 8 + 1
        return batch

    
    @torch.no_grad()
    def forward(self, batch: ForwardBatch, inference_args: InferenceArgs) -> ForwardBatch:
        logger.info(f"Running pipeline stages: {self._stage_name_mapping.keys()}")
        logger.info(f"Batch: {batch}")
        # for stage in self._stages:
            # batch = stage(batch, inference_args)

        # or 

        batch = self.input_validation_stage(batch, inference_args)
        batch = self.prompt_encoding_stage_primary(batch, inference_args)
        batch = self.prompt_encoding_stage_secondary(batch, inference_args)
        batch = self.conditioning_stage(batch, inference_args)
        batch = self.timestep_preparation_stage(batch, inference_args)

        # custom logic 
        batch = self.adjust_video_length(batch, inference_args)

        batch = self.latent_preparation_stage(batch, inference_args)
        batch = self.denoising_stage(batch, inference_args)
        batch = self.decoding_stage(batch, inference_args)

        return DiffusionPipelineOutput(videos=batch.videos) 
