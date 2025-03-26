"""
HunYuan video diffusion pipeline implementation.

This module contains an implementation of the HunYuan video diffusion pipeline
using the modular pipeline architecture.
"""

from typing import Union
import torch
from diffusers.image_processor import VaeImageProcessor

from fastvideo.v1.pipelines.composed_pipeline_base import ComposedPipelineBase
from fastvideo.v1.pipelines.stages import (
    InputValidationStage,
    TimestepPreparationStage,
    LatentPreparationStage,
    ConditioningStage,
    DenoisingStage,
    DecodingStage,
    LlamaEncodingStage,
    CLIPTextEncodingStage,
)
from fastvideo.v1.inference_args import InferenceArgs
from fastvideo.v1.pipelines.pipeline_batch_info import ForwardBatch
# TODO(will): move PRECISION_TO_TYPE to better place

from diffusers.utils import BaseOutput
import numpy as np
from dataclasses import dataclass

from fastvideo.v1.logger import init_logger

logger = init_logger(__name__)


@dataclass
class DiffusionPipelineOutput(BaseOutput):
    videos: Union[torch.Tensor, np.ndarray]


class HunyuanVideoPipeline(ComposedPipelineBase):

    def required_config_modules(self):
        return [
            "text_encoder", "text_encoder_2", "tokenizer", "tokenizer_2", "vae",
            "transformer", "scheduler"
        ]


    def create_pipeline_stages(self, inference_args: InferenceArgs):
        """Set up pipeline stages with proper dependency injection."""
        
        self.add_stage(
            stage_name="input_validation_stage",
            stage=InputValidationStage()
        )
        
        self.add_stage(
            stage_name="prompt_encoding_stage_primary",
            stage=LlamaEncodingStage(
                text_encoder=self.get_module("text_encoder"),
                tokenizer=self.get_module("tokenizer"),
            )
        )
        
        self.add_stage(
            stage_name="prompt_encoding_stage_secondary",
            stage=CLIPTextEncodingStage(
                text_encoder=self.get_module("text_encoder_2"),
                tokenizer=self.get_module("tokenizer_2"),
            )
        )
        
        self.add_stage(
            stage_name="conditioning_stage",
            stage=ConditioningStage()
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
                scheduler=self.get_module("scheduler")
            )
        )
        
        self.add_stage(
            stage_name="denoising_stage",
            stage=DenoisingStage(
                transformer=self.get_module("transformer"),
                scheduler=self.get_module("scheduler")
            )
        )
        
        self.add_stage(
            stage_name="decoding_stage",
            stage=DecodingStage(
                vae=self.get_module("vae")
            )
        )
    
    def initialize_pipeline(self, inference_args: InferenceArgs):
        """
        Initialize the pipeline.
        """
        vae_scale_factor = 2**(len(self.get_module("vae").block_out_channels) -
                               1)
        inference_args.vae_scale_factor = vae_scale_factor

        self.image_processor = VaeImageProcessor(
            vae_scale_factor=vae_scale_factor)
        self.add_module("image_processor", self.image_processor)

        num_channels_latents = self.get_module("transformer").in_channels
        inference_args.num_channels_latents = num_channels_latents

    # TODO
    def adjust_video_length(self, batch: ForwardBatch,
                            inference_args: InferenceArgs):
        """Adjust video length based on VAE version"""
        video_length = batch.num_frames
        batch.num_frames = (video_length - 1) // 4 + 1
        return batch

    @torch.no_grad()
    def forward(self, batch: ForwardBatch,
                inference_args: InferenceArgs) -> ForwardBatch:
        logger.info(
            f"Running pipeline stages: {self._stage_name_mapping.keys()}")
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


EntryClass = HunyuanVideoPipeline
