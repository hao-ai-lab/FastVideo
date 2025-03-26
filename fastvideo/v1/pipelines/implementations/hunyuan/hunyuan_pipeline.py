"""
HunYuan video diffusion pipeline implementation.

This module contains an implementation of the HunYuan video diffusion pipeline
using the modular pipeline architecture.
"""

from typing import Union
import torch
from diffusers.image_processor import VaeImageProcessor

from fastvideo.v1.pipelines.composed import ComposedPipelineBase
from fastvideo.v1.pipelines.stages import (
    InputValidationStage,
    PromptEncodingStage,
    TimestepPreparationStage,
    LatentPreparationStage,
    ConditioningStage,
    DenoisingStage,
    DecodingStage,
)
from fastvideo.v1.pipelines.stages.prompt_encoding import PromptEncodingStage
from fastvideo.v1.inference_args import InferenceArgs
from fastvideo.v1.pipelines.pipeline_batch_info import ForwardBatch
# TODO(will): move PRECISION_TO_TYPE to better place
from .constants import PROMPT_TEMPLATE, PRECISION_TO_TYPE

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

    def initialize_encoders(self, inference_args: InferenceArgs):
        self.initialize_encoders_v1(inference_args)

    # TODO(will): the user API for this functionality needs to be much cleaner
    # and simpler. Currently, the user needs to manually pop the encoders/tokenizers
    # and replace them with the TextEncoder or ImageEncoder.
    def initialize_encoders_v1(self, inference_args: InferenceArgs):
        """
        Initialize the encoders. Will remove the encoders/tokenizers modules from the
        modules. Will add the TextEncoder or ImageEncoder to the modules.
        """
        from fastvideo.v1.models.text_encoder import TextEncoder

        crop_start = PROMPT_TEMPLATE["video"].get("crop_start", 0)

        max_length = inference_args.text_len + crop_start

        # prompt_template
        prompt_template = PROMPT_TEMPLATE["image"]

        # prompt_template_video
        prompt_template_video = PROMPT_TEMPLATE["video"]

        encoder_1 = self.modules.pop("text_encoder")
        assert encoder_1 is not None, "Text encoder is not found"
        encoder_1.to(inference_args.device)
        encoder_1.to(
            dtype=PRECISION_TO_TYPE[inference_args.text_encoder_precision])
        encoder_1.requires_grad_(False)

        print(f"keys: {self.modules.keys()}")
        tokenizer_1 = self.modules.pop("tokenizer")
        assert tokenizer_1 is not None, "Tokenizer is not found"

        text_encoder = TextEncoder(
            text_encoder=encoder_1,
            tokenizer=tokenizer_1,
            # text_encoder_type="text_encoder"
            max_length=max_length,
            # text_encoder_precision=inference_args.text_encoder_precision,
            prompt_template=prompt_template,
            prompt_template_video=prompt_template_video,
            hidden_state_skip_layer=inference_args.hidden_state_skip_layer,
            apply_final_norm=False,
            device=inference_args.device
            if not inference_args.use_cpu_offload else "cpu",
        )

        encoder_2 = self.modules.pop("text_encoder_2")
        assert encoder_2 is not None, "Text encoder 2 is not found"
        encoder_2.to(inference_args.device)
        encoder_2.to(
            dtype=PRECISION_TO_TYPE[inference_args.text_encoder_precision])
        encoder_2.requires_grad_(False)

        tokenizer_2 = self.modules.pop("tokenizer_2")
        assert tokenizer_2 is not None, "Tokenizer 2 is not found"

        text_encoder_2 = TextEncoder(
            text_encoder=encoder_2,
            tokenizer=tokenizer_2,
            # text_encoder_type="text_encoder_2",
            max_length=inference_args.text_len_2,
            # text_encoder_precision=inference_args.text_encoder_precision,
            device=inference_args.device
            if not inference_args.use_cpu_offload else "cpu",
        )
        self.modules["text_encoder"] = text_encoder
        self.modules["text_encoder_2"] = text_encoder_2

    def create_pipeline_stages(self, inference_args: InferenceArgs):
        """Set up pipeline stages with proper dependency injection."""

        self.add_stage(stage_name="input_validation_stage",
                       stage=InputValidationStage())

        self.add_stage(stage_name="prompt_encoding_stage_primary",
                       stage=PromptEncodingStage(
                           text_encoder=self.get_module("text_encoder"),
                           is_secondary=False))

        self.add_stage(stage_name="prompt_encoding_stage_secondary",
                       stage=PromptEncodingStage(
                           text_encoder=self.get_module("text_encoder_2"),
                           is_secondary=True))

        self.add_stage(stage_name="conditioning_stage",
                       stage=ConditioningStage())

        self.add_stage(stage_name="timestep_preparation_stage",
                       stage=TimestepPreparationStage(
                           scheduler=self.get_module("scheduler")))

        self.add_stage(stage_name="latent_preparation_stage",
                       stage=LatentPreparationStage(
                           scheduler=self.get_module("scheduler")))

        self.add_stage(stage_name="denoising_stage",
                       stage=DenoisingStage(
                           transformer=self.get_module("transformer"),
                           scheduler=self.get_module("scheduler")))

        self.add_stage(stage_name="decoding_stage",
                       stage=DecodingStage(vae=self.get_module("vae")))

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
