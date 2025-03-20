"""
HunYuan video diffusion pipeline implementation.

This module contains an implementation of the HunYuan video diffusion pipeline
using the modular pipeline architecture.
"""

from typing import Union, Any, Dict
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
    PostProcessingStage,
)
from fastvideo.v1.pipelines.stages.prompt_encoding import PromptEncodingStage
# from fastvideo.v1.pipelines.stages.timestep_preparation import FlowMatchingTimestepPreparationStage
from fastvideo.v1.inference_args import InferenceArgs
# from fastvideo.v1.pipelines.composed.composed_pipeline_base import DiffusionPipelineOutput
from fastvideo.v1.pipelines.pipeline_batch_info import ForwardBatch
# TODO(will): move PRECISION_TO_TYPE to better place
from .constants import PROMPT_TEMPLATE, PRECISION_TO_TYPE

from diffusers.utils import BaseOutput
import numpy as np
from dataclasses import dataclass

from fastvideo.v1.logger import init_logger
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


class HunyuanVideoPipeline(ComposedPipelineBase):

    def initialize_encoders(self, modules: Dict[str, Any], inference_args: InferenceArgs):
        self.initialize_encoders_v1(modules, inference_args)


    # TODO(will): the user API for this functionality needs to be much cleaner
    # and simpler. Currently, the user needs to manually pop the encoders/tokenizers
    # and replace them with the TextEncoder or ImageEncoder.
    def initialize_encoders_v1(self, modules: Dict[str, Any], inference_args:
    InferenceArgs):
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

        encoder_1 = modules.pop("text_encoder")
        assert encoder_1 is not None, "Text encoder is not found"
        encoder_1.to(inference_args.device)
        encoder_1.to(dtype=PRECISION_TO_TYPE[inference_args.text_encoder_precision])
        encoder_1.requires_grad_(False)

        tokenizer_1 = modules.pop("tokenizer")
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
            device=inference_args.device if not inference_args.use_cpu_offload else "cpu",
        )

        encoder_2 = modules.pop("text_encoder_2")
        assert encoder_2 is not None, "Text encoder 2 is not found"
        encoder_2.to(inference_args.device)
        encoder_2.to(dtype=PRECISION_TO_TYPE[inference_args.text_encoder_precision])
        encoder_2.requires_grad_(False)

        tokenizer_2 = modules.pop("tokenizer_2")
        assert tokenizer_2 is not None, "Tokenizer 2 is not found"

        text_encoder_2 = TextEncoder(
            text_encoder=encoder_2,
            tokenizer=tokenizer_2,
            # text_encoder_type="text_encoder_2",
            max_length=inference_args.text_len_2,
            # text_encoder_precision=inference_args.text_encoder_precision,
            device=inference_args.device if not inference_args.use_cpu_offload else "cpu",
        )
        modules["text_encoder"] = text_encoder
        modules["text_encoder_2"] = text_encoder_2


    def setup_pipeline(self, inference_args: InferenceArgs):
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


        vae_scale_factor = 2**(len(self.vae.block_out_channels) - 1)
        inference_args.vae_scale_factor = vae_scale_factor

        self.image_processor = VaeImageProcessor(vae_scale_factor=vae_scale_factor)
        self.register_modules({"image_processor": self.image_processor})


        num_channels_latents = self.transformer.in_channels
        inference_args.num_channels_latents = num_channels_latents


    # TODO
    def adjust_video_length(self, batch: ForwardBatch, inference_args: InferenceArgs):
        """Adjust video length based on VAE version"""
        video_length = batch.num_frames
        batch.num_frames = (video_length - 1) // 4 + 1
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

EntryClass = HunyuanVideoPipeline