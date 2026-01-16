# SPDX-License-Identifier: Apache-2.0
"""
GLM-Image diffusion pipeline implementation.
"""

from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.logger import init_logger
from fastvideo.models.schedulers.scheduling_flow_match_euler_discrete import (
    FlowMatchEulerDiscreteScheduler, )
from fastvideo.pipelines import ComposedPipelineBase, LoRAPipeline
from fastvideo.pipelines.stages import InputValidationStage
from fastvideo.pipelines.stages.glm_image_before_denoising import (
    GlmImageBeforeDenoisingStage, )
from fastvideo.pipelines.stages.glm_image_denoising import GlmImageDenoisingStage
from fastvideo.pipelines.stages.glm_image_decoding import GlmImageDecodingStage

logger = init_logger(__name__)


class GlmImagePipeline(LoRAPipeline, ComposedPipelineBase):
    pipeline_name = "GlmImagePipeline"

    _required_config_modules = [
        "text_encoder",
        "tokenizer",
        "vae",
        "transformer",
        "scheduler",
        "vision_language_encoder",  # AR model for prior tokens - critical for quality
        "processor",  # Processor for AR model
    ]

    _optional_config_modules = []

    def initialize_pipeline(self, fastvideo_args: FastVideoArgs) -> None:
        # Initial shift, will be updated dynamically in BeforeDenoisingStage
        self.modules["scheduler"] = FlowMatchEulerDiscreteScheduler(shift=1.0)

        # Validate configuration if provided
        if fastvideo_args.pipeline_config is not None:
            # Basic validation or usage if needed, otherwise just pass
            pass

    def create_pipeline_stages(self, fastvideo_args: FastVideoArgs) -> None:
        self.add_stage(
            stage_name="input_validation_stage",
            stage=InputValidationStage(),
        )

        self.add_stage(
            stage_name="glm_image_before_denoising_stage",
            stage=GlmImageBeforeDenoisingStage(
                vae=self.get_module("vae"),
                text_encoder=self.get_module("text_encoder"),
                tokenizer=self.get_module("tokenizer"),
                processor=self.get_module("processor"),
                transformer=self.get_module("transformer"),
                scheduler=self.get_module("scheduler"),
                vision_language_encoder=self.get_module(
                    "vision_language_encoder"),
            ),
        )

        self.add_stage(
            stage_name="denoising_stage",
            stage=GlmImageDenoisingStage(
                transformer=self.get_module("transformer"),
                scheduler=self.get_module("scheduler"),
                vae=self.get_module("vae"),
                pipeline=self,
            ),
        )

        self.add_stage(
            stage_name="decoding_stage",
            stage=GlmImageDecodingStage(
                vae=self.get_module("vae"),
                pipeline=self,
            ),
        )


EntryClass = GlmImagePipeline
