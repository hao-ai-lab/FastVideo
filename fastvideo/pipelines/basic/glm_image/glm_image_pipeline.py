# SPDX-License-Identifier: Apache-2.0
from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.logger import init_logger
from fastvideo.models.schedulers.scheduling_flow_match_euler_discrete import (
    FlowMatchEulerDiscreteScheduler, )
from fastvideo.pipelines import ComposedPipelineBase, LoRAPipeline
from fastvideo.pipelines.basic.glm_image.stages import (
    GlmImageBeforeDenoisingStage,
    GlmImageConditionEncodingStage,
    GlmImageDecodingStage,
    GlmImageDenoisingStage,
)
from fastvideo.pipelines.stages import InputValidationStage

logger = init_logger(__name__)


class GlmImagePipeline(LoRAPipeline, ComposedPipelineBase):
    pipeline_name = "GlmImagePipeline"

    _required_config_modules = [
        "text_encoder",
        "tokenizer",
        "vae",
        "transformer",
        "scheduler",
        "vision_language_encoder",
        "processor",
    ]

    _optional_config_modules: list[str] = []

    def initialize_pipeline(self, fastvideo_args: FastVideoArgs) -> None:
        self.modules["scheduler"] = FlowMatchEulerDiscreteScheduler(shift=1.0)

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
                vision_language_encoder=self.get_module("vision_language_encoder"),
            ),
        )

        self.add_stage(
            stage_name="glm_image_condition_encoding_stage",
            stage=GlmImageConditionEncodingStage(
                vae=self.get_module("vae"),
                transformer=self.get_module("transformer"),
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
