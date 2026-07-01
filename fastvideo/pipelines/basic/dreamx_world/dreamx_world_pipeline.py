# SPDX-License-Identifier: Apache-2.0
"""DreamX-World video pipeline entrypoint."""

from fastvideo.api.sampling_param import SamplingParam
from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.logger import init_logger
from fastvideo.models.schedulers.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler
from fastvideo.pipelines import ComposedPipelineBase, LoRAPipeline
from fastvideo.configs.pipelines.dreamx_world import DreamXWorld5BCamPipelineConfig
from fastvideo.pipelines.basic.dreamx_world.stages import DreamXWorldCameraConditioningStage
from fastvideo.pipelines.stages import (
    ConditioningStage,
    DecodingStage,
    DenoisingStage,
    InputValidationStage,
    LatentPreparationStage,
    TextEncodingStage,
    TimestepPreparationStage,
)

logger = init_logger(__name__)


class DreamXWorldPipeline(LoRAPipeline, ComposedPipelineBase):
    """DreamX-World-5B-Cam pipeline with native FastVideo camera conditioning."""

    _required_config_modules = ["text_encoder", "tokenizer", "vae", "transformer", "scheduler"]
    pipeline_config_cls = DreamXWorld5BCamPipelineConfig
    sampling_params_cls = SamplingParam

    def initialize_pipeline(self, fastvideo_args: FastVideoArgs):
        self.modules["scheduler"] = FlowMatchEulerDiscreteScheduler(shift=fastvideo_args.pipeline_config.flow_shift)

    def create_pipeline_stages(self, fastvideo_args: FastVideoArgs) -> None:
        self.add_stage(stage_name="input_validation_stage", stage=InputValidationStage())

        self.add_stage(
            stage_name="prompt_encoding_stage",
            stage=TextEncodingStage(
                text_encoders=[self.get_module("text_encoder")],
                tokenizers=[self.get_module("tokenizer")],
            ),
        )

        self.add_stage(stage_name="conditioning_stage", stage=ConditioningStage())

        self.add_stage(
            stage_name="timestep_preparation_stage",
            stage=TimestepPreparationStage(scheduler=self.get_module("scheduler")),
        )

        self.add_stage(
            stage_name="latent_preparation_stage",
            stage=LatentPreparationStage(
                scheduler=self.get_module("scheduler"),
                transformer=self.get_module("transformer", None),
            ),
        )

        self.add_stage(stage_name="dreamx_camera_conditioning_stage", stage=DreamXWorldCameraConditioningStage())

        self.add_stage(
            stage_name="denoising_stage",
            stage=DenoisingStage(
                transformer=self.get_module("transformer"),
                transformer_2=self.get_module("transformer_2", None),
                scheduler=self.get_module("scheduler"),
                vae=self.get_module("vae"),
                pipeline=self,
            ),
        )

        self.add_stage(stage_name="decoding_stage", stage=DecodingStage(vae=self.get_module("vae"), pipeline=self))

        logger.info("DreamXWorldPipeline initialized with native camera conditioning")


EntryClass = DreamXWorldPipeline
