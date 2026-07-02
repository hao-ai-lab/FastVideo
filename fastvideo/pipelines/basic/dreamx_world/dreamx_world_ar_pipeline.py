# SPDX-License-Identifier: Apache-2.0
"""DreamX-World-5B autoregressive pipeline entrypoint."""

from fastvideo.api.sampling_param import SamplingParam
from fastvideo.configs.pipelines.dreamx_world import DreamXWorld5BARPipelineConfig
from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.logger import init_logger
from fastvideo.models.schedulers.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler
from fastvideo.pipelines import ComposedPipelineBase, LoRAPipeline
from fastvideo.pipelines.basic.dreamx_world.ar_denoising import DreamXWorldARCausalDenoisingStage
from fastvideo.pipelines.basic.dreamx_world.stages import DreamXWorldCameraConditioningStage
from fastvideo.pipelines.stages import (
    ConditioningStage,
    DecodingStage,
    InputValidationStage,
    LatentPreparationStage,
    TextEncodingStage,
    TimestepPreparationStage,
)

logger = init_logger(__name__)


class DreamXWorldARPipeline(LoRAPipeline, ComposedPipelineBase):
    """DreamX-World-5B autoregressive causal camera pipeline."""

    _required_config_modules = ["text_encoder", "tokenizer", "vae", "transformer", "scheduler"]
    pipeline_config_cls = DreamXWorld5BARPipelineConfig
    sampling_params_cls = SamplingParam

    def initialize_pipeline(self, fastvideo_args: FastVideoArgs):
        self.modules["scheduler"] = FlowMatchEulerDiscreteScheduler(shift=fastvideo_args.pipeline_config.flow_shift)
        self.modules["scheduler"].set_timesteps(1000)

    def create_pipeline_stages(self, fastvideo_args: FastVideoArgs) -> None:
        self.add_stage(stage_name="input_validation_stage", stage=InputValidationStage())
        self.add_stage(stage_name="prompt_encoding_stage",
                       stage=TextEncodingStage(
                           text_encoders=[self.get_module("text_encoder")],
                           tokenizers=[self.get_module("tokenizer")],
                       ))
        self.add_stage(stage_name="conditioning_stage", stage=ConditioningStage())
        self.add_stage(stage_name="timestep_preparation_stage",
                       stage=TimestepPreparationStage(scheduler=self.get_module("scheduler")))
        self.add_stage(stage_name="latent_preparation_stage",
                       stage=LatentPreparationStage(
                           scheduler=self.get_module("scheduler"),
                           transformer=self.get_module("transformer", None),
                       ))
        self.add_stage(stage_name="dreamx_camera_conditioning_stage", stage=DreamXWorldCameraConditioningStage())
        self.add_stage(stage_name="denoising_stage",
                       stage=DreamXWorldARCausalDenoisingStage(
                           transformer=self.get_module("transformer"),
                           scheduler=self.get_module("scheduler"),
                           vae=self.get_module("vae"),
                           pipeline=self,
                       ))
        self.add_stage(stage_name="decoding_stage", stage=DecodingStage(vae=self.get_module("vae"), pipeline=self))
        logger.info("DreamXWorldARPipeline initialized with autoregressive causal denoising")


EntryClass = DreamXWorldARPipeline
