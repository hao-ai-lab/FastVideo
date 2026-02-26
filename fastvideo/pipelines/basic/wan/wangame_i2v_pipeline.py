# SPDX-License-Identifier: Apache-2.0
"""WanGame image-to-video pipeline implementation.

This module contains an implementation of the WanGame image-to-video pipeline
using the modular pipeline architecture.
"""

from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.logger import init_logger
from fastvideo.pipelines.composed_pipeline_base import ComposedPipelineBase
from fastvideo.pipelines.lora_pipeline import LoRAPipeline
from fastvideo.pipelines.samplers.wan import (
    build_wan_scheduler,
    get_wan_sampler_kind,
    wan_use_btchw_layout,
)

# isort: off
from fastvideo.pipelines.stages import (
    ConditioningStage,
    DecodingStage,
    DenoisingStage,
    ImageEncodingStage,
    ImageVAEEncodingStage,
    InputValidationStage,
    LatentPreparationStage,
    SdeDenoisingStage,
    TimestepPreparationStage,
)

# isort: on

logger = init_logger(__name__)


class WanGameActionImageToVideoPipeline(LoRAPipeline, ComposedPipelineBase):

    _required_config_modules = [
        "vae",
        "transformer",
        "scheduler",
        "image_encoder",
        "image_processor",
    ]

    def initialize_pipeline(self, fastvideo_args: FastVideoArgs):
        sampler_kind = get_wan_sampler_kind(fastvideo_args)
        self.modules["scheduler"] = build_wan_scheduler(fastvideo_args, sampler_kind)

    def create_pipeline_stages(self, fastvideo_args: FastVideoArgs):
        """Set up pipeline stages with proper dependency injection."""

        sampler_kind = get_wan_sampler_kind(fastvideo_args)
        use_btchw_layout = wan_use_btchw_layout(sampler_kind)

        self.add_stage(stage_name="input_validation_stage",
                       stage=InputValidationStage())

        self.add_stage(
            stage_name="image_encoding_stage",
            stage=ImageEncodingStage(
                image_encoder=self.get_module("image_encoder"),
                image_processor=self.get_module("image_processor"),
            ),
        )

        self.add_stage(stage_name="conditioning_stage",
                       stage=ConditioningStage())

        if sampler_kind == "ode":
            self.add_stage(stage_name="timestep_preparation_stage",
                           stage=TimestepPreparationStage(
                               scheduler=self.get_module("scheduler")))

        self.add_stage(stage_name="latent_preparation_stage",
                       stage=LatentPreparationStage(
                           scheduler=self.get_module("scheduler"),
                           transformer=self.get_module("transformer"),
                           use_btchw_layout=use_btchw_layout))

        self.add_stage(stage_name="image_latent_preparation_stage",
                       stage=ImageVAEEncodingStage(vae=self.get_module("vae")))

        if sampler_kind == "sde":
            self.add_stage(stage_name="denoising_stage",
                           stage=SdeDenoisingStage(
                               transformer=self.get_module("transformer"),
                               scheduler=self.get_module("scheduler")))
        else:
            self.add_stage(stage_name="denoising_stage",
                           stage=DenoisingStage(
                               transformer=self.get_module("transformer"),
                               scheduler=self.get_module("scheduler")))

        self.add_stage(stage_name="decoding_stage",
                       stage=DecodingStage(vae=self.get_module("vae")))


class WanLingBotImageToVideoPipeline(WanGameActionImageToVideoPipeline):
    pass


EntryClass = [WanGameActionImageToVideoPipeline, WanLingBotImageToVideoPipeline]
