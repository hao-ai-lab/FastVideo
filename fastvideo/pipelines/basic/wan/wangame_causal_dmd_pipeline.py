# SPDX-License-Identifier: Apache-2.0
"""Wangame causal DMD pipeline implementations."""

from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.logger import init_logger
from fastvideo.pipelines import ComposedPipelineBase, LoRAPipeline

from fastvideo.pipelines.stages import (
    ConditioningStage, DecodingStage, MatrixGameCausalDenoisingStage,
    MatrixGameCausalOdeDenoisingStage, MatrixGameImageEncodingStage,
    InputValidationStage, LatentPreparationStage, TextEncodingStage,
    TimestepPreparationStage)
from fastvideo.pipelines.stages.image_encoding import (
    MatrixGameImageVAEEncodingStage)

logger = init_logger(__name__)


class _WangameCausalDMDPipelineBase(LoRAPipeline, ComposedPipelineBase):
    requires_timestep_preparation: bool
    requires_dmd_denoising_steps: bool
    denoising_stage_cls: type[MatrixGameCausalDenoisingStage]

    _required_config_modules = [
        "vae", "transformer", "scheduler", "image_encoder", "image_processor"
    ]

    def create_pipeline_stages(self, fastvideo_args: FastVideoArgs) -> None:
        del fastvideo_args
        self.add_stage(stage_name="input_validation_stage",
                       stage=InputValidationStage())

        if (self.get_module("text_encoder", None) is not None
                and self.get_module("tokenizer", None) is not None):
            self.add_stage(stage_name="prompt_encoding_stage",
                           stage=TextEncodingStage(
                               text_encoders=[self.get_module("text_encoder")],
                               tokenizers=[self.get_module("tokenizer")],
                           ))

        if (self.get_module("image_encoder", None) is not None
                and self.get_module("image_processor", None) is not None):
            self.add_stage(
                stage_name="image_encoding_stage",
                stage=MatrixGameImageEncodingStage(
                    image_encoder=self.get_module("image_encoder"),
                    image_processor=self.get_module("image_processor"),
                ))

        self.add_stage(stage_name="conditioning_stage",
                       stage=ConditioningStage())

        if self.requires_timestep_preparation:
            self.add_stage(stage_name="timestep_preparation_stage",
                           stage=TimestepPreparationStage(
                               scheduler=self.get_module("scheduler")))

        self.add_stage(stage_name="latent_preparation_stage",
                       stage=LatentPreparationStage(
                           scheduler=self.get_module("scheduler"),
                           transformer=self.get_module("transformer", None)))

        self.add_stage(
            stage_name="image_latent_preparation_stage",
            stage=MatrixGameImageVAEEncodingStage(vae=self.get_module("vae")))

        denoising_stage = self.denoising_stage_cls(
            transformer=self.get_module("transformer"),
            transformer_2=self.get_module("transformer_2", None),
            scheduler=self.get_module("scheduler"),
            pipeline=self,
            vae=self.get_module("vae"),
        )

        self.add_stage(stage_name="denoising_stage", stage=denoising_stage)

        self.add_stage(stage_name="decoding_stage",
                       stage=DecodingStage(vae=self.get_module("vae")))

        logger.info("%s initialized with action support",
                    self.__class__.__name__)

class WangameCausalOdeDMDPipeline(_WangameCausalDMDPipelineBase):
    requires_timestep_preparation = True
    requires_dmd_denoising_steps = False
    denoising_stage_cls = MatrixGameCausalOdeDenoisingStage


class WangameCausalSdeDMDPipeline(_WangameCausalDMDPipelineBase):
    requires_timestep_preparation = False
    requires_dmd_denoising_steps = True
    denoising_stage_cls = MatrixGameCausalDenoisingStage

EntryClass = [WangameCausalOdeDMDPipeline, WangameCausalSdeDMDPipeline]
