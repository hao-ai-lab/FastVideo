# SPDX-License-Identifier: Apache-2.0
"""
Wan causal DMD pipeline implementation.

This module wires the causal DMD denoising stage into the modular pipeline.
"""

from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.logger import init_logger
from fastvideo.pipelines import ComposedPipelineBase, LoRAPipeline

# isort: off
from fastvideo.pipelines.stages import (ConditioningStage, DecodingStage,
                                        CausalDMDDenosingStage,
                                        InputValidationStage,
                                        TimestepPreparationStage,
                                        Hy15ImageEncodingStage,
                                        LatentPreparationStage,
                                        TextEncodingStage)
# isort: on

logger = init_logger(__name__)


class Hy15CausalDMDPipeline(LoRAPipeline, ComposedPipelineBase):

    _required_config_modules = [
        "text_encoder", "text_encoder_2", "tokenizer", "tokenizer_2", "vae",
        "transformer", "scheduler"
    ]

    def create_pipeline_stages(self, fastvideo_args: FastVideoArgs) -> None:
        """Set up pipeline stages with proper dependency injection."""

        self.add_stage(stage_name="input_validation_stage",
                       stage=InputValidationStage())

        self.add_stage(stage_name="prompt_encoding_stage",
                       stage=TextEncodingStage(
                           text_encoders=[
                               self.get_module("text_encoder"),
                               self.get_module("text_encoder_2")
                           ],
                           tokenizers=[
                               self.get_module("tokenizer"),
                               self.get_module("tokenizer_2")
                           ],
                       ))

        self.add_stage(stage_name="conditioning_stage",
                       stage=ConditioningStage())

        # self.add_stage(stage_name="timestep_preparation_stage",
        #                stage=TimestepPreparationStage(
        #                    scheduler=self.get_module("scheduler")))

        self.add_stage(stage_name="latent_preparation_stage",
                       stage=LatentPreparationStage(
                           scheduler=self.get_module("scheduler"),
                           transformer=self.get_module("transformer", None)))

        self.add_stage(stage_name="image_encoding_stage",
                        stage=Hy15ImageEncodingStage(image_encoder=None,
                                                    image_processor=None))

        self.add_stage(stage_name="denoising_stage",
                       stage=CausalDMDDenosingStage(
                           transformer=self.get_module("transformer"),
                           scheduler=self.get_module("scheduler"),
                           vae=self.get_module("vae")))

        self.add_stage(stage_name="decoding_stage",
                       stage=DecodingStage(vae=self.get_module("vae")))


EntryClass = Hy15CausalDMDPipeline
