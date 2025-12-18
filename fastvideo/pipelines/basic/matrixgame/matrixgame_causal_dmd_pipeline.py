# SPDX-License-Identifier: Apache-2.0
"""Matrix-Game causal DMD pipeline implementation."""

from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.logger import init_logger
from fastvideo.pipelines import ComposedPipelineBase, LoRAPipeline

from fastvideo.pipelines.stages import (ConditioningStage, DecodingStage,
                                        InputValidationStage,
                                        LatentPreparationStage,
                                        TextEncodingStage,
                                        MatrixGameImageEncodingStage,
                                        MatrixGameCausalDenoisingStage)
from fastvideo.pipelines.stages.image_encoding import (
    MatrixGameImageVAEEncodingStage)

logger = init_logger(__name__)


class MatrixGameCausalDMDPipeline(LoRAPipeline, ComposedPipelineBase):
    _required_config_modules = [
        "vae", "transformer", "scheduler", "image_encoder", "image_processor"
    ]

    def create_pipeline_stages(self, fastvideo_args: FastVideoArgs) -> None:
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

        self.add_stage(stage_name="latent_preparation_stage",
                       stage=LatentPreparationStage(
                           scheduler=self.get_module("scheduler"),
                           transformer=self.get_module("transformer", None)))

        self.add_stage(
            stage_name="image_latent_preparation_stage",
            stage=MatrixGameImageVAEEncodingStage(vae=self.get_module("vae")))

        self.add_stage(stage_name="denoising_stage",
                       stage=MatrixGameCausalDenoisingStage(
                           transformer=self.get_module("transformer"),
                           transformer_2=self.get_module("transformer_2", None),
                           scheduler=self.get_module("scheduler"),
                           pipeline=self,
                           vae=self.get_module("vae")))

        self.add_stage(stage_name="decoding_stage",
                       stage=DecodingStage(vae=self.get_module("vae")))

        logger.info(
            "MatrixGameCausalDMDPipeline initialized with action support")


EntryClass = [MatrixGameCausalDMDPipeline]
