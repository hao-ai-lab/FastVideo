# SPDX-License-Identifier: Apache-2.0
"""Matrix-Game 3.0 I2V pipeline implementation."""

from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.logger import init_logger
from fastvideo.models.schedulers.scheduling_flow_unipc_multistep import FlowUniPCMultistepScheduler
from fastvideo.pipelines.composed_pipeline_base import ComposedPipelineBase
from fastvideo.pipelines.lora_pipeline import LoRAPipeline
from fastvideo.pipelines.stages import (ConditioningStage, DecodingStage, InputValidationStage, LatentPreparationStage,
                                        MatrixGame3DenoisingStage, TextEncodingStage)
from fastvideo.pipelines.stages.image_encoding import MatrixGame3ImageVAEEncodingStage

logger = init_logger(__name__)


class MatrixGame3I2VPipeline(LoRAPipeline, ComposedPipelineBase):
    _required_config_modules = ["vae", "transformer", "scheduler", "text_encoder", "tokenizer"]

    def initialize_pipeline(self, fastvideo_args: FastVideoArgs):
        self.modules["scheduler"] = FlowUniPCMultistepScheduler(shift=fastvideo_args.pipeline_config.flow_shift)

    def create_pipeline_stages(self, fastvideo_args: FastVideoArgs):
        self.add_stage(stage_name="input_validation_stage", stage=InputValidationStage())

        self.add_stage(stage_name="prompt_encoding_stage",
                       stage=TextEncodingStage(
                           text_encoders=[self.get_module("text_encoder")],
                           tokenizers=[self.get_module("tokenizer")],
                       ))

        self.add_stage(stage_name="conditioning_stage", stage=ConditioningStage())

        self.add_stage(stage_name="latent_preparation_stage",
                       stage=LatentPreparationStage(scheduler=self.get_module("scheduler"),
                                                    transformer=self.get_module("transformer")))

        self.add_stage(stage_name="image_latent_preparation_stage",
                       stage=MatrixGame3ImageVAEEncodingStage(vae=self.get_module("vae")))

        self.add_stage(stage_name="denoising_stage",
                       stage=MatrixGame3DenoisingStage(transformer=self.get_module("transformer"),
                                                       scheduler=self.get_module("scheduler"),
                                                       pipeline=self,
                                                       vae=self.get_module("vae")))

        self.add_stage(stage_name="decoding_stage", stage=DecodingStage(vae=self.get_module("vae")))

        logger.info("MatrixGame3I2VPipeline initialized with text, action, and camera support")


EntryClass = [MatrixGame3I2VPipeline]
