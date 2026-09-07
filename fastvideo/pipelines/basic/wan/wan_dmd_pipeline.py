# SPDX-License-Identifier: Apache-2.0
"""
Wan video diffusion pipeline implementation.

This module contains an implementation of the Wan video diffusion pipeline
using the modular pipeline architecture.
"""

from fastvideo.pipelines.basic.wan.stages.dmd import DmdDenoisingStage
from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.logger import init_logger
from fastvideo.models.wan.definition import DMD_TRAINING_NOISE_SHIFT
from fastvideo.models.schedulers.scheduling_flow_match_euler_discrete import (FlowMatchEulerDiscreteScheduler)
from fastvideo.pipelines import ComposedPipelineBase, LoRAPipeline

# isort: off
from fastvideo.pipelines.stages import (ConditioningStage, DecodingStage, InputValidationStage, LatentPreparationStage,
                                        TextEncodingStage, TimestepPreparationStage)
# isort: on

logger = init_logger(__name__)


class WanDMDPipeline(LoRAPipeline, ComposedPipelineBase):
    """
    Wan video diffusion pipeline with LoRA support.
    """

    _required_config_modules = ["text_encoder", "tokenizer", "vae", "transformer", "scheduler"]

    def initialize_pipeline(self, fastvideo_args: FastVideoArgs):

        self.modules["scheduler"] = FlowMatchEulerDiscreteScheduler(shift=fastvideo_args.pipeline_config.flow_shift)

    def create_pipeline_stages(self, fastvideo_args: FastVideoArgs) -> None:
        """Set up pipeline stages with proper dependency injection."""

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
                       stage=LatentPreparationStage(scheduler=self.get_module("scheduler"),
                                                    transformer=self.get_module("transformer", None),
                                                    use_btchw_layout=True))

        # DMD needs the complete training-noise table, separate from the
        # inference scheduler mutated by TimestepPreparationStage.
        self.add_stage(stage_name="denoising_stage",
                       stage=DmdDenoisingStage(
                           transformer=self.get_module("transformer"),
                           scheduler=FlowMatchEulerDiscreteScheduler(shift=DMD_TRAINING_NOISE_SHIFT)))

        self.add_stage(stage_name="decoding_stage", stage=DecodingStage(vae=self.get_module("vae")))


EntryClass = WanDMDPipeline
