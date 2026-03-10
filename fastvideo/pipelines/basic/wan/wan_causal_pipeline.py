# SPDX-License-Identifier: Apache-2.0
"""Wan causal pipeline with standard multi-step denoising.

Block-by-block causal inference with KV caching, using the full
scheduler timestep schedule (40-50 steps) rather than DMD few-step.
"""

from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.logger import init_logger
from fastvideo.models.schedulers.scheduling_flow_unipc_multistep import (
    FlowUniPCMultistepScheduler)
from fastvideo.pipelines import ComposedPipelineBase, LoRAPipeline

# isort: off
from fastvideo.pipelines.stages import (
    CausalDenoisingStage,
    ConditioningStage,
    DecodingStage,
    InputValidationStage,
    LatentPreparationStage,
    TextEncodingStage,
    TimestepPreparationStage,
)
# isort: on

logger = init_logger(__name__)


class WanCausalPipeline(LoRAPipeline, ComposedPipelineBase):
    """Wan causal pipeline with standard multi-step denoising."""

    _required_config_modules = [
        "text_encoder",
        "tokenizer",
        "vae",
        "transformer",
        "scheduler",
    ]

    def initialize_pipeline(
        self,
        fastvideo_args: FastVideoArgs,
    ):
        self.modules["scheduler"] = (FlowUniPCMultistepScheduler(
            shift=fastvideo_args.pipeline_config.flow_shift, ))

    def create_pipeline_stages(
        self,
        fastvideo_args: FastVideoArgs,
    ) -> None:
        self.add_stage(
            stage_name="input_validation_stage",
            stage=InputValidationStage(),
        )

        self.add_stage(
            stage_name="prompt_encoding_stage",
            stage=TextEncodingStage(
                text_encoders=[self.get_module("text_encoder")],
                tokenizers=[self.get_module("tokenizer")],
            ),
        )

        self.add_stage(
            stage_name="conditioning_stage",
            stage=ConditioningStage(),
        )

        self.add_stage(
            stage_name="timestep_preparation_stage",
            stage=TimestepPreparationStage(
                scheduler=self.get_module("scheduler"), ),
        )

        self.add_stage(
            stage_name="latent_preparation_stage",
            stage=LatentPreparationStage(
                scheduler=self.get_module("scheduler"),
                transformer=self.get_module("transformer", None),
            ),
        )

        self.add_stage(
            stage_name="denoising_stage",
            stage=CausalDenoisingStage(
                transformer=self.get_module("transformer"),
                scheduler=self.get_module("scheduler"),
            ),
        )

        self.add_stage(
            stage_name="decoding_stage",
            stage=DecodingStage(vae=self.get_module("vae")),
        )


EntryClass = WanCausalPipeline
