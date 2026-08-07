# SPDX-License-Identifier: Apache-2.0
"""Causal Self-Forcing WanTrack I2V pipeline."""

from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.models.schedulers.scheduling_flow_match_euler_discrete import (
    FlowMatchEulerDiscreteScheduler, )
from fastvideo.pipelines import ComposedPipelineBase, LoRAPipeline
from fastvideo.pipelines.stages import (
    ConditioningStage,
    DecodingStage,
    ImageEncodingStage,
    ImageVAEEncodingStage,
    InputValidationStage,
    LatentPreparationStage,
    TextEncodingStage,
)
from fastvideo.pipelines.stages.wantrack_causal_denoising import (
    WanTrackCausalDenoisingStage, )


class WanTrackCausalDMDPipeline(LoRAPipeline, ComposedPipelineBase):
    """Wan I2V encode stages + causal TrackWan DMD denoising."""

    _required_config_modules = [
        "text_encoder",
        "tokenizer",
        "vae",
        "transformer",
        "scheduler",
        "image_encoder",
        "image_processor",
    ]

    def initialize_pipeline(self, fastvideo_args: FastVideoArgs) -> None:
        self.modules["scheduler"] = FlowMatchEulerDiscreteScheduler(
            shift=fastvideo_args.pipeline_config.flow_shift)

    def create_pipeline_stages(self, fastvideo_args: FastVideoArgs) -> None:
        self.add_stage(stage_name="input_validation_stage", stage=InputValidationStage())
        self.add_stage(
            stage_name="prompt_encoding_stage",
            stage=TextEncodingStage(
                text_encoders=[self.get_module("text_encoder")],
                tokenizers=[self.get_module("tokenizer")],
            ),
        )
        self.add_stage(
            stage_name="image_encoding_stage",
            stage=ImageEncodingStage(
                image_encoder=self.get_module("image_encoder"),
                image_processor=self.get_module("image_processor"),
            ),
        )
        self.add_stage(stage_name="conditioning_stage", stage=ConditioningStage())
        self.add_stage(
            stage_name="latent_preparation_stage",
            stage=LatentPreparationStage(
                scheduler=self.get_module("scheduler"),
                transformer=self.get_module("transformer", None),
            ),
        )
        self.add_stage(
            stage_name="image_latent_preparation_stage",
            stage=ImageVAEEncodingStage(vae=self.get_module("vae")),
        )
        self.add_stage(
            stage_name="denoising_stage",
            stage=WanTrackCausalDenoisingStage(
                transformer=self.get_module("transformer"),
                scheduler=self.get_module("scheduler"),
                vae=self.get_module("vae"),
            ),
        )
        self.add_stage(stage_name="decoding_stage", stage=DecodingStage(vae=self.get_module("vae")))


EntryClass = WanTrackCausalDMDPipeline
