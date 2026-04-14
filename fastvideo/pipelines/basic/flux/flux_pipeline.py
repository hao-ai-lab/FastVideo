# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.pipelines.composed_pipeline_base import ComposedPipelineBase
from fastvideo.pipelines.stages.flux_stages import (
    FluxConditioningStage,
    FluxDecodingStage,
    FluxDenoisingStage,
    FluxInputValidationStage,
    FluxLatentPreparationStage,
    FluxTimestepPreparationStage,
)
from fastvideo.pipelines.stages.text_encoding import TextEncodingStage


class FluxPipeline(ComposedPipelineBase):
    """FLUX.1-dev T2I (Diffusers module layout, packed latents, embedded guidance)."""

    _required_config_modules = [
        "scheduler",
        "transformer",
        "vae",
        "text_encoder",
        "text_encoder_2",
        "tokenizer",
        "tokenizer_2",
    ]

    def create_pipeline_stages(self, fastvideo_args: FastVideoArgs) -> None:
        self.add_stage(stage_name="input_validation_stage", stage=FluxInputValidationStage())

        self.add_stage(
            stage_name="text_encoding_stage",
            stage=TextEncodingStage(
                text_encoders=[
                    self.get_module("text_encoder"),
                    self.get_module("text_encoder_2"),
                ],
                tokenizers=[
                    self.get_module("tokenizer"),
                    self.get_module("tokenizer_2"),
                ],
            ),
        )

        self.add_stage(stage_name="flux_conditioning_stage", stage=FluxConditioningStage())

        self.add_stage(
            stage_name="timestep_preparation_stage",
            stage=FluxTimestepPreparationStage(scheduler=self.get_module("scheduler")),
        )

        self.add_stage(
            stage_name="latent_preparation_stage",
            stage=FluxLatentPreparationStage(scheduler=self.get_module("scheduler")),
        )

        self.add_stage(
            stage_name="denoising_stage",
            stage=FluxDenoisingStage(
                transformer=self.get_module("transformer"),
                scheduler=self.get_module("scheduler"),
            ),
        )

        self.add_stage(
            stage_name="decoding_stage",
            stage=FluxDecodingStage(vae=self.get_module("vae")),
        )


EntryClass = FluxPipeline
