# SPDX-License-Identifier: Apache-2.0
"""Kandinsky5 DMD (few-step distilled) validation pipeline.

Composes the same stages as ``Kandinsky5T2VPipeline`` (text encoding, latent
prep, decoding) but swaps in ``Kandinsky5DmdDenoisingStage`` for the
denoising loop, mirroring ``wan_dmd_pipeline.py``'s split from the base T2V
pipeline. This exists solely to drive the stage-2 DMD training config's
validation callback.
"""

from __future__ import annotations

from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.pipelines.composed_pipeline_base import ComposedPipelineBase
from fastvideo.pipelines.stages.input_validation import InputValidationStage
from fastvideo.pipelines.stages.kandinsky5 import (
    Kandinsky5DecodingStage,
    Kandinsky5DmdDenoisingStage,
    Kandinsky5LatentPreparationStage,
)
from fastvideo.pipelines.stages.text_encoding import TextEncodingStage
from fastvideo.pipelines.stages.timestep_preparation import TimestepPreparationStage


class Kandinsky5DMDPipeline(ComposedPipelineBase):
    """Kandinsky-5.0 Lite DMD (few-step) text-to-video pipeline."""

    _required_config_modules = [
        "scheduler",
        "text_encoder",
        "text_encoder_2",
        "tokenizer",
        "tokenizer_2",
        "transformer",
        "vae",
    ]

    def create_pipeline_stages(self, fastvideo_args: FastVideoArgs) -> None:
        self.add_stage(stage_name="input_validation_stage", stage=InputValidationStage())

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

        self.add_stage(
            stage_name="timestep_preparation_stage",
            stage=TimestepPreparationStage(scheduler=self.get_module("scheduler")),
        )

        self.add_stage(
            stage_name="latent_preparation_stage",
            stage=Kandinsky5LatentPreparationStage(
                scheduler=self.get_module("scheduler"),
                transformer=self.get_module("transformer"),
            ),
        )

        self.add_stage(
            stage_name="denoising_stage",
            stage=Kandinsky5DmdDenoisingStage(
                transformer=self.get_module("transformer"),
                scheduler=self.get_module("scheduler"),
            ),
        )

        self.add_stage(
            stage_name="decoding_stage",
            stage=Kandinsky5DecodingStage(vae=self.get_module("vae"), pipeline=self),
        )


EntryClass = Kandinsky5DMDPipeline
