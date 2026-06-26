# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.pipelines.composed_pipeline_base import ComposedPipelineBase
from fastvideo.pipelines.stages.input_validation import InputValidationStage
from fastvideo.pipelines.stages.kandinsky5 import (
    Kandinsky5DecodingStage,
    Kandinsky5DenoisingStage,
    Kandinsky5LatentPreparationStage,
)
from fastvideo.pipelines.stages.text_encoding import TextEncodingStage
from fastvideo.pipelines.stages.timestep_preparation import TimestepPreparationStage


class Kandinsky5I2VPipeline(ComposedPipelineBase):
    """Kandinsky-5.0 image-to-video pipeline."""

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
        raise NotImplementedError("Class to be implemented yet")
        
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
        
        self.add_stage(stage_name="image_encoding_stage",
                       stage=Kandinsky5ImageEncodingStage(vae=self.get_module("vae")))


        self.add_stage(
            stage_name="latent_preparation_stage",
            stage=Kandinsky5LatentPreparationStage(
                scheduler=self.get_module("scheduler"),
                transformer=self.get_module("transformer"),
            ),
        )
        
        self.add_stage(
            stage_name="decoding_stage",
            stage=Kandinsky5DecodingStage(vae=self.get_module("vae"), pipeline=self),
        )

        self.add_stage(
            stage_name="denoising_stage",
            stage=Kandinsky5DenoisingStage(
                transformer=self.get_module("transformer"),
                scheduler=self.get_module("scheduler"),
            ),
        )

        self.add_stage(
            stage_name="decoding_stage",
            stage=Kandinsky5DecodingStage(vae=self.get_module("vae"), pipeline=self),
        )


EntryClass = Kandinsky5T2VPipeline
