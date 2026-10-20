# SPDX-License-Identifier: Apache-2.0
"""FastVideo pipeline for Helios-Distilled text-to-video generation."""

from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.pipelines import ComposedPipelineBase, LoRAPipeline
from fastvideo.pipelines.basic.helios.stages import (
    HeliosChunkDecodingStage,
    HeliosInputValidationStage,
    HeliosPyramidDenoisingStage,
)
from fastvideo.pipelines.stages import ConditioningStage, TextEncodingStage


class HeliosPyramidPipeline(LoRAPipeline, ComposedPipelineBase):
    """Autoregressive three-level spatial-pyramid pipeline for Helios."""

    _required_config_modules = [
        "text_encoder",
        "tokenizer",
        "vae",
        "transformer",
        "scheduler",
    ]

    def create_pipeline_stages(self, fastvideo_args: FastVideoArgs) -> None:
        self.add_stage(
            stage_name="input_validation_stage",
            stage=HeliosInputValidationStage(),
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
            stage_name="pyramid_denoising_stage",
            stage=HeliosPyramidDenoisingStage(
                transformer=self.get_module("transformer"),
                scheduler=self.get_module("scheduler"),
                pipeline=self,
            ),
        )
        self.add_stage(
            stage_name="chunk_decoding_stage",
            stage=HeliosChunkDecodingStage(
                vae=self.get_module("vae"),
                pipeline=self,
            ),
        )


EntryClass = HeliosPyramidPipeline
