# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from fastvideo.configs.pipelines.zimage import ZImagePipelineConfig
from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.pipelines.composed_pipeline_base import ComposedPipelineBase
from fastvideo.pipelines.stages.text_encoding import TextEncodingStage

from .stages import (
    ZImageConditioningStage,
    ZImageDecodingStage,
    ZImageDenoisingStage,
    ZImageInputValidationStage,
    ZImageLatentPreparationStage,
    ZImageTimestepPreparationStage,
)


class ZImagePipeline(ComposedPipelineBase):
    """Native Z-Image text-to-image pipeline."""

    pipeline_config_cls: type[ZImagePipelineConfig] = ZImagePipelineConfig
    _required_config_modules = [
        "scheduler",
        "text_encoder",
        "tokenizer",
        "transformer",
        "vae",
    ]

    def create_pipeline_stages(self, fastvideo_args: FastVideoArgs) -> None:
        scheduler = self.get_module("scheduler")
        transformer = self.get_module("transformer")

        self.add_stage("input_validation_stage", ZImageInputValidationStage())
        self.add_stage(
            "text_encoding_stage",
            TextEncodingStage(
                text_encoders=[self.get_module("text_encoder")],
                tokenizers=[self.get_module("tokenizer")],
            ),
        )
        self.add_stage("zimage_conditioning_stage", ZImageConditioningStage())
        self.add_stage(
            "latent_preparation_stage",
            ZImageLatentPreparationStage(transformer=transformer),
        )
        self.add_stage(
            "timestep_preparation_stage",
            ZImageTimestepPreparationStage(scheduler=scheduler),
        )
        self.add_stage(
            "denoising_stage",
            ZImageDenoisingStage(transformer=transformer, scheduler=scheduler),
        )
        self.add_stage(
            "decoding_stage",
            ZImageDecodingStage(vae=self.get_module("vae")),
        )


EntryClass = ZImagePipeline
