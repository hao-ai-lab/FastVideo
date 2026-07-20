# SPDX-License-Identifier: Apache-2.0
"""Native MMAudio video/text-to-audio pipeline."""

from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.pipelines.basic.mmaudio.stages import (
    MMAudioDecodingStage,
    MMAudioDenoisingStage,
    MMAudioInputValidationStage,
    MMAudioLatentPreparationStage,
    MMAudioTextConditioningStage,
    MMAudioVideoConditioningStage,
)
from fastvideo.pipelines.composed_pipeline_base import ComposedPipelineBase


class MMAudioPipeline(ComposedPipelineBase):
    """MMAudio large-44k-v2 V2A/T2A inference pipeline."""

    _required_config_modules = [
        "transformer",
        "scheduler",
        "text_encoder",
        "tokenizer",
        "image_encoder",
        "image_encoder_2",
        "audio_vae",
        "vocoder",
    ]

    def create_pipeline_stages(self, fastvideo_args: FastVideoArgs) -> None:
        transformer = self.get_module("transformer")
        self.add_stage("input_validation_stage", MMAudioInputValidationStage())
        self.add_stage(
            "video_conditioning_stage",
            MMAudioVideoConditioningStage(
                image_encoder=self.get_module("image_encoder"),
                sync_encoder=self.get_module("image_encoder_2"),
                transformer=transformer,
            ),
        )
        self.add_stage(
            "text_conditioning_stage",
            MMAudioTextConditioningStage(
                text_encoder=self.get_module("text_encoder"),
                tokenizer=self.get_module("tokenizer"),
                transformer=transformer,
            ),
        )
        self.add_stage("latent_preparation_stage", MMAudioLatentPreparationStage(transformer))
        self.add_stage(
            "denoising_stage",
            MMAudioDenoisingStage(transformer, self.get_module("scheduler")),
        )
        self.add_stage(
            "decoding_stage",
            MMAudioDecodingStage(
                audio_vae=self.get_module("audio_vae"),
                vocoder=self.get_module("vocoder"),
            ),
        )


EntryClass = MMAudioPipeline

