# SPDX-License-Identifier: Apache-2.0
"""Native FastVideo preprocessing pipeline for MMAudio training."""

from __future__ import annotations

from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.models.audio.mmaudio_processing import build_mmaudio_mel_converter
from fastvideo.pipelines.composed_pipeline_base import ComposedPipelineBase
from fastvideo.pipelines.preprocess.mmaudio.stages import MMAudioFeatureExtractionStage


class MMAudioPreprocessPipeline(ComposedPipelineBase):
    """Encode raw audio, video, and captions into MMAudio training features."""

    _required_config_modules = [
        "audio_vae",
        "text_encoder",
        "tokenizer",
        "image_encoder",
        "image_encoder_2",
    ]

    def __init__(self, model_path: str, fastvideo_args: FastVideoArgs, **kwargs) -> None:
        # Official training features are extracted in fp32. These overrides
        # are local to preprocess mode and do not change inference defaults.
        config = fastvideo_args.pipeline_config
        config.text_encoder_precisions = ("fp32", )
        config.image_encoder_precisions = ("fp32", "fp32")
        config.audio_encoder_precision = "fp32"
        super().__init__(model_path, fastvideo_args, **kwargs)

    def initialize_pipeline(self, fastvideo_args: FastVideoArgs) -> None:
        audio_vae = self.get_module("audio_vae")
        if not hasattr(audio_vae, "encoder"):
            raise ValueError("MMAudio preprocessing requires an audio_vae component converted "
                             "with need_encoder=true. Use the converter's --preprocessor-only mode.")
        if getattr(audio_vae, "mode", None) != "44k":
            raise ValueError("The current MMAudio training integration requires the 44k audio VAE")

    def create_pipeline_stages(self, fastvideo_args: FastVideoArgs) -> None:
        mel_converter = build_mmaudio_mel_converter("44k")
        self.add_stage(
            "feature_extraction_stage",
            MMAudioFeatureExtractionStage(
                audio_vae=self.get_module("audio_vae"),
                mel_converter=mel_converter,
                image_encoder=self.get_module("image_encoder"),
                sync_encoder=self.get_module("image_encoder_2"),
                text_encoder=self.get_module("text_encoder"),
                tokenizer=self.get_module("tokenizer"),
            ),
        )


EntryClass = MMAudioPreprocessPipeline
