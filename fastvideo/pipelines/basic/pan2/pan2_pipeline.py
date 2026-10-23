# SPDX-License-Identifier: Apache-2.0
"""PAN2 video generation: text-to-video, and image-to-video from a first frame."""

from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.pipelines.basic.pan2.stages import (
    PAN2ConditioningStage,
    PAN2DenoisingStage,
    PAN2InputValidationStage,
    PAN2TextEncodingStage,
    PAN2TimestepPreparationStage,
)
from fastvideo.pipelines.composed_pipeline_base import ComposedPipelineBase
from fastvideo.pipelines.stages import DecodingStage, LatentPreparationStage


class PAN2Pipeline(ComposedPipelineBase):
    """One stage chain serves both modes: a request with ``image_path`` (or a PIL ``pil_image``) runs image-to-video
    from that first frame, and one without runs text-to-video."""

    _required_config_modules = ["text_encoder", "tokenizer", "vae", "transformer", "scheduler"]

    def create_pipeline_stages(self, fastvideo_args: FastVideoArgs) -> None:
        self.add_stage(stage_name="input_validation_stage", stage=PAN2InputValidationStage())
        self.add_stage(stage_name="prompt_encoding_stage",
                       stage=PAN2TextEncodingStage(
                           text_encoders=[self.get_module("text_encoder")],
                           tokenizers=[self.get_module("tokenizer")],
                       ))
        self.add_stage(stage_name="timestep_preparation_stage",
                       stage=PAN2TimestepPreparationStage(scheduler=self.get_module("scheduler")))
        self.add_stage(stage_name="latent_preparation_stage",
                       stage=LatentPreparationStage(scheduler=self.get_module("scheduler"),
                                                    transformer=self.get_module("transformer")))
        self.add_stage(stage_name="conditioning_stage", stage=PAN2ConditioningStage(vae=self.get_module("vae")))
        self.add_stage(stage_name="denoising_stage",
                       stage=PAN2DenoisingStage(transformer=self.get_module("transformer"),
                                                scheduler=self.get_module("scheduler"),
                                                pipeline=self))
        self.add_stage(stage_name="decoding_stage", stage=DecodingStage(vae=self.get_module("vae"), pipeline=self))


class PAN2ModularPipeline(PAN2Pipeline):
    """The pipeline class PAN2 checkpoints declare in their ``modular_model_index.json``."""


EntryClass = [PAN2Pipeline, PAN2ModularPipeline]
