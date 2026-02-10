# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.logger import init_logger
from fastvideo.pipelines.composed_pipeline_base import ComposedPipelineBase
from fastvideo.pipelines.stages.input_validation import InputValidationStage
from fastvideo.pipelines.stages.text_encoding import TextEncodingStage
from fastvideo.pipelines.stages.timestep_preparation import (
    TimestepPreparationStage, )
from fastvideo.pipelines.stages.sd35_conditioning import (
    SD35ConditioningStage,
    SD35DecodingStage,
    SD35DenoisingStage,
    SD35LatentPreparationStage,
)

logger = init_logger(__name__)


class SD35Pipeline(ComposedPipelineBase):
    """Minimal SD3.5 Medium text-to-image pipeline (treat as num_frames=1)."""

    _required_config_modules = [
        "scheduler",
        "transformer",
        "vae",
        "text_encoder",
        "text_encoder_2",
        "text_encoder_3",
        "tokenizer",
        "tokenizer_2",
        "tokenizer_3",
    ]

    def initialize_pipeline(self, fastvideo_args: FastVideoArgs) -> None:
        te_cfgs = list(fastvideo_args.pipeline_config.text_encoder_configs)
        if len(te_cfgs) >= 2:
            for i in (0, 1):
                te_cfgs[i].tokenizer_kwargs.setdefault("padding", "max_length")
                te_cfgs[i].tokenizer_kwargs.setdefault("max_length", 77)
                te_cfgs[i].tokenizer_kwargs.setdefault("truncation", True)
                te_cfgs[i].tokenizer_kwargs.setdefault("return_tensors", "pt")
        if len(te_cfgs) >= 3:
            te_cfgs[2].tokenizer_kwargs["max_length"] = min(
                int(te_cfgs[2].tokenizer_kwargs.get("max_length", 256)), 256)
            te_cfgs[2].tokenizer_kwargs.setdefault("padding", "max_length")
            te_cfgs[2].tokenizer_kwargs.setdefault("truncation", True)
            te_cfgs[2].tokenizer_kwargs.setdefault("return_tensors", "pt")

    def create_pipeline_stages(self, fastvideo_args: FastVideoArgs) -> None:
        self.add_stage(stage_name="input_validation_stage",
                       stage=InputValidationStage())

        self.add_stage(
            stage_name="text_encoding_stage",
            stage=TextEncodingStage(
                text_encoders=[
                    self.get_module("text_encoder"),
                    self.get_module("text_encoder_2"),
                    self.get_module("text_encoder_3"),
                ],
                tokenizers=[
                    self.get_module("tokenizer"),
                    self.get_module("tokenizer_2"),
                    self.get_module("tokenizer_3"),
                ],
            ),
        )

        self.add_stage(
            stage_name="timestep_preparation_stage",
            stage=TimestepPreparationStage(
                scheduler=self.get_module("scheduler")),
        )

        self.add_stage(
            stage_name="latent_preparation_stage",
            stage=SD35LatentPreparationStage(
                scheduler=self.get_module("scheduler"), ),
        )

        self.add_stage(
            stage_name="sd35_conditioning_stage",
            stage=SD35ConditioningStage(
                text_encoders=[
                    self.get_module("text_encoder"),
                    self.get_module("text_encoder_2"),
                    self.get_module("text_encoder_3"),
                ],
                tokenizers=[
                    self.get_module("tokenizer"),
                    self.get_module("tokenizer_2"),
                    self.get_module("tokenizer_3"),
                ],
            ),
        )

        self.add_stage(
            stage_name="denoising_stage",
            stage=SD35DenoisingStage(
                transformer=self.get_module("transformer"),
                scheduler=self.get_module("scheduler"),
            ),
        )

        self.add_stage(
            stage_name="decoding_stage",
            stage=SD35DecodingStage(vae=self.get_module("vae"), ),
        )


class StableDiffusion3Pipeline(SD35Pipeline):
    """Alias name to match SD3.5 diffusers `model_index.json` _class_name."""


EntryClass = [SD35Pipeline, StableDiffusion3Pipeline]
