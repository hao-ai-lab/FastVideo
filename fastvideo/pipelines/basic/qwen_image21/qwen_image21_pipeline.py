# SPDX-License-Identifier: Apache-2.0
"""Native Qwen-Image-2.1 pipeline for generation and image-conditioned edits."""

from fastvideo.api.sampling_param import SamplingParam
from fastvideo.configs.pipelines.qwen_image21 import QwenImage21PipelineConfig
from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.pipelines.composed_pipeline_base import ComposedPipelineBase

from .stages import (
    QwenImage21DecodingStage,
    QwenImage21DenoisingStage,
    QwenImage21InputStage,
    QwenImage21LatentStage,
    QwenImage21ScheduleStage,
    QwenImage21TextEncodingStage,
)


class QwenImage21Pipeline(ComposedPipelineBase):
    pipeline_config_cls = QwenImage21PipelineConfig
    sampling_params_cls = SamplingParam
    _required_config_modules = ["processor", "text_encoder", "transformer", "vae", "scheduler"]
    _lazy_module_names = ("text_encoder", "transformer", "vae")

    def create_pipeline_stages(self, fastvideo_args: FastVideoArgs) -> None:
        self.add_stage("input_preparation", QwenImage21InputStage())
        self.add_stage("text_encoding",
                       QwenImage21TextEncodingStage(self.get_module("text_encoder"), self.get_module("processor")))
        self.add_stage("latent_preparation", QwenImage21LatentStage(self.get_module("vae")))
        self.add_stage("timestep_preparation", QwenImage21ScheduleStage(self.get_module("scheduler")))
        self.add_stage("denoising",
                       QwenImage21DenoisingStage(self.get_module("transformer"), self.get_module("scheduler")))
        self.add_stage("decoding", QwenImage21DecodingStage(self.get_module("vae")))


EntryClass = QwenImage21Pipeline
