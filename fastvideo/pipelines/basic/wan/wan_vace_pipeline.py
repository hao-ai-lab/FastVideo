# SPDX-License-Identifier: Apache-2.0
"""Wan-VACE controllable video generation pipeline."""

from fastvideo.distributed import get_local_torch_device
from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.logger import init_logger
from fastvideo.models.schedulers.scheduling_flow_unipc_multistep import FlowUniPCMultistepScheduler
from fastvideo.pipelines import ComposedPipelineBase, LoRAPipeline
from fastvideo.pipelines.basic.wan.stages.vace_conditioning import WanVACEContextStage
from fastvideo.pipelines.basic.wan.stages.vace_decoding import WanVACEDecodingStage
from fastvideo.pipelines.basic.wan.stages.vace_denoising import WanVACEDenoisingStage
from fastvideo.pipelines.basic.wan.stages.vace_input import WanVACEInputStage
from fastvideo.pipelines.basic.wan.stages.vace_latent_preparation import WanVACELatentPreparationStage
from fastvideo.pipelines.stages import (ConditioningStage, InputValidationStage, TextEncodingStage,
                                        TimestepPreparationStage)

logger = init_logger(__name__)


class WanVACEPipeline(LoRAPipeline, ComposedPipelineBase):
    _required_config_modules = ["text_encoder", "tokenizer", "vae", "transformer", "scheduler"]

    def initialize_pipeline(self, fastvideo_args: FastVideoArgs):
        self.modules["scheduler"] = FlowUniPCMultistepScheduler(shift=fastvideo_args.pipeline_config.flow_shift)

    def create_pipeline_stages(self, fastvideo_args: FastVideoArgs) -> None:
        self.add_stage(stage_name="input_validation_stage", stage=InputValidationStage())
        self.add_stage(stage_name="prompt_encoding_stage",
                       stage=TextEncodingStage(
                           text_encoders=[self.get_module("text_encoder")],
                           tokenizers=[self.get_module("tokenizer")],
                       ))
        self.add_stage(stage_name="conditioning_stage", stage=ConditioningStage())
        self.add_stage(stage_name="vace_input_stage",
                       stage=WanVACEInputStage(device=get_local_torch_device()))
        self.add_stage(stage_name="timestep_preparation_stage",
                       stage=TimestepPreparationStage(scheduler=self.get_module("scheduler")))
        self.add_stage(stage_name="latent_preparation_stage",
                       stage=WanVACELatentPreparationStage(scheduler=self.get_module("scheduler"),
                                                         transformer=self.get_module("transformer")))
        self.add_stage(stage_name="vace_context_stage",
                       stage=WanVACEContextStage(vae=self.get_module("vae")))
        self.add_stage(stage_name="denoising_stage",
                       stage=WanVACEDenoisingStage(transformer=self.get_module("transformer"),
                                                   scheduler=self.get_module("scheduler"),
                                                   pipeline=self))
        self.add_stage(stage_name="decoding_stage",
                       stage=WanVACEDecodingStage(vae=self.get_module("vae"), pipeline=self))


EntryClass = WanVACEPipeline
