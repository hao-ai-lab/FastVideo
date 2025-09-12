# SPDX-License-Identifier: Apache-2.0
"""
Wan causal DMD pipeline implementation.

This module wires the causal DMD denoising stage into the modular pipeline.
"""

from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.logger import init_logger
from fastvideo.models.schedulers.scheduling_flow_match_euler_discrete import (
    FlowMatchEulerDiscreteScheduler)
from fastvideo.pipelines import ComposedPipelineBase, LoRAPipeline

# isort: off
from fastvideo.pipelines.stages import (ConditioningStage, DecodingStage,
                                        CausalDMDDenosingStage,
                                        InputValidationStage,
                                        LatentPreparationStage,
                                        TextEncodingStage)
# isort: on

import torch

from fastvideo.sf_utils.wan_wrapper import WanDiffusionWrapper

logger = init_logger(__name__)

from fastvideo.distributed import get_local_torch_device


class WanCausalDMDPipeline(LoRAPipeline, ComposedPipelineBase):

    _required_config_modules = [
        "text_encoder", "tokenizer", "vae", "transformer", "scheduler"
    ]

    def create_pipeline_stages(self, fastvideo_args: FastVideoArgs) -> None:
        """Set up pipeline stages with proper dependency injection."""
        if fastvideo_args.use_sf_wan:
            # timestep shift is 5.0 for self-forcing Wan model
            # see https://github.com/guandeh17/Self-Forcing/blob/33593df3e81fa3ec10239271dd2c100facac6de1/configs/self_forcing_dmd.yaml#L50
            config = self.get_module("transformer").config
            sf_transformer = WanDiffusionWrapper(
                model_name="Wan2.1-T2V-1.3B", timestep_shift=5.0, is_causal=True, config=config)
            del self.modules["transformer"]
            state_dict = torch.load('checkpoints/self_forcing_dmd.pt')
            sf_transformer.load_state_dict(state_dict['generator_ema'])
            sf_transformer.to(get_local_torch_device())
            self.modules["transformer"] = sf_transformer
            logger.info("Using self-forcing Wan model for DMD inference")

        self.add_stage(stage_name="input_validation_stage",
                       stage=InputValidationStage())

        self.add_stage(stage_name="prompt_encoding_stage",
                       stage=TextEncodingStage(
                           text_encoders=[self.get_module("text_encoder")],
                           tokenizers=[self.get_module("tokenizer")],
                       ))

        self.add_stage(stage_name="conditioning_stage",
                       stage=ConditioningStage())

        self.add_stage(stage_name="latent_preparation_stage",
                       stage=LatentPreparationStage(
                           scheduler=self.get_module("scheduler"),
                           transformer=self.get_module("transformer", None)))

        self.add_stage(stage_name="denoising_stage",
                       stage=CausalDMDDenosingStage(
                           transformer=self.get_module("transformer"),
                           scheduler=self.get_module("scheduler")))

        self.add_stage(stage_name="decoding_stage",
                       stage=DecodingStage(vae=self.get_module("vae")))


EntryClass = WanCausalDMDPipeline
