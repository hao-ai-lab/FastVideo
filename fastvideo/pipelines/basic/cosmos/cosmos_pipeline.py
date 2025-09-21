# SPDX-License-Identifier: Apache-2.0
"""
Cosmos video diffusion pipeline implementation.

This module contains an implementation of the Cosmos video diffusion pipeline
using the modular pipeline architecture.
"""

import os
import numpy as np
import torch

# TEMPORARY: Import diffusers VAE for comparison
import sys
sys.path.insert(0, '/workspace/diffusers/src')
from diffusers.models.autoencoders.autoencoder_kl_wan import AutoencoderKLWan as DiffusersAutoencoderKLWan

from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.logger import init_logger
from fastvideo.pipelines.composed_pipeline_base import ComposedPipelineBase
from fastvideo.pipelines.stages import (ConditioningStage, DecodingStage,
                                           CosmosDenoisingStage, InputValidationStage,
                                           CosmosLatentPreparationStage,
                                           TextEncodingStage,
                                           TimestepPreparationStage)
from fastvideo.pipelines.stages.base import PipelineStage
from fastvideo.models.schedulers.scheduling_flow_match_euler_discrete import (
    FlowMatchEulerDiscreteScheduler)

logger = init_logger(__name__)


class Cosmos2VideoToWorldPipeline(ComposedPipelineBase):

    _required_config_modules = [
        "text_encoder", "tokenizer", "vae", "transformer", "scheduler", "safety_checker"
    ]

    def initialize_pipeline(self, fastvideo_args: FastVideoArgs):

        # TEMPORARY: Replace FastVideo VAE with diffusers VAE for testing
        print("[TEMPORARY] Replacing FastVideo VAE with diffusers VAE...")
        original_vae = self.modules["vae"]
        print(f"[TEMPORARY] Original VAE type: {type(original_vae)}")

        # Load diffusers VAE with same config
        diffusers_vae = DiffusersAutoencoderKLWan.from_pretrained(
            self.model_path,
            subfolder="vae",
            torch_dtype=torch.bfloat16,
        )
        print(f"[TEMPORARY] Diffusers VAE type: {type(diffusers_vae)}")

        # Replace the VAE module
        self.modules["vae"] = diffusers_vae
        print("[TEMPORARY] VAE replacement complete!")

        self.modules["scheduler"] = FlowMatchEulerDiscreteScheduler(
            shift=fastvideo_args.pipeline_config.flow_shift)
        
        # Configure Cosmos-specific scheduler parameters (matching diffusers)
        # Source: /workspace/diffusers/src/diffusers/pipelines/cosmos/pipeline_cosmos2_video2world.py:209-219
        sigma_max = 80.0
        sigma_min = 0.002
        sigma_data = 1.0
        final_sigmas_type = "sigma_min"
        
        if self.modules["scheduler"] is not None:
            # Update scheduler config and attributes directly
            scheduler = self.modules["scheduler"]
            scheduler.config.sigma_max = sigma_max
            scheduler.config.sigma_min = sigma_min
            scheduler.config.sigma_data = sigma_data
            scheduler.config.final_sigmas_type = final_sigmas_type
            # Also set the direct attributes used by the scheduler
            scheduler.sigma_max = sigma_max
            scheduler.sigma_min = sigma_min
            scheduler.sigma_data = sigma_data

    def create_pipeline_stages(self, fastvideo_args: FastVideoArgs):
        """Set up pipeline stages with proper dependency injection."""

        # Input validation - corresponds to diffusers check_inputs method
        # Source: /workspace/diffusers/src/diffusers/pipelines/cosmos/pipeline_cosmos2_video2world.py:427-456
        self.add_stage(stage_name="input_validation_stage",
                       stage=InputValidationStage())

        # Text encoding - corresponds to diffusers encode_prompt method
        # Source: /workspace/diffusers/src/diffusers/pipelines/cosmos/pipeline_cosmos2_video2world.py:265-346
        # Also uses _get_t5_prompt_embeds method: lines 222-262
        self.add_stage(stage_name="prompt_encoding_stage",
                       stage=TextEncodingStage(
                           text_encoders=[self.get_module("text_encoder")],
                           tokenizers=[self.get_module("tokenizer")],
                       ))

        # Conditioning preparation - part of main __call__ method setup
        # Source: /workspace/diffusers/src/diffusers/pipelines/cosmos/pipeline_cosmos2_video2world.py:607-628
        self.add_stage(stage_name="conditioning_stage",
                       stage=ConditioningStage())

        # Timestep preparation - corresponds to timestep setup in __call__
        # Source: /workspace/diffusers/src/diffusers/pipelines/cosmos/pipeline_cosmos2_video2world.py:630-637
        # Uses retrieve_timesteps function: lines 81-137
        self.add_stage(stage_name="timestep_preparation_stage",
                       stage=TimestepPreparationStage(
                           scheduler=self.get_module("scheduler")))

        # Latent preparation - corresponds to prepare_latents method
        # Source: /workspace/diffusers/src/diffusers/pipelines/cosmos/pipeline_cosmos2_video2world.py:348-424
        # Also includes video preprocessing: lines 642-661
        self.add_stage(stage_name="latent_preparation_stage",
                       stage=CosmosLatentPreparationStage(
                           scheduler=self.get_module("scheduler"),
                           transformer=self.get_module("transformer"),
                           vae=self.get_module("vae")))

        # Denoising loop - corresponds to main denoising loop in __call__
        # Source: /workspace/diffusers/src/diffusers/pipelines/cosmos/pipeline_cosmos2_video2world.py:673-752
        self.add_stage(stage_name="denoising_stage",
                       stage=CosmosDenoisingStage(
                           transformer=self.get_module("transformer"),
                           scheduler=self.get_module("scheduler")))

        # VAE decoding - corresponds to final decoding section in __call__
        # Source: /workspace/diffusers/src/diffusers/pipelines/cosmos/pipeline_cosmos2_video2world.py:755-784
        self.add_stage(stage_name="decoding_stage",
                       stage=DecodingStage(vae=self.get_module("vae")))
        


EntryClass = Cosmos2VideoToWorldPipeline