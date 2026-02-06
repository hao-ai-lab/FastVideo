# SPDX-License-Identifier: Apache-2.0
"""
HunyuanGameCraft video diffusion pipeline implementation.

This module contains an implementation of the HunyuanGameCraft video diffusion pipeline
using the modular pipeline architecture. HunyuanGameCraft extends HunyuanVideo with
camera pose conditioning via CameraNet using Plücker coordinate representation.
"""

from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.logger import init_logger
from fastvideo.models.schedulers.scheduling_flow_match_euler_discrete import (
    FlowMatchEulerDiscreteScheduler)
from fastvideo.pipelines.composed_pipeline_base import ComposedPipelineBase
from fastvideo.pipelines.stages import (ConditioningStage, DecodingStage,
                                        InputValidationStage,
                                        LatentPreparationStage,
                                        TextEncodingStage,
                                        TimestepPreparationStage)
from fastvideo.pipelines.stages.gamecraft_denoising import GameCraftDenoisingStage

logger = init_logger(__name__)


class HunyuanGameCraftPipeline(ComposedPipelineBase):
    """
    HunyuanGameCraft video diffusion pipeline.
    
    This pipeline extends the HunyuanVideo architecture with camera pose conditioning
    for controllable game video generation. Camera poses are represented as Plücker
    coordinates (6D ray representation) and processed through CameraNet.
    
    The pipeline structure is similar to HunyuanVideo but the transformer includes
    CameraNet for camera conditioning. Camera states can be passed through the
    ForwardBatch.camera_states field.
    
    Required modules:
        - text_encoder: LLaMA-based text encoder (primary)
        - text_encoder_2: CLIP text encoder (secondary)
        - tokenizer: LLaMA tokenizer
        - tokenizer_2: CLIP tokenizer
        - vae: HunyuanVideo VAE (causal 3D)
        - transformer: HunyuanGameCraftTransformer3DModel with CameraNet
        - scheduler: Flow matching scheduler
    """

    _required_config_modules = [
        "text_encoder", "text_encoder_2", "tokenizer", "tokenizer_2", "vae",
        "transformer", "scheduler"
    ]

    def initialize_pipeline(self, fastvideo_args: FastVideoArgs):
        """Initialize the scheduler with correct shift parameter.
        
        Official HunyuanGameCraft uses flow_shift=5.0 with the
        FlowMatchDiscreteScheduler that applies the time-shift ONCE in
        set_timesteps. The FlowMatchEulerDiscreteScheduler applies the shift
        in __init__ (affecting sigma_min/sigma_max) AND again in set_timesteps,
        causing a double-shift bug. To avoid this we initialise with shift=1.0
        (identity) and then set the real shift value afterward, so only the
        set_timesteps call applies the shift.
        """
        flow_shift = fastvideo_args.pipeline_config.flow_shift
        logger.info(
            f"Initializing FlowMatchEulerDiscreteScheduler for GameCraft "
            f"(shift={flow_shift}, avoiding double-shift)"
        )
        # Create scheduler with shift=1.0 so __init__ does NOT shift
        # sigma_min/sigma_max. Then set the real shift so set_timesteps
        # applies it exactly once -- matching the official
        # FlowMatchDiscreteScheduler behaviour.
        scheduler = FlowMatchEulerDiscreteScheduler(
            shift=1.0,
            num_train_timesteps=1000,
        )
        scheduler.set_shift(flow_shift)
        self.modules["scheduler"] = scheduler

    def create_pipeline_stages(self, fastvideo_args: FastVideoArgs):
        """Set up pipeline stages with proper dependency injection."""

        self.add_stage(stage_name="input_validation_stage",
                       stage=InputValidationStage())

        self.add_stage(stage_name="prompt_encoding_stage_primary",
                       stage=TextEncodingStage(
                           text_encoders=[
                               self.get_module("text_encoder"),
                               self.get_module("text_encoder_2")
                           ],
                           tokenizers=[
                               self.get_module("tokenizer"),
                               self.get_module("tokenizer_2")
                           ],
                       ))

        self.add_stage(stage_name="conditioning_stage",
                       stage=ConditioningStage())

        self.add_stage(stage_name="timestep_preparation_stage",
                       stage=TimestepPreparationStage(
                           scheduler=self.get_module("scheduler")))

        self.add_stage(stage_name="latent_preparation_stage",
                       stage=LatentPreparationStage(
                           scheduler=self.get_module("scheduler"),
                           transformer=self.get_module("transformer")))

        self.add_stage(stage_name="denoising_stage",
                       stage=GameCraftDenoisingStage(
                           transformer=self.get_module("transformer"),
                           scheduler=self.get_module("scheduler"),
                           vae=self.get_module("vae")))

        self.add_stage(stage_name="decoding_stage",
                       stage=DecodingStage(vae=self.get_module("vae")))


EntryClass = HunyuanGameCraftPipeline
