# SPDX-License-Identifier: Apache-2.0
"""
Ovis-Image text-to-image diffusion pipeline implementation.

This module implements the Ovis-Image T2I pipeline using Diffusers components directly.
This is Approach A - using existing Diffusers classes for quick integration.
"""

from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.logger import init_logger
from fastvideo.models.schedulers.scheduling_flow_match_euler_discrete import (
    FlowMatchEulerDiscreteScheduler)
from fastvideo.pipelines.composed_pipeline_base import ComposedPipelineBase
from fastvideo.pipelines.stages import (ConditioningStage, DecodingStage,
                                        DenoisingStage, InputValidationStage,
                                        LatentPreparationStage,
                                        TextEncodingStage,
                                        TimestepPreparationStage)

logger = init_logger(__name__)


class OvisImagePipeline(ComposedPipelineBase):
    """
    Pipeline for Ovis-Image text-to-image generation.

    Ovis-Image is a 7B parameter model optimized for high-quality text rendering
    in generated images. It uses:
    - OvisImageTransformer2DModel: 2D diffusion transformer
    - Qwen3Model: Text encoder based on Ovis2.5-2B
    - AutoencoderKL: VAE for image encoding/decoding
    - FlowMatchEulerDiscreteScheduler: Flow-matching scheduler
    """

    _required_config_modules = [
        "text_encoder", "tokenizer", "vae", "transformer", "scheduler"
    ]

    def initialize_pipeline(self, fastvideo_args: FastVideoArgs):
        """
        Initialize pipeline-specific configurations.

        Sets up the scheduler with Ovis-Image specific parameters.
        """
        # Use the scheduler from model config
        # The scheduler is already loaded from the model, we just need to configure it
        if self.modules.get("scheduler") is None:
            self.modules["scheduler"] = FlowMatchEulerDiscreteScheduler(
                num_train_timesteps=1000,
                shift=3.0,
                use_dynamic_shifting=False,  # Disable for Approach A
            )

        # Configure scheduler parameters if needed
        scheduler = self.modules["scheduler"]
        if hasattr(scheduler, 'config'):
            # Disable dynamic shifting for Approach A (requires mu parameter)
            scheduler.config.use_dynamic_shifting = False
            # Update config if needed based on fastvideo_args
            if hasattr(fastvideo_args.pipeline_config, 'flow_shift'):
                scheduler.config.shift = fastvideo_args.pipeline_config.flow_shift

    def create_pipeline_stages(self, fastvideo_args: FastVideoArgs):
        """
        Set up pipeline stages for Ovis-Image T2I generation.

        Pipeline flow:
        1. Input validation - check dimensions
        2. Text encoding - encode prompt with Qwen3
        3. Conditioning - prepare CFG guidance
        4. Timestep preparation - setup diffusion schedule
        5. Latent preparation - initialize noise
        6. Denoising - iterative denoising with transformer
        7. Decoding - VAE decode to image
        """

        # Stage 1: Validate input dimensions
        self.add_stage(stage_name="input_validation_stage",
                       stage=InputValidationStage())

        # Stage 2: Encode text prompts with Qwen3 encoder
        self.add_stage(stage_name="prompt_encoding_stage",
                       stage=TextEncodingStage(
                           text_encoders=[self.get_module("text_encoder")],
                           tokenizers=[self.get_module("tokenizer")],
                       ))

        # Stage 3: Prepare conditioning for classifier-free guidance
        self.add_stage(stage_name="conditioning_stage",
                       stage=ConditioningStage())

        # Stage 4: Prepare timesteps for diffusion process
        self.add_stage(stage_name="timestep_preparation_stage",
                       stage=TimestepPreparationStage(
                           scheduler=self.get_module("scheduler")))

        # Stage 5: Prepare initial latent noise
        # For T2I, num_frames=1 (single image)
        self.add_stage(
            stage_name="latent_preparation_stage",
            stage=LatentPreparationStage(
                scheduler=self.get_module("scheduler"),
                transformer=self.get_module("transformer"),
                use_btchw_layout=False  # Use standard layout
            ))

        # Stage 6: Denoising loop with OvisImageTransformer2DModel
        self.add_stage(stage_name="denoising_stage",
                       stage=DenoisingStage(
                           transformer=self.get_module("transformer"),
                           scheduler=self.get_module("scheduler"),
                           vae=self.get_module("vae")))

        # Stage 7: Decode latents to image
        self.add_stage(stage_name="decoding_stage",
                       stage=DecodingStage(vae=self.get_module("vae")))


# Entry point for pipeline registry
EntryClass = OvisImagePipeline
