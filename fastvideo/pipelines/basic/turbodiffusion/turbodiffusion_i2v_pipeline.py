# SPDX-License-Identifier: Apache-2.0
"""
TurboDiffusion I2V (Image-to-Video) Pipeline Implementation.

This module contains an implementation of the TurboDiffusion I2V pipeline
for 1-4 step image-to-video generation using rCM (recurrent Consistency Model)
sampling with SLA (Sparse-Linear Attention).

Key differences from T2V:
- Uses dual models (high/low noise) with boundary switching
- sigma_max=200 (vs 80 for T2V)
- Mask conditioning with encoded first frame
"""

from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.logger import init_logger
from fastvideo.models.schedulers.scheduling_rcm import RCMScheduler
from fastvideo.pipelines import ComposedPipelineBase, LoRAPipeline
from fastvideo.pipelines.stages import (ConditioningStage, DecodingStage,
                                        DenoisingStage, ImageVAEEncodingStage,
                                        InputValidationStage,
                                        LatentPreparationStage,
                                        TextEncodingStage,
                                        TimestepPreparationStage)

logger = init_logger(__name__)


class TurboDiffusionI2VPipeline(LoRAPipeline, ComposedPipelineBase):
    """
    TurboDiffusion I2V pipeline for 1-4 step image-to-video generation.

    Uses RCM scheduler, SLA attention, and dual model switching for
    high-quality I2V generation.
    """

    _required_config_modules = [
        "text_encoder", "tokenizer", "vae", "transformer", "transformer_2", "scheduler"
    ]

    def initialize_pipeline(self, fastvideo_args: FastVideoArgs):
        # Use RCM scheduler with higher sigma_max for I2V
        logger.info("Initializing RCM scheduler for TurboDiffusion I2V (sigma_max=200)")
        self.modules["scheduler"] = RCMScheduler(sigma_max=200.0)

        # Store checkpoint paths for high/low noise models
        self._turbodiffusion_checkpoint = getattr(fastvideo_args,
                                                  'turbodiffusion_checkpoint',
                                                  None)
        # Boundary for switching high->low noise model (default 0.9)
        self._boundary = getattr(fastvideo_args, 'boundary', 0.9)

    def create_pipeline_stages(self, fastvideo_args: FastVideoArgs) -> None:
        """Set up pipeline stages with proper dependency injection."""

        self.add_stage(stage_name="input_validation_stage",
                       stage=InputValidationStage())

        self.add_stage(stage_name="prompt_encoding_stage",
                       stage=TextEncodingStage(
                           text_encoders=[self.get_module("text_encoder")],
                           tokenizers=[self.get_module("tokenizer")],
                       ))

        self.add_stage(stage_name="conditioning_stage",
                       stage=ConditioningStage())

        self.add_stage(stage_name="timestep_preparation_stage",
                       stage=TimestepPreparationStage(
                           scheduler=self.get_module("scheduler")))

        self.add_stage(stage_name="latent_preparation_stage",
                       stage=LatentPreparationStage(
                           scheduler=self.get_module("scheduler"),
                           transformer=self.get_module("transformer", None)))

        # I2V: Encode initial image to latent space
        self.add_stage(stage_name="image_latent_preparation_stage",
                       stage=ImageVAEEncodingStage(vae=self.get_module("vae")))

        self.add_stage(stage_name="denoising_stage",
                       stage=DenoisingStage(
                           transformer=self.get_module("transformer"),
                           transformer_2=self.get_module("transformer_2", None),
                           scheduler=self.get_module("scheduler"),
                           vae=self.get_module("vae"),
                           pipeline=self))

        self.add_stage(stage_name="decoding_stage",
                       stage=DecodingStage(vae=self.get_module("vae"),
                                           pipeline=self))


EntryClass = TurboDiffusionI2VPipeline
