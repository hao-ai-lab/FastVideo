# SPDX-License-Identifier: Apache-2.0
"""
Hunyuan video diffusion pipeline implementation.

This module contains an implementation of the Hunyuan video diffusion pipeline
using the modular pipeline architecture.
"""
import torch
import time
from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.pipelines.pipeline_batch_info import ForwardBatch
from fastvideo.logger import init_logger
from fastvideo.pipelines.composed_pipeline_base import ComposedPipelineBase
from fastvideo.pipelines.stages import (ConditioningStage, DecodingStage,
                                        DenoisingStage, InputValidationStage,
                                        LatentPreparationStage,
                                        TextEncodingStage,
                                        TimestepPreparationStage,
                                        Hy15ImageEncodingStage,
                                        SRDenoisingStage)

# TODO(will): move PRECISION_TO_TYPE to better place

logger = init_logger(__name__)


class HunyuanVideo15SRPipeline(ComposedPipelineBase):

    _required_config_modules = [
        "text_encoder", "text_encoder_2", "tokenizer", "tokenizer_2", "vae",
        "transformer", "transformer_2", "scheduler", "upsampler"
    ]

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

        self.add_stage(stage_name="image_encoding_stage",
                       stage=Hy15ImageEncodingStage(image_encoder=None,
                                                    image_processor=None))

        self.add_stage(stage_name="denoising_stage",
                       stage=DenoisingStage(
                           transformer=self.get_module("transformer"),
                           scheduler=self.get_module("scheduler")))

        self.add_stage(stage_name="sr_latent_preparation_stage",
                        stage=LatentPreparationStage(
                            scheduler=self.get_module("scheduler"),
                            transformer=self.get_module("transformer_2")))

        self.add_stage(stage_name="sr_denoising_stage",
                       stage=SRDenoisingStage(
                           transformer=self.get_module("transformer_2"),
                           scheduler=self.get_module("scheduler"),
                           upsampler=self.get_module("upsampler")))

        self.add_stage(stage_name="decoding_stage",
                       stage=DecodingStage(vae=self.get_module("vae")))

    @torch.no_grad()
    def forward(
        self,
        batch: ForwardBatch,
        fastvideo_args: FastVideoArgs,
    ) -> ForwardBatch:
        """
        Generate a video or image using the pipeline.
        
        Args:
            batch: The batch to generate from.
            fastvideo_args: The inference arguments.
        Returns:
            ForwardBatch: The batch with the generated video or image.
        """
        if not self.post_init_called:
            self.post_init()

        # Execute each stage
        logger.info("Running pipeline stages: %s",
                    self._stage_name_mapping.keys())
        # logger.info("Batch: %s", batch)
        batch = self.input_validation_stage(batch, fastvideo_args)
        batch = self.prompt_encoding_stage_primary(batch, fastvideo_args)
        batch = self.conditioning_stage(batch, fastvideo_args)
        batch = self.timestep_preparation_stage(batch, fastvideo_args)
        batch = self.latent_preparation_stage(batch, fastvideo_args)
        batch = self.image_encoding_stage(batch, fastvideo_args)
        batch = self.denoising_stage(batch, fastvideo_args)
        # self.get_module("transformer").to("cpu")

        batch.lq_latents = batch.latents
        batch.latents = None
        batch.height = batch.height_sr
        batch.width = batch.width_sr
        batch = self.sr_latent_preparation_stage(batch, fastvideo_args)
        batch = self.image_encoding_stage(batch, fastvideo_args)
        batch = self.sr_denoising_stage(batch, fastvideo_args)

        start_time = time.time()
        batch = self.decoding_stage(batch, fastvideo_args)
        end_time = time.time()
        logger.info(f"Decoding time: {end_time - start_time} seconds")

        # Return the output
        return batch

EntryClass = HunyuanVideo15SRPipeline
