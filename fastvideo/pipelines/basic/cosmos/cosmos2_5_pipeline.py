# SPDX-License-Identifier: Apache-2.0
"""Cosmos 2.5 pipeline entry (staged pipeline)."""

from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.logger import init_logger
from fastvideo.models.schedulers.scheduling_cosmos25_distilled import Cosmos25DistilledScheduler
from fastvideo.pipelines.composed_pipeline_base import ComposedPipelineBase
from fastvideo.pipelines.pipeline_batch_info import ForwardBatch
from fastvideo.pipelines.stages import (ConditioningStage, Cosmos25AutoDenoisingStage,
                                        Cosmos25AutoLatentPreparationStage, DecodingStage, InputValidationStage,
                                        Cosmos25DistilledT2WDenoisingStage, Cosmos25DistilledT2WLatentPreparationStage,
                                        Cosmos25TextEncodingStage, Cosmos25TimestepPreparationStage)

logger = init_logger(__name__)


class Cosmos25DistilledInputValidationStage(InputValidationStage):
    """Reject conditioning and classic CFG unsupported by the released student."""

    def forward(self, batch: ForwardBatch, fastvideo_args: FastVideoArgs) -> ForwardBatch:
        conditioning_inputs = (
            batch.image_path,
            batch.pil_image,
            batch.preprocessed_image,
            batch.video_path,
            batch.video_latent,
        )
        if any(value is not None for value in conditioning_inputs):
            raise ValueError("Cosmos Predict2.5 distilled currently supports text-to-world generation only")
        if batch.do_classifier_free_guidance:
            raise ValueError("Cosmos Predict2.5 distilled does not use classifier-free guidance; set guidance_scale=1")
        if not 1 <= batch.num_inference_steps <= 4:
            raise ValueError("Cosmos Predict2.5 distilled supports 1 to 4 inference steps")
        return super().forward(batch, fastvideo_args)


class Cosmos2_5Pipeline(ComposedPipelineBase):
    """Cosmos 2.5 video generation pipeline."""

    _required_config_modules = ["text_encoder", "tokenizer", "vae", "transformer", "scheduler", "safety_checker"]

    def create_pipeline_stages(self, fastvideo_args: FastVideoArgs):
        logger.info("Creating Cosmos 2.5 pipeline stages...")

        scheduler = self.get_module("scheduler")
        is_distilled = isinstance(scheduler, Cosmos25DistilledScheduler)

        input_validation = Cosmos25DistilledInputValidationStage() if is_distilled else InputValidationStage()
        self.add_stage(stage_name="input_validation_stage", stage=input_validation)

        self.add_stage(
            stage_name="prompt_encoding_stage",
            stage=Cosmos25TextEncodingStage(text_encoder=self.get_module("text_encoder"), ),
        )

        self.add_stage(stage_name="conditioning_stage", stage=ConditioningStage())

        self.add_stage(stage_name="timestep_preparation_stage",
                       stage=Cosmos25TimestepPreparationStage(scheduler=scheduler))

        if is_distilled:
            latent_stage = Cosmos25DistilledT2WLatentPreparationStage(
                scheduler=scheduler,
                transformer=self.get_module("transformer"),
            )
            denoising_stage = Cosmos25DistilledT2WDenoisingStage(
                transformer=self.get_module("transformer"),
                scheduler=scheduler,
            )
        else:
            latent_stage = Cosmos25AutoLatentPreparationStage(
                scheduler=scheduler,
                transformer=self.get_module("transformer"),
                vae=self.get_module("vae"),
            )
            denoising_stage = Cosmos25AutoDenoisingStage(
                transformer=self.get_module("transformer"),
                scheduler=scheduler,
            )

        self.add_stage(stage_name="latent_preparation_stage", stage=latent_stage)
        self.add_stage(stage_name="denoising_stage", stage=denoising_stage)

        self.add_stage(stage_name="decoding_stage", stage=DecodingStage(vae=self.get_module("vae")))
        logger.info("Cosmos 2.5 pipeline stages created")


# Entry point for pipeline registry
EntryClass = Cosmos2_5Pipeline
