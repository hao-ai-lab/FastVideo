# SPDX-License-Identifier: Apache-2.0
"""Cosmos 2.5 pipeline entry (staged pipeline)."""

from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.logger import init_logger
from fastvideo.models.schedulers.scheduling_cosmos25_distilled import Cosmos25DistilledScheduler
from fastvideo.models.schedulers.scheduling_cosmos25_dfd import Cosmos25DFDScheduler
from fastvideo.pipelines.composed_pipeline_base import ComposedPipelineBase
from fastvideo.pipelines.pipeline_batch_info import ForwardBatch
from fastvideo.pipelines.stages import (ConditioningStage, Cosmos25AutoDenoisingStage,
                                        Cosmos25AutoLatentPreparationStage, DecodingStage, InputValidationStage,
                                        Cosmos25DistilledT2WDenoisingStage, Cosmos25DistilledT2WLatentPreparationStage,
                                        Cosmos25DFDV2WDenoisingStage, Cosmos25DFDV2WLatentPreparationStage,
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


class Cosmos25DFDInputValidationStage(InputValidationStage):
    """Validate the fixed public DFD V2W inference contract."""

    def forward(self, batch: ForwardBatch, fastvideo_args: FastVideoArgs) -> ForwardBatch:
        image_inputs = (batch.image_path, batch.pil_image, batch.preprocessed_image)
        if not any(value is not None for value in image_inputs):
            raise ValueError("Cosmos Predict2.5 DFD requires one conditioning image")
        if batch.video_path is not None or batch.video_latent is not None:
            raise ValueError("Cosmos Predict2.5 DFD currently accepts a single image, not a conditioning video")
        if batch.do_classifier_free_guidance or batch.guidance_scale != 1:
            raise ValueError("Cosmos Predict2.5 DFD does not use classifier-free guidance; set guidance_scale=1")
        if batch.num_inference_steps != 4:
            raise ValueError("Cosmos Predict2.5 DFD requires exactly 4 inference steps")
        if batch.num_frames != 81:
            raise ValueError("Cosmos Predict2.5 DFD requires exactly 81 output frames")
        if batch.height != 704 or batch.width != 1280:
            raise ValueError("Cosmos Predict2.5 DFD requires height=704 and width=1280")
        if batch.fps not in (None, 24):
            raise ValueError("Cosmos Predict2.5 DFD was trained at 24 fps")
        batch.fps = 24
        return super().forward(batch, fastvideo_args)


class Cosmos2_5Pipeline(ComposedPipelineBase):
    """Cosmos 2.5 video generation pipeline."""

    _required_config_modules = ["text_encoder", "tokenizer", "vae", "transformer", "scheduler", "safety_checker"]

    def create_pipeline_stages(self, fastvideo_args: FastVideoArgs):
        logger.info("Creating Cosmos 2.5 pipeline stages...")

        scheduler = self.get_module("scheduler")
        is_distilled = isinstance(scheduler, Cosmos25DistilledScheduler)
        is_dfd = isinstance(scheduler, Cosmos25DFDScheduler)

        if is_dfd:
            input_validation = Cosmos25DFDInputValidationStage()
        elif is_distilled:
            input_validation = Cosmos25DistilledInputValidationStage()
        else:
            input_validation = InputValidationStage()
        self.add_stage(stage_name="input_validation_stage", stage=input_validation)

        self.add_stage(
            stage_name="prompt_encoding_stage",
            stage=Cosmos25TextEncodingStage(text_encoder=self.get_module("text_encoder"), ),
        )

        self.add_stage(stage_name="conditioning_stage", stage=ConditioningStage())

        self.add_stage(stage_name="timestep_preparation_stage",
                       stage=Cosmos25TimestepPreparationStage(scheduler=scheduler))

        if is_dfd:
            latent_stage = Cosmos25DFDV2WLatentPreparationStage(
                scheduler=scheduler,
                transformer=self.get_module("transformer"),
                vae=self.get_module("vae"),
            )
            denoising_stage = Cosmos25DFDV2WDenoisingStage(
                transformer=self.get_module("transformer"),
                scheduler=scheduler,
            )
        elif is_distilled:
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
