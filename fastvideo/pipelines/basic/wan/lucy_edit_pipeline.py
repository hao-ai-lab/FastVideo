# SPDX-License-Identifier: Apache-2.0
"""Lucy Edit video editing pipeline.

Lucy Edit uses a Wan2.2 5B transformer with an input video latent appended to
the noisy latent channels. The stage topology is therefore closest to Wan V2V,
but the model repo does not include CLIP image-encoder components.
"""

from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.logger import init_logger
from fastvideo.pipelines.basic.wan.wan_v2v_pipeline import WanVideoToVideoPipeline
from fastvideo.pipelines.stages import (
    ConditioningStage,
    DecodingStage,
    DenoisingStage,
    InputValidationStage,
    LatentPreparationStage,
    TextEncodingStage,
    TimestepPreparationStage,
    VideoVAEEncodingStage,
)

logger = init_logger(__name__)


class LucyEditPipeline(WanVideoToVideoPipeline):
    """FastVideo pipeline for decart-ai/Lucy-Edit-Dev."""

    _required_config_modules = [
        "text_encoder",
        "tokenizer",
        "vae",
        "transformer",
        "scheduler",
    ]

    def create_pipeline_stages(self, fastvideo_args: FastVideoArgs):
        self.add_stage(stage_name="input_validation_stage", stage=InputValidationStage())

        self.add_stage(
            stage_name="prompt_encoding_stage",
            stage=TextEncodingStage(
                text_encoders=[self.get_module("text_encoder")],
                tokenizers=[self.get_module("tokenizer")],
            ),
        )

        self.add_stage(stage_name="conditioning_stage", stage=ConditioningStage())

        self.add_stage(
            stage_name="timestep_preparation_stage",
            stage=TimestepPreparationStage(scheduler=self.get_module("scheduler")),
        )

        self.add_stage(
            stage_name="latent_preparation_stage",
            stage=LatentPreparationStage(
                scheduler=self.get_module("scheduler"),
                transformer=self.get_module("transformer"),
            ),
        )

        self.add_stage(
            stage_name="video_latent_preparation_stage",
            stage=VideoVAEEncodingStage(vae=self.get_module("vae")),
        )

        self.add_stage(
            stage_name="denoising_stage",
            stage=DenoisingStage(
                transformer=self.get_module("transformer"),
                transformer_2=self.get_module("transformer_2"),
                scheduler=self.get_module("scheduler"),
            ),
        )

        self.add_stage(stage_name="decoding_stage", stage=DecodingStage(vae=self.get_module("vae")))


EntryClass = LucyEditPipeline
