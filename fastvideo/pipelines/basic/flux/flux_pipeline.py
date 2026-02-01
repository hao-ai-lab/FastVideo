# # SPDX-License-Identifier: Apache-2.0
# """
# Flux text-to-image diffusion pipeline implementation.
# """

from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.logger import init_logger
from fastvideo.pipelines.composed_pipeline_base import ComposedPipelineBase
from fastvideo.pipelines.stages import (ConditioningStage, DecodingStage,
                                        DenoisingStage, InputValidationStage,
                                        LatentPreparationStage,
                                        TextEncodingStage,
                                        FluxTimestepPreparationStage)

# logger = init_logger(__name__)


# class FluxPromptEncodingStage(TextEncodingStage):
#     def forward(self, batch, fastvideo_args):
#         batch = super().forward(batch, fastvideo_args)
#         if len(batch.prompt_embeds) > 1:
#             batch.clip_embedding_pos = batch.prompt_embeds[1]
#         if batch.do_classifier_free_guidance and batch.negative_prompt_embeds:
#             if len(batch.negative_prompt_embeds) > 1:
#                 batch.clip_embedding_neg = batch.negative_prompt_embeds[1]
#         return batch


# class FluxPipeline(ComposedPipelineBase):
#     _required_config_modules = [
#         "text_encoder", "text_encoder_2", "tokenizer", "tokenizer_2", "vae",
#         "transformer", "scheduler"
#     ]

#     def create_pipeline_stages(self, fastvideo_args: FastVideoArgs):
#         self.add_stage(stage_name="input_validation_stage",
#                        stage=InputValidationStage())

#         self.add_stage(stage_name="prompt_encoding_stage_primary",
#                        stage=FluxPromptEncodingStage(
#                            text_encoders=[
#                                self.get_module("text_encoder_2"),
#                                self.get_module("text_encoder")
#                            ],
#                            tokenizers=[
#                                self.get_module("tokenizer_2"),
#                                self.get_module("tokenizer")
#                            ],
#                        ))

#         self.add_stage(stage_name="conditioning_stage",
#                        stage=ConditioningStage())

#         self.add_stage(stage_name="timestep_preparation_stage",
#                        stage=FluxTimestepPreparationStage(
#                            scheduler=self.get_module("scheduler")))

#         self.add_stage(stage_name="latent_preparation_stage",
#                        stage=LatentPreparationStage(
#                            scheduler=self.get_module("scheduler"),
#                            transformer=self.get_module("transformer")))

#         self.add_stage(stage_name="denoising_stage",
#                        stage=DenoisingStage(
#                            transformer=self.get_module("transformer"),
#                            scheduler=self.get_module("scheduler")))

#         self.add_stage(stage_name="decoding_stage",
#                        stage=DecodingStage(vae=self.get_module("vae")))


# EntryClass = FluxPipeline

# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

# SPDX-License-Identifier: Apache-2.0

from fastvideo.pipelines import LoRAPipeline
from fastvideo.pipelines import (
    ComposedPipelineBase,
)
from fastvideo.pipelines_core.schedule_batch import Req
from fastvideo.pipelines import (
    ConditioningStage,
    DecodingStage,
    DenoisingStage,
    InputValidationStage,
    LatentPreparationStage,
    TextEncodingStage,
    TimestepPreparationStage,
)
from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.logger import init_logger

# TODO(will): move PRECISION_TO_TYPE to better place

logger = init_logger(__name__)


def calculate_shift(
    image_seq_len,
    base_seq_len: int = 256,
    max_seq_len: int = 4096,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
):
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    mu = image_seq_len * m + b
    return mu


def prepare_mu(batch: Req, server_args: FastVideoArgs):
    height = batch.height
    width = batch.width
    vae_scale_factor = (
        server_args.pipeline_config.vae_config.arch_config.vae_scale_factor
    )
    image_seq_len = (int(height) // vae_scale_factor) * (int(width) // vae_scale_factor)

    mu = calculate_shift(
        image_seq_len,
        # hard code, since scheduler_config is not in PipelineConfig now
        256,
        4096,
        0.5,
        1.15,
    )
    return "mu", mu


class FluxPipeline(LoRAPipeline, ComposedPipelineBase):
    pipeline_name = "FluxPipeline"

    _required_config_modules = [
        "text_encoder",
        "text_encoder_2",
        "tokenizer",
        "tokenizer_2",
        "vae",
        "transformer",
        "scheduler",
    ]

    def create_pipeline_stages(self, server_args: FastVideoArgs):
        """Set up pipeline stages with proper dependency injection."""

        self.add_stage(
            stage_name="input_validation_stage", stage=InputValidationStage()
        )

        self.add_stage(
            stage_name="prompt_encoding_stage_primary",
            stage=TextEncodingStage(
                text_encoders=[
                    self.get_module("text_encoder"),
                    self.get_module("text_encoder_2"),
                ],
                tokenizers=[
                    self.get_module("tokenizer"),
                    self.get_module("tokenizer_2"),
                ],
            ),
        )

        self.add_stage(stage_name="conditioning_stage", stage=ConditioningStage())

        self.add_stage(
            stage_name="timestep_preparation_stage",
            stage=TimestepPreparationStage(
                scheduler=self.get_module("scheduler"),
                prepare_extra_set_timesteps_kwargs=[prepare_mu],
            ),
        )

        self.add_stage(
            stage_name="latent_preparation_stage",
            stage=LatentPreparationStage(
                scheduler=self.get_module("scheduler"),
                transformer=self.get_module("transformer"),
            ),
        )

        self.add_stage(
            stage_name="denoising_stage",
            stage=DenoisingStage(
                transformer=self.get_module("transformer"),
                scheduler=self.get_module("scheduler"),
            ),
        )

        self.add_stage(
            stage_name="decoding_stage", stage=DecodingStage(vae=self.get_module("vae"))
        )


EntryClass = FluxPipeline