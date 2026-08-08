# SPDX-License-Identifier: Apache-2.0
"""LongCat causal DMD pipeline.

Wires :class:`LongCatCausalDMDDenoisingStage` into the modular pipeline
so that self-forcing-trained LongCat student weights can be evaluated
in the same chunked, KV-cached rollout pattern they were trained with.
Mirrors :class:`WanCausalDMDPipeline` (the Wan counterpart used by
``self_forcing_causal_t2v.yaml``).
"""

from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.logger import init_logger
from fastvideo.pipelines import ComposedPipelineBase, LoRAPipeline
from fastvideo.pipelines.stages import (
    ConditioningStage,
    DecodingStage,
    InputValidationStage,
    LatentPreparationStage,
    TextEncodingStage,
)
from fastvideo.pipelines.stages.longcat_causal_dmd_denoising import (
    LongCatCausalDMDDenoisingStage, )

logger = init_logger(__name__)


class LongCatCausalDMDPipeline(LoRAPipeline, ComposedPipelineBase):

    _required_config_modules = [
        "text_encoder",
        "tokenizer",
        "vae",
        "transformer",
        "scheduler",
    ]

    def create_pipeline_stages(self, fastvideo_args: FastVideoArgs) -> None:
        self.add_stage(
            stage_name="input_validation_stage",
            stage=InputValidationStage(),
        )

        self.add_stage(
            stage_name="prompt_encoding_stage",
            stage=TextEncodingStage(
                text_encoders=[self.get_module("text_encoder")],
                tokenizers=[self.get_module("tokenizer")],
            ),
        )

        self.add_stage(
            stage_name="conditioning_stage",
            stage=ConditioningStage(),
        )

        self.add_stage(
            stage_name="latent_preparation_stage",
            stage=LatentPreparationStage(
                scheduler=self.get_module("scheduler"),
                transformer=self.get_module("transformer", None),
            ),
        )

        self.add_stage(
            stage_name="denoising_stage",
            stage=LongCatCausalDMDDenoisingStage(
                transformer=self.get_module("transformer"),
                scheduler=self.get_module("scheduler"),
                vae=self.get_module("vae"),
            ),
        )

        self.add_stage(
            stage_name="decoding_stage",
            stage=DecodingStage(vae=self.get_module("vae")),
        )


EntryClass = LongCatCausalDMDPipeline
