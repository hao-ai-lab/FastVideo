# SPDX-License-Identifier: Apache-2.0
"""Stage-composed LingBot-Video Dense and MoE/refiner T2V pipeline."""

from typing import Any

from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.pipelines import ComposedPipelineBase, LoRAPipeline
from fastvideo.pipelines.basic.lingbot_video.stages import (
    LingBotVideoDenoisingStage,
    LingBotVideoInputValidationStage,
    LingBotVideoLatentPreparationStage,
    LingBotVideoRefinerPreparationStage,
)
from fastvideo.pipelines.stages import (
    ConditioningStage,
    DecodingStage,
    TextEncodingStage,
    TimestepPreparationStage,
)


class LingBotVideoPipeline(LoRAPipeline, ComposedPipelineBase):
    """T2V pipeline with optional released MoE pixel-space refinement."""

    is_video_pipeline = True
    _required_config_modules = ["text_encoder", "tokenizer", "vae", "transformer", "scheduler"]

    def load_modules(
        self,
        fastvideo_args: FastVideoArgs,
        loaded_modules: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Load the optional refiner DiT and the VAE encoder only when declared."""
        model_index = self._load_config(self.model_path)
        required = list(type(self)._required_config_modules)
        load_refiner = "transformer_2" in model_index and getattr(fastvideo_args, "refine_enabled", None) is not False
        if load_refiner:
            required.append("transformer_2")
            fastvideo_args.pipeline_config.vae_config.load_encoder = True
        self._required_config_modules = required
        return super().load_modules(fastvideo_args, loaded_modules)

    def initialize_pipeline(self, fastvideo_args: FastVideoArgs) -> None:
        """Apply the released runtime flow shift to the loaded scheduler."""
        shift = fastvideo_args.pipeline_config.flow_shift
        if shift is None:
            raise ValueError("LingBot-Video requires a flow shift")
        self.get_module("scheduler").set_shift(float(shift))

    def create_pipeline_stages(self, fastvideo_args: FastVideoArgs) -> None:
        """Create base generation and the optional decoded-video refiner stages."""
        refiner = self.get_module("transformer_2")
        self.add_stage(
            "input_validation_stage",
            LingBotVideoInputValidationStage(refiner_enabled=refiner is not None),
        )
        self.add_stage(
            "prompt_encoding_stage",
            TextEncodingStage(
                text_encoders=[self.get_module("text_encoder")],
                tokenizers=[self.get_module("tokenizer")],
            ),
        )
        self.add_stage("conditioning_stage", ConditioningStage())
        self.add_stage(
            "timestep_preparation_stage",
            TimestepPreparationStage(scheduler=self.get_module("scheduler")),
        )
        self.add_stage(
            "latent_preparation_stage",
            LingBotVideoLatentPreparationStage(transformer=self.get_module("transformer")),
        )
        self.add_stage(
            "denoising_stage",
            LingBotVideoDenoisingStage(
                transformer=self.get_module("transformer"),
                scheduler=self.get_module("scheduler"),
            ),
        )
        self.add_stage(
            "decoding_stage",
            DecodingStage(vae=self.get_module("vae"), pipeline=self),
        )
        if refiner is not None:
            self.add_stage(
                "refiner_preparation_stage",
                LingBotVideoRefinerPreparationStage(
                    vae=self.get_module("vae"),
                    scheduler=self.get_module("scheduler"),
                ),
            )
            self.add_stage(
                "refiner_denoising_stage",
                LingBotVideoDenoisingStage(
                    transformer=refiner,
                    scheduler=self.get_module("scheduler"),
                    refiner=True,
                ),
            )
            self.add_stage(
                "refiner_decoding_stage",
                DecodingStage(vae=self.get_module("vae"), pipeline=self),
            )


EntryClass = LingBotVideoPipeline
