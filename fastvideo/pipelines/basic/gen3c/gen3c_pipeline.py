# SPDX-License-Identifier: Apache-2.0
"""GEN3C video diffusion pipeline wiring."""

from diffusers import EDMEulerScheduler

from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.logger import init_logger
from fastvideo.pipelines.composed_pipeline_base import ComposedPipelineBase
from fastvideo.pipelines.stages import (DecodingStage, Gen3CCFGPolicyStage, Gen3CConditioningStage, Gen3CDenoisingStage,
                                        Gen3CLatentPreparationStage, InputValidationStage, TextEncodingStage,
                                        TimestepPreparationStage)

logger = init_logger(__name__)


class Gen3CPipeline(ComposedPipelineBase):
    """
    GEN3C Video Generation Pipeline.

    This pipeline extends Cosmos with 3D cache support for camera-controlled
    video generation. When an input image is provided, it runs the full
    3D cache conditioning pipeline (depth estimation -> point cloud ->
    camera trajectory -> forward warping -> VAE encoding).
    """

    _required_config_modules = [
        "text_encoder",
        "tokenizer",
        "vae",
        "transformer",
        "scheduler",
    ]

    def initialize_pipeline(self, fastvideo_args: FastVideoArgs):
        scheduler = self.modules.get("scheduler")
        if scheduler is not None and hasattr(scheduler, "precondition_inputs"):
            return

        # GEN3C denoising uses EDM preconditioning terms. The converted
        # model_index may point to FlowMatch scheduler configs that don't
        # expose precondition_inputs, so force the official EDM scheduler here.
        logger.warning(
            "Replacing loaded scheduler (%s) with EDMEulerScheduler for GEN3C parity.",
            type(scheduler).__name__ if scheduler is not None else "None",
        )
        self.modules["scheduler"] = EDMEulerScheduler(
            sigma_max=80.0,
            sigma_min=0.0002,
            sigma_data=float(getattr(fastvideo_args.pipeline_config, "sigma_data", 0.5)),
        )

    def create_pipeline_stages(self, fastvideo_args: FastVideoArgs):
        """Set up pipeline stages with proper dependency injection."""

        self.add_stage(stage_name="cfg_policy_stage", stage=Gen3CCFGPolicyStage())

        self.add_stage(stage_name="input_validation_stage", stage=InputValidationStage())

        self.add_stage(stage_name="prompt_encoding_stage",
                       stage=TextEncodingStage(
                           text_encoders=[self.get_module("text_encoder")],
                           tokenizers=[self.get_module("tokenizer")],
                       ))

        self.add_stage(stage_name="conditioning_stage", stage=Gen3CConditioningStage(vae=self.get_module("vae")))

        self.add_stage(stage_name="timestep_preparation_stage",
                       stage=TimestepPreparationStage(scheduler=self.get_module("scheduler")))

        self.add_stage(stage_name="latent_preparation_stage",
                       stage=Gen3CLatentPreparationStage(scheduler=self.get_module("scheduler"),
                                                         transformer=self.get_module("transformer"),
                                                         vae=self.get_module("vae")))

        self.add_stage(stage_name="denoising_stage",
                       stage=Gen3CDenoisingStage(transformer=self.get_module("transformer"),
                                                 scheduler=self.get_module("scheduler")))

        self.add_stage(stage_name="decoding_stage", stage=DecodingStage(vae=self.get_module("vae")))


EntryClass = Gen3CPipeline
