# SPDX-License-Identifier: Apache-2.0

import torch

from fastvideo.distributed import get_local_torch_device
from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.logger import init_logger
from fastvideo.models.schedulers.scheduling_flow_match_euler_discrete import (FlowMatchEulerDiscreteScheduler)
from fastvideo.pipelines.composed_pipeline_base import ComposedPipelineBase
from fastvideo.pipelines.stages import (ConditioningStage, DecodingStage, DenoisingStage, InputValidationStage,
                                        LatentPreparationStage, TextEncodingStage)
from fastvideo.pipelines.stages.timestep_preparation import (OvisImageTimestepPreparationStage)
from fastvideo.utils import PRECISION_TO_TYPE

logger = init_logger(__name__)


class OvisImageDecodingStage(DecodingStage):
    """VAE decode for the 2D Ovis-Image AutoencoderKL.

    Differs from the generic video `DecodingStage` in two ways the reused
    diffusers `AutoencoderKL` requires: the singleton temporal axis carried by
    the shared denoise loop (B, C, 1, H, W) is dropped before the 2D conv decode
    and restored after, and `decode()` returns a diffusers `DecoderOutput` whose
    `.sample` must be unwrapped.
    """

    @torch.no_grad()
    def decode(self, latents: torch.Tensor, fastvideo_args: FastVideoArgs) -> torch.Tensor:
        device = get_local_torch_device()
        self.vae = self.vae.to(device)
        latents = latents.to(device)
        if latents.ndim == 5:
            latents = latents.squeeze(2)

        latents = self._denormalize_latents(latents)

        vae_dtype = PRECISION_TO_TYPE[fastvideo_args.pipeline_config.vae_precision]
        vae_autocast_enabled = ((vae_dtype != torch.float32) and not fastvideo_args.disable_autocast)
        with torch.autocast(device_type="cuda", dtype=vae_dtype, enabled=vae_autocast_enabled):
            if fastvideo_args.pipeline_config.vae_tiling:
                self.vae.enable_tiling()
            if not vae_autocast_enabled:
                latents = latents.to(vae_dtype)
            decoded = self.vae.decode(latents)

        image = decoded.sample if hasattr(decoded, "sample") else decoded
        image = (image / 2 + 0.5).clamp(0, 1)
        return image.unsqueeze(2)


class OvisImagePipeline(ComposedPipelineBase):
    """Ovis-Image-7B text-to-image: native DiT + Qwen3 encoder, reused
    AutoencoderKL VAE, flow-match scheduler."""

    _required_config_modules = ["text_encoder", "tokenizer", "vae", "transformer", "scheduler"]

    def initialize_pipeline(self, fastvideo_args: FastVideoArgs):
        """
        Ensure a flow-match scheduler is present.

        The scheduler normally loads from the model with
        `use_dynamic_shifting=True`; `OvisImageTimestepPreparationStage` supplies
        the per-resolution `mu`. Dynamic shifting is left enabled so the timestep
        schedule matches the reference pipeline.
        """
        if self.modules.get("scheduler") is None:
            self.modules["scheduler"] = FlowMatchEulerDiscreteScheduler(
                num_train_timesteps=1000,
                shift=3.0,
                use_dynamic_shifting=True,
            )

    def create_pipeline_stages(self, fastvideo_args: FastVideoArgs):
        self.add_stage(stage_name="input_validation_stage", stage=InputValidationStage())

        self.add_stage(stage_name="prompt_encoding_stage",
                       stage=TextEncodingStage(
                           text_encoders=[self.get_module("text_encoder")],
                           tokenizers=[self.get_module("tokenizer")],
                       ))

        self.add_stage(stage_name="conditioning_stage", stage=ConditioningStage())

        self.add_stage(stage_name="timestep_preparation_stage",
                       stage=OvisImageTimestepPreparationStage(scheduler=self.get_module("scheduler")))

        self.add_stage(stage_name="latent_preparation_stage",
                       stage=LatentPreparationStage(
                           scheduler=self.get_module("scheduler"),
                           transformer=self.get_module("transformer"),
                           use_btchw_layout=False,
                       ))

        self.add_stage(stage_name="denoising_stage",
                       stage=DenoisingStage(transformer=self.get_module("transformer"),
                                            scheduler=self.get_module("scheduler"),
                                            vae=self.get_module("vae")))

        self.add_stage(stage_name="decoding_stage", stage=OvisImageDecodingStage(vae=self.get_module("vae")))


EntryClass = OvisImagePipeline
