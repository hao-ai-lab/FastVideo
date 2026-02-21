# SPDX-License-Identifier: Apache-2.0
from collections.abc import Callable
from dataclasses import dataclass, field

import torch

from fastvideo.configs.models import DiTConfig, EncoderConfig, VAEConfig
from fastvideo.configs.models.dits import WanVideoConfig
from fastvideo.configs.models.encoders import BaseEncoderOutput, T5Config
from fastvideo.configs.models.vaes import WanVAEConfig
from fastvideo.configs.pipelines.base import PipelineConfig


def t5_postprocess_text(outputs: BaseEncoderOutput) -> torch.Tensor:
    mask: torch.Tensor = outputs.attention_mask
    hidden_state: torch.Tensor = outputs.last_hidden_state
    seq_lens = mask.gt(0).sum(dim=1).long()
    assert torch.isnan(hidden_state).sum() == 0
    prompt_embeds = [u[:v] for u, v in zip(hidden_state, seq_lens, strict=True)]
    prompt_embeds_tensor: torch.Tensor = torch.stack(
        [
            torch.cat([u, u.new_zeros(512 - u.size(0), u.size(1))])
            for u in prompt_embeds
        ],
        dim=0,
    )
    return prompt_embeds_tensor


@dataclass
class LucyEditConfig(PipelineConfig):
    """Configuration for Lucy-Edit-Dev video editing pipeline.

    Lucy-Edit-Dev is built on the Wan2.2 5B architecture. It performs
    instruction-guided video editing by encoding the input video through
    the VAE and concatenating the video latents with noise latents along
    the channel dimension before denoising.

    Transformer: in_channels=96 (48 noise + 48 video latent), out_channels=48
    VAE: z_dim=48, scale_factor_spatial=16, patch_size=2 (Wan2.2 enhanced VAE)
    """

    # DiT config
    dit_config: DiTConfig = field(default_factory=WanVideoConfig)

    # VAE config
    vae_config: VAEConfig = field(default_factory=WanVAEConfig)
    vae_tiling: bool = False
    vae_sp: bool = False

    # Denoising stage
    flow_shift: float | None = 5.0

    # Wan2.2 expand_timesteps flag
    expand_timesteps: bool = True

    # Text encoding stage
    text_encoder_configs: tuple[EncoderConfig, ...] = field(
        default_factory=lambda: (T5Config(),)
    )
    postprocess_text_funcs: tuple[
        Callable[[BaseEncoderOutput], torch.Tensor], ...
    ] = field(default_factory=lambda: (t5_postprocess_text,))

    # Precision for each component
    precision: str = "bf16"
    vae_precision: str = "fp32"
    text_encoder_precisions: tuple[str, ...] = field(
        default_factory=lambda: ("fp32",)
    )

    # self-forcing params
    warp_denoising_step: bool = True

    def __post_init__(self):
        # Lucy-Edit needs both VAE encoder (for input video) and decoder
        self.vae_config.load_encoder = True
        self.vae_config.load_decoder = True
        self.dit_config.expand_timesteps = self.expand_timesteps
