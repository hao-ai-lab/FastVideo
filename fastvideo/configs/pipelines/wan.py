# SPDX-License-Identifier: Apache-2.0
from collections.abc import Callable
from dataclasses import dataclass, field

import torch

from fastvideo.configs.models import DiTConfig, EncoderConfig, VAEConfig
from fastvideo.configs.models.dits import WanVideoConfig
from fastvideo.configs.models.dits.wanvideo import WanVideoArchConfig
from fastvideo.configs.models.encoders import (BaseEncoderOutput, CLIPVisionConfig, T5Config,
                                               WAN2_1ControlCLIPVisionConfig)
from fastvideo.configs.models.vaes import WanVAEConfig
from fastvideo.configs.models.vaes.wanvae import WanVAEArchConfig
from fastvideo.configs.pipelines.base import PipelineConfig


def t5_postprocess_text(outputs: BaseEncoderOutput) -> torch.Tensor:
    mask: torch.Tensor = outputs.attention_mask
    hidden_state: torch.Tensor = outputs.last_hidden_state
    seq_lens = mask.gt(0).sum(dim=1).long()
    assert torch.isnan(hidden_state).sum() == 0
    prompt_embeds = [u[:v] for u, v in zip(hidden_state, seq_lens, strict=True)]
    prompt_embeds_tensor: torch.Tensor = torch.stack(
        [torch.cat([u, u.new_zeros(512 - u.size(0), u.size(1))]) for u in prompt_embeds], dim=0)
    return prompt_embeds_tensor


@dataclass
class WanT2V480PConfig(PipelineConfig):
    """Base configuration for Wan T2V 1.3B pipeline architecture."""

    # WanConfig-specific parameters with defaults
    # DiT
    dit_config: DiTConfig = field(default_factory=WanVideoConfig)
    # VAE
    vae_config: VAEConfig = field(default_factory=WanVAEConfig)
    vae_tiling: bool = False
    vae_sp: bool = False

    # Denoising stage
    flow_shift: float | None = 3.0

    # Text encoding stage
    text_encoder_configs: tuple[EncoderConfig, ...] = field(default_factory=lambda: (T5Config(), ))
    postprocess_text_funcs: tuple[Callable[[BaseEncoderOutput], torch.Tensor],
                                  ...] = field(default_factory=lambda: (t5_postprocess_text, ))

    # Precision for each component
    precision: str = "bf16"
    vae_precision: str = "fp32"
    text_encoder_precisions: tuple[str, ...] = field(default_factory=lambda: ("fp32", ))

    # self-forcing params
    warp_denoising_step: bool = True

    # WanConfig-specific added parameters

    def __post_init__(self):
        self.vae_config.load_encoder = False
        self.vae_config.load_decoder = True


@dataclass
class WanT2V720PConfig(WanT2V480PConfig):
    """Base configuration for Wan T2V 14B 720P pipeline architecture."""

    # WanConfig-specific parameters with defaults

    # Denoising stage
    flow_shift: float | None = 5.0


@dataclass
class WanI2V480PConfig(WanT2V480PConfig):
    """Base configuration for Wan I2V 14B 480P pipeline architecture."""

    # WanConfig-specific parameters with defaults

    # Precision for each component
    image_encoder_config: EncoderConfig = field(default_factory=CLIPVisionConfig)
    image_encoder_precision: str = "fp32"

    def __post_init__(self) -> None:
        self.vae_config.load_encoder = True
        self.vae_config.load_decoder = True


@dataclass
class WanI2V720PConfig(WanI2V480PConfig):
    """Base configuration for Wan I2V 14B 720P pipeline architecture."""

    # WanConfig-specific parameters with defaults

    # Denoising stage
    flow_shift: float | None = 5.0


@dataclass
class WANV2VConfig(WanI2V480PConfig):
    """Configuration for WAN2.1 1.3B Control pipeline."""

    image_encoder_config: EncoderConfig = field(default_factory=WAN2_1ControlCLIPVisionConfig)
    # CLIP encoder precision
    image_encoder_precision: str = 'bf16'


@dataclass
class FastWan2_1_T2V_480P_Config(WanT2V480PConfig):
    """Base configuration for FastWan T2V 1.3B 480P pipeline architecture with DMD"""

    # WanConfig-specific parameters with defaults

    # Denoising stage
    flow_shift: float | None = 8.0
    dmd_denoising_steps: list[int] | None = field(default_factory=lambda: [1000, 757, 522])


@dataclass
class Wan2_2_TI2V_5B_Config(WanT2V480PConfig):
    flow_shift: float | None = 5.0
    ti2v_task: bool = True
    expand_timesteps: bool = True

    def __post_init__(self) -> None:
        self.vae_config.load_encoder = True
        self.vae_config.load_decoder = True
        self.dit_config.expand_timesteps = self.expand_timesteps


@dataclass
class LucyEditDevConfig(Wan2_2_TI2V_5B_Config):
    """Configuration for Decart Lucy Edit Dev video editing."""

    dit_config: DiTConfig = field(default_factory=lambda: WanVideoConfig(arch_config=WanVideoArchConfig(
        num_attention_heads=24,
        in_channels=96,
        out_channels=48,
        ffn_dim=14336,
        num_layers=30,
    )))
    vae_config: VAEConfig = field(default_factory=lambda: WanVAEConfig(arch_config=WanVAEArchConfig(
        base_dim=160,
        decoder_base_dim=256,
        z_dim=48,
        in_channels=12,
        out_channels=12,
        scale_factor_spatial=16,
        patch_size=2,
        is_residual=True,
        clip_output=False,
        latents_mean=(
            -0.2289, -0.0052, -0.1323, -0.2339, -0.2799, 0.0174,
            0.1838, 0.1557, -0.1382, 0.0542, 0.2813, 0.0891,
            0.1570, -0.0098, 0.0375, -0.1825, -0.2246, -0.1207,
            -0.0698, 0.5109, 0.2665, -0.2108, -0.2158, 0.2502,
            -0.2055, -0.0322, 0.1109, 0.1567, -0.0729, 0.0899,
            -0.2799, -0.1230, -0.0313, -0.1649, 0.0117, 0.0723,
            -0.2839, -0.2083, -0.0520, 0.3748, 0.0152, 0.1957,
            0.1433, -0.2944, 0.3573, -0.0548, -0.1681, -0.0667,
        ),
        latents_std=(
            0.4765, 1.0364, 0.4514, 1.1677, 0.5313, 0.4990,
            0.4818, 0.5013, 0.8158, 1.0344, 0.5894, 1.0901,
            0.6885, 0.6165, 0.8454, 0.4978, 0.5759, 0.3523,
            0.7135, 0.6804, 0.5833, 1.4146, 0.8986, 0.5659,
            0.7069, 0.5338, 0.4889, 0.4917, 0.4069, 0.4999,
            0.6866, 0.4093, 0.5709, 0.6065, 0.6415, 0.4944,
            0.5726, 1.2042, 0.5458, 1.6887, 0.3971, 1.0600,
            0.3943, 0.5537, 0.5444, 0.4089, 0.7468, 0.7744,
        ),
    )))
    ti2v_task: bool = False
    lucy_edit_task: bool = True

    def __post_init__(self) -> None:
        # Lucy uses Wan2.2's enhanced 48-channel VAE latents. Denoising
        # concatenates noise + video latents, matching the 96-channel
        # transformer input declared above.
        self.vae_config.load_encoder = True
        self.vae_config.load_decoder = True
        self.dit_config.expand_timesteps = self.expand_timesteps


@dataclass
class FastWan2_2_TI2V_5B_Config(Wan2_2_TI2V_5B_Config):
    flow_shift: float | None = 5.0
    dmd_denoising_steps: list[int] | None = field(default_factory=lambda: [1000, 757, 522])


@dataclass
class Wan2_2_T2V_A14B_Config(WanT2V480PConfig):
    flow_shift: float | None = 12.0
    boundary_ratio: float | None = 0.875

    # self-forcing params
    dmd_denoising_steps: list[int] | None = field(default_factory=lambda: [1000, 750, 500, 250])
    warp_denoising_step: bool = True

    def __post_init__(self) -> None:
        self.dit_config.boundary_ratio = self.boundary_ratio


@dataclass
class Wan2_2_I2V_A14B_Config(WanI2V480PConfig):
    flow_shift: float | None = 5.0
    boundary_ratio: float | None = 0.900

    def __post_init__(self) -> None:
        super().__post_init__()
        self.dit_config.boundary_ratio = self.boundary_ratio


# =============================================
# ============= Causal Self-Forcing =============
# =============================================
@dataclass
class SelfForcingWanT2V480PConfig(WanT2V480PConfig):
    is_causal: bool = True
    flow_shift: float | None = 5.0
    dmd_denoising_steps: list[int] | None = field(default_factory=lambda: [1000, 750, 500, 250])
    warp_denoising_step: bool = True


@dataclass
class SelfForcingWan2_2_T2V480PConfig(Wan2_2_T2V_A14B_Config):
    is_causal: bool = True
    flow_shift: float | None = 12.0
    boundary_ratio: float | None = 0.875
    dmd_denoising_steps: list[int] | None = field(default_factory=lambda: [1000, 850, 700, 550, 350, 275, 200, 125])
    warp_denoising_step: bool = True

    def __post_init__(self) -> None:
        self.vae_config.load_encoder = True
        self.vae_config.load_decoder = True
