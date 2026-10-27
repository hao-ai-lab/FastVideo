# SPDX-License-Identifier: Apache-2.0
"""Qwen-Image-2.1 RGBA autoencoder configuration."""

from dataclasses import dataclass, field

from fastvideo.configs.models.vaes.base import VAEArchConfig, VAEConfig


@dataclass
class QwenImage21VAEArchConfig(VAEArchConfig):
    _class_name: str = "AutoencoderKLQwenImage21"
    base_dim: int = 96
    decoder_base_dim: int | None = 144
    z_dim: int = 64
    dim_mult: tuple[int, ...] = (1, 2, 4, 8, 8)
    num_res_blocks: int = 2
    attn_scales: tuple[float, ...] = ()
    temperal_downsample: tuple[bool, ...] = (False, True, True, True)
    dropout: float = 0.0
    is_residual: bool = True
    in_channels: int = 4
    out_channels: int = 4
    patch_size: int | None = None
    scale_factor_spatial: int = 16
    scale_factor_temporal: int = 8
    latents_mean: tuple[float, ...] = (
        0.5126,
        0.7721,
        -0.0631,
        1.3506,
        -0.7855,
        -2.1025,
        -0.3458,
        1.3722,
        1.8873,
        -1.7177,
        -0.6510,
        0.2732,
        0.7562,
        -0.6163,
        -1.0277,
        3.8363,
        2.0210,
        0.0472,
        0.9320,
        2.0087,
        2.4954,
        -0.1391,
        -1.4249,
        1.8464,
        -0.5236,
        1.2826,
        3.7046,
        -1.3035,
        2.7286,
        -1.4518,
        -1.9036,
        -1.9955,
        -0.0342,
        -1.0265,
        -0.7636,
        3.0555,
        0.0746,
        -3.0751,
        -0.1076,
        1.7376,
        -1.0914,
        -1.9435,
        -0.2784,
        -1.3680,
        0.4809,
        -0.4433,
        0.3764,
        0.5729,
        -2.0595,
        1.0960,
        -1.3260,
        -2.0211,
        -5.0179,
        0.5275,
        4.0162,
        1.8505,
        0.3026,
        1.9373,
        1.4937,
        0.2632,
        0.5547,
        -1.7121,
        -0.1562,
        0.0304,
    )
    latents_std: tuple[float, ...] = (
        3.2001,
        3.2936,
        3.4321,
        3.0091,
        3.1061,
        4.0379,
        4.0705,
        3.7910,
        3.0785,
        3.6500,
        3.9308,
        3.0904,
        2.8778,
        3.7675,
        3.7320,
        5.0756,
        3.2864,
        4.0397,
        3.1317,
        4.0443,
        2.9249,
        3.9454,
        3.0988,
        4.2489,
        3.4896,
        3.8513,
        3.9323,
        3.4719,
        3.7498,
        4.2830,
        3.5694,
        4.2467,
        3.9037,
        3.2947,
        5.0770,
        3.5075,
        3.2700,
        3.4767,
        2.8063,
        5.1125,
        3.5327,
        4.7833,
        3.1286,
        4.1819,
        3.8527,
        3.8312,
        3.5605,
        4.3875,
        3.9624,
        4.0168,
        3.5643,
        4.0550,
        5.5614,
        4.2963,
        4.4080,
        3.4959,
        3.8747,
        3.7608,
        3.5735,
        3.1490,
        3.7662,
        3.6746,
        3.4563,
        3.8161,
    )
    scaling_factor: float = 1.0
    temporal_compression_ratio: int = 1
    spatial_compression_ratio: int = 16
    param_names_mapping: dict[str, str] = field(default_factory=lambda: {r"^(.*)$": r"\1"})

    @property
    def latent_channels(self) -> int:
        return self.z_dim

    def __post_init__(self) -> None:
        if len(self.dim_mult) < 2 or len(self.temperal_downsample) != len(self.dim_mult) - 1:
            raise ValueError("`temperal_downsample` must have one entry per downsampling stage.")
        if len(self.latents_mean) != self.z_dim or len(self.latents_std) != self.z_dim:
            raise ValueError("Qwen-Image-2.1 latent mean/std must have one entry per latent channel.")
        if any(value <= 0 for value in self.latents_std):
            raise ValueError("Qwen-Image-2.1 latent standard deviations must be positive.")
        if self.patch_size is not None and self.patch_size < 1:
            raise ValueError("`patch_size` must be positive or None.")
        actual_spatial_scale = 2**(len(self.dim_mult) - 1) * (self.patch_size or 1)
        if self.scale_factor_spatial != actual_spatial_scale:
            raise ValueError("`scale_factor_spatial` must match the VAE downsampling stages and patch size.")
        self.spatial_compression_ratio = self.scale_factor_spatial


@dataclass
class QwenImage21VAEConfig(VAEConfig):
    arch_config: QwenImage21VAEArchConfig = field(default_factory=QwenImage21VAEArchConfig)
    use_tiling: bool = False
    use_temporal_tiling: bool = False
    use_parallel_tiling: bool = False
