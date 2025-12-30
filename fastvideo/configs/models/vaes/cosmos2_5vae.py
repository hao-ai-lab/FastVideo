"""Cosmos 2.5 (Wan2.1-style) VAE config and checkpoint-key mapping."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import torch

from fastvideo.configs.models.vaes.base import VAEArchConfig, VAEConfig


@dataclass
class Cosmos25VAEArchConfig(VAEArchConfig):
    _name_or_path: str = ""
    base_dim: int = 96
    decoder_base_dim: int | None = None
    z_dim: int = 16
    dim_mult: tuple[int, ...] = (1, 2, 4, 4)
    num_res_blocks: int = 2
    attn_scales: tuple[float, ...] = ()
    temperal_downsample: tuple[bool, ...] = (False, True, True)
    dropout: float = 0.0
    is_residual: bool = False
    in_channels: int = 3
    out_channels: int = 3
    patch_size: int | None = None
    scale_factor_temporal: int = 4
    scale_factor_spatial: int = 8
    clip_output: bool = True

    latents_mean: tuple[float, ...] = (
        -0.7571,
        -0.7089,
        -0.9113,
        0.1075,
        -0.1745,
        0.9653,
        -0.1517,
        1.5508,
        0.4134,
        -0.0715,
        0.5517,
        -0.3632,
        -0.1922,
        -0.9497,
        0.2503,
        -0.2921,
    )
    latents_std: tuple[float, ...] = (
        2.8184,
        1.4541,
        2.3275,
        2.6558,
        1.2196,
        1.7708,
        2.6052,
        2.0743,
        3.2687,
        2.1526,
        2.8652,
        1.5579,
        1.6382,
        1.1253,
        2.8251,
        1.9160,
    )

    # Simple 1:1 renames. More complex decoder remapping is handled by
    # `map_official_key()`.
    param_names_mapping: dict[str, str] = field(
        default_factory=lambda: {
            r"^conv1\.(.*)$": r"quant_conv.\1",
            r"^conv2\.(.*)$": r"post_quant_conv.\1",
            r"^encoder\.conv1\.(.*)$": r"encoder.conv_in.\1",
            r"^decoder\.conv1\.(.*)$": r"decoder.conv_in.\1",
            r"^encoder\.head\.0\.gamma$": r"encoder.norm_out.gamma",
            r"^encoder\.head\.2\.(.*)$": r"encoder.conv_out.\1",
            r"^decoder\.head\.0\.gamma$": r"decoder.norm_out.gamma",
            r"^decoder\.head\.2\.(.*)$": r"decoder.conv_out.\1",
        })

    @staticmethod
    def map_official_key(key: str) -> str | None:
        """Map a single official checkpoint key into FastVideo key space."""

        def map_residual_subkey(prefix: str, sub: str) -> str | None:
            if re.match(r"^residual\.0\.gamma$", sub):
                return f"{prefix}.norm1.gamma"
            m = re.match(r"^residual\.2\.(weight|bias)$", sub)
            if m:
                return f"{prefix}.conv1.{m.group(1)}"
            if re.match(r"^residual\.3\.gamma$", sub):
                return f"{prefix}.norm2.gamma"
            m = re.match(r"^residual\.6\.(weight|bias)$", sub)
            if m:
                return f"{prefix}.conv2.{m.group(1)}"
            m = re.match(r"^shortcut\.(weight|bias)$", sub)
            if m:
                return f"{prefix}.conv_shortcut.{m.group(1)}"
            return None

        def map_attn_subkey(prefix: str, sub: str) -> str | None:
            if re.match(r"^norm\.gamma$", sub):
                return f"{prefix}.norm.gamma"
            m = re.match(r"^to_qkv\.(weight|bias)$", sub)
            if m:
                return f"{prefix}.to_qkv.{m.group(1)}"
            m = re.match(r"^proj\.(weight|bias)$", sub)
            if m:
                return f"{prefix}.proj.{m.group(1)}"
            return None

        def map_resample_subkey(prefix: str, sub: str) -> str | None:
            m = re.match(r"^resample\.1\.(weight|bias)$", sub)
            if m:
                return f"{prefix}.resample.1.{m.group(1)}"
            m = re.match(r"^time_conv\.(weight|bias)$", sub)
            if m:
                return f"{prefix}.time_conv.{m.group(1)}"
            return None

        m = re.match(r"^conv1\.(weight|bias)$", key)
        if m:
            return f"quant_conv.{m.group(1)}"
        m = re.match(r"^conv2\.(weight|bias)$", key)
        if m:
            return f"post_quant_conv.{m.group(1)}"
        m = re.match(r"^(encoder|decoder)\.conv1\.(weight|bias)$", key)
        if m:
            return f"{m.group(1)}.conv_in.{m.group(2)}"
        m = re.match(r"^(encoder|decoder)\.head\.0\.gamma$", key)
        if m:
            return f"{m.group(1)}.norm_out.gamma"
        m = re.match(r"^(encoder|decoder)\.head\.2\.(weight|bias)$", key)
        if m:
            return f"{m.group(1)}.conv_out.{m.group(2)}"

        m = re.match(r"^(encoder|decoder)\.middle\.0\.(.*)$", key)
        if m:
            return map_residual_subkey(f"{m.group(1)}.mid_block.resnets.0",
                                       m.group(2))
        m = re.match(r"^(encoder|decoder)\.middle\.1\.(.*)$", key)
        if m:
            return map_attn_subkey(f"{m.group(1)}.mid_block.attentions.0",
                                   m.group(2))
        m = re.match(r"^(encoder|decoder)\.middle\.2\.(.*)$", key)
        if m:
            return map_residual_subkey(f"{m.group(1)}.mid_block.resnets.1",
                                       m.group(2))

        m = re.match(r"^encoder\.downsamples\.(\d+)\.(.*)$", key)
        if m:
            idx = int(m.group(1))
            sub = m.group(2)
            if sub.startswith("residual.") or sub.startswith("shortcut."):
                return map_residual_subkey(f"encoder.down_blocks.{idx}", sub)
            if sub.startswith("resample.") or sub.startswith("time_conv."):
                return map_resample_subkey(f"encoder.down_blocks.{idx}", sub)
            return None

        m = re.match(r"^decoder\.upsamples\.(\d+)\.(.*)$", key)
        if m:
            uidx = int(m.group(1))
            sub = m.group(2)

            if uidx in (0, 1, 2):
                block_i, res_i = 0, uidx
            elif uidx == 3:
                block_i, res_i = 0, None
            elif uidx in (4, 5, 6):
                block_i, res_i = 1, uidx - 4
            elif uidx == 7:
                block_i, res_i = 1, None
            elif uidx in (8, 9, 10):
                block_i, res_i = 2, uidx - 8
            elif uidx == 11:
                block_i, res_i = 2, None
            elif uidx in (12, 13, 14):
                block_i, res_i = 3, uidx - 12
            else:
                return None

            if res_i is None:
                return map_resample_subkey(
                    f"decoder.up_blocks.{block_i}.upsamplers.0",
                    sub,
                )

            return map_residual_subkey(
                f"decoder.up_blocks.{block_i}.resnets.{res_i}",
                sub,
            )

        return None

    temporal_compression_ratio: int = 4
    spatial_compression_ratio: int = 8

    def __post_init__(self):
        self.scaling_factor: torch.Tensor = 1.0 / torch.tensor(
            self.latents_std).view(1, self.z_dim, 1, 1, 1)
        self.shift_factor: torch.Tensor = torch.tensor(self.latents_mean).view(
            1, self.z_dim, 1, 1, 1)
        self.temporal_compression_ratio = self.scale_factor_temporal
        self.spatial_compression_ratio = self.scale_factor_spatial


@dataclass
class Cosmos25VAEConfig(VAEConfig):
    """Cosmos2.5 VAE config."""

    arch_config: Cosmos25VAEArchConfig = field(
        default_factory=Cosmos25VAEArchConfig)

    use_feature_cache: bool = True
    use_tiling: bool = False
    use_temporal_tiling: bool = False
    use_parallel_tiling: bool = False

    def __post_init__(self):
        self.blend_num_frames = (self.tile_sample_min_num_frames -
                                 self.tile_sample_stride_num_frames) * 2
