"""Cosmos3 (Wan2.2-TI2V-5B) VAE config and checkpoint-key mapping.

The Cosmos3 checkpoint VAE is literally ``Wan-AI/Wan2.2-TI2V-5B-Diffusers``
(diffusers ``AutoencoderKLWan``), so this config locks the Wan2.2 geometry:
residual down/up blocks, ``patch_size=2``, ``z_dim=48``, ``base_dim=160``,
``decoder_base_dim=256``, and ``scale_factor_spatial=16``. The 48-dim
``latents_mean``/``latents_std`` are taken verbatim from the Cosmos3
checkpoint's ``vae/config.json`` (identical to the canonical Wan2.2-TI2V-5B
statistics).

Mirrors the :class:`Cosmos25VAEArchConfig` pattern. ``param_names_mapping`` /
``map_official_key`` translate the *official* Wan2.2 VAE state-dict keys
(nested-residual naming, e.g. ``encoder.downsamples.{b}.downsamples.{j}`` and
``decoder.upsamples.{b}.upsamples.{j}``) into FastVideo's ``AutoencoderKLWan``
key space. The standard diffusers checkpoint already ships native FastVideo
keys, so these helpers exist for parity tooling and official ``.pth`` loading.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from fastvideo.configs.models.vaes.wanvae import WanVAEArchConfig, WanVAEConfig


@dataclass
class Cosmos3VAEArchConfig(WanVAEArchConfig):
    # Wan2.2-TI2V-5B geometry (differs from the Wan2.1 WanVAEArchConfig
    # defaults: residual blocks, patch_size=2, z_dim=48, base_dim=160,
    # decoder_base_dim=256, scale_factor_spatial=16, 12 patch channels).
    _name_or_path: str = "Wan-AI/Wan2.2-TI2V-5B-Diffusers"
    base_dim: int = 160
    decoder_base_dim: int | None = 256
    z_dim: int = 48
    dim_mult: tuple[int, ...] = (1, 2, 4, 4)
    num_res_blocks: int = 2
    attn_scales: tuple[float, ...] = ()
    temperal_downsample: tuple[bool, ...] = (False, True, True)
    dropout: float = 0.0
    is_residual: bool = True
    in_channels: int = 12
    out_channels: int = 12
    patch_size: int | None = 2
    scale_factor_temporal: int = 4
    scale_factor_spatial: int = 16
    clip_output: bool = False

    # 48-dim statistics copied verbatim from the Cosmos3 checkpoint
    # (official_weights/cosmos3/vae/config.json).
    latents_mean: tuple[float, ...] = (
        -0.2289,
        -0.0052,
        -0.1323,
        -0.2339,
        -0.2799,
        0.0174,
        0.1838,
        0.1557,
        -0.1382,
        0.0542,
        0.2813,
        0.0891,
        0.157,
        -0.0098,
        0.0375,
        -0.1825,
        -0.2246,
        -0.1207,
        -0.0698,
        0.5109,
        0.2665,
        -0.2108,
        -0.2158,
        0.2502,
        -0.2055,
        -0.0322,
        0.1109,
        0.1567,
        -0.0729,
        0.0899,
        -0.2799,
        -0.123,
        -0.0313,
        -0.1649,
        0.0117,
        0.0723,
        -0.2839,
        -0.2083,
        -0.052,
        0.3748,
        0.0152,
        0.1957,
        0.1433,
        -0.2944,
        0.3573,
        -0.0548,
        -0.1681,
        -0.0667,
    )
    latents_std: tuple[float, ...] = (
        0.4765,
        1.0364,
        0.4514,
        1.1677,
        0.5313,
        0.499,
        0.4818,
        0.5013,
        0.8158,
        1.0344,
        0.5894,
        1.0901,
        0.6885,
        0.6165,
        0.8454,
        0.4978,
        0.5759,
        0.3523,
        0.7135,
        0.6804,
        0.5833,
        1.4146,
        0.8986,
        0.5659,
        0.7069,
        0.5338,
        0.4889,
        0.4917,
        0.4069,
        0.4999,
        0.6866,
        0.4093,
        0.5709,
        0.6065,
        0.6415,
        0.4944,
        0.5726,
        1.2042,
        0.5458,
        1.6887,
        0.3971,
        1.06,
        0.3943,
        0.5537,
        0.5444,
        0.4089,
        0.7468,
        0.7744,
    )

    # Simple 1:1 renames. The nested-residual block remapping (encoder
    # downsamples / decoder upsamples / middle / head) is handled by
    # ``map_official_key()``.
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
        """Map a single official Wan2.2 VAE key into FastVideo key space.

        Handles the residual (Wan2.2) module layout where each down/up block
        is a nested ``Sequential`` (``downsamples.{b}.downsamples.{j}`` /
        ``upsamples.{b}.upsamples.{j}``) rather than the flat Wan2.1 indexing.
        Returns ``None`` for keys with no FastVideo counterpart.
        """

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
            return map_residual_subkey(f"{m.group(1)}.mid_block.resnets.0", m.group(2))
        m = re.match(r"^(encoder|decoder)\.middle\.1\.(.*)$", key)
        if m:
            return map_attn_subkey(f"{m.group(1)}.mid_block.attentions.0", m.group(2))
        m = re.match(r"^(encoder|decoder)\.middle\.2\.(.*)$", key)
        if m:
            return map_residual_subkey(f"{m.group(1)}.mid_block.resnets.1", m.group(2))

        # Encoder: downsamples.{block}.downsamples.{j}.* (nested residual layout)
        m = re.match(r"^encoder\.downsamples\.(\d+)\.downsamples\.(\d+)\.(.*)$", key)
        if m:
            block_i, res_i, sub = int(m.group(1)), int(m.group(2)), m.group(3)
            if sub.startswith("resample.") or sub.startswith("time_conv."):
                return map_resample_subkey(f"encoder.down_blocks.{block_i}.downsampler", sub)
            return map_residual_subkey(f"encoder.down_blocks.{block_i}.resnets.{res_i}", sub)

        # Decoder: upsamples.{block}.upsamples.{j}.* (nested residual layout)
        m = re.match(r"^decoder\.upsamples\.(\d+)\.upsamples\.(\d+)\.(.*)$", key)
        if m:
            block_i, res_i, sub = int(m.group(1)), int(m.group(2)), m.group(3)
            if sub.startswith("resample.") or sub.startswith("time_conv."):
                return map_resample_subkey(f"decoder.up_blocks.{block_i}.upsampler", sub)
            return map_residual_subkey(f"decoder.up_blocks.{block_i}.resnets.{res_i}", sub)

        return None

    # ``__post_init__`` (scaling_factor / shift_factor / compression ratios) is
    # inherited unchanged from ``WanVAEArchConfig``.


@dataclass
class Cosmos3VAEConfig(WanVAEConfig):
    """Cosmos3 VAE config (reuses FastVideo's Wan2.2 ``AutoencoderKLWan``).

    Subclasses :class:`WanVAEConfig` so the model reads the same runtime flags
    (``use_feature_cache``, ``use_light_vae``, tiling) and only swaps in the
    Cosmos3 = Wan2.2 ``arch_config``.
    """

    arch_config: Cosmos3VAEArchConfig = field(default_factory=Cosmos3VAEArchConfig)

    use_feature_cache: bool = True
    use_tiling: bool = False
    use_temporal_tiling: bool = False
    use_parallel_tiling: bool = False

    # ``__post_init__`` (blend_num_frames) is inherited from ``WanVAEConfig``.
