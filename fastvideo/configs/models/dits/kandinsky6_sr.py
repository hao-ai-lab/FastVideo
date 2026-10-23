# SPDX-License-Identifier: Apache-2.0
"""Arch config of the Kandinsky6 video super-resolution (SR) DiT.

Fields mirror ``transformer/config.json`` of the official ``Kandinsky6SRTransformer3DModel`` bundles, including the
nested ``sr_params`` (sampling parameters of the SR training setup).  ``out_visual_dim`` is the full head width: the
flow-matching checkpoint predicts ``in_visual_dim`` channels, the pi-Flow checkpoint ``n_grid * in_visual_dim`` (``n_grid``
lives in its ``PiflowScheduler`` config).  ``update_model_arch`` drops undeclared keys, so configurations the port does
not implement are rejected by name instead of silently running a different model.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from fastvideo.configs.models.dits.base import DiTArchConfig, DiTConfig
from fastvideo.platforms import AttentionBackendEnum

_METADATA_KEYS = frozenset({"_class_name", "_diffusers_version", "_name_or_path"})
# The released checkpoints' sampling setup; the pipeline implements exactly this one.
_SUPPORTED_SR_PARAMS = {
    "lq_noise_type": "ddpm",
    "lq_channel_noise_scale": 0.0,
    "cap_noise_timestep": False,
}


def _is_sr_visual_block(name: str, module: Any) -> bool:
    return "visual_transformer_blocks" in name and name.split(".")[-1].isdigit()


@dataclass
class Kandinsky6SRArchConfig(DiTArchConfig):
    _fsdp_shard_conditions: list = field(default_factory=lambda: [_is_sr_visual_block])
    # NABLA sparse attention requested by ``attention_params`` is not wired; the DiT runs dense attention.
    _supported_attention_backends: tuple[AttentionBackendEnum, ...] = (
        AttentionBackendEnum.FLASH_ATTN,
        AttentionBackendEnum.TORCH_SDPA,
    )
    # Map the current Diffusers checkpoint names to native FastVideo layers. The loader applies only
    # the FIRST matching rule: decoder renames must include the FFN rename in the same rule.
    # Text-tower attn is self-attention; decoder self/cross-attention names are unchanged.
    param_names_mapping: dict = field(
        default_factory=lambda: {
            r"^(time_embeddings)\.timestep_embedder\.linear_1\.(weight|bias)$": r"\1.in_layer.\2",
            r"^(time_embeddings)\.timestep_embedder\.linear_2\.(weight|bias)$": r"\1.out_layer.\2",
            r"^(.*feed_forward)\.net\.0\.proj\.(weight|bias)$": r"\1.mlp.fc_in.\2",
            r"^(.*feed_forward)\.net\.2\.(weight|bias)$": r"\1.mlp.fc_out.\2",
        })
    reverse_param_names_mapping: dict = field(default_factory=lambda: {})

    in_visual_dim: int = 64
    out_visual_dim: int = 64
    time_dim: int = 512
    patch_size: tuple[int, int, int] = (1, 1, 1)
    model_dim: int = 1792
    ff_dim: int = 7168
    num_visual_blocks: int = 32
    axes_dims: tuple[int, int, int] = (16, 24, 24)
    visual_cond: bool = True
    instruct_type: str = "hybrid_anchor"
    attention_params: dict | None = None
    # Text-tower arguments of the shared constructor; the SR DiT is text-free.
    use_text: bool = False
    in_text_dim: int = 3584
    in_text_dim2: int = 768
    num_text_blocks: int = 2
    attribute_overrides: dict | None = None
    sr_params: dict = field(default_factory=dict)
    # New top-level fields the diffusers refactor registers on the checkpoint config (replacing the old
    # attention_params/sr_params-nested scheme); the DiT does not read these yet (still dense-only), but they
    # must be known so loading a checkpoint saved from the current diffusers code doesn't hard-fail on an
    # "unsupported key". Default None/unset so an old-schema checkpoint (no such key in its config.json) keeps
    # today's behavior exactly -- unlike diffusers' own constructor defaults, which would apply to every
    # checkpoint that omits these keys, including ones never trained with NABLA.
    scale_factor: tuple[float, float, float] | None = None
    nabla_threshold: float | None = None
    nabla_window: tuple[int, int, int] | None = None
    tile_sizes: tuple[tuple[int, int], ...] | None = None

    def __post_init__(self) -> None:
        super().__post_init__()
        self.patch_size = tuple(int(p) for p in self.patch_size)  # type: ignore[assignment]
        self.axes_dims = tuple(int(a) for a in self.axes_dims)  # type: ignore[assignment]
        unsupported = {
            "use_text": self.use_text,
            "attribute_overrides": self.attribute_overrides,
        }
        for name, value in unsupported.items():
            if value:
                raise NotImplementedError(f"Kandinsky6 SR: {name}={value!r} is not supported.")
        if self.instruct_type != "hybrid_anchor" or not self.visual_cond:
            raise NotImplementedError("Kandinsky6 SR supports instruct_type='hybrid_anchor' with visual_cond=True, got "
                                      f"{self.instruct_type!r} / visual_cond={self.visual_cond}.")
        if self.patch_size[0] != 1:
            raise ValueError(f"Kandinsky6 SR requires patch_size[0] == 1, got {self.patch_size}")
        head_dim = sum(self.axes_dims)
        if self.model_dim % head_dim:
            raise ValueError(f"model_dim ({self.model_dim}) must be divisible by head_dim ({head_dim})")
        if self.sr_params:
            for key, expected in _SUPPORTED_SR_PARAMS.items():
                if self.sr_params.get(key, expected) != expected:
                    raise NotImplementedError(f"Kandinsky6 SR: sr_params.{key}={self.sr_params[key]!r} is not "
                                              f"supported (expected {expected!r}).")

        self.hidden_size = self.model_dim
        self.num_attention_heads = self.model_dim // head_dim
        self.in_channels = self.in_visual_dim
        self.out_channels = self.out_visual_dim
        self.num_channels_latents = self.in_visual_dim

    @property
    def input_channels(self) -> int:
        """``[noised latent | anchor | anchor mask]``."""
        return 2 * self.in_visual_dim + 1

    @property
    def visual_size(self) -> int:
        size = self.sr_params.get("visual_size", 512)
        return int(size[0] if isinstance(size, list | tuple) else size)

    @property
    def rope_scale_factor(self) -> tuple[float, ...]:
        """RoPE frequency scale per (t, h, w) axis for :attr:`visual_size`."""
        factors = self.sr_params.get("scale_factor", {})
        values = factors.get(str(self.visual_size), factors.get(self.visual_size, (1.0, 1.0, 1.0)))
        return tuple(float(v) for v in values)

    @property
    def lq_noise_scale(self) -> float:
        return float(self.sr_params.get("lq_noise_scale", 0.7))

    def known_config_keys(self) -> frozenset[str]:
        return frozenset(self.__dataclass_fields__) | _METADATA_KEYS


@dataclass
class Kandinsky6SRConfig(DiTConfig):
    arch_config: DiTArchConfig = field(default_factory=Kandinsky6SRArchConfig)
    prefix: str = "Kandinsky6SR"
