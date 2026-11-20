# SPDX-License-Identifier: Apache-2.0
"""Qwen3.5 text encoder configuration (hybrid Gated DeltaNet + gated attention)."""
from dataclasses import dataclass, field
from typing import Any

from fastvideo.configs.models.encoders.base import (
    TextEncoderArchConfig,
    TextEncoderConfig,
)

_LEGACY_LAYER_TYPES = {
    "mamba": "linear_attention",
    "conv": "linear_attention",
    "attention": "full_attention",
}
_DEFAULT_ROPE_THETA = 10000.0
_DEFAULT_PARTIAL_ROTARY_FACTOR = 0.25


def _is_transformer_layer(n: str, m: Any) -> bool:
    return "layers" in n and str.isdigit(n.split(".")[-1])


def _is_embeddings(n: str, m: Any) -> bool:
    return n.endswith("embed_tokens")


def _is_final_norm(n: str, m: Any) -> bool:
    return n == "norm"


@dataclass
class Qwen3_5TextArchConfig(TextEncoderArchConfig):
    """Architecture config for the Qwen3.5 text model (``Qwen3_5TextModel``).

    Layers alternate between Gated DeltaNet linear attention and gated full
    attention according to ``layer_types``. The defaults describe the 9B text
    tower; checkpoint ``config.json`` values override them at load time.
    """

    vocab_size: int = 248320
    hidden_size: int = 4096
    intermediate_size: int = 12288
    num_hidden_layers: int = 32
    num_attention_heads: int = 16
    num_key_value_heads: int = 4
    head_dim: int = 256
    hidden_act: str = "silu"
    max_position_embeddings: int = 32768
    initializer_range: float = 0.02
    rms_norm_eps: float = 1e-6
    use_cache: bool = True
    tie_word_embeddings: bool = False
    attention_bias: bool = False
    attention_dropout: float = 0.0
    linear_conv_kernel_dim: int = 4
    linear_key_head_dim: int = 128
    linear_value_head_dim: int = 128
    linear_num_key_heads: int = 16
    linear_num_value_heads: int = 32
    layer_types: list[str] | None = None
    full_attention_interval: int = 4
    # RoPE settings as written by Transformers 5 (``rope_parameters``) or by
    # older configs (top-level ``rope_theta`` / ``partial_rotary_factor`` /
    # ``rope_scaling``). Values inside ``rope_parameters`` take precedence.
    rope_parameters: dict[str, Any] | None = None
    rope_scaling: dict[str, Any] | None = None
    rope_theta: float | None = None
    partial_rotary_factor: float | None = None
    pad_token_id: int | None = None
    bos_token_id: int | None = None
    eos_token_id: int | list[int] | None = None

    _fsdp_shard_conditions: list = field(
        default_factory=lambda: [_is_transformer_layer, _is_embeddings, _is_final_norm])

    def __post_init__(self) -> None:
        super().__post_init__()
        # Validate eagerly; the resolved views are recomputed on demand because
        # ``update_model_arch`` re-runs this hook after replacing raw fields.
        self.resolved_layer_types()
        self.resolved_rope_parameters()

    def resolved_layer_types(self) -> list[str]:
        """Return one ``linear_attention`` / ``full_attention`` entry per layer."""
        if self.layer_types is None:
            interval = self.full_attention_interval
            layer_types = [
                "linear_attention" if (index + 1) % interval else "full_attention"
                for index in range(self.num_hidden_layers)
            ]
        else:
            layer_types = [_LEGACY_LAYER_TYPES.get(kind, kind) for kind in self.layer_types]
        if len(layer_types) < self.num_hidden_layers:
            raise ValueError(f"Qwen3.5 layer_types has {len(layer_types)} entries for "
                             f"{self.num_hidden_layers} layers")
        unknown = set(layer_types) - {"linear_attention", "full_attention"}
        if unknown:
            raise ValueError(f"Unsupported Qwen3.5 layer types: {sorted(unknown)}")
        return layer_types[:self.num_hidden_layers]

    def resolved_rope_parameters(self) -> dict[str, Any]:
        """Return the effective text RoPE settings with Transformers' precedence."""
        rope = dict(self.rope_scaling or self.rope_parameters or {})
        rope.setdefault("rope_type", rope.get("type", "default"))
        rope.setdefault("rope_theta", self.rope_theta if self.rope_theta is not None else _DEFAULT_ROPE_THETA)
        rope.setdefault(
            "partial_rotary_factor",
            self.partial_rotary_factor if self.partial_rotary_factor is not None else _DEFAULT_PARTIAL_ROTARY_FACTOR,
        )
        return rope


@dataclass
class Qwen3_5TextConfig(TextEncoderConfig):
    """Top-level config for the Qwen3.5 text encoder."""

    arch_config: TextEncoderArchConfig = field(default_factory=Qwen3_5TextArchConfig)
    prefix: str = "qwen3_5"
