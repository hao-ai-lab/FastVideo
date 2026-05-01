# SPDX-License-Identifier: Apache-2.0
"""Config for the Stable Audio Open 1.0 DiT.

Note: the SA pipeline bypasses the standard `ComposedPipelineBase`
component loader because the published HF repo ships a single monolithic
`model.safetensors` (no Diffusers-style `model_index.json` or
per-subfolder layout). The arch fields and `param_names_mapping` here
document the architecture and key remap so the same conventions used by
the rest of the DiT family apply (FSDP shard conditions, supported
attention backends, future loader integrations) — they are not currently
consumed by `fastvideo/models/loader/fsdp_load.py` for SA.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from fastvideo.configs.models.dits.base import DiTArchConfig, DiTConfig
from fastvideo.platforms import AttentionBackendEnum


def _is_transformer_layer(n: str, m) -> bool:
    # Matches `transformer.layers.{i}` in the SA DiT module tree.
    parts = n.split(".")
    return (len(parts) >= 3 and parts[-3] == "transformer" and parts[-2] == "layers" and parts[-1].isdigit())


@dataclass
class StableAudioArchConfig(DiTArchConfig):
    _fsdp_shard_conditions: list = field(default_factory=lambda: [_is_transformer_layer])

    # SA's checkpoint is `stable_audio_tools` raw format (not Diffusers),
    # so the only remaps are: strip the `model.model.` host-pipeline
    # prefix, and rename `nn.LayerNorm`'s `gamma`/`beta` to torch's
    # canonical `weight`/`bias`. Linear / cross-attention naming already
    # matches FastVideo's conventions, so no further remap is needed.
    param_names_mapping: dict = field(
        default_factory=lambda: {
            r"^model\.model\.(.*?)\.gamma$": r"\1.weight",
            r"^model\.model\.(.*?)\.beta$": r"\1.bias",
            r"^model\.model\.(.*)$": r"\1",
        })

    # SA only supports backends compatible with single-GPU LocalAttention.
    _supported_attention_backends: tuple[AttentionBackendEnum, ...] = (
        AttentionBackendEnum.FLASH_ATTN,
        AttentionBackendEnum.TORCH_SDPA,
    )

    # Architecture constants (from the published `model_config.json` for
    # `stabilityai/stable-audio-open-1.0`).
    io_channels: int = 64
    embed_dim: int = 1536
    depth: int = 24
    num_attention_heads: int = 24
    cond_token_dim: int = 768
    global_cond_dim: int = 1536
    project_cond_tokens: bool = False
    project_global_cond: bool = True

    def __post_init__(self) -> None:
        super().__post_init__()
        self.hidden_size = self.embed_dim
        self.in_channels = self.io_channels
        self.out_channels = self.io_channels
        self.num_channels_latents = self.io_channels
        self.attention_head_dim = self.embed_dim // self.num_attention_heads


@dataclass
class StableAudioConfig(DiTConfig):
    arch_config: DiTArchConfig = field(default_factory=StableAudioArchConfig)

    prefix: str = "StableAudio"
