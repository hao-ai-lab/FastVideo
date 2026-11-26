# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Bytedance Ltd. and/or its affiliate
# Copyright 2025 The Wan Team and The HuggingFace Team. All rights reserved.
# Source-ID RoPE and token conditioning adapted from ByteDance/Bernini
# (Apache-2.0), bernini/models/transformer_wan.py.
"""Native single-sample Bernini-R transformer, reusing the Wan tower.

Reference tokens precede target tokens. Both use their own (t,h,w) grid;
the reference has source ID 1 and the target source ID 0. Only the target
is unpatchified. Output patches use (pT,pH,pW,C), as in the official model.
The first integration supports one GPU; packed multi-sample/SP is rejected.
"""

import math
from functools import lru_cache

import torch

from fastvideo.configs.models.dits.bernini import BerniniRConfig
from fastvideo.distributed.parallel_state import get_sp_world_size, get_tp_world_size
from fastvideo.models.wan.transformer import WanTransformer3DModel, WanTransformerBlock
from fastvideo.platforms import AttentionBackendEnum


@lru_cache(maxsize=32)
def source_rotary_embedding(grid: tuple[int, int, int], head_dim: int,
                            source_id: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Interleaved float64 cos/sin, matching the reference's complex128 RoPE.

    Float32 tables change rare BF16 rounding bins and accumulate across the
    released model's 30 layers, even when FP32 output looks nearly identical.
    """
    widths = (head_dim - 4 * (head_dim // 6), 2 * (head_dim // 6), 2 * (head_dim // 6))
    coordinates = torch.meshgrid(*(torch.arange(n, dtype=torch.float64) for n in grid), indexing="ij")
    phases = torch.cat([
        coord.flatten()[:, None] / (10000**(torch.arange(0, width, 2, dtype=torch.float64) / width))
        for coord, width in zip(coordinates, widths, strict=True)
    ], dim=-1)
    phases = phases + source_id / (10000**(torch.arange(0, head_dim, 2, dtype=torch.float64) / head_dim))
    return phases.cos().repeat_interleave(2, dim=-1), phases.sin().repeat_interleave(2, dim=-1)


def pack_latents(latents: torch.Tensor) -> torch.Tensor:
    """[B,C,T,H,W] -> [B,T*H/2*W/2,4*C], patch order (pH,pW,C)."""
    b, c, t, h, w = latents.shape
    return latents.reshape(b, c, t, h // 2, 2, w // 2, 2).permute(0, 2, 3, 5, 4, 6, 1).reshape(b, -1, 4 * c)


def unpack_latents(tokens: torch.Tensor, shape: tuple[int, ...]) -> torch.Tensor:
    b, c, t, h, w = shape
    return tokens.reshape(b, t, h // 2, w // 2, 2, 2, c).permute(0, 6, 1, 2, 4, 3, 5).reshape(shape)


class BerniniRTransformerBlock(WanTransformerBlock):
    """Wan-owned layers with the released Bernini residual rounding points.

    In BF16, the residual is rounded before each FP32 norm. The shared Wan
    fused residual/norm chain deliberately has different rounding points.
    """

    def forward(self, hidden_states, encoder_hidden_states, temb, freqs_cis, original_seq_len):
        dtype = hidden_states.dtype
        shift, scale, gate, ff_shift, ff_scale, ff_gate = (
            self.scale_shift_table.float() + temb.float()).chunk(6, dim=1)
        normalized = (self.norm1(hidden_states.float()) * (1 + scale) + shift).to(dtype)
        query, _ = self.to_q(normalized)
        key, _ = self.to_k(normalized)
        value, _ = self.to_v(normalized)
        query, key = self.norm_q(query), self.norm_k(key)
        query, key, value = (x.unflatten(2, (self.num_attention_heads, -1)) for x in (query, key, value))
        attended, _ = self.attn1(query, key, value, original_seq_len, freqs_cis=freqs_cis)
        attended, _ = self.to_out(attended.flatten(2))
        hidden_states = (hidden_states.float() + attended * gate).to(dtype)
        normalized = self.self_attn_residual_norm.norm(hidden_states.float()).to(dtype)
        hidden_states = hidden_states + self.attn2(normalized, context=encoder_hidden_states, context_lens=None)
        normalized = (self.cross_attn_residual_norm.norm(hidden_states.float()) * (1 + ff_scale) + ff_shift).to(dtype)
        return (hidden_states.float() + self.ffn(normalized).float() * ff_gate).to(dtype)


class BerniniRTransformer3DModel(WanTransformer3DModel):
    _supported_attention_backends = (AttentionBackendEnum.FLASH_ATTN, AttentionBackendEnum.TORCH_SDPA)
    _keep_in_fp32_modules = [
        "condition_embedder.time_embedder", "scale_shift_table", "self_attn_residual_norm.norm"
    ]

    def _get_parameter_dtype(self, name: str, default_dtype: torch.dtype) -> torch.dtype:
        # Mirror the official loader's time_embedder/scale_shift_table/norm2
        # exceptions before casting checkpoint tensors, not after losing bits.
        if (name.startswith("condition_embedder.time_embedder.")
                or "scale_shift_table" in name.split(".")
                or ".self_attn_residual_norm.norm." in name):
            return torch.float32
        return default_dtype

    def __init__(self, config: BerniniRConfig, hf_config: dict):
        if get_sp_world_size() != 1 or get_tp_world_size() != 1:
            raise ValueError("Bernini-R currently supports sp_size=tp_size=1 only")
        super().__init__(config, hf_config)
        self.blocks = torch.nn.ModuleList([
            BerniniRTransformerBlock(self.hidden_size, config.ffn_dim, config.num_attention_heads,
                                     config.qk_norm, config.cross_attn_norm, config.eps,
                                     config.added_kv_proj_dim, self._supported_attention_backends,
                                     quant_config=config.quant_config, prefix=f"{config.prefix}.blocks.{i}")
            for i in range(config.num_layers)
        ])

    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor,
                timestep: torch.Tensor, source_latents: torch.Tensor | None = None, **kwargs) -> torch.Tensor:
        if hidden_states.ndim != 5 or hidden_states.shape[0] != 1:
            raise ValueError("Bernini-R requires a single [1,C,T,H,W] target")
        if tuple(self.patch_size) != (1, 2, 2):
            raise ValueError("Bernini-R requires patch_size=(1,2,2)")
        if any(n % 2 for n in hidden_states.shape[-2:]):
            raise ValueError("Bernini-R latent height/width must be divisible by 2")
        streams, tables = [], []
        for latent, sid in ((source_latents, 1), (hidden_states, 0)):
            if latent is None:
                continue
            if latent.ndim != 5 or latent.shape[:2] != hidden_states.shape[:2]:
                raise ValueError("Source and target must have matching batch/channel dimensions")
            if any(n % 2 for n in latent.shape[-2:]):
                raise ValueError("Bernini-R source latent height/width must be divisible by 2")
            patches = self.patch_embedding(latent)
            streams.append(patches.flatten(2).transpose(1, 2))
            tables.append(source_rotary_embedding(tuple(patches.shape[2:]), self.config.attention_head_dim, sid))
        target_tokens = streams[-1].shape[1]
        tokens = torch.cat(streams, dim=1)
        rope = tuple(torch.cat([table[i] for table in tables]).to(tokens.device) for i in range(2))
        # The reference computes time embedding in FP32, then rounds to the
        # text dtype *before* SiLU/time projection. Shared Wan does not use
        # this Bernini-specific boundary, so keep the override local.
        time_embedder = self.condition_embedder.time_embedder.mlp
        half = self.config.freq_dim // 2
        # Diffusers builds frequencies on the timestep device. Building them
        # on CPU then copying changes FP32 last bits and BF16 modulation bins.
        exponent = -math.log(10000) * torch.arange(half, dtype=torch.float32, device=timestep.device) / half
        phase = timestep[:, None].float() * exponent.exp()[None]
        frequencies = torch.cat([phase.cos(), phase.sin()], dim=-1).to(time_embedder.fc_in.weight.dtype)
        temb = time_embedder(frequencies).to(encoder_hidden_states.dtype)
        modulation = self.condition_embedder.time_modulation(temb)
        context = self.condition_embedder.text_embedder(encoder_hidden_states)
        modulation = modulation.unflatten(1, (6, -1))
        for block in self.blocks:
            tokens = block(tokens, context, modulation, rope, tokens.shape[1])
        shift, scale = (self.scale_shift_table.float() + temb.float().unsqueeze(1)).chunk(2, dim=1)
        normalized = self.norm_out.norm(tokens.float()) * (1 + scale) + shift
        tokens = self.proj_out(normalized.to(tokens.dtype))
        return unpack_latents(tokens[:, -target_tokens:], tuple(hidden_states.shape))


EntryClass = BerniniRTransformer3DModel
