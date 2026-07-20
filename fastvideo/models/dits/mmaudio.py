# SPDX-License-Identifier: Apache-2.0
"""MMAudio multimodal flow-prediction transformer.

The model operates on one-dimensional audio latent sequences and jointly
attends to semantic video, synchronization-video, and text conditions. The
initial implementation intentionally uses torch SDPA so its single-GPU numeric
contract remains explicit; sequence/tensor parallel support is a later,
separately verified optimization.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from einops import rearrange
from einops.layers.torch import Rearrange
from torch import nn

from fastvideo.configs.models.dits.mmaudio import MMAudioTransformerConfig
from fastvideo.models.dits.base import BaseDiT
from fastvideo.models.loader.utils import get_param_names_mapping


def compute_rope_rotations(
    length: int, dim: int, theta: int, *, freq_scaling: float = 1.0, device: torch.device | str = "cpu"
) -> torch.Tensor:
    if dim % 2 != 0:
        raise ValueError(f"RoPE dimension must be even, got {dim}.")
    with torch.amp.autocast(device_type="cuda", enabled=False):
        pos = torch.arange(length, dtype=torch.float32, device=device)
        freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim))
        freqs *= freq_scaling
        rotations = torch.einsum("..., f -> ... f", pos, freqs)
        rotations = torch.stack(
            [
                torch.cos(rotations),
                -torch.sin(rotations),
                torch.sin(rotations),
                torch.cos(rotations),
            ],
            dim=-1,
        )
        return rearrange(rotations, "n d (i j) -> 1 n d i j", i=2, j=2)


def apply_rope(x: torch.Tensor, rotations: torch.Tensor) -> torch.Tensor:
    with torch.amp.autocast(device_type="cuda", enabled=False):
        source = x.float().view(*x.shape[:-1], -1, 1, 2)
        output = rotations[..., 0] * source[..., 0] + rotations[..., 1] * source[..., 1]
        return output.reshape(*x.shape).to(dtype=x.dtype)


class ChannelLastConv1d(nn.Conv1d):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(x.permute(0, 2, 1)).permute(0, 2, 1)


class MMAudioMLP(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, multiple_of: int = 256) -> None:
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class MMAudioConvMLP(nn.Module):
    def __init__(
        self, dim: int, hidden_dim: int, multiple_of: int = 256, kernel_size: int = 3, padding: int = 1
    ) -> None:
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
        self.w1 = ChannelLastConv1d(dim, hidden_dim, bias=False, kernel_size=kernel_size, padding=padding)
        self.w2 = ChannelLastConv1d(hidden_dim, dim, bias=False, kernel_size=kernel_size, padding=padding)
        self.w3 = ChannelLastConv1d(dim, hidden_dim, bias=False, kernel_size=kernel_size, padding=padding)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class TimestepEmbedder(nn.Module):
    def __init__(self, dim: int, frequency_embedding_size: int, max_period: int) -> None:
        super().__init__()
        if dim % 2 != 0:
            raise ValueError(f"Timestep embedding dim must be even, got {dim}.")
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )
        self.dim = dim
        self.max_period = max_period
        with torch.autocast("cuda", enabled=False):
            freqs = 1.0 / (
                10000 ** (torch.arange(0, frequency_embedding_size, 2, dtype=torch.float32) / frequency_embedding_size)
            )
            self.register_buffer("freqs", (10000 / max_period) * freqs, persistent=False)

    def timestep_embedding(self, timestep: torch.Tensor) -> torch.Tensor:
        args = timestep[:, None].float() * self.freqs[None]
        return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)

    def forward(self, timestep: torch.Tensor) -> torch.Tensor:
        embedding = self.timestep_embedding(timestep).to(timestep.dtype)
        return self.mlp(embedding)


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x * (1 + scale) + shift


def mmaudio_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    output = F.scaled_dot_product_attention(q.contiguous(), k.contiguous(), v.contiguous())
    return rearrange(output, "b h n d -> b n (h d)").contiguous()


class SelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int) -> None:
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.q_norm = nn.RMSNorm(dim // num_heads)
        self.k_norm = nn.RMSNorm(dim // num_heads)
        self.split_into_heads = Rearrange(
            "b n (h d j) -> b h n d j",
            h=num_heads,
            d=dim // num_heads,
            j=3,
        )

    def pre_attention(
        self, x: torch.Tensor, rotations: torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        q, k, v = self.split_into_heads(self.qkv(x)).chunk(3, dim=-1)
        q = self.q_norm(q.squeeze(-1))
        k = self.k_norm(k.squeeze(-1))
        v = v.squeeze(-1)
        if rotations is not None:
            q = apply_rope(q, rotations)
            k = apply_rope(k, rotations)
        return q, k, v

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return mmaudio_attention(*self.pre_attention(x, None))


class MMDitSingleBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        pre_only: bool = False,
        kernel_size: int = 7,
        padding: int = 3,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False)
        self.attn = SelfAttention(dim, num_heads)
        self.pre_only = pre_only
        if pre_only:
            self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, 2 * dim, bias=True))
        else:
            self.linear1 = (
                nn.Linear(dim, dim)
                if kernel_size == 1
                else ChannelLastConv1d(dim, dim, kernel_size=kernel_size, padding=padding)
            )
            self.norm2 = nn.LayerNorm(dim, elementwise_affine=False)
            self.ffn = (
                MMAudioMLP(dim, int(dim * mlp_ratio))
                if kernel_size == 1
                else MMAudioConvMLP(
                    dim,
                    int(dim * mlp_ratio),
                    kernel_size=kernel_size,
                    padding=padding,
                )
            )
            self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim, bias=True))

    def pre_attention(self, x: torch.Tensor, condition: torch.Tensor, rotations: torch.Tensor | None):
        modulation = self.adaLN_modulation(condition)
        if self.pre_only:
            shift_msa, scale_msa = modulation.chunk(2, dim=-1)
            gate_msa = shift_mlp = scale_mlp = gate_mlp = None
        else:
            (shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp) = modulation.chunk(6, dim=-1)
        normalized = modulate(self.norm1(x), shift_msa, scale_msa)
        qkv = self.attn.pre_attention(normalized, rotations)
        return qkv, (gate_msa, shift_mlp, scale_mlp, gate_mlp)

    def post_attention(self, x: torch.Tensor, attention_output: torch.Tensor, condition):
        if self.pre_only:
            return x
        gate_msa, shift_mlp, scale_mlp, gate_mlp = condition
        x = x + self.linear1(attention_output) * gate_msa
        residual = modulate(self.norm2(x), shift_mlp, scale_mlp)
        return x + self.ffn(residual) * gate_mlp

    def forward(self, x: torch.Tensor, condition: torch.Tensor, rotations: torch.Tensor | None) -> torch.Tensor:
        qkv, modulation = self.pre_attention(x, condition, rotations)
        return self.post_attention(x, mmaudio_attention(*qkv), modulation)


class JointBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0, pre_only: bool = False) -> None:
        super().__init__()
        self.pre_only = pre_only
        self.latent_block = MMDitSingleBlock(dim, num_heads, mlp_ratio, pre_only=False, kernel_size=3, padding=1)
        self.clip_block = MMDitSingleBlock(dim, num_heads, mlp_ratio, pre_only=pre_only, kernel_size=3, padding=1)
        self.text_block = MMDitSingleBlock(dim, num_heads, mlp_ratio, pre_only=pre_only, kernel_size=1)

    def forward(
        self,
        latent: torch.Tensor,
        clip_features: torch.Tensor,
        text_features: torch.Tensor,
        global_condition: torch.Tensor,
        extended_condition: torch.Tensor,
        latent_rotations: torch.Tensor,
        clip_rotations: torch.Tensor,
    ):
        latent_qkv, latent_mod = self.latent_block.pre_attention(latent, extended_condition, latent_rotations)
        clip_qkv, clip_mod = self.clip_block.pre_attention(clip_features, global_condition, clip_rotations)
        text_qkv, text_mod = self.text_block.pre_attention(text_features, global_condition, None)

        latent_len = latent.shape[1]
        clip_len = clip_features.shape[1]
        joint_qkv = [torch.cat([latent_qkv[i], clip_qkv[i], text_qkv[i]], dim=2) for i in range(3)]
        attention_output = mmaudio_attention(*joint_qkv)
        latent_output = attention_output[:, :latent_len]
        clip_output = attention_output[:, latent_len : latent_len + clip_len]
        text_output = attention_output[:, latent_len + clip_len :]

        latent = self.latent_block.post_attention(latent, latent_output, latent_mod)
        if not self.pre_only:
            clip_features = self.clip_block.post_attention(clip_features, clip_output, clip_mod)
            text_features = self.text_block.post_attention(text_features, text_output, text_mod)
        return latent, clip_features, text_features


class FinalBlock(nn.Module):
    def __init__(self, dim: int, out_dim: int) -> None:
        super().__init__()
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, 2 * dim, bias=True))
        self.norm = nn.LayerNorm(dim, elementwise_affine=False)
        self.conv = ChannelLastConv1d(dim, out_dim, kernel_size=7, padding=3)

    def forward(self, latent: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        shift, scale = self.adaLN_modulation(condition).chunk(2, dim=-1)
        return self.conv(modulate(self.norm(latent), shift, scale))


@dataclass
class PreprocessedConditions:
    clip_f: torch.Tensor
    sync_f: torch.Tensor
    text_f: torch.Tensor
    clip_f_c: torch.Tensor
    text_f_c: torch.Tensor


_DEFAULT_CONFIG = MMAudioTransformerConfig()


class MMAudioTransformer(BaseDiT):
    _fsdp_shard_conditions = _DEFAULT_CONFIG.arch_config._fsdp_shard_conditions
    _compile_conditions = _DEFAULT_CONFIG.arch_config._compile_conditions
    _supported_attention_backends = _DEFAULT_CONFIG.arch_config._supported_attention_backends
    param_names_mapping = get_param_names_mapping(_DEFAULT_CONFIG.arch_config.param_names_mapping)
    reverse_param_names_mapping = _DEFAULT_CONFIG.arch_config.reverse_param_names_mapping

    def __init__(self, config: MMAudioTransformerConfig, hf_config: dict[str, Any], **kwargs) -> None:
        del kwargs
        super().__init__(config=config, hf_config=hf_config)
        arch = config.arch_config
        self.v2 = arch.v2
        self.latent_dim = arch.latent_dim
        self._latent_seq_len = arch.latent_seq_len
        self._clip_seq_len = arch.clip_seq_len
        self._sync_seq_len = arch.sync_seq_len
        self._text_seq_len = arch.text_seq_len
        self.hidden_dim = arch.hidden_dim
        self.num_heads = arch.num_heads
        self.hidden_size = arch.hidden_size
        self.num_attention_heads = arch.num_attention_heads
        self.num_channels_latents = arch.num_channels_latents

        activation = nn.SiLU if arch.v2 else nn.SELU
        self.audio_input_proj = nn.Sequential(
            ChannelLastConv1d(arch.latent_dim, arch.hidden_dim, kernel_size=7, padding=3),
            activation(),
            MMAudioConvMLP(arch.hidden_dim, arch.hidden_dim * 4, kernel_size=7, padding=3),
        )
        clip_layers: list[nn.Module] = [nn.Linear(arch.clip_dim, arch.hidden_dim)]
        if arch.v2:
            clip_layers.append(nn.SiLU())
        clip_layers.append(MMAudioConvMLP(arch.hidden_dim, arch.hidden_dim * 4, kernel_size=3, padding=1))
        self.clip_input_proj = nn.Sequential(*clip_layers)
        self.sync_input_proj = nn.Sequential(
            ChannelLastConv1d(arch.sync_dim, arch.hidden_dim, kernel_size=7, padding=3),
            activation(),
            MMAudioConvMLP(arch.hidden_dim, arch.hidden_dim * 4, kernel_size=3, padding=1),
        )
        text_layers: list[nn.Module] = [nn.Linear(arch.text_dim, arch.hidden_dim)]
        if arch.v2:
            text_layers.append(nn.SiLU())
        text_layers.append(MMAudioMLP(arch.hidden_dim, arch.hidden_dim * 4))
        self.text_input_proj = nn.Sequential(*text_layers)

        self.clip_cond_proj = nn.Linear(arch.hidden_dim, arch.hidden_dim)
        self.text_cond_proj = nn.Linear(arch.hidden_dim, arch.hidden_dim)
        self.global_cond_mlp = MMAudioMLP(arch.hidden_dim, arch.hidden_dim * 4)
        self.sync_pos_emb = nn.Parameter(torch.zeros((1, 1, 8, arch.sync_dim)))
        self.final_layer = FinalBlock(arch.hidden_dim, arch.latent_dim)
        self.t_embed = TimestepEmbedder(
            arch.hidden_dim,
            frequency_embedding_size=(arch.hidden_dim if arch.v2 else 256),
            max_period=(1 if arch.v2 else 10000),
        )
        self.joint_blocks = nn.ModuleList(
            [
                JointBlock(
                    arch.hidden_dim,
                    arch.num_heads,
                    mlp_ratio=arch.mlp_ratio,
                    pre_only=(index == arch.depth - arch.fused_depth - 1),
                )
                for index in range(arch.depth - arch.fused_depth)
            ]
        )
        self.fused_blocks = nn.ModuleList(
            [
                MMDitSingleBlock(arch.hidden_dim, arch.num_heads, mlp_ratio=arch.mlp_ratio, kernel_size=3, padding=1)
                for _ in range(arch.fused_depth)
            ]
        )

        self.latent_mean = nn.Parameter(torch.full((1, 1, arch.latent_dim), float("nan")), requires_grad=False)
        self.latent_std = nn.Parameter(torch.full((1, 1, arch.latent_dim), float("nan")), requires_grad=False)
        self.empty_string_feat = nn.Parameter(torch.zeros((arch.text_seq_len, arch.text_dim)), requires_grad=False)
        self.empty_clip_feat = nn.Parameter(torch.zeros(1, arch.clip_dim), requires_grad=True)
        self.empty_sync_feat = nn.Parameter(torch.zeros(1, arch.sync_dim), requires_grad=True)

        self.initialize_weights()
        self.initialize_rotations()
        self.__post_init__()

    @property
    def device(self) -> torch.device:
        return self.latent_mean.device

    @property
    def latent_seq_len(self) -> int:
        return self._latent_seq_len

    @property
    def clip_seq_len(self) -> int:
        return self._clip_seq_len

    @property
    def sync_seq_len(self) -> int:
        return self._sync_seq_len

    def initialize_rotations(self) -> None:
        head_dim = self.hidden_dim // self.num_heads
        latent_rotations = compute_rope_rotations(self._latent_seq_len, head_dim, 10000, device=self.device)
        clip_rotations = compute_rope_rotations(
            self._clip_seq_len,
            head_dim,
            10000,
            freq_scaling=self._latent_seq_len / self._clip_seq_len,
            device=self.device,
        )
        self.register_buffer("latent_rot", latent_rotations, persistent=False)
        self.register_buffer("clip_rot", clip_rotations, persistent=False)

    def update_seq_lengths(self, latent_seq_len: int, clip_seq_len: int, sync_seq_len: int) -> None:
        self._latent_seq_len = latent_seq_len
        self._clip_seq_len = clip_seq_len
        self._sync_seq_len = sync_seq_len
        self.initialize_rotations()

    def initialize_weights(self) -> None:

        def basic_init(module: nn.Module) -> None:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(basic_init)
        nn.init.normal_(self.t_embed.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embed.mlp[2].weight, std=0.02)
        for block in self.joint_blocks:
            for stream in (block.latent_block, block.clip_block, block.text_block):
                nn.init.constant_(stream.adaLN_modulation[-1].weight, 0)
                nn.init.constant_(stream.adaLN_modulation[-1].bias, 0)
        for block in self.fused_blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.conv.weight, 0)
        nn.init.constant_(self.final_layer.conv.bias, 0)
        nn.init.constant_(self.sync_pos_emb, 0)
        nn.init.constant_(self.empty_clip_feat, 0)
        nn.init.constant_(self.empty_sync_feat, 0)

    def normalize(self, latent: torch.Tensor) -> torch.Tensor:
        return latent.sub_(self.latent_mean).div_(self.latent_std)

    def unnormalize(self, latent: torch.Tensor) -> torch.Tensor:
        return latent.mul_(self.latent_std).add_(self.latent_mean)

    def preprocess_conditions(
        self, clip_features: torch.Tensor, sync_features: torch.Tensor, text_features: torch.Tensor
    ) -> PreprocessedConditions:
        if clip_features.shape[1] != self._clip_seq_len:
            raise ValueError(f"Expected {self._clip_seq_len} CLIP tokens, got {clip_features.shape}.")
        if sync_features.shape[1] != self._sync_seq_len:
            raise ValueError(f"Expected {self._sync_seq_len} sync tokens, got {sync_features.shape}.")
        if text_features.shape[1] != self._text_seq_len:
            raise ValueError(f"Expected {self._text_seq_len} text tokens, got {text_features.shape}.")
        batch_size = clip_features.shape[0]
        sync_features = sync_features.view(batch_size, self._sync_seq_len // 8, 8, -1) + self.sync_pos_emb
        sync_features = sync_features.flatten(1, 2)
        clip_features = self.clip_input_proj(clip_features)
        sync_features = self.sync_input_proj(sync_features)
        text_features = self.text_input_proj(text_features)
        sync_features = F.interpolate(
            sync_features.transpose(1, 2),
            size=self._latent_seq_len,
            mode="nearest-exact",
        ).transpose(1, 2)
        return PreprocessedConditions(
            clip_f=clip_features,
            sync_f=sync_features,
            text_f=text_features,
            clip_f_c=self.clip_cond_proj(clip_features.mean(dim=1)),
            text_f_c=self.text_cond_proj(text_features.mean(dim=1)),
        )

    def predict_flow(
        self, latent: torch.Tensor, timestep: torch.Tensor, conditions: PreprocessedConditions
    ) -> torch.Tensor:
        if latent.shape[1] != self._latent_seq_len:
            raise ValueError(f"Expected latent length {self._latent_seq_len}, got {latent.shape}.")
        clip_features = conditions.clip_f
        text_features = conditions.text_f
        latent = self.audio_input_proj(latent)
        global_condition = self.global_cond_mlp(conditions.clip_f_c + conditions.text_f_c)
        global_condition = self.t_embed(timestep).unsqueeze(1) + global_condition.unsqueeze(1)
        extended_condition = global_condition + conditions.sync_f
        for block in self.joint_blocks:
            latent, clip_features, text_features = block(
                latent,
                clip_features,
                text_features,
                global_condition,
                extended_condition,
                self.latent_rot,
                self.clip_rot,
            )
        for block in self.fused_blocks:
            latent = block(latent, extended_condition, self.latent_rot)
        # The released checkpoint was trained with global rather than sync-
        # extended conditioning at the final layer; preserve that contract.
        return self.final_layer(latent, global_condition)

    def get_empty_string_sequence(self, batch_size: int) -> torch.Tensor:
        return self.empty_string_feat.unsqueeze(0).expand(batch_size, -1, -1)

    def get_empty_clip_sequence(self, batch_size: int) -> torch.Tensor:
        return self.empty_clip_feat.unsqueeze(0).expand(batch_size, self._clip_seq_len, -1)

    def get_empty_sync_sequence(self, batch_size: int) -> torch.Tensor:
        return self.empty_sync_feat.unsqueeze(0).expand(batch_size, self._sync_seq_len, -1)

    def get_empty_conditions(
        self,
        batch_size: int,
        *,
        negative_text_features: torch.Tensor | None = None,
    ) -> PreprocessedConditions:
        empty_text = negative_text_features if negative_text_features is not None else self.get_empty_string_sequence(1)
        conditions = self.preprocess_conditions(
            self.get_empty_clip_sequence(1),
            self.get_empty_sync_sequence(1),
            empty_text,
        )
        conditions.clip_f = conditions.clip_f.expand(batch_size, -1, -1)
        conditions.sync_f = conditions.sync_f.expand(batch_size, -1, -1)
        conditions.clip_f_c = conditions.clip_f_c.expand(batch_size, -1)
        if negative_text_features is None:
            conditions.text_f = conditions.text_f.expand(batch_size, -1, -1)
            conditions.text_f_c = conditions.text_f_c.expand(batch_size, -1)
        return conditions

    def guided_flow(
        self,
        timestep: torch.Tensor,
        latent: torch.Tensor,
        conditions: PreprocessedConditions,
        empty_conditions: PreprocessedConditions,
        guidance_scale: float,
    ) -> torch.Tensor:
        timestep = timestep * torch.ones(len(latent), device=latent.device, dtype=latent.dtype)
        if guidance_scale < 1.0:
            return self.predict_flow(latent, timestep, conditions)
        return guidance_scale * self.predict_flow(latent, timestep, conditions) + (
            1 - guidance_scale
        ) * self.predict_flow(latent, timestep, empty_conditions)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: PreprocessedConditions
        | tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        | list[torch.Tensor]
        | dict[str, torch.Tensor],
        timestep: torch.LongTensor,
        encoder_hidden_states_image: torch.Tensor | list[torch.Tensor] | None = None,
        guidance=None,
        **kwargs,
    ) -> torch.Tensor:
        del encoder_hidden_states_image, guidance, kwargs
        if isinstance(encoder_hidden_states, PreprocessedConditions):
            conditions = encoder_hidden_states
        elif isinstance(encoder_hidden_states, dict):
            conditions = self.preprocess_conditions(
                encoder_hidden_states["clip_features"],
                encoder_hidden_states["sync_features"],
                encoder_hidden_states["text_features"],
            )
        elif isinstance(encoder_hidden_states, (tuple, list)) and len(encoder_hidden_states) == 3:
            conditions = self.preprocess_conditions(*encoder_hidden_states)
        else:
            raise TypeError(
                "MMAudio encoder_hidden_states must be PreprocessedConditions, "
                "a (clip, sync, text) triple, or a named condition dict."
            )
        return self.predict_flow(hidden_states, timestep, conditions)


EntryClass = MMAudioTransformer
