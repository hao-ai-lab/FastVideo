# SPDX-License-Identifier: Apache-2.0
"""FastVideo-native LingBot-Video Dense and MoE diffusion transformers."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from fastvideo.configs.models.dits.lingbot_video import LingBotVideoConfig, _is_lingbot_video_block
from fastvideo.distributed.communication_op import (
    sequence_model_parallel_all_gather_with_unpad,
    sequence_model_parallel_all_to_all_4D,
    sequence_model_parallel_shard,
)
from fastvideo.distributed.parallel_state import get_sp_world_size, model_parallel_is_initialized
from fastvideo.layers.linear import ReplicatedLinear
from fastvideo.layers.visual_embedding import Timesteps
from fastvideo.models.dits.base import BaseDiT
from fastvideo.platforms import AttentionBackendEnum


@dataclass
class LingBotVideoTransformerOutput:
    """Output container matching the released transformer contract."""

    sample: torch.Tensor


_FP32_MODULE_NAMES = (
    "time_embedder",
    "time_modulation",
    "scale_shift_table",
    "norm",
    "norm1",
    "norm2",
    "norm_q",
    "norm_k",
    "norm_post_attn",
    "norm_post_ffn",
    "norm_out",
    "norm_out_modulation",
    "router",
)


def _keep_in_fp32(name: str) -> bool:
    """Return whether a released checkpoint module keeps fp32 parameters."""
    return any(module_name in name.split(".") for module_name in _FP32_MODULE_NAMES)


def _sequence_parallel_world_size() -> int:
    """Use standalone single-rank behavior before distributed initialization."""
    return get_sp_world_size() if model_parallel_is_initialized() else 1


class LingBotVideoLinear(ReplicatedLinear):
    """Replicated FastVideo linear with the tensor-only official call contract."""

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Return the projected tensor while preserving the normal weight surface."""
        output, _ = super().forward(hidden_states)
        return output


class LingBotVideoRMSNorm(nn.Module):
    """RMSNorm with fp32 accumulation and input-dtype output."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Normalize the last dimension using the official accumulation order."""
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return (self.weight * hidden_states).to(input_dtype)


def _apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    """Apply complex 3D rotary embeddings to `(B, S, H, D)` tensors."""
    with torch.amp.autocast("cuda", enabled=False):
        x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
        output = torch.view_as_real(x_complex * freqs_cis.unsqueeze(2)).flatten(3)
        return output.type_as(x)


class LingBotVideoRotaryEmbedding(nn.Module):
    """Complex64 rotary table indexed by temporal and spatial positions."""

    def __init__(self, axes_dims: tuple[int, ...], axes_lens: tuple[int, ...], theta: float) -> None:
        super().__init__()
        self.axes_dims = tuple(axes_dims)
        self.axes_lens = list(axes_lens)
        self.theta = theta
        self.freqs_cis: list[torch.Tensor] | None = None

    @staticmethod
    def _precompute(dims: tuple[int, ...], lengths: tuple[int, ...], theta: float) -> list[torch.Tensor]:
        """Build the per-axis complex frequency tables on CPU."""
        tables: list[torch.Tensor] = []
        for dim, length in zip(dims, lengths, strict=True):
            frequencies = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float64, device="cpu") / dim))
            positions = torch.arange(length, device=frequencies.device, dtype=torch.float64)
            phases = torch.outer(positions, frequencies).float()
            tables.append(torch.polar(torch.ones_like(phases), phases).to(torch.complex64))
        return tables

    def forward(self, position_ids: torch.Tensor) -> torch.Tensor:
        """Gather and concatenate rotary frequencies for `(S, 3)` positions."""
        device = position_ids.device
        maxima = position_ids.max(dim=0).values.tolist()
        rebuild = self.freqs_cis is None or any(
            maximum >= length for maximum, length in zip(maxima, self.axes_lens, strict=True)
        )
        if rebuild:
            for index, maximum in enumerate(maxima):
                if maximum >= self.axes_lens[index]:
                    self.axes_lens[index] = int(maximum * 1.5) + 1
            self.freqs_cis = self._precompute(self.axes_dims, tuple(self.axes_lens), self.theta)
            self.freqs_cis = [table.to(device) for table in self.freqs_cis]
        elif self.freqs_cis[0].device != device:
            self.freqs_cis = [table.to(device) for table in self.freqs_cis]
        return torch.cat(
            [self.freqs_cis[index][position_ids[:, index]] for index in range(len(self.axes_dims))],
            dim=-1,
        )


def _make_joint_position_ids(
    text_len: int,
    grid_t: int,
    grid_h: int,
    grid_w: int,
    device: torch.device,
) -> torch.Tensor:
    """Create official `[video; text]` 3D positions for one sample."""
    temporal = torch.arange(grid_t, device=device, dtype=torch.int32) + text_len + 1
    height = torch.arange(grid_h, device=device, dtype=torch.int32)
    width = torch.arange(grid_w, device=device, dtype=torch.int32)
    video_positions = torch.stack(torch.meshgrid(temporal, height, width, indexing="ij"), dim=-1).flatten(0, 2)
    text_temporal = torch.arange(text_len, device=device, dtype=torch.int32) + 1
    text_positions = torch.stack(
        [text_temporal, torch.zeros_like(text_temporal), torch.zeros_like(text_temporal)], dim=-1
    )
    return torch.cat([video_positions, text_positions], dim=0)


class LingBotVideoTextEmbedder(nn.Module):
    """Project Qwen3-VL hidden states into the DiT hidden dimension."""

    def __init__(self, text_dim: int, hidden_size: int) -> None:
        super().__init__()
        self.norm = LingBotVideoRMSNorm(text_dim, eps=1e-6)
        self.linear_1 = LingBotVideoLinear(text_dim, hidden_size, bias=True)
        self.linear_2 = LingBotVideoLinear(hidden_size, hidden_size, bias=True)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Apply RMSNorm followed by the released two-layer SiLU projection."""
        hidden_states = self.norm(hidden_states)
        return self.linear_2(F.silu(self.linear_1(hidden_states)))


class LingBotVideoAttention(nn.Module):
    """Joint video-text attention shared by the Dense and MoE variants."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        norm_eps: float,
        qkv_bias: bool,
        out_bias: bool,
    ) -> None:
        """Create released QKV projections, per-head norms, and output projection."""
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.to_q = LingBotVideoLinear(hidden_size, hidden_size, bias=qkv_bias)
        self.to_k = LingBotVideoLinear(hidden_size, hidden_size, bias=qkv_bias)
        self.to_v = LingBotVideoLinear(hidden_size, hidden_size, bias=qkv_bias)
        self.norm_q = LingBotVideoRMSNorm(self.head_dim, norm_eps)
        self.norm_k = LingBotVideoRMSNorm(self.head_dim, norm_eps)
        self.to_out = LingBotVideoLinear(hidden_size, hidden_size, bias=out_bias)

    def forward(
        self,
        hidden_states: torch.Tensor,
        rotary_emb: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        original_seq_len: int | None = None,
    ) -> torch.Tensor:
        """Project QKV, apply rotary embeddings, and run non-causal SDPA."""
        batch, sequence, _ = hidden_states.shape
        query = self.to_q(hidden_states).unflatten(2, (self.num_heads, self.head_dim))
        key = self.to_k(hidden_states).unflatten(2, (self.num_heads, self.head_dim))
        value = self.to_v(hidden_states).unflatten(2, (self.num_heads, self.head_dim))
        query = _apply_rotary_emb(self.norm_q(query), rotary_emb)
        key = _apply_rotary_emb(self.norm_k(key), rotary_emb)

        if original_seq_len is not None:
            # Attention needs the full unpadded joint sequence while projections
            # and residual blocks stay sharded over tokens.
            qkv = torch.cat([query, key, value], dim=0)
            qkv = sequence_model_parallel_all_to_all_4D(qkv, scatter_dim=2, gather_dim=1)
            padded_seq_len = qkv.shape[1]
            query, key, value = qkv[:, :original_seq_len].chunk(3, dim=0)
        output = F.scaled_dot_product_attention(
            query.transpose(1, 2),
            key.transpose(1, 2),
            value.transpose(1, 2),
            attn_mask=attention_mask,
            dropout_p=0.0,
            is_causal=False,
        ).transpose(1, 2)
        if original_seq_len is not None:
            output = F.pad(output, (0, 0, 0, 0, 0, padded_seq_len - original_seq_len))
            output = sequence_model_parallel_all_to_all_4D(output, scatter_dim=1, gather_dim=2)
        return self.to_out(output.reshape(batch, sequence, -1).type_as(hidden_states))


class LingBotVideoMLP(nn.Module):
    """Dense SwiGLU feed-forward network."""

    def __init__(self, hidden_size: int, intermediate_size: int) -> None:
        super().__init__()
        self.gate_proj = LingBotVideoLinear(hidden_size, intermediate_size, bias=False)
        self.up_proj = LingBotVideoLinear(hidden_size, intermediate_size, bias=False)
        self.down_proj = LingBotVideoLinear(intermediate_size, hidden_size, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Apply the released SiLU-gated MLP ordering."""
        return self.down_proj(F.silu(self.gate_proj(hidden_states)) * self.up_proj(hidden_states))


class LingBotVideoRouter(nn.Module):
    """Released token-choice router with bias-only expert selection correction."""

    def __init__(
        self,
        hidden_size: int,
        num_experts: int,
        top_k: int,
        score_func: str,
        norm_topk_prob: bool,
        n_group: int | None,
        topk_group: int | None,
        route_scale: float,
    ) -> None:
        """Create fp32-routed expert scores with the released persistent bias."""
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.score_func = score_func
        self.norm_topk_prob = norm_topk_prob
        self.n_group = n_group
        self.topk_group = topk_group
        self.route_scale = route_scale
        self.weight = nn.Parameter(torch.empty(num_experts, hidden_size))
        self.register_buffer("e_score_correction_bias", torch.zeros(num_experts), persistent=True)

    def _group_limited_topk(self, scores_for_choice: torch.Tensor) -> torch.Tensor:
        """Restrict token choices to groups with the two strongest expert scores."""
        sequence_length = scores_for_choice.shape[0]
        experts_per_group = self.num_experts // self.n_group
        grouped = scores_for_choice.view(sequence_length, self.n_group, experts_per_group)
        group_scores = grouped.topk(2, dim=-1)[0].sum(dim=-1)
        group_indices = torch.topk(group_scores, k=self.topk_group, dim=-1, sorted=False)[1]
        group_mask = torch.zeros_like(group_scores)
        group_mask.scatter_(1, group_indices, 1)
        score_mask = (
            group_mask.unsqueeze(-1)
            .expand(sequence_length, self.n_group, experts_per_group)
            .reshape(sequence_length, -1)
        )
        masked_scores = scores_for_choice.masked_fill(~score_mask.bool(), float("-inf"))
        return torch.topk(masked_scores, k=self.top_k, dim=-1, sorted=False)[1]

    def forward(
        self, tokens: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Score in fp32, select with correction bias, and weight without it."""
        with torch.amp.autocast(tokens.device.type, enabled=False):
            logits = F.linear(tokens.float(), self.weight.float())
        scores = F.softmax(logits, dim=-1) if self.score_func == "softmax" else logits.sigmoid()
        scores_for_choice = scores + self.e_score_correction_bias.unsqueeze(0)
        if self.n_group is not None and self.n_group > 1:
            top_indices = self._group_limited_topk(scores_for_choice)
        else:
            top_indices = torch.topk(scores_for_choice, k=self.top_k, dim=-1, sorted=False)[1]
        top_scores = scores.gather(1, top_indices)
        if self.top_k > 1 and self.norm_topk_prob:
            top_scores = top_scores / (top_scores.sum(dim=-1, keepdim=True) + 1e-20)
        top_scores = top_scores * self.route_scale
        return top_indices, top_scores.to(tokens.dtype), logits, scores, scores_for_choice


class LingBotVideoGroupedExperts(nn.Module):
    """Released grouped-expert parameter layout: w1/w3 `[E,I,H]`, w2 `[E,H,I]`."""

    def __init__(self, num_experts: int, hidden_size: int, intermediate_size: int) -> None:
        super().__init__()
        self.num_experts = num_experts
        self.w1 = nn.Parameter(torch.empty(num_experts, intermediate_size, hidden_size))
        self.w2 = nn.Parameter(torch.empty(num_experts, hidden_size, intermediate_size))
        self.w3 = nn.Parameter(torch.empty(num_experts, intermediate_size, hidden_size))


def _round_up_to_multiple(value: int, multiple: int) -> int:
    """Round an integer up to the next multiple."""
    return ((value + multiple - 1) // multiple) * multiple


class LingBotVideoSparseMoeBlock(nn.Module):
    """Token-choice sparse feed-forward block matching the released state surface."""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_experts: int,
        top_k: int,
        moe_intermediate_size: int,
        score_func: str,
        norm_topk_prob: bool,
        n_group: int | None,
        topk_group: int | None,
        routed_scaling_factor: float,
        n_shared_experts: int | None,
    ) -> None:
        """Create routed and optional shared experts with released parameter names."""
        super().__init__()
        del intermediate_size
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.router = LingBotVideoRouter(
            hidden_size,
            num_experts,
            top_k,
            score_func,
            norm_topk_prob,
            n_group,
            topk_group,
            routed_scaling_factor,
        )
        self.experts = LingBotVideoGroupedExperts(num_experts, hidden_size, moe_intermediate_size)
        self.shared_experts = None
        if n_shared_experts is not None and n_shared_experts > 0:
            self.shared_experts = LingBotVideoMLP(hidden_size, moe_intermediate_size * n_shared_experts)

    @staticmethod
    def _reorder_tokens(
        tokens: torch.Tensor,
        top_scores: torch.Tensor,
        top_indices: torch.Tensor,
        num_experts: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int, int]:
        """Pack active token choices into stable expert-major order."""
        num_tokens = tokens.shape[0]
        top_k = top_indices.shape[1]
        flat_scores = top_scores.reshape(-1)
        flat_indices = top_indices.reshape(-1)
        active_positions = torch.where(flat_scores != 0)[0]
        active_experts = flat_indices[active_positions]
        counts = torch.zeros(num_experts, device=tokens.device, dtype=torch.int64)
        counts.scatter_add_(0, active_experts, torch.ones_like(active_experts, dtype=torch.int64))
        sort_order = torch.argsort(active_experts, stable=True)
        sorted_positions = active_positions[sort_order]
        sorted_scores = flat_scores[sorted_positions]
        original_token_indices = sorted_positions // top_k
        permuted_tokens = tokens[original_token_indices]
        return permuted_tokens, counts, sorted_positions, sorted_scores, num_tokens, top_k

    @staticmethod
    def _pad_grouped_tokens(
        tokens: torch.Tensor, counts: torch.Tensor, align: int = 8
    ) -> tuple[torch.Size, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Align each expert segment for `torch._grouped_mm` and retain unpad indices."""
        num_tokens = tokens.shape[0]
        num_experts = int(counts.shape[0])
        max_length = _round_up_to_multiple(num_tokens + num_experts * align, align)
        counts_i64 = counts.to(torch.int64)
        total_per_expert = torch.clamp_min(counts_i64, align)
        aligned_counts = ((total_per_expert + align - 1) // align * align).to(torch.int32)
        write_offsets = torch.cumsum(aligned_counts, dim=0) - aligned_counts
        start_indices = torch.cumsum(counts_i64, dim=0) - counts_i64
        permuted_indices = torch.full(
            (max_length,), num_tokens, dtype=torch.int64, device=tokens.device
        )
        for expert_index in range(num_experts):
            length = int(counts_i64[expert_index].item())
            if length == 0:
                continue
            write_start = int(write_offsets[expert_index].item())
            start = int(start_indices[expert_index].item())
            permuted_indices[write_start : write_start + length] = torch.arange(
                start, start + length, device=tokens.device, dtype=torch.int64
            )
        tokens_with_pad = torch.vstack((tokens, tokens.new_zeros((tokens.shape[-1],))))
        input_shape = tokens_with_pad.shape
        return input_shape, tokens_with_pad[permuted_indices], permuted_indices, aligned_counts

    @staticmethod
    def _unpad_grouped_tokens(
        output: torch.Tensor, input_shape: torch.Size, permuted_indices: torch.Tensor
    ) -> torch.Tensor:
        """Undo per-expert alignment while dropping the shared padding row."""
        unpermuted = output.new_empty(input_shape)
        unpermuted[permuted_indices, :] = output
        return unpermuted[:-1]

    def _run_grouped_experts(self, tokens: torch.Tensor, counts: torch.Tensor) -> torch.Tensor:
        """Use the released bf16 grouped matmuls on CUDA and an eager CPU fallback."""
        if tokens.device.type == "cpu" or not hasattr(torch, "_grouped_mm"):
            return self._run_experts_for_loop(tokens, counts)
        input_shape, padded_tokens, permuted_indices, aligned_counts = self._pad_grouped_tokens(tokens, counts)
        offsets = torch.cumsum(aligned_counts, dim=0, dtype=torch.int32)
        hidden = F.silu(
            torch._grouped_mm(
                padded_tokens.bfloat16(),
                self.experts.w1.bfloat16().transpose(-2, -1),
                offs=offsets,
            )
        )
        hidden = hidden * torch._grouped_mm(
            padded_tokens.bfloat16(),
            self.experts.w3.bfloat16().transpose(-2, -1),
            offs=offsets,
        )
        output = torch._grouped_mm(
            hidden,
            self.experts.w2.bfloat16().transpose(-2, -1),
            offs=offsets,
        ).type_as(padded_tokens)
        return self._unpad_grouped_tokens(output, input_shape, permuted_indices)

    def _run_experts_for_loop(self, tokens: torch.Tensor, counts: torch.Tensor) -> torch.Tensor:
        """Evaluate contiguous expert segments eagerly for CPU correctness tests."""
        splits = torch.split(tokens, counts.tolist(), dim=0)
        outputs: list[torch.Tensor] = []
        for expert_index, expert_tokens in enumerate(splits):
            if expert_tokens.numel() == 0:
                continue
            hidden = F.silu(expert_tokens @ self.experts.w1[expert_index].transpose(-2, -1))
            hidden = hidden * (expert_tokens @ self.experts.w3[expert_index].transpose(-2, -1))
            outputs.append(hidden @ self.experts.w2[expert_index].transpose(-2, -1))
        if not outputs:
            return tokens.new_zeros(tokens.shape)
        return torch.cat(outputs, dim=0)

    @staticmethod
    def _restore_tokens(
        expert_output: torch.Tensor,
        sorted_positions: torch.Tensor,
        sorted_scores: torch.Tensor,
        num_tokens: int,
        top_k: int,
    ) -> torch.Tensor:
        """Restore token order and combine expert outputs with fp32 weighted sums."""
        hidden_size = expert_output.shape[-1]
        unsorted = torch.zeros(
            (num_tokens * top_k, hidden_size),
            dtype=expert_output.dtype,
            device=expert_output.device,
        )
        unsorted[sorted_positions] = expert_output
        unsorted = unsorted.reshape(num_tokens, top_k, hidden_size)
        scores_unsorted = torch.zeros(
            num_tokens * top_k,
            dtype=sorted_scores.dtype,
            device=sorted_scores.device,
        )
        scores_unsorted[sorted_positions] = sorted_scores
        scores_unsorted = scores_unsorted.reshape(num_tokens, top_k, 1)
        return (unsorted.float() * scores_unsorted).sum(dim=1).to(expert_output.dtype)

    def _run_selected_experts(
        self,
        tokens: torch.Tensor,
        top_scores: torch.Tensor,
        top_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Dispatch routed choices, execute experts, and restore token-major order."""
        permuted_tokens, counts, sorted_positions, sorted_scores, num_tokens, top_k = self._reorder_tokens(
            tokens, top_scores, top_indices, self.router.num_experts
        )
        expert_output = self._run_grouped_experts(permuted_tokens, counts)
        return self._restore_tokens(expert_output, sorted_positions, sorted_scores, num_tokens, top_k)

    def forward(self, hidden_states: torch.Tensor, padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        """Route token choices, zero padded routes, and add optional shared experts."""
        batch = hidden_states.shape[0]
        tokens = hidden_states.view(-1, self.hidden_size)
        top_indices, top_scores, logits, scores, scores_for_choice = self.router(tokens)
        del logits, scores, scores_for_choice
        if padding_mask is not None:
            mask = padding_mask.unsqueeze(-1).to(top_scores.dtype)
            top_scores = top_scores * mask
            top_scores = top_scores / (top_scores.sum(dim=-1, keepdim=True) + 1e-9)
            top_scores = top_scores * self.router.route_scale
        output = self._run_selected_experts(tokens, top_scores, top_indices)
        output = output.view(batch, -1, self.hidden_size)
        if self.shared_experts is not None:
            output = output + self.shared_experts(hidden_states)
        return output


class LingBotVideoBlock(nn.Module):
    """One Dense or sparse LingBot-Video transformer block."""

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        intermediate_size: int,
        norm_eps: float,
        qkv_bias: bool,
        out_bias: bool,
        num_experts: int,
        num_experts_per_tok: int,
        moe_intermediate_size: int,
        decoder_sparse_step: int,
        mlp_only_layers: tuple[int, ...] | list[int],
        n_shared_experts: int | None,
        score_func: str,
        norm_topk_prob: bool,
        n_group: int | None,
        topk_group: int | None,
        routed_scaling_factor: float,
        layer_idx: int,
    ) -> None:
        """Select the released Dense or sparse feed-forward structure for one layer."""
        super().__init__()
        self.layer_idx = layer_idx
        self.scale_shift_table = nn.Parameter(torch.zeros(1, 6 * hidden_size))
        self.norm1 = LingBotVideoRMSNorm(hidden_size, norm_eps)
        self.attn = LingBotVideoAttention(hidden_size, num_attention_heads, norm_eps, qkv_bias, out_bias)
        self.norm_post_attn = LingBotVideoRMSNorm(hidden_size, norm_eps)
        self.norm2 = LingBotVideoRMSNorm(hidden_size, norm_eps)
        if layer_idx not in mlp_only_layers and (
            num_experts > 0 and (layer_idx + 1) % decoder_sparse_step == 0
        ):
            self.ffn = LingBotVideoSparseMoeBlock(
                hidden_size,
                intermediate_size,
                num_experts,
                num_experts_per_tok,
                moe_intermediate_size,
                score_func,
                norm_topk_prob,
                n_group,
                topk_group,
                routed_scaling_factor,
                n_shared_experts,
            )
        else:
            self.ffn = LingBotVideoMLP(hidden_size, intermediate_size)
        self.norm_post_ffn = LingBotVideoRMSNorm(hidden_size, norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        temb6: torch.Tensor,
        rotary_emb: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        moe_padding_mask: torch.Tensor | None = None,
        original_seq_len: int | None = None,
    ) -> torch.Tensor:
        """Run attention and configured feed-forward residual branches with fp32 AdaLN."""
        expected_tokens = hidden_states.shape[0] * hidden_states.shape[1]
        if temb6.ndim != 2 or temb6.shape[0] != expected_tokens:
            raise ValueError(
                "LingBotVideoBlock expects token-level temb6 with shape "
                f"(B*S, 6D); got {tuple(temb6.shape)} for {tuple(hidden_states.shape)}."
            )
        modulation = temb6.view(*hidden_states.shape[:2], -1) + self.scale_shift_table.unsqueeze(0)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = modulation.chunk(6, dim=-1)
        gate_msa, gate_mlp = gate_msa.tanh(), gate_mlp.tanh()
        scale_msa, scale_mlp = 1.0 + scale_msa, 1.0 + scale_mlp
        bulk_dtype = self.attn.to_q.weight.dtype

        attention_input = (self.norm1(hidden_states) * scale_msa + shift_msa).to(bulk_dtype)
        attention_output = self.attn(attention_input, rotary_emb, attention_mask, original_seq_len)
        hidden_states = hidden_states + (gate_msa * self.norm_post_attn(attention_output)).to(hidden_states.dtype)
        mlp_input = (self.norm2(hidden_states) * scale_mlp + shift_mlp).to(bulk_dtype)
        if isinstance(self.ffn, LingBotVideoSparseMoeBlock):
            mlp_output = self.ffn(mlp_input, padding_mask=moe_padding_mask)
        else:
            mlp_output = self.ffn(mlp_input)
        mlp_output = self.norm_post_ffn(mlp_output)
        return hidden_states + (gate_mlp * mlp_output).to(hidden_states.dtype)


class LingBotVideoTimestepEmbedding(nn.Module):
    """Two-layer timestep embedding with released parameter names."""

    def __init__(self, input_dim: int, hidden_size: int, bias: bool) -> None:
        super().__init__()
        self.linear_1 = LingBotVideoLinear(input_dim, hidden_size, bias=bias)
        self.linear_2 = LingBotVideoLinear(hidden_size, hidden_size, bias=bias)

    def forward(self, sample: torch.Tensor) -> torch.Tensor:
        """Apply the official linear-SiLU-linear timestep projection."""
        return self.linear_2(F.silu(self.linear_1(sample)))


class LingBotVideoTransformer3DModel(BaseDiT):
    """LingBot-Video DiT with a source-compatible Dense or MoE state surface."""

    _fsdp_shard_conditions = [_is_lingbot_video_block]
    _compile_conditions = [_is_lingbot_video_block]
    _supported_attention_backends = (
        AttentionBackendEnum.TORCH_SDPA,
        AttentionBackendEnum.FLASH_ATTN,
    )
    param_names_mapping = {r"^(.*)$": r"\1"}
    reverse_param_names_mapping = {r"^(.*)$": r"\1"}

    def _get_parameter_dtype(self, name: str, default_dtype: torch.dtype) -> torch.dtype:
        """Select the released mixed-precision dtype while loading each parameter."""
        return torch.float32 if _keep_in_fp32(name) else default_dtype

    def __init__(self, config: LingBotVideoConfig, hf_config: dict[str, Any]) -> None:
        """Construct the Dense or MoE variant from its released transformer config."""
        config.update_model_arch(hf_config)
        super().__init__(config, hf_config)
        head_dim = config.hidden_size // config.num_attention_heads
        if head_dim != sum(config.axes_dims):
            raise ValueError(f"head_dim {head_dim} != sum(axes_dims) {sum(config.axes_dims)}")
        sp_world_size = _sequence_parallel_world_size()
        assert config.num_attention_heads % sp_world_size == 0, (
            f"The number of attention heads ({config.num_attention_heads}) must be divisible by "
            f"the sequence parallel size ({sp_world_size})"
        )
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.num_channels_latents = config.in_channels
        self.patch_embedder = LingBotVideoLinear(
            config.in_channels * math.prod(config.patch_size),
            config.hidden_size,
            bias=config.patch_embed_bias,
        )
        self.time_proj = Timesteps(config.freq_dim, flip_sin_to_cos=True, downscale_freq_shift=0)
        self.time_embedder = LingBotVideoTimestepEmbedding(
            config.freq_dim, config.hidden_size, config.timestep_mlp_bias
        )
        self.time_modulation = nn.Sequential(nn.SiLU(), LingBotVideoLinear(config.hidden_size, 6 * config.hidden_size))
        self.text_embedder = LingBotVideoTextEmbedder(config.text_dim, config.hidden_size)
        self.rope = LingBotVideoRotaryEmbedding(tuple(config.axes_dims), tuple(config.axes_lens), config.rope_theta)
        self.blocks = nn.ModuleList(
            [
                LingBotVideoBlock(
                    config.hidden_size,
                    config.num_attention_heads,
                    config.intermediate_size,
                    config.norm_eps,
                    config.qkv_bias,
                    config.out_bias,
                    config.num_experts,
                    config.num_experts_per_tok,
                    config.moe_intermediate_size,
                    config.decoder_sparse_step,
                    config.mlp_only_layers,
                    config.n_shared_experts,
                    config.score_func,
                    config.norm_topk_prob,
                    config.n_group,
                    config.topk_group,
                    config.routed_scaling_factor,
                    layer_index,
                )
                for layer_index in range(config.depth)
            ]
        )
        self.norm_out = nn.LayerNorm(config.hidden_size, elementwise_affine=False, eps=config.norm_eps)
        self.norm_out_modulation = nn.Sequential(
            nn.SiLU(), LingBotVideoLinear(config.hidden_size, 2 * config.hidden_size)
        )
        self.proj_out = LingBotVideoLinear(config.hidden_size, math.prod(config.patch_size) * config.out_channels)
        self.__post_init__()

    def to(self, *args: Any, **kwargs: Any) -> LingBotVideoTransformer3DModel:
        """Cast bulk weights while retaining the released fp32-sensitive modules."""
        device, dtype, non_blocking, _ = torch._C._nn._parse_to(*args, **kwargs)
        if dtype is None or dtype == torch.float32:
            return super().to(*args, **kwargs)
        if not torch.is_floating_point(torch.empty((), dtype=dtype)):
            return super().to(*args, **kwargs)
        if device is not None:
            super().to(device=device, non_blocking=non_blocking)
        for name, parameter in self.named_parameters():
            if torch.is_floating_point(parameter):
                target_dtype = torch.float32 if _keep_in_fp32(name) else dtype
                parameter.data = parameter.data.to(target_dtype, non_blocking=non_blocking)
                if parameter.grad is not None:
                    parameter.grad.data = parameter.grad.data.to(target_dtype, non_blocking=non_blocking)
        for name, buffer in self.named_buffers():
            if torch.is_floating_point(buffer):
                target_dtype = torch.float32 if _keep_in_fp32(name) else dtype
                buffer.data = buffer.data.to(target_dtype, non_blocking=non_blocking)
        return self

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor | list[torch.Tensor],
        encoder_hidden_states_image: torch.Tensor | list[torch.Tensor] | None = None,
        guidance: torch.Tensor | None = None,
        encoder_attention_mask: torch.Tensor | None = None,
        return_dict: bool = True,
        **kwargs: Any,
    ) -> LingBotVideoTransformerOutput | tuple[torch.Tensor]:
        """Denoise video latents with joint video-text attention."""
        del encoder_hidden_states_image, guidance, kwargs
        if isinstance(encoder_hidden_states, list):
            if len(encoder_hidden_states) != 1:
                raise ValueError("LingBot-Video expects one text-encoder output tensor.")
            encoder_hidden_states = encoder_hidden_states[0]
        batch, channels, frames, height, width = hidden_states.shape
        patch_t, patch_h, patch_w = self.config.patch_size
        grid_t, grid_h, grid_w = frames // patch_t, height // patch_h, width // patch_w
        video_tokens = grid_t * grid_h * grid_w
        text_tokens = encoder_hidden_states.shape[1]
        device = hidden_states.device
        if encoder_attention_mask is None:
            encoder_attention_mask = torch.ones((batch, text_tokens), device=device, dtype=torch.bool)
        text_lengths = encoder_attention_mask.sum(dim=-1).long()

        patches = hidden_states.reshape(batch, channels, grid_t, patch_t, grid_h, patch_h, grid_w, patch_w)
        patches = patches.permute(0, 2, 4, 6, 3, 5, 7, 1).reshape(
            batch, video_tokens, patch_t * patch_h * patch_w * channels
        )
        video_hidden = self.patch_embedder(patches)
        text_hidden = self.text_embedder(encoder_hidden_states)
        joint = torch.cat([video_hidden, text_hidden], dim=1)

        rotary_parts: list[torch.Tensor] = []
        for index in range(batch):
            real_text_length = int(text_lengths[index].item())
            positions = _make_joint_position_ids(real_text_length, grid_t, grid_h, grid_w, device)
            rotary = self.rope(positions)
            if real_text_length < text_tokens:
                padding = torch.zeros(
                    text_tokens - real_text_length,
                    rotary.shape[-1],
                    device=device,
                    dtype=rotary.dtype,
                )
                rotary = torch.cat([rotary, padding], dim=0)
            rotary_parts.append(rotary)
        rotary_emb = torch.stack(rotary_parts, dim=0)

        attention_mask = None
        moe_padding_mask = None
        if bool((text_lengths < text_tokens).any()):
            key_mask = torch.cat(
                [
                    torch.ones(batch, video_tokens, dtype=torch.bool, device=device),
                    encoder_attention_mask.bool(),
                ],
                dim=1,
            )
            attention_mask = key_mask[:, None, None, :]
            moe_padding_mask = key_mask
        timestep_projection = self.time_proj(timestep.float())
        timestep_embedding = self.time_embedder(timestep_projection)
        token_embedding = timestep_embedding.unsqueeze(1).expand(batch, joint.shape[1], -1)
        original_joint_length: int | None = None
        if _sequence_parallel_world_size() > 1:
            # Match the official CP order: project token modulation before
            # placing every token-aligned tensor on the same padded shard.
            temb6 = self.time_modulation(token_embedding.reshape(-1, self.hidden_size)).reshape(
                batch, joint.shape[1], -1
            )
            joint, original_joint_length = sequence_model_parallel_shard(joint, dim=1)
            rotary_emb, _ = sequence_model_parallel_shard(rotary_emb, dim=1)
            token_embedding, _ = sequence_model_parallel_shard(token_embedding, dim=1)
            temb6, _ = sequence_model_parallel_shard(temb6, dim=1)
            if moe_padding_mask is None:
                moe_padding_mask = torch.ones(
                    batch, original_joint_length, dtype=torch.bool, device=device
                )
            moe_padding_mask, _ = sequence_model_parallel_shard(moe_padding_mask, dim=1)
            moe_padding_mask = moe_padding_mask.reshape(-1)
            temb6 = temb6.reshape(-1, 6 * self.hidden_size)
        else:
            temb6 = self.time_modulation(token_embedding.reshape(-1, self.hidden_size))
            if moe_padding_mask is not None:
                moe_padding_mask = moe_padding_mask.reshape(-1)

        for block in self.blocks:
            joint = block(
                joint,
                temb6,
                rotary_emb,
                attention_mask,
                moe_padding_mask,
                original_joint_length,
            )

        final_modulation = self.norm_out_modulation(token_embedding.reshape(-1, self.hidden_size))
        shift, scale = final_modulation.reshape(*joint.shape[:2], -1).chunk(2, dim=-1)
        final_hidden = self.norm_out(joint) * (1.0 + scale) + shift
        projected = self.proj_out(final_hidden.to(self.proj_out.weight.dtype))
        if original_joint_length is not None:
            projected = sequence_model_parallel_all_gather_with_unpad(projected, original_joint_length, dim=1)
        projected = projected[:, :video_tokens]
        output_channels = self.config.out_channels
        output = projected.reshape(batch, grid_t, grid_h, grid_w, patch_t, patch_h, patch_w, output_channels)
        output = output.permute(0, 7, 1, 4, 2, 5, 3, 6).reshape(batch, output_channels, frames, height, width)
        if not return_dict:
            return (output,)
        return LingBotVideoTransformerOutput(sample=output)


EntryClass = LingBotVideoTransformer3DModel
