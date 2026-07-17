# SPDX-License-Identifier: Apache-2.0
"""LingBot World 2 causal-fast DiT implemented inside FastVideo."""

import math
import warnings
from typing import Any

from einops import rearrange
import torch
import torch.nn as nn
import torch.nn.functional as F

from fastvideo.configs.models.dits.lingbotworld2 import (
    LingBotWorld2CausalFastVideoConfig,
)
from fastvideo.distributed.communication_op import (
    sequence_model_parallel_all_gather,
    sequence_model_parallel_all_to_all_4D,
)
from fastvideo.distributed.parallel_state import get_sp_parallel_rank, get_sp_world_size
from fastvideo.models.dits.base import BaseDiT
from fastvideo.platforms import AttentionBackendEnum

try:
    import flash_attn_interface

    FLASH_ATTN_3_AVAILABLE = True
except ModuleNotFoundError:
    FLASH_ATTN_3_AVAILABLE = False

try:
    import flash_attn

    FLASH_ATTN_2_AVAILABLE = True
except ModuleNotFoundError:
    FLASH_ATTN_2_AVAILABLE = False


def is_blocks(n: str, m) -> bool:
    return "blocks" in n and str.isdigit(n.split(".")[-1])


def sinusoidal_embedding_1d(dim: int, position: torch.Tensor) -> torch.Tensor:
    """Build Wan/LingBot World 2 sinusoidal timestep embeddings."""
    assert dim % 2 == 0
    half = dim // 2
    position = position.type(torch.float64)
    sinusoid = torch.outer(
        position,
        torch.pow(10000, -torch.arange(half, device=position.device).to(position).div(half)),
    )
    return torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)


@torch.amp.autocast("cuda", enabled=False)
def rope_params(max_seq_len: int, dim: int, theta: int = 10000) -> torch.Tensor:
    """Return complex RoPE frequencies used by the released LingBot World 2 model."""
    assert dim % 2 == 0
    freqs = torch.outer(
        torch.arange(max_seq_len),
        1.0 / torch.pow(theta, torch.arange(0, dim, 2).to(torch.float64).div(dim)),
    )
    return torch.polar(torch.ones_like(freqs), freqs)


def flash_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_lens: torch.Tensor | None = None,
    k_lens: torch.Tensor | None = None,
    dropout_p: float = 0.0,
    softmax_scale: float | None = None,
    q_scale: float | None = None,
    causal: bool = False,
    window_size: tuple[int, int] = (-1, -1),
    deterministic: bool = False,
    dtype: torch.dtype = torch.bfloat16,
    version: int | None = None,
) -> torch.Tensor:
    """Run LingBot World 2-compatible FlashAttention on packed varlen inputs."""
    half_dtypes = (torch.float16, torch.bfloat16)
    assert dtype in half_dtypes
    assert q.device.type == "cuda" and q.size(-1) <= 256

    b, lq, lk, out_dtype = q.size(0), q.size(1), k.size(1), q.dtype

    def half(x: torch.Tensor) -> torch.Tensor:
        return x if x.dtype in half_dtypes else x.to(dtype)

    if q_lens is None:
        q = half(q.flatten(0, 1))
        q_lens = torch.tensor([lq] * b, dtype=torch.int32, device=q.device)
    else:
        q = half(torch.cat([u[:v] for u, v in zip(q, q_lens, strict=True)]))

    if k_lens is None:
        k = half(k.flatten(0, 1))
        v = half(v.flatten(0, 1))
        k_lens = torch.tensor([lk] * b, dtype=torch.int32, device=k.device)
    else:
        k = half(torch.cat([u[:v] for u, v in zip(k, k_lens, strict=True)]))
        v = half(torch.cat([u[:v] for u, v in zip(v, k_lens, strict=True)]))

    q = q.to(v.dtype)
    k = k.to(v.dtype)
    if q_scale is not None:
        q = q * q_scale

    if version == 3 and not FLASH_ATTN_3_AVAILABLE:
        warnings.warn("FlashAttention 3 is not available; using FlashAttention 2.")

    cu_q = torch.cat([q_lens.new_zeros([1]), q_lens]).cumsum(
        0, dtype=torch.int32
    ).to(q.device, non_blocking=True)
    cu_k = torch.cat([k_lens.new_zeros([1]), k_lens]).cumsum(
        0, dtype=torch.int32
    ).to(k.device, non_blocking=True)

    if (version is None or version == 3) and FLASH_ATTN_3_AVAILABLE:
        x = flash_attn_interface.flash_attn_varlen_func(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=cu_q,
            cu_seqlens_k=cu_k,
            seqused_q=None,
            seqused_k=None,
            max_seqlen_q=lq,
            max_seqlen_k=lk,
            softmax_scale=softmax_scale,
            causal=causal,
            deterministic=deterministic,
        ).unflatten(0, (b, lq))
    else:
        assert FLASH_ATTN_2_AVAILABLE
        x = flash_attn.flash_attn_varlen_func(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=cu_q,
            cu_seqlens_k=cu_k,
            max_seqlen_q=lq,
            max_seqlen_k=lk,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=window_size,
            deterministic=deterministic,
        ).unflatten(0, (b, lq))
    return x.type(out_dtype)


def attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_lens: torch.Tensor | None = None,
    k_lens: torch.Tensor | None = None,
    dropout_p: float = 0.0,
    softmax_scale: float | None = None,
    q_scale: float | None = None,
    causal: bool = False,
    window_size: tuple[int, int] = (-1, -1),
    deterministic: bool = False,
    dtype: torch.dtype = torch.bfloat16,
    fa_version: int | None = None,
) -> torch.Tensor:
    """Dispatch LingBot World 2 attention to FlashAttention when available."""
    if FLASH_ATTN_2_AVAILABLE or FLASH_ATTN_3_AVAILABLE:
        return flash_attention(
            q=q,
            k=k,
            v=v,
            q_lens=q_lens,
            k_lens=k_lens,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            q_scale=q_scale,
            causal=causal,
            window_size=window_size,
            deterministic=deterministic,
            dtype=dtype,
            version=fa_version,
        )

    if q_lens is not None or k_lens is not None:
        warnings.warn("Padding masks are disabled without FlashAttention.")
    q = q.transpose(1, 2).to(dtype)
    k = k.transpose(1, 2).to(dtype)
    v = v.transpose(1, 2).to(dtype)
    out = F.scaled_dot_product_attention(
        q, k, v, attn_mask=None, is_causal=causal, dropout_p=dropout_p
    )
    return out.transpose(1, 2).contiguous()


@torch.amp.autocast("cuda", enabled=False)
def causal_rope_apply(
    x: torch.Tensor,
    grid_sizes: torch.Tensor,
    freqs: torch.Tensor,
    start_frame: int = 0,
) -> torch.Tensor:
    """Apply LingBot World 2 causal RoPE with the current chunk frame offset."""
    n, c = x.size(2), x.size(3) // 2
    freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)
    output = []
    for i, (f, h, w) in enumerate(grid_sizes.tolist()):
        seq_len = f * h * w
        x_i = torch.view_as_complex(x[i, :seq_len].to(torch.float64).reshape(seq_len, n, -1, 2))
        freqs_i = torch.cat(
            [
                freqs[0][start_frame : start_frame + f].view(f, 1, 1, -1).expand(f, h, w, -1),
                freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
                freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1),
            ],
            dim=-1,
        ).reshape(seq_len, 1, -1)
        x_i = torch.view_as_real(x_i * freqs_i).flatten(2)
        x_i = torch.cat([x_i, x[i, seq_len:]])
        output.append(x_i)
    return torch.stack(output).type_as(x)


class WanRMSNorm(nn.Module):
    """RMSNorm used by Wan/LingBot World 2 attention projections."""

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize the last dimension in fp32 and restore input dtype."""
        return self._norm(x.float()).type_as(x) * self.weight

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)


class WanLayerNorm(nn.LayerNorm):
    """LayerNorm variant that computes in fp32 and returns the input dtype."""

    def __init__(self, dim: int, eps: float = 1e-6, elementwise_affine: bool = False):
        super().__init__(dim, elementwise_affine=elementwise_affine, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply layer norm in fp32 for LingBot World 2 numerical parity."""
        return super().forward(x.float()).type_as(x)


class CausalWanSelfAttention(nn.Module):
    """LingBot World 2 causal self-attention with rolling KV cache."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        local_attn_size: int = -1,
        sink_size: int = 0,
        qk_norm: bool = True,
        eps: float = 1e-6,
    ):
        super().__init__()
        assert dim % num_heads == 0
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.local_attn_size = local_attn_size
        self.sink_size = sink_size
        self.qk_norm = qk_norm
        self.eps = eps
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        seq_lens: torch.Tensor,
        grid_sizes: torch.Tensor,
        freqs: torch.Tensor,
        kv_cache: dict,
        current_start: int = 0,
        max_attention_size: int = 1_000_000,
        frame_seqlen: int | None = None,
        seq_lens_int: int | None = None,
    ) -> torch.Tensor:
        """Project QKV, update the rolling cache, and attend to its active window."""
        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim
        q = self.norm_q(self.q(x)).view(b, s, n, d)
        k = self.norm_k(self.k(x)).view(b, s, n, d)
        v = self.v(x).view(b, s, n, d)

        if frame_seqlen is None:
            frame_seqlen = math.prod(grid_sizes[0][1:]).item()
        current_start_frame = current_start // frame_seqlen
        if seq_lens_int is None:
            seq_lens_int = int(seq_lens[0].item() if seq_lens.dim() > 0 else seq_lens.item())

        sp_size = get_sp_world_size()
        if sp_size > 1:
            q = sequence_model_parallel_all_to_all_4D(q, scatter_dim=2, gather_dim=1)
            k = sequence_model_parallel_all_to_all_4D(k, scatter_dim=2, gather_dim=1)
            v = sequence_model_parallel_all_to_all_4D(v, scatter_dim=2, gather_dim=1)
            padded_seq_len = s * sp_size
            roped_query = causal_rope_apply(q, grid_sizes, freqs, start_frame=current_start_frame).type_as(v)
            roped_key = causal_rope_apply(k, grid_sizes, freqs, start_frame=current_start_frame).type_as(v)
            roped_query = roped_query[:, :seq_lens_int]
            roped_key = roped_key[:, :seq_lens_int]
            v = v[:, :seq_lens_int]
            num_new_tokens = seq_lens_int
        else:
            padded_seq_len = s
            roped_query = causal_rope_apply(q, grid_sizes, freqs, start_frame=current_start_frame).type_as(v)
            roped_key = causal_rope_apply(k, grid_sizes, freqs, start_frame=current_start_frame).type_as(v)
            num_new_tokens = roped_query.shape[1]

        current_end = current_start + num_new_tokens
        sink_tokens = self.sink_size * frame_seqlen
        kv_cache_size = kv_cache["k"].shape[1]

        if self.local_attn_size == -1:
            local_end_index = current_start + num_new_tokens
            local_start_index = current_start
            kv_cache["k"][:, local_start_index:local_end_index] = roped_key
            kv_cache["v"][:, local_start_index:local_end_index] = v
        elif (current_end > kv_cache["global_end_index"].item()) and (
            num_new_tokens + kv_cache["local_end_index"].item() > kv_cache_size
        ):
            num_evicted_tokens = num_new_tokens + kv_cache["local_end_index"].item() - kv_cache_size
            num_rolled_tokens = kv_cache["local_end_index"].item() - num_evicted_tokens - sink_tokens
            kv_cache["k"][:, sink_tokens : sink_tokens + num_rolled_tokens] = kv_cache["k"][
                :, sink_tokens + num_evicted_tokens : sink_tokens + num_evicted_tokens + num_rolled_tokens
            ].clone()
            kv_cache["v"][:, sink_tokens : sink_tokens + num_rolled_tokens] = kv_cache["v"][
                :, sink_tokens + num_evicted_tokens : sink_tokens + num_evicted_tokens + num_rolled_tokens
            ].clone()
            local_end_index = kv_cache["local_end_index"].item() + current_end - kv_cache["global_end_index"].item() - num_evicted_tokens
            local_start_index = local_end_index - num_new_tokens
            kv_cache["k"][:, local_start_index:local_end_index] = roped_key
            kv_cache["v"][:, local_start_index:local_end_index] = v
        else:
            local_end_index = kv_cache["local_end_index"].item() + current_end - kv_cache["global_end_index"].item()
            local_start_index = local_end_index - num_new_tokens
            kv_cache["k"][:, local_start_index:local_end_index] = roped_key
            kv_cache["v"][:, local_start_index:local_end_index] = v

        k_cache = kv_cache["k"][:, max(0, local_end_index - max_attention_size) : local_end_index]
        v_cache = kv_cache["v"][:, max(0, local_end_index - max_attention_size) : local_end_index]
        x = attention(roped_query, k_cache, v_cache)
        kv_cache["global_end_index"].fill_(current_end)
        kv_cache["local_end_index"].fill_(local_end_index)

        if sp_size > 1:
            sp_pad = padded_seq_len - seq_lens_int
            if sp_pad > 0:
                x = torch.cat([x, x.new_zeros(b, sp_pad, x.size(2), d)], dim=1)
            x = sequence_model_parallel_all_to_all_4D(x, scatter_dim=1, gather_dim=2)
        return self.o(x.flatten(2))

class WanCrossAttention(CausalWanSelfAttention):
    """LingBot World 2 cross-attention with reusable text K/V cache."""

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        context_lens: torch.Tensor | None,
        crossattn_cache: dict | None = None,
        cross_attn_first_call: bool | None = None,
    ) -> torch.Tensor:
        """Attend hidden states to text context, populating cache on first use."""
        b, n, d = x.size(0), self.num_heads, self.head_dim
        q = self.norm_q(self.q(x)).view(b, -1, n, d)
        if crossattn_cache is not None:
            is_first = crossattn_cache["is_init"].item() == 0 if cross_attn_first_call is None else cross_attn_first_call
            if is_first:
                crossattn_cache["is_init"].fill_(1)
                k = self.norm_k(self.k(context)).view(b, -1, n, d)
                v = self.v(context).view(b, -1, n, d)
                crossattn_cache["k"].copy_(k)
                crossattn_cache["v"].copy_(v)
            else:
                k = crossattn_cache["k"]
                v = crossattn_cache["v"]
        else:
            k = self.norm_k(self.k(context)).view(b, -1, n, d)
            v = self.v(context).view(b, -1, n, d)
        x = flash_attention(q, k, v, k_lens=context_lens)
        return self.o(x.flatten(2))


class CausalWanAttentionBlock(nn.Module):
    """One LingBot World 2 causal transformer block including camera injection."""

    def __init__(
        self,
        dim: int,
        ffn_dim: int,
        num_heads: int,
        local_attn_size: int = -1,
        sink_size: int = 0,
        qk_norm: bool = True,
        cross_attn_norm: bool = False,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        self.local_attn_size = local_attn_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps
        self.norm1 = WanLayerNorm(dim, eps)
        self.self_attn = CausalWanSelfAttention(dim, num_heads, local_attn_size, sink_size, qk_norm, eps)
        self.norm3 = WanLayerNorm(dim, eps, elementwise_affine=True) if cross_attn_norm else nn.Identity()
        self.cross_attn = WanCrossAttention(dim, num_heads, qk_norm=qk_norm, eps=eps)
        self.norm2 = WanLayerNorm(dim, eps)
        self.ffn = nn.Sequential(nn.Linear(dim, ffn_dim), nn.GELU(approximate="tanh"), nn.Linear(ffn_dim, dim))
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)
        self.cam_injector_layer1 = nn.Linear(dim, dim)
        self.cam_injector_layer2 = nn.Linear(dim, dim)
        self.cam_scale_layer = nn.Linear(dim, dim)
        self.cam_shift_layer = nn.Linear(dim, dim)

    def forward(
        self,
        x: torch.Tensor,
        e: torch.Tensor,
        seq_lens: torch.Tensor,
        grid_sizes: torch.Tensor,
        freqs: torch.Tensor,
        context: torch.Tensor,
        context_lens: torch.Tensor | None,
        dit_cond_dict: dict[str, Any] | None = None,
        kv_cache: dict | None = None,
        crossattn_cache: dict | None = None,
        current_start: int = 0,
        max_attention_size: int = 1_000_000,
        frame_seqlen: int | None = None,
        cross_attn_first_call: bool | None = None,
        seq_lens_int: int | None = None,
    ) -> torch.Tensor:
        """Apply self-attention, camera modulation, cross-attention, and FFN."""
        assert kv_cache is not None
        assert e.dtype == torch.float32
        with torch.amp.autocast("cuda", dtype=torch.float32):
            e = (self.modulation.unsqueeze(0) + e).chunk(6, dim=2)

        y = self.self_attn(
            self.norm1(x).float() * (1 + e[1].squeeze(2)) + e[0].squeeze(2),
            seq_lens,
            grid_sizes,
            freqs,
            kv_cache,
            current_start,
            max_attention_size,
            frame_seqlen=frame_seqlen,
            seq_lens_int=seq_lens_int,
        )
        with torch.amp.autocast("cuda", dtype=torch.float32):
            x = x + y * e[2].squeeze(2)

        if dit_cond_dict is not None and "c2ws_plucker_emb" in dit_cond_dict:
            c2ws_plucker_emb = dit_cond_dict["c2ws_plucker_emb"]
            c2ws_hidden_states = self.cam_injector_layer2(
                F.silu(self.cam_injector_layer1(c2ws_plucker_emb))
            )
            c2ws_hidden_states = c2ws_hidden_states + c2ws_plucker_emb
            x = (1.0 + self.cam_scale_layer(c2ws_hidden_states)) * x + self.cam_shift_layer(c2ws_hidden_states)

        x = x + self.cross_attn(
            self.norm3(x),
            context,
            context_lens,
            crossattn_cache=crossattn_cache,
            cross_attn_first_call=cross_attn_first_call,
        )
        y = self.ffn(self.norm2(x).float() * (1 + e[4].squeeze(2)) + e[3].squeeze(2))
        with torch.amp.autocast("cuda", dtype=torch.float32):
            x = x + y * e[5].squeeze(2)
        return x


class CausalHead(nn.Module):
    """Output projection head for LingBot World 2 causal-fast DiT."""

    def __init__(self, dim: int, out_dim: int, patch_size: tuple[int, int, int], eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim
        self.patch_size = patch_size
        self.eps = eps
        self.norm = WanLayerNorm(dim, eps)
        self.head = nn.Linear(dim, math.prod(patch_size) * out_dim)
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)

    def forward(self, x: torch.Tensor, e: torch.Tensor) -> torch.Tensor:
        """Normalize, modulate, and project hidden states to latent patches."""
        assert e.dtype == torch.float32
        with torch.amp.autocast("cuda", dtype=torch.float32):
            e = (self.modulation.unsqueeze(0) + e.unsqueeze(2)).chunk(2, dim=2)
            x = self.head(self.norm(x) * (1 + e[1].squeeze(2)) + e[0].squeeze(2))
        return x


class LingBotWorld2CausalFastTransformer3DModel(BaseDiT):
    """Released LingBot World 2 14B causal-fast model with native FastVideo loading."""

    _fsdp_shard_conditions = [is_blocks]
    _compile_conditions: list = []
    _supported_attention_backends = (
        AttentionBackendEnum.FLASH_ATTN,
        AttentionBackendEnum.TORCH_SDPA,
    )
    param_names_mapping: dict = {}
    reverse_param_names_mapping: dict = {}
    lora_param_names_mapping: dict = {}

    def __init__(self, config: LingBotWorld2CausalFastVideoConfig, hf_config: dict[str, Any]) -> None:
        super().__init__(config=config, hf_config=hf_config)
        self.model_type = config.model_type
        self.patch_size = tuple(config.patch_size)
        self.text_len = config.text_len
        self.in_dim = config.in_dim
        self.dim = config.dim
        self.hidden_size = config.dim
        self.ffn_dim = config.ffn_dim
        self.freq_dim = config.freq_dim
        self.text_dim = config.text_dim
        self.out_dim = config.out_dim
        self.out_channels = config.out_dim
        self.num_heads = config.num_heads
        self.num_attention_heads = config.num_heads
        self.attention_head_dim = config.dim // config.num_heads
        self.num_layers = config.num_layers
        self.local_attn_size = config.local_attn_size
        self.sink_size = config.sink_size
        self.qk_norm = config.qk_norm
        self.cross_attn_norm = config.cross_attn_norm
        self.eps = config.eps
        self.num_channels_latents = config.out_dim

        control_dim = 6
        self.patch_embedding = nn.Conv3d(self.in_dim, self.dim, kernel_size=self.patch_size, stride=self.patch_size)
        self.patch_embedding_wancamctrl = nn.Linear(
            control_dim * 64 * self.patch_size[0] * self.patch_size[1] * self.patch_size[2],
            self.dim,
        )
        self.c2ws_hidden_states_layer1 = nn.Linear(self.dim, self.dim)
        self.c2ws_hidden_states_layer2 = nn.Linear(self.dim, self.dim)
        self.text_embedding = nn.Sequential(
            nn.Linear(self.text_dim, self.dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(self.dim, self.dim),
        )
        self.time_embedding = nn.Sequential(
            nn.Linear(self.freq_dim, self.dim),
            nn.SiLU(),
            nn.Linear(self.dim, self.dim),
        )
        self.time_projection = nn.Sequential(nn.SiLU(), nn.Linear(self.dim, self.dim * 6))
        self.blocks = nn.ModuleList(
            [
                CausalWanAttentionBlock(
                    self.dim,
                    self.ffn_dim,
                    self.num_heads,
                    self.local_attn_size,
                    self.sink_size,
                    self.qk_norm,
                    self.cross_attn_norm,
                    self.eps,
                )
                for _ in range(self.num_layers)
            ]
        )
        self.head = CausalHead(self.dim, self.out_dim, self.patch_size, self.eps)
        self.freqs: torch.Tensor | None = None
        self.init_weights()
        self.__post_init__()

    def _get_freqs(self, device: torch.device) -> torch.Tensor:
        """Materialize the non-persistent RoPE frequency table outside meta init."""
        if self.freqs is None or self.freqs.is_meta or self.freqs.device != device:
            d = self.dim // self.num_heads
            self.freqs = torch.cat(
                [
                    rope_params(1024, d - 4 * (d // 6)),
                    rope_params(1024, 2 * (d // 6)),
                    rope_params(1024, 2 * (d // 6)),
                ],
                dim=1,
            ).to(device)
        return self.freqs

    def forward(
        self,
        hidden_states: torch.Tensor | list[torch.Tensor] | None = None,
        encoder_hidden_states: torch.Tensor | list[torch.Tensor] | None = None,
        timestep: torch.Tensor | None = None,
        encoder_hidden_states_image: torch.Tensor | list[torch.Tensor] | None = None,
        guidance=None,
        *,
        x: list[torch.Tensor] | None = None,
        t: torch.Tensor | None = None,
        context: list[torch.Tensor] | torch.Tensor | None = None,
        seq_len: int | None = None,
        y: list[torch.Tensor] | None = None,
        dit_cond_dict: dict[str, Any] | None = None,
        kv_cache: list[dict] | None = None,
        crossattn_cache: list[dict] | None = None,
        current_start: int = 0,
        max_attention_size: int = 1_000_000,
        frame_seqlen: int | None = None,
        cross_attn_first_call: bool | None = None,
        **kwargs,
    ) -> list[torch.Tensor]:
        """Run one cached causal-fast DiT forward using the released LingBot World 2 ABI."""
        del encoder_hidden_states_image, guidance, kwargs
        if x is None:
            assert isinstance(hidden_states, torch.Tensor)
            x = [hidden_states[0]]
        if t is None:
            assert timestep is not None
            t = timestep
        if context is None:
            context = encoder_hidden_states
        if isinstance(context, torch.Tensor):
            context = [u for u in context]
        assert context is not None
        assert seq_len is not None
        assert kv_cache is not None
        assert crossattn_cache is not None
        if self.model_type == "i2v":
            assert y is not None

        device = self.patch_embedding.weight.device
        freqs = self._get_freqs(device)
        if y is not None:
            x = [torch.cat([u, v], dim=0) for u, v in zip(x, y, strict=True)]

        x = [self.patch_embedding(u.unsqueeze(0)) for u in x]
        grid_sizes = torch.stack([torch.tensor(u.shape[2:], dtype=torch.long, device=u.device) for u in x])
        x = [u.flatten(2).transpose(1, 2) for u in x]
        seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long, device=device)
        assert seq_lens.max() <= seq_len
        x = torch.cat(x)
        seq_lens_int = int(seq_lens[0].item())
        sp_size = get_sp_world_size()
        sp_rank = get_sp_parallel_rank()
        padded_seq_len = ((seq_lens_int + sp_size - 1) // sp_size) * sp_size
        sp_pad_len = padded_seq_len - seq_lens_int
        if sp_pad_len > 0:
            x = torch.cat([x, x.new_zeros(x.size(0), sp_pad_len, x.size(2))], dim=1)

        if t.dim() == 1:
            t = t.expand(t.size(0), padded_seq_len)
        with torch.amp.autocast("cuda", dtype=torch.float32):
            bt = t.size(0)
            t = t.flatten()
            e = self.time_embedding(
                sinusoidal_embedding_1d(self.freq_dim, t).unflatten(0, (bt, padded_seq_len)).float()
            )
            e0 = self.time_projection(e).unflatten(2, (6, self.dim))

        context_lens = None
        context = self.text_embedding(
            torch.stack([torch.cat([u, u.new_zeros(self.text_len - u.size(0), u.size(1))]) for u in context])
        )

        if dit_cond_dict is not None and "c2ws_plucker_emb" in dit_cond_dict:
            c2ws_plucker_emb = dit_cond_dict["c2ws_plucker_emb"]
            c2ws_plucker_emb = [
                rearrange(
                    i,
                    "1 c (f c1) (h c2) (w c3) -> 1 (f h w) (c c1 c2 c3)",
                    c1=self.patch_size[0],
                    c2=self.patch_size[1],
                    c3=self.patch_size[2],
                )
                for i in c2ws_plucker_emb
            ]
            c2ws_plucker_emb = torch.cat(c2ws_plucker_emb, dim=1)
            c2ws_plucker_emb = self.patch_embedding_wancamctrl(c2ws_plucker_emb)
            c2ws_hidden_states = self.c2ws_hidden_states_layer2(
                F.silu(self.c2ws_hidden_states_layer1(c2ws_plucker_emb))
            )
            c2ws_plucker_emb = c2ws_plucker_emb + c2ws_hidden_states
            cam_len = c2ws_plucker_emb.size(1)
            if cam_len < padded_seq_len:
                c2ws_plucker_emb = torch.cat(
                    [
                        c2ws_plucker_emb,
                        c2ws_plucker_emb.new_zeros(
                            c2ws_plucker_emb.size(0),
                            padded_seq_len - cam_len,
                            c2ws_plucker_emb.size(2),
                        ),
                    ],
                    dim=1,
                )
            elif cam_len > padded_seq_len:
                c2ws_plucker_emb = c2ws_plucker_emb[:, :padded_seq_len, :]
            if sp_size > 1:
                c2ws_plucker_emb = torch.chunk(c2ws_plucker_emb, sp_size, dim=1)[sp_rank]
            dit_cond_dict = dict(dit_cond_dict)
            dit_cond_dict["c2ws_plucker_emb"] = c2ws_plucker_emb

        if sp_size > 1:
            x = torch.chunk(x, sp_size, dim=1)[sp_rank]
            e = torch.chunk(e, sp_size, dim=1)[sp_rank]
            e0 = torch.chunk(e0, sp_size, dim=1)[sp_rank]

        for block_index, block in enumerate(self.blocks):
            x = block(
                x,
                e=e0,
                seq_lens=seq_lens,
                grid_sizes=grid_sizes,
                freqs=freqs,
                context=context,
                context_lens=context_lens,
                dit_cond_dict=dit_cond_dict,
                kv_cache=kv_cache[block_index],
                crossattn_cache=crossattn_cache[block_index],
                current_start=current_start,
                max_attention_size=max_attention_size,
                frame_seqlen=frame_seqlen,
                cross_attn_first_call=cross_attn_first_call,
                seq_lens_int=seq_lens_int,
            )

        x = self.head(x, e)
        if sp_size > 1:
            x = sequence_model_parallel_all_gather(x, dim=1)
        return [u.float() for u in self.unpatchify(x, grid_sizes)]

    def unpatchify(self, x: torch.Tensor, grid_sizes: torch.Tensor) -> list[torch.Tensor]:
        """Reconstruct latent videos from flattened patch tokens."""
        c = self.out_dim
        out = []
        for u, v in zip(x, grid_sizes.tolist(), strict=True):
            u = u[: math.prod(v)].view(*v, *self.patch_size, c)
            u = torch.einsum("fhwpqrc->cfphqwr", u)
            u = u.reshape(c, *[i * j for i, j in zip(v, self.patch_size, strict=True)])
            out.append(u)
        return out

    def init_weights(self) -> None:
        """Initialize modules for non-meta construction; checkpoint load overwrites them."""
        if self.patch_embedding.weight.is_meta:
            return
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        nn.init.xavier_uniform_(self.patch_embedding.weight.flatten(1))
        for m in self.text_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
        for m in self.time_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
        nn.init.zeros_(self.head.head.weight)


EntryClass = LingBotWorld2CausalFastTransformer3DModel
