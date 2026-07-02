# SPDX-License-Identifier: Apache-2.0
"""DreamX-World autoregressive causal DiT.

Adapted from DreamX-World's Apache-2.0
``wan/modules/causal_camera_model_2_2_prope_infinity.py``. The implementation is
kept native to FastVideo: no production import from DreamX, Diffusers, or
Transformers is required.
"""

from typing import Any

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from fastvideo.configs.models.dits.dreamx_world import DreamXWorldARConfig
from fastvideo.models.dits.base import BaseDiT
from fastvideo.models.dits.dreamx_world import (_dreamx_apply_tiled_projmat,
                                               _dreamx_prope_qkv)


def attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    q_bhld = q.transpose(1, 2)
    k_bhld = k.transpose(1, 2)
    v_bhld = v.transpose(1, 2)
    out = F.scaled_dot_product_attention(q_bhld, k_bhld, v_bhld, dropout_p=0.0)
    return out.transpose(1, 2)


def prope_qkv(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
              viewmats: torch.Tensor, Ks: torch.Tensor):
    q, k, v, output_projection = _dreamx_prope_qkv(q, k, v, viewmats, Ks)

    def apply_fn_o(x: torch.Tensor) -> torch.Tensor:
        return _dreamx_apply_tiled_projmat(x, output_projection)

    return q, k, v, apply_fn_o


def sinusoidal_embedding_1d(dim, position):
    assert dim % 2 == 0
    half = dim // 2
    position = position.type(torch.float64)
    sinusoid = torch.outer(
        position, torch.pow(10000, -torch.arange(half).to(position).div(half)))
    return torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)


def rope_params(max_seq_len, dim, theta=10000):
    assert dim % 2 == 0
    freqs = torch.outer(
        torch.arange(max_seq_len),
        1.0 / torch.pow(theta,
                        torch.arange(0, dim, 2).to(torch.float64).div(dim)))
    return torch.polar(torch.ones_like(freqs), freqs)


class WanRMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return self._norm(x).type_as(x) * self.weight

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)


class WanLayerNorm(nn.LayerNorm):
    def __init__(self, dim, eps=1e-6, elementwise_affine=False):
        super().__init__(dim, elementwise_affine=elementwise_affine, eps=eps)

    def forward(self, x):
        return super().forward(x).type_as(x)


class WanCrossAttention(nn.Module):
    def __init__(self, dim, num_heads, window_size=(-1, -1), qk_norm=True, eps=1e-6):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()

    def forward(self, x, context, context_lens, crossattn_cache=None):
        b, n, d = x.size(0), self.num_heads, self.head_dim
        q = self.norm_q(self.q(x)).view(b, -1, n, d)

        if crossattn_cache is not None:
            if not crossattn_cache["is_init"]:
                crossattn_cache["is_init"] = True
                k = self.norm_k(self.k(context)).view(b, -1, n, d)
                v = self.v(context).view(b, -1, n, d)
                crossattn_cache["k"] = k
                crossattn_cache["v"] = v
            else:
                k = crossattn_cache["k"]
                v = crossattn_cache["v"]
        else:
            k = self.norm_k(self.k(context)).view(b, -1, n, d)
            v = self.v(context).view(b, -1, n, d)

        x = attention(q, k, v)
        x = x.flatten(2)
        return self.o(x)


def block_relativistic_rope(x, grid_sizes, freqs, start_frame=0, relative_frame_indices=None):
    """
    Apply Block-Relativistic RoPE to input tensor.
    Adapted from Infinity-RoPE (https://arxiv.org/abs/2511.20649).

    Args:
        x: Input tensor [B, L, num_heads, head_dim]
        grid_sizes: Tensor [B, 3] containing (F, H, W)
        freqs: RoPE frequencies
        start_frame: Starting frame index for sequential RoPE
        relative_frame_indices: Optional tensor [F] specifying explicit frame indices
                               for Block-Relativistic RoPE. Overrides start_frame if provided.
    """
    n, c = x.size(2), x.size(3) // 2
    freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)

    output = []
    for i, (f, h, w) in enumerate(grid_sizes.tolist()):
        seq_len = f * h * w
        x_i = torch.view_as_complex(x[i, :seq_len].to(torch.float64).reshape(
            seq_len, n, -1, 2))

        if relative_frame_indices is not None:
            frame_indices = relative_frame_indices.long()
            freqs_temporal = freqs[0][frame_indices].view(f, 1, 1, -1).expand(f, h, w, -1)
        else:
            freqs_temporal = freqs[0][start_frame:start_frame + f].view(f, 1, 1, -1).expand(f, h, w, -1)

        freqs_i = torch.cat([
            freqs_temporal,
            freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
        ], dim=-1).reshape(seq_len, 1, -1)

        x_i = torch.view_as_real(x_i * freqs_i).flatten(2)
        x_i = torch.cat([x_i, x[i, seq_len:]])
        output.append(x_i)

    return torch.stack(output).type_as(x)


class CausalWanSelfAttention(nn.Module):
    """Self-attention with KV cache and Block-Relativistic RoPE for causal inference."""

    def __init__(self, dim, num_heads, local_attn_size=6, sink_size=1,
                 qk_norm=True, eps=1e-6):
        assert dim % num_heads == 0
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.local_attn_size = local_attn_size
        self.sink_size = sink_size
        self.qk_norm = qk_norm
        self.eps = eps
        self.max_attention_size = 39600 if local_attn_size == -1 else local_attn_size * 880

        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()

    def forward(self, x, seq_lens, grid_sizes, freqs, kv_cache,
                current_start=0, cache_start=None, sink_recache_after_switch=False):
        """
        Args:
            x: Shape [B, L, C]
            seq_lens: Shape [B]
            grid_sizes: Shape [B, 3] containing (F, H, W)
            freqs: RoPE frequencies [1024, head_dim / 2]
            kv_cache: Dict with 'k', 'v', 'global_end_index', 'local_end_index'
            current_start: Current position in the global token sequence
        """
        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim
        if cache_start is None:
            cache_start = current_start

        q = self.norm_q(self.q(x)).view(b, s, n, d)
        k = self.norm_k(self.k(x)).view(b, s, n, d)
        v = self.v(x).view(b, s, n, d)

        frame_seqlen = math.prod(grid_sizes[0][1:]).item()
        num_new_frames = grid_sizes[0][0].item()
        current_end = current_start + q.shape[1]
        sink_tokens = self.sink_size * frame_seqlen
        kv_cache_size = kv_cache["k"].shape[1]
        num_new_tokens = q.shape[1]

        cache_update_info = None
        is_recompute = current_end <= kv_cache["global_end_index"].item() and current_start > 0

        if self.local_attn_size != -1 and (current_end > kv_cache["global_end_index"].item()) and (
                num_new_tokens + kv_cache["local_end_index"].item() > kv_cache_size):
            # === ROLLING MODE: cache full, evict oldest non-sink tokens ===
            num_evicted_tokens = num_new_tokens + kv_cache["local_end_index"].item() - kv_cache_size
            num_rolled_tokens = kv_cache["local_end_index"].item() - num_evicted_tokens - sink_tokens
            local_end_index = kv_cache["local_end_index"].item() + current_end - \
                kv_cache["global_end_index"].item() - num_evicted_tokens
            local_start_index = local_end_index - num_new_tokens

            temp_k = kv_cache["k"].detach().clone()
            temp_v = kv_cache["v"].detach().clone()
            temp_k[:, sink_tokens:sink_tokens + num_rolled_tokens] = \
                temp_k[:, sink_tokens + num_evicted_tokens:sink_tokens + num_evicted_tokens + num_rolled_tokens].clone()
            temp_v[:, sink_tokens:sink_tokens + num_rolled_tokens] = \
                temp_v[:, sink_tokens + num_evicted_tokens:sink_tokens + num_evicted_tokens + num_rolled_tokens].clone()

            write_start_index = max(local_start_index, sink_tokens) if is_recompute else local_start_index
            roped_offset = max(0, write_start_index - local_start_index)
            write_len = max(0, local_end_index - write_start_index)
            if write_len > 0:
                temp_k[:, write_start_index:local_end_index] = k[:, roped_offset:roped_offset + write_len]
                temp_v[:, write_start_index:local_end_index] = v[:, roped_offset:roped_offset + write_len]

            # Block-Relativistic RoPE: query uses window-relative indices
            query_relative_indices = torch.arange(
                self.local_attn_size - num_new_frames, self.local_attn_size, device=q.device)
            roped_query = block_relativistic_rope(
                q, grid_sizes, freqs, relative_frame_indices=query_relative_indices).type_as(v)

            # Block-Relativistic RoPE: cached K uses position-in-window indices
            num_cache_frames = local_end_index // frame_seqlen
            cache_relative_indices = torch.arange(0, num_cache_frames, device=k.device)
            cache_grid_sizes = grid_sizes.clone()
            cache_grid_sizes[0, 0] = num_cache_frames
            roped_temp_k = block_relativistic_rope(
                temp_k[:, :local_end_index].view(b, num_cache_frames, frame_seqlen, n, d).flatten(1, 2),
                cache_grid_sizes, freqs, relative_frame_indices=cache_relative_indices).type_as(v)

            cache_update_info = {
                "action": "roll_and_insert",
                "sink_tokens": sink_tokens,
                "num_rolled_tokens": num_rolled_tokens,
                "num_evicted_tokens": num_evicted_tokens,
                "local_start_index": local_start_index,
                "local_end_index": local_end_index,
                "write_start_index": write_start_index,
                "write_end_index": local_end_index,
                "new_k": k[:, roped_offset:roped_offset + write_len],
                "new_v": v[:, roped_offset:roped_offset + write_len],
                "current_end": current_end,
                "is_recompute": is_recompute
            }
        else:
            # === DIRECT INSERT MODE: cache not yet full ===
            local_end_index = kv_cache["local_end_index"].item() + current_end - kv_cache["global_end_index"].item()
            local_start_index = local_end_index - num_new_tokens

            temp_k = kv_cache["k"].detach().clone()
            temp_v = kv_cache["v"].detach().clone()

            write_start_index = max(local_start_index, sink_tokens) if is_recompute else local_start_index
            if sink_recache_after_switch:
                write_start_index = local_start_index
            roped_offset = max(0, write_start_index - local_start_index)
            write_len = max(0, local_end_index - write_start_index)
            if write_len > 0:
                temp_k[:, write_start_index:local_end_index] = k[:, roped_offset:roped_offset + write_len]
                temp_v[:, write_start_index:local_end_index] = v[:, roped_offset:roped_offset + write_len]

            # RoPE with relative indices (growing sequentially before cache fills)
            current_frame_in_window = local_start_index // frame_seqlen
            query_relative_indices = torch.arange(
                current_frame_in_window, current_frame_in_window + num_new_frames, device=q.device)
            roped_query = block_relativistic_rope(
                q, grid_sizes, freqs, relative_frame_indices=query_relative_indices).type_as(v)

            num_cache_frames = local_end_index // frame_seqlen
            cache_relative_indices = torch.arange(0, num_cache_frames, device=k.device)
            cache_grid_sizes = grid_sizes.clone()
            cache_grid_sizes[0, 0] = num_cache_frames
            roped_temp_k = block_relativistic_rope(
                temp_k[:, :local_end_index].view(b, num_cache_frames, frame_seqlen, n, d).flatten(1, 2),
                cache_grid_sizes, freqs, relative_frame_indices=cache_relative_indices).type_as(v)

            cache_update_info = {
                "action": "direct_insert",
                "local_start_index": local_start_index,
                "local_end_index": local_end_index,
                "write_start_index": write_start_index,
                "write_end_index": local_end_index,
                "new_k": k[:, roped_offset:roped_offset + write_len],
                "new_v": v[:, roped_offset:roped_offset + write_len],
                "current_end": current_end,
                "is_recompute": is_recompute
            }

        # Attention: sink tokens + local window
        if sink_tokens > 0:
            local_budget = self.max_attention_size - sink_tokens
            k_sink = roped_temp_k[:, :sink_tokens]
            v_sink = temp_v[:, :sink_tokens]
            if local_budget > 0:
                local_start_for_window = max(sink_tokens, local_end_index - local_budget)
                k_local = roped_temp_k[:, local_start_for_window:local_end_index]
                v_local = temp_v[:, local_start_for_window:local_end_index]
                k_cat = torch.cat([k_sink, k_local], dim=1)
                v_cat = torch.cat([v_sink, v_local], dim=1)
            else:
                k_cat = k_sink
                v_cat = v_sink
            x = attention(roped_query, k_cat, v_cat)
        else:
            window_start = max(0, local_end_index - self.max_attention_size)
            x = attention(
                roped_query,
                roped_temp_k[:, window_start:local_end_index],
                temp_v[:, window_start:local_end_index])

        x = x.flatten(2)
        x = self.o(x)
        return x, (current_end, local_end_index, cache_update_info)


class CausalPropeSelfAttention(nn.Module):
    """PRoPE self-attention with optional KV cache for camera-controlled inference."""

    def __init__(self, dim, attn_dim, num_heads, window_size=(-1, -1),
                 local_attn_size=-1, sink_size=0, qk_norm=True, eps=1e-6):
        assert dim % num_heads == 0
        assert attn_dim % num_heads == 0
        super().__init__()
        self.dim = dim
        self.attn_dim = attn_dim
        self.num_heads = num_heads
        self.head_dim = attn_dim // num_heads
        self.local_attn_size = local_attn_size
        self.sink_size = sink_size
        self.qk_norm = qk_norm
        self.eps = eps
        self.window_size = window_size
        self.max_attention_size = 39600 if local_attn_size == -1 else local_attn_size * 880

        self.q_proj = nn.Linear(dim, attn_dim)
        self.k_proj = nn.Linear(dim, attn_dim)
        self.v_proj = nn.Linear(dim, attn_dim)
        self.out_proj = nn.Linear(attn_dim, dim)

        self.norm_q = WanRMSNorm(attn_dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = WanRMSNorm(attn_dim, eps=eps) if qk_norm else nn.Identity()

        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, x, cam_viewmats, cam_K, seq_lens, grid_sizes, freqs,
                kv_cache=None, current_start=0, cache_start=None,
                sink_recache_after_switch=False, cache_update_policy="commit_detached"):
        """
        Args:
            x: Shape [B, L, C]
            cam_viewmats: Camera view matrices
            cam_K: Camera intrinsics
            kv_cache: Optional KV cache dict. When None, runs full attention over current chunk.
        """
        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim
        if cache_start is None:
            cache_start = current_start

        q = self.norm_q(self.q_proj(x)).view(b, s, n, d)
        k = self.norm_k(self.k_proj(x)).view(b, s, n, d)
        v = self.v_proj(x).view(b, s, n, d)

        # Apply PRoPE (Positional Rotary Position Embedding from camera parameters)
        q_t, k_t, v_t, apply_fn_o = prope_qkv(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
            viewmats=cam_viewmats, Ks=cam_K)
        proped_q = q_t.transpose(1, 2)
        proped_k = k_t.transpose(1, 2)
        proped_v = v_t.transpose(1, 2)

        if kv_cache is None:
            # No cache: full attention over current chunk
            x_out = attention(proped_q, proped_k, proped_v)
        else:
            # KV cache mode with rolling cache support
            frame_seqlen = math.prod(grid_sizes[0][1:]).item()
            num_new_tokens = s
            current_end = current_start + num_new_tokens
            sink_tokens = self.sink_size * frame_seqlen
            kv_cache_size = kv_cache["k"].shape[1]
            is_recompute = (current_end <= kv_cache["global_end_index"].item()) and (current_start > 0)

            if self.local_attn_size != -1 and (current_end > kv_cache["global_end_index"].item()) and (
                    num_new_tokens + kv_cache["local_end_index"].item() > kv_cache_size):
                # === ROLLING MODE ===
                num_evicted_tokens = num_new_tokens + kv_cache["local_end_index"].item() - kv_cache_size
                num_rolled_tokens = kv_cache["local_end_index"].item() - num_evicted_tokens - sink_tokens
                local_end_index = kv_cache["local_end_index"].item() + current_end - \
                    kv_cache["global_end_index"].item() - num_evicted_tokens
                local_start_index = local_end_index - num_new_tokens

                if cache_update_policy != "none":
                    with torch.no_grad():
                        kv_cache["k"][:, sink_tokens:sink_tokens + num_rolled_tokens] = \
                            kv_cache["k"][:, sink_tokens + num_evicted_tokens:sink_tokens + num_evicted_tokens + num_rolled_tokens].clone()
                        kv_cache["v"][:, sink_tokens:sink_tokens + num_rolled_tokens] = \
                            kv_cache["v"][:, sink_tokens + num_evicted_tokens:sink_tokens + num_evicted_tokens + num_rolled_tokens].clone()

                    write_start_index = max(local_start_index, sink_tokens) if is_recompute else local_start_index
                    roped_offset = max(0, write_start_index - local_start_index)
                    write_len = max(0, local_end_index - write_start_index)
                    if write_len > 0:
                        with torch.no_grad():
                            kv_cache["k"][:, write_start_index:local_end_index] = proped_k[:, roped_offset:roped_offset + write_len].detach()
                            kv_cache["v"][:, write_start_index:local_end_index] = proped_v[:, roped_offset:roped_offset + write_len].detach()
            else:
                # === DIRECT INSERT MODE ===
                local_end_index = kv_cache["local_end_index"].item() + current_end - kv_cache["global_end_index"].item()
                local_start_index = local_end_index - num_new_tokens

                if cache_update_policy != "none":
                    write_start_index = max(local_start_index, sink_tokens) if is_recompute else local_start_index
                    if sink_recache_after_switch:
                        write_start_index = local_start_index
                    roped_offset = max(0, write_start_index - local_start_index)
                    write_len = max(0, local_end_index - write_start_index)
                    if write_len > 0:
                        with torch.no_grad():
                            kv_cache["k"][:, write_start_index:local_end_index] = proped_k[:, roped_offset:roped_offset + write_len].detach()
                            kv_cache["v"][:, write_start_index:local_end_index] = proped_v[:, roped_offset:roped_offset + write_len].detach()

            # Attention: sink tokens + local window
            if sink_tokens > 0:
                local_budget = self.max_attention_size - sink_tokens
                k_sink = kv_cache["k"][:, :sink_tokens].detach()
                v_sink = kv_cache["v"][:, :sink_tokens].detach()
                if local_budget > 0:
                    local_start_for_window = max(sink_tokens, local_end_index - local_budget)
                    k_local = kv_cache["k"][:, local_start_for_window:local_end_index].detach()
                    v_local = kv_cache["v"][:, local_start_for_window:local_end_index].detach()
                    k_cat = torch.cat([k_sink, k_local], dim=1)
                    v_cat = torch.cat([v_sink, v_local], dim=1)
                else:
                    k_cat = k_sink
                    v_cat = v_sink
                x_out = attention(proped_q, k_cat, v_cat)
            else:
                window_start = max(0, local_end_index - self.max_attention_size)
                x_out = attention(
                    proped_q,
                    kv_cache["k"][:, window_start:local_end_index].detach(),
                    kv_cache["v"][:, window_start:local_end_index].detach())

            if not is_recompute and cache_update_policy != "none":
                kv_cache["global_end_index"].fill_(current_end)
                kv_cache["local_end_index"].fill_(local_end_index)

        # Apply inverse PRoPE
        x = apply_fn_o(x_out.transpose(1, 2)).transpose(1, 2)
        x = x.flatten(2)
        x = self.out_proj(x)
        return x


class CausalWanAttentionBlock(nn.Module):

    def __init__(self, dim, ffn_dim, num_heads, local_attn_size=-1, sink_size=0,
                 qk_norm=True, cross_attn_norm=False, eps=1e-6, **kwargs):
        super().__init__()
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        self.local_attn_size = local_attn_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps

        self.add_control_adapter = kwargs.get('add_control_adapter', False)
        self.cam_method = kwargs.get('cam_method')
        self.attn_compress = kwargs.get('attn_compress', 1)
        self.layer_idx = kwargs.get('layer_idx')
        cam_self_attn_layers = kwargs.get('cam_self_attn_layers')

        # layers
        self.norm1 = WanLayerNorm(dim, eps)
        self.self_attn = CausalWanSelfAttention(
            dim, num_heads, local_attn_size, sink_size, qk_norm, eps)
        self.norm3 = WanLayerNorm(
            dim, eps, elementwise_affine=True) if cross_attn_norm else nn.Identity()
        self.cross_attn = WanCrossAttention(dim, num_heads, (-1, -1), qk_norm, eps)
        self.norm2 = WanLayerNorm(dim, eps)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim), nn.GELU(approximate='tanh'),
            nn.Linear(ffn_dim, dim))

        # PRoPE self-attention branch for camera control
        add_cam_attn = self.add_control_adapter and self.cam_method == 'prope'
        if add_cam_attn and cam_self_attn_layers is not None:
            add_cam_attn = self.layer_idx in cam_self_attn_layers
        if add_cam_attn:
            self.cam_self_attn = CausalPropeSelfAttention(
                dim, dim // self.attn_compress, num_heads,
                local_attn_size=local_attn_size, sink_size=sink_size,
                qk_norm=qk_norm, eps=eps)

        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

    def forward(self, x, e, seq_lens, grid_sizes, freqs, context, context_lens,
                kv_cache, crossattn_cache=None, current_start=0, cache_start=None,
                cam_viewmats=None, cam_K=None, sink_recache_after_switch=False,
                cache_update_policy="commit_detached"):
        num_frames, frame_seqlen = e.shape[1], x.shape[1] // e.shape[1]
        e = (self.modulation.unsqueeze(1) + e).chunk(6, dim=2)

        # self-attention
        attn_input = (self.norm1(x).unflatten(
            dim=1, sizes=(num_frames, frame_seqlen)) * (1 + e[1]) + e[0]).flatten(1, 2)
        y, cache_update_info = self.self_attn(
            attn_input, seq_lens, grid_sizes, freqs, kv_cache,
            current_start, cache_start, sink_recache_after_switch)

        # PRoPE camera attention (parallel branch)
        if hasattr(self, 'cam_self_attn') and cam_viewmats is not None and cam_K is not None:
            prope_kv_cache = None
            if kv_cache is not None and "prope_k" in kv_cache:
                prope_kv_cache = {
                    "k": kv_cache["prope_k"],
                    "v": kv_cache["prope_v"],
                    "global_end_index": kv_cache["prope_global_end_index"],
                    "local_end_index": kv_cache["prope_local_end_index"],
                }
            y = y + self.cam_self_attn(
                attn_input, cam_viewmats, cam_K, seq_lens, grid_sizes, freqs,
                kv_cache=prope_kv_cache, current_start=current_start,
                cache_start=cache_start, cache_update_policy=cache_update_policy)

        x = x + (y.unflatten(dim=1, sizes=(num_frames, frame_seqlen)) * e[2]).flatten(1, 2)

        # cross-attention & FFN
        x = x + self.cross_attn(self.norm3(x), context, context_lens,
                                crossattn_cache=crossattn_cache)
        y = self.ffn(
            (self.norm2(x).unflatten(dim=1, sizes=(num_frames, frame_seqlen))
             * (1 + e[4]) + e[3]).flatten(1, 2))
        x = x + (y.unflatten(dim=1, sizes=(num_frames, frame_seqlen)) * e[5]).flatten(1, 2)

        return x, cache_update_info


class CausalHead(nn.Module):

    def __init__(self, dim, out_dim, patch_size, eps=1e-6):
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim
        self.patch_size = patch_size
        self.eps = eps

        out_dim = math.prod(patch_size) * out_dim
        self.norm = WanLayerNorm(dim, eps)
        self.head = nn.Linear(dim, out_dim)
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)

    def forward(self, x, e):
        num_frames, frame_seqlen = e.shape[1], x.shape[1] // e.shape[1]
        e = (self.modulation.unsqueeze(1) + e).chunk(2, dim=2)
        x = self.head(
            self.norm(x).unflatten(dim=1, sizes=(num_frames, frame_seqlen))
            * (1 + e[1]) + e[0])
        return x


class DreamXWorldARTransformer3DModel(BaseDiT):
    """DreamX-World-5B autoregressive causal transformer."""

    _fsdp_shard_conditions = DreamXWorldARConfig()._fsdp_shard_conditions
    _compile_conditions = DreamXWorldARConfig()._compile_conditions
    _supported_attention_backends = DreamXWorldARConfig()._supported_attention_backends
    param_names_mapping = DreamXWorldARConfig().param_names_mapping
    reverse_param_names_mapping = DreamXWorldARConfig().reverse_param_names_mapping
    lora_param_names_mapping = DreamXWorldARConfig().lora_param_names_mapping
    _no_split_modules = ["CausalWanAttentionBlock"]

    def __init__(self, config: DreamXWorldARConfig, hf_config: dict[str, Any]) -> None:
        super().__init__(config=config, hf_config=hf_config)

        model_type = config.model_type
        patch_size = config.patch_size
        text_len = config.text_len
        in_dim = config.in_channels
        dim = config.hidden_size
        ffn_dim = config.ffn_dim
        freq_dim = config.freq_dim
        text_dim = config.text_dim
        out_dim = config.out_channels
        num_heads = config.num_attention_heads
        num_layers = config.num_layers
        local_attn_size = config.local_attn_size
        sink_size = config.sink_size
        qk_norm = bool(config.qk_norm)
        cross_attn_norm = config.cross_attn_norm
        eps = config.eps
        add_control_adapter = config.add_control_adapter
        cam_method = config.cam_method
        attn_compress = config.attn_compress
        cam_self_attn_layers = config.cam_self_attn_layers

        assert model_type in ['t2v', 'i2v', 'ti2v']
        self.model_type = model_type
        self.patch_size = patch_size
        self.text_len = text_len
        self.in_dim = in_dim
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.freq_dim = freq_dim
        self.text_dim = text_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.local_attn_size = local_attn_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps

        # embeddings
        self.patch_embedding = nn.Conv3d(
            in_dim, dim, kernel_size=patch_size, stride=patch_size)
        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, dim), nn.GELU(approximate='tanh'),
            nn.Linear(dim, dim))
        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.time_projection = nn.Sequential(
            nn.SiLU(), nn.Linear(dim, dim * 6))

        # transformer blocks
        self.blocks = nn.ModuleList([
            CausalWanAttentionBlock(
                dim, ffn_dim, num_heads, local_attn_size, sink_size,
                qk_norm, cross_attn_norm, eps,
                add_control_adapter=add_control_adapter,
                cam_method=cam_method,
                attn_compress=attn_compress,
                layer_idx=layer_idx,
                cam_self_attn_layers=cam_self_attn_layers)
            for layer_idx in range(num_layers)
        ])
        for layer_idx, block in enumerate(self.blocks):
            block.self_attn.layer_idx = layer_idx
            block.self_attn.num_layers = self.num_layers

        # head
        self.head = CausalHead(dim, out_dim, patch_size, eps)

        # RoPE frequencies
        assert (dim % num_heads) == 0 and (dim // num_heads) % 2 == 0
        d = dim // num_heads
        self.freqs = torch.cat([
            rope_params(1024, d - 4 * (d // 6)),
            rope_params(1024, 2 * (d // 6)),
            rope_params(1024, 2 * (d // 6))
        ], dim=1)

        self.num_attention_heads = num_heads
        self.attention_head_dim = dim // num_heads
        self.hidden_size = dim
        self.in_channels = in_dim
        self.out_channels = out_dim
        self.num_channels_latents = out_dim
        self.init_weights()
        self.num_frame_per_block = config.arch_config.num_frames_per_block
        self.__post_init__()

    def forward(self, x=None, t=None, context=None, seq_len=None, y=None, y_camera=None,
                kv_cache=None, crossattn_cache=None, current_start=0,
                cache_start=0, cache_update_policy="commit_detached",
                hidden_states=None, encoder_hidden_states=None, timestep=None, **kwargs):
        """
        Causal inference with KV caching.
        See Algorithm 2 of CausVid (https://arxiv.org/abs/2412.07772).

        Args:
            x: List of input video tensors [C_in, F, H, W]
            t: Timestep tensor [B, L]
            context: List of text embeddings [L, C]
            seq_len: Maximum sequence length for positional encoding
            y: Optional conditional video inputs (I2V mode)
            y_camera: Camera parameters dict {'viewmats': ..., 'K': ...}
            kv_cache: List of KV cache dicts per transformer block
            crossattn_cache: List of cross-attention cache dicts
            current_start: Current position in global token sequence
            cache_start: Cache start position
            cache_update_policy: Cache update strategy ('commit_detached' or 'none')

        Returns:
            Stacked output tensors [B, C_out, F, H/8, W/8]
        """
        if x is None and hidden_states is not None:
            x = [sample for sample in hidden_states]
        if t is None and timestep is not None:
            t = timestep
        if context is None and encoder_hidden_states is not None:
            if isinstance(encoder_hidden_states, torch.Tensor):
                context = [sample for sample in encoder_hidden_states]
            else:
                context = encoder_hidden_states
        if seq_len is None:
            if torch.is_tensor(t):
                seq_len = int(t.shape[1]) if t.dim() > 1 else int(t.numel())
            elif x is not None:
                sample = x[0]
                seq_len = (sample.shape[1] // self.patch_size[0]) * (sample.shape[2] // self.patch_size[1]) * (sample.shape[3] // self.patch_size[2])
        if x is None or t is None or context is None or seq_len is None:
            raise ValueError("DreamXWorldARTransformer3DModel requires x/t/context/seq_len or FastVideo aliases")

        device = self.patch_embedding.weight.device
        if self.freqs.is_meta or self.freqs.device != device:
            d = self.dim // self.num_heads
            self.freqs = torch.cat([
                rope_params(1024, d - 4 * (d // 6)),
                rope_params(1024, 2 * (d // 6)),
                rope_params(1024, 2 * (d // 6)),
            ], dim=1).to(device)

        if y is not None:
            x = [torch.cat([u, v], dim=0) for u, v in zip(x, y, strict=True)]

        # patch embedding
        x = [self.patch_embedding(u.unsqueeze(0)) for u in x]
        grid_sizes = torch.stack(
            [torch.tensor(u.shape[2:], dtype=torch.long) for u in x])
        x = [u.flatten(2).transpose(1, 2) for u in x]
        seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
        assert seq_lens.max() <= seq_len
        x = torch.cat(x)

        # time embedding
        e = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, t.flatten()).type_as(x))
        e0 = self.time_projection(e).unflatten(
            1, (6, self.dim)).unflatten(dim=0, sizes=t.shape)

        # text embedding
        context_lens = None
        context = self.text_embedding(
            torch.stack([
                torch.cat(
                    [u, u.new_zeros(self.text_len - u.size(0), u.size(1))])
                for u in context
            ]))

        # camera parameters
        if y_camera is not None and isinstance(y_camera, dict):
            cam_viewmats = y_camera['viewmats']
            cam_K = y_camera['K']
        else:
            cam_viewmats = None
            cam_K = None

        block_kwargs = dict(
            e=e0, seq_lens=seq_lens, grid_sizes=grid_sizes, freqs=self.freqs,
            context=context, context_lens=context_lens,
            cam_viewmats=cam_viewmats, cam_K=cam_K,
            cache_update_policy=cache_update_policy,
        )

        cache_update_infos = []
        for block_index, block in enumerate(self.blocks):
            block_kwargs.update({
                "kv_cache": kv_cache[block_index] if kv_cache is not None else None,
                "crossattn_cache": crossattn_cache[block_index] if crossattn_cache is not None else None,
                "current_start": current_start,
                "cache_start": cache_start,
            })
            x, block_cache_update_info = block(x, **block_kwargs)
            if kv_cache is not None:
                cache_update_infos.append((block_index, block_cache_update_info))

        # Apply deferred cache updates
        if kv_cache is not None and cache_update_infos and cache_update_policy != "none":
            self._apply_cache_updates(kv_cache, cache_update_infos)

        # head & unpatchify
        x = self.head(x, e.unflatten(dim=0, sizes=t.shape).unsqueeze(2))
        x = self.unpatchify(x, grid_sizes)
        return torch.stack(x)

    def _apply_cache_updates(self, kv_cache, cache_update_infos):
        """Apply deferred cache updates collected from all transformer blocks.

        For Block-Relativistic RoPE, this stores un-roped K values in the cache.
        RoPE is applied dynamically during attention based on each token's current
        relative position in the sliding window.
        """
        with torch.no_grad():
            for block_index, (current_end, local_end_index, update_info) in cache_update_infos:
                if update_info is not None:
                    cache = kv_cache[block_index]

                    if update_info["action"] == "roll_and_insert":
                        sink_tokens = update_info["sink_tokens"]
                        num_rolled_tokens = update_info["num_rolled_tokens"]
                        num_evicted_tokens = update_info["num_evicted_tokens"]
                        write_start_index = update_info.get("write_start_index", update_info["local_start_index"])
                        write_end_index = update_info.get("write_end_index", update_info["local_end_index"])
                        new_k = update_info["new_k"].detach()
                        new_v = update_info["new_v"].detach()

                        cache["k"][:, sink_tokens:sink_tokens + num_rolled_tokens] = \
                            cache["k"][:, sink_tokens + num_evicted_tokens:sink_tokens + num_evicted_tokens + num_rolled_tokens].clone()
                        cache["v"][:, sink_tokens:sink_tokens + num_rolled_tokens] = \
                            cache["v"][:, sink_tokens + num_evicted_tokens:sink_tokens + num_evicted_tokens + num_rolled_tokens].clone()

                        if write_end_index > write_start_index and new_k.shape[1] == (write_end_index - write_start_index):
                            cache["k"][:, write_start_index:write_end_index] = new_k
                            cache["v"][:, write_start_index:write_end_index] = new_v

                    elif update_info["action"] == "direct_insert":
                        write_start_index = update_info.get("write_start_index", update_info["local_start_index"])
                        write_end_index = update_info.get("write_end_index", update_info["local_end_index"])
                        new_k = update_info["new_k"].detach()
                        new_v = update_info["new_v"].detach()

                        if write_end_index > write_start_index and new_k.shape[1] == (write_end_index - write_start_index):
                            cache["k"][:, write_start_index:write_end_index] = new_k
                            cache["v"][:, write_start_index:write_end_index] = new_v

                is_recompute = False if update_info is None else update_info.get("is_recompute", False)
                if not is_recompute:
                    kv_cache[block_index]["global_end_index"].fill_(current_end)
                    kv_cache[block_index]["local_end_index"].fill_(local_end_index)

    def unpatchify(self, x, grid_sizes):
        """Reconstruct video tensors from patch embeddings."""
        c = self.out_dim
        out = []
        for u, v in zip(x, grid_sizes.tolist(), strict=True):
            u = u[:math.prod(v)].view(*v, *self.patch_size, c)
            u = torch.einsum('fhwpqrc->cfphqwr', u)
            u = u.reshape(c, *[i * j for i, j in zip(v, self.patch_size, strict=True)])
            out.append(u)
        return out

    def init_weights(self):
        """Initialize model parameters using Xavier initialization."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        nn.init.xavier_uniform_(self.patch_embedding.weight.flatten(1))
        for m in self.text_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=.02)
        for m in self.time_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=.02)

        nn.init.zeros_(self.head.head.weight)


EntryClass = DreamXWorldARTransformer3DModel
