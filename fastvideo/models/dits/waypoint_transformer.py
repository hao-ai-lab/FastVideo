# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2025 Hugging Face Team and Overworld
"""FastVideo-native Waypoint-1-Small transformer."""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch.nn.attention.flex_attention import (
        create_block_mask,
        flex_attention,
    )

    _FLEX_ATTN_AVAILABLE = True
except ImportError:
    _FLEX_ATTN_AVAILABLE = False

from fastvideo.configs.models.dits.waypoint_transformer import WaypointConfig
from fastvideo.layers.linear import ReplicatedLinear
from fastvideo.layers.mlp import MLP
from fastvideo.models.dits.base import BaseDiT

_DEFAULT_WAYPOINT_CONFIG = WaypointConfig()
_DEFAULT_WAYPOINT_ARCH = _DEFAULT_WAYPOINT_CONFIG.arch_config

if _FLEX_ATTN_AVAILABLE:
    _flex_attention = torch.compile(
        flex_attention, dynamic=False, mode="max-autotune-no-cudagraphs"
    )
else:
    _flex_attention = None


def rms_norm(x: torch.Tensor, eps: float | None = None) -> torch.Tensor:
    return F.rms_norm(x, (x.size(-1), ), eps=eps)


class AdaLN(nn.Module):

    def __init__(
        self,
        d_model: int,
        bias: bool = False,
        eps: float = 1e-6,
        params_dtype: torch.dtype | None = None,
        prefix: str = "",
    ):
        super().__init__()
        self.eps = eps
        self.fc = ReplicatedLinear(
            d_model,
            2 * d_model,
            bias=bias,
            params_dtype=params_dtype,
            prefix=prefix,
        )

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape
        N = cond.shape[1]
        h = F.silu(cond)
        ab, _ = self.fc(h)
        ab = ab.view(B, N, 1,
                     2 * D).expand(-1, -1, L // N, -1).reshape(B, L, 2 * D)
        scale, shift = ab.chunk(2, dim=-1)
        return rms_norm(x) * (1 + scale) + shift


def ada_rmsnorm(
    x: torch.Tensor,
    scale: torch.Tensor,
    bias: torch.Tensor,
    eps: float | None = None,
) -> torch.Tensor:
    B, L, D = x.shape
    N = scale.shape[1]
    x = rms_norm(x, eps)
    scale = scale.unsqueeze(2).expand(-1, -1, L // N, -1).reshape(B, L, D)
    bias = bias.unsqueeze(2).expand(-1, -1, L // N, -1).reshape(B, L, D)
    return x * (1 + scale) + bias


def ada_gate(x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    B, L, D = x.shape
    N = gate.shape[1]
    gate = gate.unsqueeze(2).expand(-1, -1, L // N, -1).reshape(B, L, D)
    return x * gate


def _waypoint_mlp(in_dim: int, hidden_dim: int, out_dim: int,
                  bias: bool = False) -> MLP:
    return MLP(
        input_dim=in_dim,
        mlp_hidden_dim=hidden_dim,
        output_dim=out_dim,
        bias=bias,
        act_type="silu",
    )


class CFG(nn.Module):
    
    def __init__(self, d_model: int, dropout: float = 0.0):
        super().__init__()
        self.dropout = dropout
        self.null_emb = nn.Parameter(torch.zeros(1, 1, d_model))
    
    def forward(
        self, x: torch.Tensor, is_conditioned: Optional[bool] = None
    ) -> torch.Tensor:
        B, L, _ = x.shape
        null = self.null_emb.expand(B, L, -1)
        
        if self.training or is_conditioned is None:
            if self.dropout == 0.0:
                return x
            drop = torch.rand(B, 1, 1, device=x.device) < self.dropout
            return torch.where(drop, null, x)
        
        return x if is_conditioned else null


class ControllerInputEmbedding(nn.Module):
    def __init__(self, n_buttons: int, d_model: int, mlp_ratio: int = 4):
        super().__init__()
        self.mlp = _waypoint_mlp(n_buttons + 3, d_model * mlp_ratio, d_model)
    
    def forward(self, mouse: torch.Tensor, button: torch.Tensor, scroll: torch.Tensor) -> torch.Tensor:
        x = torch.cat((mouse, button, scroll), dim=-1)
        return self.mlp(x)


class NoiseConditioner(nn.Module):
    def __init__(self, d_model: int, freq_dim: int = 512, base: float = 10_000.0):
        super().__init__()
        half = freq_dim // 2
        assert half * 2 == freq_dim
        self.half = half
        self.base = base
        self.register_buffer(
            "freq",
            self._build_freq(),
            persistent=False,
        )
        self.mlp = _waypoint_mlp(freq_dim, d_model * 4, d_model)

    def _build_freq(self, device: torch.device | None = None) -> torch.Tensor:
        return torch.logspace(
            0,
            -1,
            steps=self.half,
            base=self.base,
            dtype=torch.float32,
            device=device,
        )

    def materialize(self, device: torch.device) -> None:
        if self.freq.is_meta:
            self._buffers["freq"] = self._build_freq(device)

    @torch.autocast("cuda", enabled=False)
    def forward(self, sigma: torch.Tensor) -> torch.Tensor:
        orig_dtype, shape = sigma.dtype, sigma.shape
        s = sigma.reshape(-1).float() * 1000.0
        phase = s.unsqueeze(1) * self.freq.unsqueeze(0)
        emb = torch.cat([torch.sin(phase), torch.cos(phase)], dim=-1)
        emb = emb * 2.0**0.5
        mlp_dtype = next(self.mlp.parameters()).dtype
        emb = self.mlp(emb.to(mlp_dtype))
        return emb.to(orig_dtype).view(*shape, -1)


def _pixel_frequencies(dim: int, max_freq: float, device: torch.device) -> torch.Tensor:
    """Return the published spatial frequency spectrum."""
    return torch.linspace(
        1.0, max_freq / 2, dim // 2, device=device, dtype=torch.float32
    ) * math.pi


def _lang_frequencies(dim: int, device: torch.device) -> torch.Tensor:
    """Return the published temporal frequency spectrum."""
    return 10.0 ** (
        -torch.arange(dim // 2, device=device, dtype=torch.float32) / 2
    )


class OrthoRoPE(nn.Module):
    """Waypoint's orthogonal temporal and spatial RoPE."""

    def __init__(self, height: int, width: int, n_frames: int, head_dim: int):
        super().__init__()
        self.height = height
        self.width = width
        self.n_frames = n_frames
        self.head_dim = head_dim
        assert head_dim // 8 + head_dim // 8 + head_dim // 4 == head_dim // 2
        # This single RoPE is shared across all layers and applied to both q and
        # k, so memoize the angles on the pos_ids dict identity (fresh per
        # model.forward) to avoid recomputing them 2*n_layers times.
        self._angle_cache: tuple[dict, torch.Tensor, torch.Tensor] | None = None

    def get_angles(
        self,
        t_pos: torch.Tensor,
        y_pos: torch.Tensor,
        x_pos: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return cosine and sine angles for the three axes."""
        device = t_pos.device
        H, W = self.height, self.width
        head_dim = self.head_dim
        max_freq = min(H, W) * 0.8

        spatial_freqs = _pixel_frequencies(head_dim // 8, max_freq, device)
        w1 = max(W - 1, 1)
        h1 = max(H - 1, 1)
        norm_x = (-1.0 + 1.0 / W) + (2.0 - 2.0 / W) * x_pos.float() / w1
        norm_y = (-1.0 + 1.0 / H) + (2.0 - 2.0 / H) * y_pos.float() / h1
        angle_x = norm_x.unsqueeze(-1) * spatial_freqs.unsqueeze(0).unsqueeze(0)
        angle_x = angle_x.repeat_interleave(2, dim=-1)
        angle_y = norm_y.unsqueeze(-1) * spatial_freqs.unsqueeze(0).unsqueeze(0)
        angle_y = angle_y.repeat_interleave(2, dim=-1)

        temporal_freqs = _lang_frequencies(head_dim // 4, device)
        angle_t = t_pos.float().unsqueeze(-1) * temporal_freqs.unsqueeze(0).unsqueeze(0)
        angle_t = angle_t.repeat_interleave(2, dim=-1)

        angles = torch.cat([angle_x, angle_y, angle_t], dim=-1)
        return angles.cos(), angles.sin()

    @torch.autocast("cuda", enabled=False)
    def forward(
        self, x: torch.Tensor, pos_ids: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Apply RoPE to consecutive feature pairs."""
        cache = self._angle_cache
        if cache is not None and cache[0] is pos_ids:
            cos, sin = cache[1], cache[2]
        else:
            cos, sin = self.get_angles(
                pos_ids["t_pos"],
                pos_ids["y_pos"],
                pos_ids["x_pos"],
            )
            self._angle_cache = (pos_ids, cos, sin)
        cos = cos.unsqueeze(2)
        sin = sin.unsqueeze(2)

        x_float = x.float()
        x0, x1 = x_float.unfold(-1, 2, 2).unbind(-1)

        y0 = x0 * cos - x1 * sin
        y1 = x1 * cos + x0 * sin
        return torch.cat((y0, y1), dim=-1).type_as(x)


class MLPFusion(nn.Module):
    """Fuses per-frame control conditioning into tokens via MLP."""

    def __init__(self, d_model: int):
        super().__init__()
        self.mlp = _waypoint_mlp(2 * d_model, d_model, d_model)
    
    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape
        N = cond.shape[1]
        tokens_per_frame = L // N

        # Split fc_in weights so x and cond are fused without concatenating them.
        Wx, Wc = self.mlp.fc_in.weight.chunk(2, dim=1)
        x_reshaped = x.view(B, N, tokens_per_frame, D)
        h = F.linear(x_reshaped, Wx) + F.linear(cond, Wc).unsqueeze(2)
        h = F.silu(h)
        y = F.linear(h, self.mlp.fc_out.weight)
        
        return y.flatten(1, 2)


class CondHead(nn.Module):
    """Produce per-layer AdaLN modulation vectors."""

    n_cond = 6

    def __init__(self, d_model: int, noise_conditioning: str = "wan"):
        super().__init__()
        self.bias_in = (
            nn.Parameter(torch.zeros(d_model))
            if noise_conditioning == "wan" else None
        )
        self.cond_proj = nn.ModuleList([
            ReplicatedLinear(d_model, d_model, bias=False)
            for _ in range(self.n_cond)
        ])

    def forward(self, cond: torch.Tensor):
        if self.bias_in is not None:
            cond = cond + self.bias_in
        h = F.silu(cond)
        return tuple(proj(h)[0] for proj in self.cond_proj)


class WaypointLayerKVCache(nn.Module):
    def __init__(
        self,
        batch_size: int,
        num_heads: int,
        history_length: int,
        head_dim: int,
        dtype: torch.dtype,
        tokens_per_frame: int,
        pinned_dilation: int = 1,
    ):
        super().__init__()
        self.tpf = tokens_per_frame
        self.history_length = history_length
        self.capacity = history_length + tokens_per_frame
        self.pinned_dilation = pinned_dilation
        self.num_buckets = (
            history_length // tokens_per_frame
        ) // pinned_dilation
        assert (history_length // tokens_per_frame) % pinned_dilation == 0

        self.register_buffer(
            "kv",
            torch.zeros(
                2,
                batch_size,
                num_heads,
                self.capacity,
                head_dim,
                dtype=dtype,
            ),
            persistent=False,
        )
        written = torch.zeros(self.capacity, dtype=torch.bool)
        written[history_length:] = True
        self.register_buffer("written", written, persistent=False)
        offsets = torch.arange(tokens_per_frame, dtype=torch.long)
        self.register_buffer("frame_offsets", offsets, persistent=False)
        self.register_buffer(
            "current_idx", offsets + history_length, persistent=False
        )

    def reset(self) -> None:
        self.kv.zero_()
        self.written.zero_()
        self.written[self.history_length:].fill_(True)

    def upsert(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        frame_t: int,
        is_frozen: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        bucket = (
            frame_t + self.pinned_dilation - 1
        ) // self.pinned_dilation
        ring_idx = (
            self.frame_offsets + bucket % self.num_buckets * self.tpf
        )
        kv = torch.stack((k, v))
        self.kv.index_copy_(3, self.current_idx, kv)

        write_step = frame_t % self.pinned_dilation == 0
        mask_written = self.written.clone()
        if write_step:
            mask_written[ring_idx] = False

        if not is_frozen:
            dst = ring_idx if write_step else self.current_idx
            self.kv.index_copy_(3, dst, kv)
            self.written[dst] = True

        key, value = self.kv.unbind(0)
        return key, value, mask_written.view(1, 1, 1, -1)


class WaypointKVCache(nn.Module):
    def __init__(self, config, batch_size: int, dtype: torch.dtype):
        super().__init__()
        tpf = config.tokens_per_frame
        period = config.global_attn_period
        offset = config.global_attn_offset % period
        self.layers = nn.ModuleList([
            WaypointLayerKVCache(
                batch_size=batch_size,
                num_heads=config.n_kv_heads,
                history_length=(
                    config.global_window
                    if (layer_idx - offset) % period == 0
                    else config.local_window
                ) * tpf,
                head_dim=config.d_model // config.n_heads,
                dtype=dtype,
                tokens_per_frame=tpf,
                pinned_dilation=(
                    config.global_pinned_dilation
                    if (layer_idx - offset) % period == 0
                    else 1
                ),
            )
            for layer_idx in range(config.n_layers)
        ])
        self.is_frozen = True

    def reset(self) -> None:
        for layer in self.layers:
            layer.reset()
        self.is_frozen = True

    def set_frozen(self, is_frozen: bool) -> None:
        self.is_frozen = is_frozen

    def upsert(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        frame_t: int,
        layer_idx: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.layers[layer_idx].upsert(
            k, v, frame_t, self.is_frozen
        )


class GatedSelfAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_kv_heads: int,
        layer_idx: int,
        gated_attn: bool = True,
        rope: Optional["OrthoRoPE"] = None,
    ):
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = d_model // n_heads
        self.layer_idx = layer_idx
        self.gated_attn = gated_attn
        self.rope = rope

        self.q_proj = ReplicatedLinear(d_model, d_model, bias=False)
        self.k_proj = ReplicatedLinear(
            d_model, n_kv_heads * self.head_dim, bias=False
        )
        self.v_proj = ReplicatedLinear(
            d_model, n_kv_heads * self.head_dim, bias=False
        )
        self.out_proj = ReplicatedLinear(d_model, d_model, bias=False)
        if gated_attn:
            self.gate_proj = ReplicatedLinear(
                n_heads, n_heads, bias=False
            )
            nn.init.zeros_(self.gate_proj.weight)

    def forward(
        self,
        x: torch.Tensor,
        pos_ids: dict[str, torch.Tensor],
        kv_cache: WaypointKVCache,
    ) -> torch.Tensor:
        B, L, D = x.shape
        q, _ = self.q_proj(x)
        k, _ = self.k_proj(x)
        v, _ = self.v_proj(x)
        q = q.view(B, L, self.n_heads, self.head_dim)
        k = k.view(B, L, self.n_kv_heads, self.head_dim)
        v = v.view(B, L, self.n_kv_heads, self.head_dim)

        q, k = rms_norm(q), rms_norm(k)
        if self.rope is not None:
            q = self.rope(q, pos_ids)
            k = self.rope(k, pos_ids)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        frame_t = int(pos_ids["t_pos"][0, 0].item())
        k, v, key_mask = kv_cache.upsert(
            k, v, frame_t, self.layer_idx
        )

        if _flex_attention is not None:
            written = key_mask.reshape(-1)

            def mask_mod(b, h, q_idx, kv_idx):
                return written[kv_idx]

            block_mask = create_block_mask(
                mask_mod,
                B=None,
                H=None,
                Q_LEN=L,
                KV_LEN=written.numel(),
                device=q.device,
                _compile=False,
            )
            y = _flex_attention(
                q,
                k,
                v,
                block_mask=block_mask,
                enable_gqa=self.n_heads != self.n_kv_heads,
            )
        else:
            y = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=key_mask,
                dropout_p=0.0,
                is_causal=False,
                enable_gqa=self.n_heads != self.n_kv_heads,
            )

        y = y.transpose(1, 2)
        if self.gated_attn:
            gates, _ = self.gate_proj(x[..., :self.n_heads])
            y = y * torch.sigmoid(gates).unsqueeze(-1)

        y = y.reshape(B, L, D)
        return self.out_proj(y)[0]


class CrossAttention(nn.Module):
    """Prompt cross-attention."""
    
    def __init__(self, d_model: int, context_dim: int, head_dim: int = 64):
        super().__init__()
        self.d_model = d_model
        self.context_dim = context_dim
        self.head_dim = head_dim
        assert context_dim % head_dim == 0, f"context_dim {context_dim} must be divisible by head_dim {head_dim}"
        self.n_heads = context_dim // head_dim

        self.q_proj = ReplicatedLinear(d_model, context_dim, bias=False)
        self.k_proj = ReplicatedLinear(context_dim, context_dim, bias=False)
        self.v_proj = ReplicatedLinear(context_dim, context_dim, bias=False)
        self.out_proj = ReplicatedLinear(context_dim, d_model, bias=False)

    
    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        context_pad_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, L, _ = x.shape
        _, S, _ = context.shape

        q, _ = self.q_proj(x)
        q = q.view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        k, _ = self.k_proj(context)
        k = k.view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        v, _ = self.v_proj(context)
        v = v.view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        q, k = rms_norm(q), rms_norm(k)

        # The published implementation intentionally does not mask prompt padding.
        if _flex_attention is not None:
            attn_out = _flex_attention(q, k, v)
        else:
            attn_out = F.scaled_dot_product_attention(q, k, v)
        attn_out = attn_out.transpose(1, 2).reshape(B, L, self.context_dim)
        out, _ = self.out_proj(attn_out)
        return out


class WaypointBlock(nn.Module):
    """Single Waypoint transformer block."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_kv_heads: int,
        mlp_ratio: int,
        layer_idx: int,
        prompt_conditioning: Optional[str],
        prompt_conditioning_period: int,
        prompt_embedding_dim: int,
        ctrl_conditioning_period: int,
        noise_conditioning: str,
        gated_attn: bool = True,
        rope: Optional[OrthoRoPE] = None,
    ):
        super().__init__()
        self.layer_idx = layer_idx

        self.attn = GatedSelfAttention(
            d_model=d_model,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            layer_idx=layer_idx,
            gated_attn=gated_attn,
            rope=rope,
        )

        self.mlp = _waypoint_mlp(d_model, d_model * mlp_ratio, d_model)
        self.cond_head = CondHead(d_model, noise_conditioning)
        
        do_prompt_cond = (
            prompt_conditioning is not None
            and layer_idx % prompt_conditioning_period == 0
        )
        self.prompt_cross_attn = (
            CrossAttention(d_model, prompt_embedding_dim)
            if do_prompt_cond else None
        )
        do_ctrl_cond = layer_idx % ctrl_conditioning_period == 0
        self.ctrl_mlpfusion = MLPFusion(d_model) if do_ctrl_cond else None
    
    def forward(
        self,
        x: torch.Tensor,
        cond: torch.Tensor,
        prompt_emb: Optional[torch.Tensor] = None,
        prompt_pad_mask: Optional[torch.Tensor] = None,
        ctrl_emb: Optional[torch.Tensor] = None,
        pos_emb: Optional[dict[str, torch.Tensor]] = None,
        kv_cache: WaypointKVCache | None = None,
    ) -> torch.Tensor:
        if pos_emb is None or kv_cache is None:
            raise ValueError("Waypoint attention requires position IDs and a KV cache")
        s0, b0, g0, s1, b1, g1 = self.cond_head(cond)

        residual = x
        x = ada_rmsnorm(x, s0, b0)
        x = self.attn(x, pos_ids=pos_emb, kv_cache=kv_cache)
        x = ada_gate(x, g0) + residual

        if self.prompt_cross_attn is not None and prompt_emb is not None:
            x = self.prompt_cross_attn(
                rms_norm(x),
                context=rms_norm(prompt_emb),
                context_pad_mask=prompt_pad_mask,
            ) + x

        if self.ctrl_mlpfusion is not None and ctrl_emb is not None:
            x = self.ctrl_mlpfusion(rms_norm(x), rms_norm(ctrl_emb)) + x

        x = ada_gate(self.mlp(ada_rmsnorm(x, s1, b1)), g1) + x

        return x


class WaypointTransformer(nn.Module):
    """Stack of Waypoint transformer blocks with shared OrthoRoPE."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        if config.value_residual:
            raise NotImplementedError(
                "Waypoint-1-Small does not use value_residual"
            )
        head_dim = config.d_model // config.n_heads
        rope = OrthoRoPE(
            height=config.height,
            width=config.width,
            n_frames=config.n_frames,
            head_dim=head_dim,
        )
        self.blocks = nn.ModuleList([
            WaypointBlock(
                d_model=config.d_model,
                n_heads=config.n_heads,
                n_kv_heads=config.n_kv_heads,
                mlp_ratio=config.mlp_ratio,
                layer_idx=idx,
                prompt_conditioning=config.prompt_conditioning,
                prompt_conditioning_period=config.prompt_conditioning_period,
                prompt_embedding_dim=config.prompt_embedding_dim,
                ctrl_conditioning_period=config.ctrl_conditioning_period,
                noise_conditioning=config.noise_conditioning,
                gated_attn=config.gated_attn,
                rope=rope,
            )
            for idx in range(config.n_layers)
        ])

        if config.noise_conditioning in ("dit_air", "wan"):
            ref_proj = self.blocks[0].cond_head.cond_proj
            for blk in self.blocks[1:]:
                for blk_mod, ref_mod in zip(blk.cond_head.cond_proj, ref_proj):
                    blk_mod.weight = ref_mod.weight

    def forward(
        self,
        x: torch.Tensor,
        cond: torch.Tensor,
        prompt_emb: Optional[torch.Tensor] = None,
        prompt_pad_mask: Optional[torch.Tensor] = None,
        ctrl_emb: Optional[torch.Tensor] = None,
        pos_emb: Optional[dict[str, torch.Tensor]] = None,
        kv_cache: WaypointKVCache | None = None,
    ) -> torch.Tensor:
        for block in self.blocks:
            x = block(
                x, cond, prompt_emb, prompt_pad_mask, ctrl_emb, pos_emb,
                kv_cache=kv_cache,
            )
        return x


class WaypointWorldModel(BaseDiT):
    """Waypoint-1-Small world model."""
    
    _fsdp_shard_conditions = _DEFAULT_WAYPOINT_ARCH._fsdp_shard_conditions
    _compile_conditions = []
    param_names_mapping = _DEFAULT_WAYPOINT_ARCH.param_names_mapping
    reverse_param_names_mapping = _DEFAULT_WAYPOINT_ARCH.reverse_param_names_mapping
    lora_param_names_mapping: dict = {}
    
    def __init__(self, config, hf_config: dict = None):
        super().__init__(config=config, hf_config=hf_config or {})
        
        self.hidden_size = config.d_model
        self.num_attention_heads = config.n_heads
        self.num_channels_latents = config.channels

        self.denoise_step_emb = NoiseConditioner(config.d_model)
        self.ctrl_emb = ControllerInputEmbedding(
            config.n_buttons, config.d_model, config.mlp_ratio
        )

        if config.ctrl_conditioning is not None:
            self.ctrl_cfg = CFG(config.d_model, config.ctrl_cond_dropout)
        else:
            self.ctrl_cfg = None

        if config.prompt_conditioning is not None:
            self.prompt_cfg = CFG(config.prompt_embedding_dim, config.prompt_cond_dropout)
        else:
            self.prompt_cfg = None

        self.transformer = WaypointTransformer(config)

        self.patch = tuple(config.patch)
        ph, pw = self.patch
        
        self.patchify = nn.Conv2d(
            config.channels,
            config.d_model,
            kernel_size=self.patch,
            stride=self.patch,
            bias=False,
        )
        
        self.unpatchify = ReplicatedLinear(
            config.d_model,
            config.channels * ph * pw,
            bias=True,
        )

        self.out_norm = AdaLN(config.d_model)

        self.__post_init__()

    def materialize_non_persistent_buffers(
        self,
        device: torch.device,
        dtype: torch.dtype | None = None,
    ) -> None:
        self.denoise_step_emb.materialize(device)

    def forward(
        self,
        x: torch.Tensor,
        sigma: torch.Tensor,
        frame_timestamp: torch.Tensor,
        prompt_emb: Optional[torch.Tensor] = None,
        prompt_pad_mask: Optional[torch.Tensor] = None,
        mouse: Optional[torch.Tensor] = None,
        button: Optional[torch.Tensor] = None,
        scroll: Optional[torch.Tensor] = None,
        kv_cache: WaypointKVCache | None = None,
    ) -> torch.Tensor:
        B, N, C, H, W = x.shape
        if B != 1 or N != 1:
            raise ValueError("WaypointWorldModel supports one frame with batch size 1")
        if kv_cache is None:
            raise ValueError("WaypointWorldModel requires a WaypointKVCache")
        ph, pw = self.patch
        
        assert H % ph == 0 and W % pw == 0, f"H={H}, W={W} must be divisible by patch={self.patch}"
        Hp, Wp = H // ph, W // pw
        # Waypoint expects tokens_per_frame = 256 (16*16); latent must match.
        expected_tokens = getattr(
            self.config, "tokens_per_frame", None
        )
        if expected_tokens is not None and Hp * Wp != expected_tokens:
            raise ValueError(
                f"Token layout mismatch: Hp*Wp={Hp * Wp} but "
                f"config.tokens_per_frame={expected_tokens}. "
                "Pipeline must use latent_h,w so that (H//ph)*(W//pw)==tokens_per_frame."
            )

        if mouse is None or button is None or scroll is None:
            raise ValueError("WaypointWorldModel requires mouse, button, and scroll")
        if prompt_emb is None:
            raise ValueError("WaypointWorldModel requires prompt embeddings")

        cond = self.denoise_step_emb(sigma)
        ctrl_emb = self.ctrl_emb(mouse, button, scroll)
        if self.ctrl_cfg is not None:
            ctrl_emb = self.ctrl_cfg(ctrl_emb)

        if prompt_emb is not None and self.prompt_cfg is not None:
            prompt_emb = self.prompt_cfg(prompt_emb)

        x = x.reshape(B * N, C, H, W)
        x = self.patchify(x)
        x = x.view(B, N, self.config.d_model, Hp, Wp)
        x = x.permute(0, 1, 3, 4, 2).reshape(B, N * Hp * Wp, self.config.d_model)

        L = N * Hp * Wp
        idx = torch.arange(Hp * Wp, device=x.device, dtype=torch.long)
        y_xy = idx // Wp
        x_xy = idx % Wp
        y_pos = (
            y_xy.unsqueeze(0).unsqueeze(0).expand(B, N, -1).reshape(B, L)
        )
        x_pos = (
            x_xy.unsqueeze(0).unsqueeze(0).expand(B, N, -1).reshape(B, L)
        )
        t_pos = (
            frame_timestamp.unsqueeze(2).expand(-1, -1, Hp * Wp).reshape(B, L)
        )
        pos_ids = {"t_pos": t_pos, "y_pos": y_pos, "x_pos": x_pos}

        x = self.transformer(
            x, cond, prompt_emb, prompt_pad_mask, ctrl_emb,
            pos_emb=pos_ids, kv_cache=kv_cache,
        )

        x = F.silu(self.out_norm(x, cond))
        x, _ = self.unpatchify(x)
        x = x.view(B, N, Hp, Wp, C, ph, pw)
        x = x.permute(0, 1, 4, 2, 5, 3, 6).reshape(B, N, C, H, W)
        
        return x
    
EntryClass = WaypointWorldModel
