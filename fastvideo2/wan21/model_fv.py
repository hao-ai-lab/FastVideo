"""Vendored fastvideo-main Wan2.1 DiT forward — the numerics authority for
fastvideo-TRAINED artifacts (FastWan, FastWan-QAD, SFWan).

Provenance: hao-ai-lab/FastVideo @ e3f47dc2de2d1fa0c68c5839a0a41ed25b04a953
    fastvideo/models/dits/wanvideo.py         (WanTransformerBlock, WanTransformer3DModel)
    fastvideo/layers/{layernorm,mlp,rotary_embedding,visual_embedding}.py
    fastvideo/attention/{layer.py,backends/flash_attn.py}   (sp=1, dense path)

Why a second Wan DiT next to ``model.py`` (the official vendor): main's
forward is measurably different math — fp32 real-arithmetic RoPE built from a
fp64 table (official: fp64 complex end to end), fp32 sinusoidal time
embedding (official: fp64), fused residual+norm ops whose fp32 promotion
points differ from official's fp32 islands, and dense ``flash_attn_func``
(official: varlen). Against official goldens main's block stack lands at
~1.7e-2 rel; artifacts distilled inside main assume MAIN's math, so cards for
those artifacts point here. Authority follows artifact provenance.

Restructured single-GPU (sp=1, tp=1, T2V only) with DIFFUSERS-NATIVE
checkpoint keys (the layout FastVideo publishes), so loading needs no key
remapping. Equivalence with main is gated bitwise by
``gates/capture_fastvideo_main.py`` goldens + ``gates/anchor.py`` — every
restructuring claim below is only as good as those anchors.

Casting rules are load-bearing; do not "clean up":
  * blocks modulate in fp32 (``scale_shift_table + temb.float()``) but the
    FINAL norm modulates in bf16 (no ``.float()`` upstream) — asymmetric on
    purpose (it's what main executes).
  * self-attn residual promotes to fp32 (``x + attn * gate_fp32``); the
    cross-attn residual does NOT (gate is the int 1 → bf16 + bf16).
  * RMSNorm casts back to input dtype BEFORE multiplying the weight.
  * the fp8 variant quantizes AFTER the bf16 load cast (fp32 ckpt → bf16 →
    fp8 codes); quantizing from fp32 gives different codes.
"""
from __future__ import annotations

import json
import math
import os
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["WanModelFV", "WanModelFVFP8", "WanModelFVVSA"]


# --------------------------------------------------------------------------- #
# RoPE — main's table (fp64, full head_dim via repeat_interleave) + fp32 apply
# --------------------------------------------------------------------------- #
_ROPE_CACHE: dict[tuple, tuple[torch.Tensor, torch.Tensor]] = {}


def _rope_cos_sin(grid: tuple[int, int, int], head_dim: int, theta: float = 10000.0):
    """cos/sin [S, head_dim] in fp64 (CPU, cached). Axis split mirrors main:
    [d - 4*(d//6), 2*(d//6), 2*(d//6)] over (f, h, w), f-major token order."""
    key = (grid, head_dim, theta)
    if key not in _ROPE_CACHE:
        d6 = head_dim // 6
        dim_list = [head_dim - 4 * d6, 2 * d6, 2 * d6]
        f, h, w = grid
        axes = [
            torch.arange(f, dtype=torch.float64).view(f, 1, 1).expand(f, h, w),
            torch.arange(h, dtype=torch.float64).view(1, h, 1).expand(f, h, w),
            torch.arange(w, dtype=torch.float64).view(1, 1, w).expand(f, h, w),
        ]
        cos_parts, sin_parts = [], []
        for dim, pos in zip(dim_list, axes):
            freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float64)[: dim // 2] / dim))
            ang = torch.outer(pos.reshape(-1), freqs)          # [S, dim/2]
            cos_parts.append(ang.cos().repeat_interleave(2, dim=-1))
            sin_parts.append(ang.sin().repeat_interleave(2, dim=-1))
        _ROPE_CACHE[key] = (torch.cat(cos_parts, dim=1), torch.cat(sin_parts, dim=1))
        while len(_ROPE_CACHE) > 16:
            _ROPE_CACHE.pop(next(iter(_ROPE_CACHE)))
    return _ROPE_CACHE[key]


def _apply_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """main's `_apply_rotary_emb`, full-head_dim branch: fp32 rotate-half over
    interleaved pairs, cast back. x: [B, S, H, D]; cos/sin: [S, D] fp32."""
    cos = cos.unsqueeze(-2)
    sin = sin.unsqueeze(-2)
    x_real, x_imag = x.float().reshape(*x.shape[:-1], -1, 2).unbind(-1)
    x_rotated = torch.stack([-x_imag, x_real], dim=-1).flatten(-2)
    return (x.float() * cos + x_rotated * sin).type_as(x)


def _attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, scale: float) -> torch.Tensor:
    """Dense attention on [B, S, H, D], main's backend order: flash_attn_func
    on CUDA (the authority path), SDPA otherwise (local testing fallback)."""
    if q.is_cuda and q.dtype in (torch.float16, torch.bfloat16):
        try:
            from flash_attn import flash_attn_func
            return flash_attn_func(q, k, v, softmax_scale=scale, causal=False)
        except ImportError:
            pass
    out = F.scaled_dot_product_attention(
        q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), scale=scale)
    return out.transpose(1, 2)


# --------------------------------------------------------------------------- #
# Norms / MLP — main's cast semantics, diffusers-native key layout
# --------------------------------------------------------------------------- #
class RMSNormFV(nn.Module):
    """main's RMSNorm.forward_native: fp32 variance, cast back to the input
    dtype, THEN multiply the (load-dtype) weight."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.variance_epsilon = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        x = x.to(torch.float32)
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.variance_epsilon)
        return x.to(orig_dtype) * self.weight


class FP32LayerNormFV(nn.LayerNorm):
    """main's FP32LayerNorm: fp32 layer_norm with explicitly fp32-cast
    weights, output cast back to the INPUT dtype."""

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        origin_dtype = inputs.dtype
        return F.layer_norm(
            inputs.float(),
            self.normalized_shape,
            self.weight.float() if self.weight is not None else None,
            self.bias.float() if self.bias is not None else None,
            self.eps,
        ).to(origin_dtype)


class _GELUProj(nn.Module):
    """diffusers' GELU wrapper (key ``.proj``) with main's tanh approximation."""

    def __init__(self, dim_in: int, dim_out: int):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.gelu(self.proj(x), approximate="tanh")


class FeedForwardFV(nn.Module):
    """Keys: ``net.0.proj`` / ``net.2`` (diffusers); math: main's MLP
    fc_in → gelu(tanh) → fc_out."""

    def __init__(self, dim: int, ffn_dim: int):
        super().__init__()
        self.net = nn.Sequential(_GELUProj(dim, ffn_dim), nn.Identity(), nn.Linear(ffn_dim, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class AttentionFV(nn.Module):
    """Projections + full-dim qk RMSNorm (``rms_norm_across_heads``). Keys:
    to_q/to_k/to_v/to_out.0/norm_q/norm_k. ``forward`` implements CROSS
    attention (main's WanT2VCrossAttention); self-attention order lives in
    the block to preserve main's exact op sequence."""

    def __init__(self, dim: int, num_heads: int, eps: float = 1e-6):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.to_q = nn.Linear(dim, dim)
        self.to_k = nn.Linear(dim, dim)
        self.to_v = nn.Linear(dim, dim)
        self.to_out = nn.Sequential(nn.Linear(dim, dim))
        self.norm_q = RMSNormFV(dim, eps=eps)
        self.norm_k = RMSNormFV(dim, eps=eps)

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        b, n, d = x.size(0), self.num_heads, self.head_dim
        q = self.norm_q(self.to_q(x)).view(b, -1, n, d)
        k = self.norm_k(self.to_k(context)).view(b, -1, n, d)
        v = self.to_v(context).view(b, -1, n, d)
        out = _attention(q, k, v, scale=d ** -0.5)
        return self.to_out(out.flatten(2))


class WanBlockFV(nn.Module):
    """main's WanTransformerBlock.forward, fused ops inlined with identical
    promotion points. Parametered norms keyed as diffusers: only ``norm2``
    (the affine LN after self-attn — main's self_attn_residual_norm.norm)."""

    def __init__(self, dim: int, ffn_dim: int, num_heads: int, eps: float = 1e-6):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.norm1 = FP32LayerNormFV(dim, eps=eps, elementwise_affine=False)
        self.attn1 = AttentionFV(dim, num_heads, eps=eps)
        self.norm2 = FP32LayerNormFV(dim, eps=eps, elementwise_affine=True)
        self.attn2 = AttentionFV(dim, num_heads, eps=eps)
        self.norm3 = FP32LayerNormFV(dim, eps=eps, elementwise_affine=False)
        self.ffn = FeedForwardFV(dim, ffn_dim)
        self.scale_shift_table = nn.Parameter(torch.zeros(1, 6, dim))

    def _self_attention(self, norm_hidden_states, freqs_cis, vsa):
        assert vsa is None, "dense block got VSA metadata — wrong block class for this card"
        a = self.attn1
        query = a.norm_q(a.to_q(norm_hidden_states)).unflatten(2, (self.num_heads, -1))
        key = a.norm_k(a.to_k(norm_hidden_states)).unflatten(2, (self.num_heads, -1))
        value = a.to_v(norm_hidden_states).unflatten(2, (self.num_heads, -1))
        cos, sin = freqs_cis
        query = _apply_rotary(query, cos, sin)
        key = _apply_rotary(key, cos, sin)
        attn_output = _attention(query, key, value, scale=self.head_dim ** -0.5)
        return a.to_out(attn_output.flatten(2))

    def forward(self, hidden_states, encoder_hidden_states, temb, freqs_cis, vsa=None):
        orig_dtype = hidden_states.dtype
        # modulation in fp32 (blocks only — the final norm stays bf16)
        e = self.scale_shift_table + temb.float()
        shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = e.chunk(6, dim=1)

        # 1. self-attention (block-level projections, main's order)
        norm_hidden_states = (self.norm1(hidden_states.float()) * (1 + scale_msa) + shift_msa).to(orig_dtype)
        attn_output = self._self_attention(norm_hidden_states, freqs_cis, vsa)

        # main's self_attn_residual_norm: residual promotes to fp32 via the
        # fp32 gate; affine LN (norm2); null shift/scale are exact no-ops.
        residual = hidden_states + attn_output * gate_msa               # fp32
        norm_hidden_states, hidden_states = self.norm2(residual).to(orig_dtype), residual.to(orig_dtype)

        # 2. cross-attention. main's cross_attn_residual_norm: gate is the
        # int 1 — residual stays bf16 (no promotion), modulation promotes.
        attn_output = self.attn2(norm_hidden_states, encoder_hidden_states)
        residual = hidden_states + attn_output                          # bf16
        norm_hidden_states = (self.norm3(residual) * (1.0 + c_scale_msa) + c_shift_msa).to(orig_dtype)
        hidden_states = residual

        # 3. feed-forward, gated residual promotes via fp32 gate
        ff_output = self.ffn(norm_hidden_states)
        hidden_states = (hidden_states + ff_output * c_gate_msa).to(orig_dtype)
        return hidden_states


class _ConditionEmbedderFV(nn.Module):
    """Keys: time_embedder.linear_1/2, time_proj, text_embedder.linear_1/2.
    Math: main's TimestepEmbedder (fp32 sinusoid) + ModulateProjection
    (silu THEN linear) + text MLP (gelu tanh)."""

    def __init__(self, dim: int, freq_dim: int, text_dim: int):
        super().__init__()
        self.freq_dim = freq_dim
        self.time_embedder = nn.ModuleDict({
            "linear_1": nn.Linear(freq_dim, dim), "linear_2": nn.Linear(dim, dim)})
        self.time_proj = nn.Linear(dim, dim * 6)
        self.text_embedder = nn.ModuleDict({
            "linear_1": nn.Linear(text_dim, dim), "linear_2": nn.Linear(dim, dim)})

    def forward(self, timestep: torch.Tensor, encoder_hidden_states: torch.Tensor):
        half = self.freq_dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(0, half, dtype=torch.float32) / half
                          ).to(device=timestep.device)
        args = timestep[:, None].float() * freqs[None]
        t_freq = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        t_freq = t_freq.to(self.time_embedder["linear_1"].weight.dtype)
        temb = self.time_embedder["linear_2"](F.silu(self.time_embedder["linear_1"](t_freq)))
        timestep_proj = self.time_proj(F.silu(temb))
        te = self.text_embedder
        encoder_hidden_states = te["linear_2"](
            F.gelu(te["linear_1"](encoder_hidden_states), approximate="tanh"))
        return temb, timestep_proj, encoder_hidden_states


class WanBlockFVVSA(WanBlockFV):
    """main's WanTransformerBlock_VSA: identical modulation/residual chain,
    with a gate_compress projection (BLOCK-level checkpoint key — FastVideo
    publishes the VSA checkpoints with ``blocks.N.to_gate_compress``, main's
    own name, inside an otherwise diffusers-keyed layout) and the block-sparse
    kernel in place of dense attention. RoPE applies to q,k only; the gate is
    tiled alongside q/k/v and consumed by the kernel."""

    def __init__(self, dim: int, ffn_dim: int, num_heads: int, eps: float = 1e-6):
        super().__init__(dim, ffn_dim, num_heads, eps=eps)
        self.to_gate_compress = nn.Linear(dim, dim)

    def _self_attention(self, norm_hidden_states, freqs_cis, vsa):
        assert vsa is not None, "VSA block requires metadata (card declares vsa_sparsity)"
        import torch as _torch
        from fastvideo2.layers.vsa import vsa_attention
        a = self.attn1
        query = a.norm_q(a.to_q(norm_hidden_states))
        key = a.norm_k(a.to_k(norm_hidden_states))
        value = a.to_v(norm_hidden_states)
        gate_compress = self.to_gate_compress(norm_hidden_states)
        h = self.num_heads
        query, key, value, gate_compress = (t.unflatten(2, (h, -1))
                                            for t in (query, key, value, gate_compress))
        cos, sin = freqs_cis
        batch = query.shape[0]
        qkvg = _torch.cat([query, key, value, gate_compress], dim=0)
        qkvg[:batch * 2] = _apply_rotary(qkvg[:batch * 2], cos, sin)
        attn_output = vsa_attention(qkvg, vsa)
        return a.to_out(attn_output.flatten(2))


class WanModelFV(nn.Module):
    """main's WanTransformer3DModel at sp=1/tp=1, T2V, diffusers-native keys."""

    block_cls: type = WanBlockFV

    def __init__(self, *, num_attention_heads: int = 12, attention_head_dim: int = 128,
                 in_channels: int = 16, out_channels: int = 16, ffn_dim: int = 8960,
                 freq_dim: int = 256, text_dim: int = 4096, num_layers: int = 30,
                 patch_size: tuple[int, int, int] = (1, 2, 2), eps: float = 1e-6):
        super().__init__()
        dim = num_attention_heads * attention_head_dim
        self.dim = dim
        self.num_attention_heads = num_attention_heads
        self.head_dim = attention_head_dim
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.patch_size = tuple(patch_size)
        self.patch_embedding = nn.Conv3d(in_channels, dim, kernel_size=self.patch_size,
                                         stride=self.patch_size)
        self.condition_embedder = _ConditionEmbedderFV(dim, freq_dim, text_dim)
        self.blocks = nn.ModuleList(
            [type(self).block_cls(dim, ffn_dim, num_attention_heads, eps=eps)
             for _ in range(num_layers)])
        self.norm_out = FP32LayerNormFV(dim, eps=eps, elementwise_affine=False)
        self.proj_out = nn.Linear(dim, out_channels * math.prod(self.patch_size))
        self.scale_shift_table = nn.Parameter(torch.zeros(1, 2, dim))

    def forward(self, hidden_states: torch.Tensor,
                encoder_hidden_states: torch.Tensor | list[torch.Tensor],
                timestep: torch.Tensor, vsa: Any = None, **_ignored: Any) -> torch.Tensor:
        orig_dtype = hidden_states.dtype
        if not isinstance(encoder_hidden_states, torch.Tensor):
            encoder_hidden_states = encoder_hidden_states[0]
        assert timestep.dim() == 1, "single-timestep T2V only"

        batch_size, _, num_frames, height, width = hidden_states.shape
        p_t, p_h, p_w = self.patch_size
        grid = (num_frames // p_t, height // p_h, width // p_w)

        cos64, sin64 = _rope_cos_sin(grid, self.head_dim)
        freqs_cis = (cos64.to(hidden_states.device).float(),
                     sin64.to(hidden_states.device).float())

        hidden_states = self.patch_embedding(hidden_states)
        hidden_states = hidden_states.flatten(2).transpose(1, 2)

        temb, timestep_proj, encoder_hidden_states = self.condition_embedder(
            timestep, encoder_hidden_states)
        timestep_proj = timestep_proj.unflatten(1, (6, -1))
        assert encoder_hidden_states.dtype == orig_dtype

        for block in self.blocks:
            hidden_states = block(hidden_states, encoder_hidden_states, timestep_proj,
                                  freqs_cis, vsa)

        # final modulation in bf16 (no .float() here — main's asymmetry),
        # norm output promoted to fp32 for the modulate, cast back.
        shift, scale = (self.scale_shift_table + temb.unsqueeze(1)).chunk(2, dim=1)
        normalized = self.norm_out(hidden_states).float()
        hidden_states = (normalized * (1.0 + scale) + shift).to(orig_dtype)
        hidden_states = self.proj_out(hidden_states)

        hidden_states = hidden_states.reshape(batch_size, *grid, p_t, p_h, p_w, -1)
        hidden_states = hidden_states.permute(0, 7, 1, 4, 2, 5, 3, 6)
        return hidden_states.flatten(6, 7).flatten(4, 5).flatten(2, 3)

    # ---------------------------------------------------------------- load #
    @classmethod
    def from_pretrained(cls, root: str, torch_dtype: Any = None,
                        subfolder: str = "transformer") -> "WanModelFV":
        folder = os.path.join(root, subfolder) if subfolder else root
        with open(os.path.join(folder, "config.json")) as f:
            cfg = json.load(f)
        assert cfg.get("_class_name") == "WanTransformer3DModel", cfg.get("_class_name")
        model = cls(num_attention_heads=cfg["num_attention_heads"],
                    attention_head_dim=cfg["attention_head_dim"],
                    in_channels=cfg["in_channels"], out_channels=cfg["out_channels"],
                    ffn_dim=cfg["ffn_dim"], freq_dim=cfg["freq_dim"],
                    text_dim=cfg["text_dim"], num_layers=cfg["num_layers"],
                    patch_size=tuple(cfg["patch_size"]), eps=cfg.get("eps", 1e-6))
        from safetensors.torch import load_file
        state: dict[str, torch.Tensor] = {}
        for name in sorted(os.listdir(folder)):
            if name.endswith(".safetensors"):
                state.update(load_file(os.path.join(folder, name)))
        model.load_state_dict(state, strict=True)  # fail closed on layout drift
        if torch_dtype is not None:
            # blanket cast, mirroring main's loader (param_dtype for ALL
            # params — no fp32 islands in main's load, unlike official's).
            model = model.to(torch_dtype)
        return model.eval()


class WanModelFVVSA(WanModelFV):
    """FastWan (VSA-distilled) serving variant: bf16, block-sparse attention.
    Loading fails closed if the checkpoint lacks ``to_gate_compress`` keys
    (strict load) or if the kernel is missing (first forward raises)."""

    block_cls = WanBlockFVVSA


class WanModelFVFP8(WanModelFV):
    """FastWan-QAD serving variant: same forward, with main's dynamic FP8 on
    exactly the linears main's suffix match hits (both attentions' q/k/v/out
    + both FFN projections; embedders and proj_out stay bf16)."""

    _FP8_LINEARS = ("attn1.to_q", "attn1.to_k", "attn1.to_v", "attn1.to_out.0",
                    "attn2.to_q", "attn2.to_k", "attn2.to_v", "attn2.to_out.0",
                    "ffn.net.0.proj", "ffn.net.2")

    @classmethod
    def from_pretrained(cls, root: str, torch_dtype: Any = None,
                        subfolder: str = "transformer") -> "WanModelFV":
        model = super().from_pretrained(root, torch_dtype=torch_dtype, subfolder=subfolder)
        from fastvideo2.layers.fp8 import quantize_fp8_
        names = [f"blocks.{i}.{leaf}" for i in range(len(model.blocks))
                 for leaf in cls._FP8_LINEARS]
        return quantize_fp8_(model, names)
