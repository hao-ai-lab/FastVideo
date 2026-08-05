# SPDX-License-Identifier: Apache-2.0
"""FastWan-oriented helpers for the experimental MLX runtime path."""

from __future__ import annotations

import json
import math
import os
import statistics
import time
from dataclasses import dataclass
from pathlib import Path

from fastvideo.logger import init_logger

logger = init_logger(__name__)


@dataclass(frozen=True)
class FastWanShape:
    height: int
    width: int
    num_frames: int
    latent_frames: int
    latent_height: int
    latent_width: int
    patch_frames: int
    patch_height: int
    patch_width: int
    tokens: int
    hidden_size: int
    num_heads: int
    head_dim: int


class UnsupportedMLXQuantizationError(ValueError):
    """A quantization mode the installed MLX build cannot execute.

    Raised by :func:`ensure_quantization_supported` before any model weights
    are loaded, so callers (CLI flags, benchmark sweeps) can fail fast with an
    actionable message -- or skip the mode -- instead of crashing deep inside
    ``mx.quantize`` mid-load.
    """


@dataclass(frozen=True)
class MLXQuantizationSpec:
    """MLX quantized-matmul configuration for DiT linear weights."""

    mode: str
    bits: int | None = None
    group_size: int | None = None

    @classmethod
    def from_name(cls, name: str | None) -> "MLXQuantizationSpec | None":
        if name is None or name in {"", "none", "fp16", "fp32"}:
            return None
        if name == "int8":
            return cls(mode="affine", bits=8, group_size=64)
        if name == "int4":
            return cls(mode="affine", bits=4, group_size=64)
        if name == "mxfp8":
            return cls(mode="mxfp8")
        if name == "mxfp4":
            return cls(mode="mxfp4")
        if name == "nvfp4":
            return cls(mode="nvfp4")
        raise ValueError(f"Unsupported MLX quantization mode: {name}")

    @property
    def label(self) -> str:
        if self.mode == "affine":
            return f"int{self.bits}"
        return self.mode


@dataclass(frozen=True)
class QuantizedMatrix:
    weight: "mx.array"
    scales: "mx.array"
    biases: "mx.array | None"
    spec: MLXQuantizationSpec
    dequantized_dtype: "mx.Dtype"


def fastwan_shape(
    *,
    height: int,
    width: int,
    num_frames: int,
    vae_temporal_compression: int = 4,
    vae_spatial_compression: int = 8,
    patch_size: tuple[int, int, int] = (1, 2, 2),
    num_heads: int = 12,
    head_dim: int = 128,
) -> FastWanShape:
    """Return the approximate DiT token shape for Wan/FastWan T2V inference."""
    latent_frames = (num_frames - 1) // vae_temporal_compression + 1
    latent_height = height // vae_spatial_compression
    latent_width = width // vae_spatial_compression
    patch_frames = latent_frames // patch_size[0]
    patch_height = latent_height // patch_size[1]
    patch_width = latent_width // patch_size[2]
    tokens = patch_frames * patch_height * patch_width
    return FastWanShape(
        height=height,
        width=width,
        num_frames=num_frames,
        latent_frames=latent_frames,
        latent_height=latent_height,
        latent_width=latent_width,
        patch_frames=patch_frames,
        patch_height=patch_height,
        patch_width=patch_width,
        tokens=tokens,
        hidden_size=num_heads * head_dim,
        num_heads=num_heads,
        head_dim=head_dim,
    )


def fastwan_shape_from_config(
    config_path: str | Path,
    *,
    height: int,
    width: int,
    num_frames: int,
) -> FastWanShape:
    config = json.loads(Path(config_path).read_text())
    return fastwan_shape(
        height=height,
        width=width,
        num_frames=num_frames,
        patch_size=tuple(config["patch_size"]),
        num_heads=int(config["num_attention_heads"]),
        head_dim=int(config["attention_head_dim"]),
    )


def replace_tokens(shape: FastWanShape, tokens: int) -> FastWanShape:
    return FastWanShape(**{**shape.__dict__, "tokens": tokens})


def median_ms(samples: list[float]) -> float:
    return statistics.median(samples) * 1000.0


def benchmark_mlx_attention(shape: FastWanShape, warmup: int, iters: int) -> float:
    import mlx.core as mx

    q = mx.random.normal((1, shape.num_heads, shape.tokens, shape.head_dim), dtype=mx.float16)
    k = mx.random.normal((1, shape.num_heads, shape.tokens, shape.head_dim), dtype=mx.float16)
    v = mx.random.normal((1, shape.num_heads, shape.tokens, shape.head_dim), dtype=mx.float16)
    scale = shape.head_dim**-0.5

    for _ in range(warmup):
        y = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale)
        mx.eval(y)

    samples = []
    for _ in range(iters):
        start = time.perf_counter()
        y = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale)
        mx.eval(y)
        samples.append(time.perf_counter() - start)
    return median_ms(samples)


def benchmark_mlx_linear(shape: FastWanShape, warmup: int, iters: int) -> float:
    import mlx.core as mx

    x = mx.random.normal((shape.tokens, shape.hidden_size), dtype=mx.float16)
    w = mx.random.normal((shape.hidden_size, shape.hidden_size), dtype=mx.float16)
    b = mx.zeros((shape.hidden_size,), dtype=mx.float16)

    for _ in range(warmup):
        y = x @ w + b
        mx.eval(y)

    samples = []
    for _ in range(iters):
        start = time.perf_counter()
        y = x @ w + b
        mx.eval(y)
        samples.append(time.perf_counter() - start)
    return median_ms(samples)


def benchmark_torch_mps_attention(shape: FastWanShape, warmup: int, iters: int) -> float | None:
    try:
        import torch
        import torch.nn.functional as F
    except ImportError:
        return None

    if not torch.backends.mps.is_available():
        return None

    device = torch.device("mps")
    q = torch.randn((1, shape.num_heads, shape.tokens, shape.head_dim), device=device, dtype=torch.float16)
    k = torch.randn((1, shape.num_heads, shape.tokens, shape.head_dim), device=device, dtype=torch.float16)
    v = torch.randn((1, shape.num_heads, shape.tokens, shape.head_dim), device=device, dtype=torch.float16)

    for _ in range(warmup):
        y = F.scaled_dot_product_attention(q, k, v)
        torch.mps.synchronize()
        _ = y

    samples = []
    for _ in range(iters):
        start = time.perf_counter()
        y = F.scaled_dot_product_attention(q, k, v)
        torch.mps.synchronize()
        _ = y
        samples.append(time.perf_counter() - start)
    return median_ms(samples)


def torch_to_mx(tensor) -> "mx.array":
    import mlx.core as mx

    return mx.array(tensor.detach().cpu().float().numpy())


def weight_dtype(weight):
    if isinstance(weight, QuantizedMatrix):
        return weight.dequantized_dtype
    return weight.dtype


_QUANT_SUPPORT_CACHE: dict[tuple[str, int | None, int | None], str | None] = {}


def quantization_support_error(spec: MLXQuantizationSpec) -> str | None:
    """Probe whether the installed MLX build supports ``spec``.

    Runs a tiny ``mx.quantize`` + ``mx.quantized_matmul`` with exactly the
    arguments :func:`quantize_matrix` / :func:`linear` use, so the result
    reflects the real runtime path. The affine (int8/int4) modes are stable
    across MLX releases, but the ``mxfp8``/``mxfp4``/``nvfp4`` mode strings
    require newer MLX builds and raise otherwise. Returns ``None`` when the
    mode works, else the underlying error message. Cached per spec.
    """
    key = (spec.mode, spec.bits, spec.group_size)
    if key not in _QUANT_SUPPORT_CACHE:
        import mlx.core as mx

        try:
            probe_dim = max(spec.group_size or 0, 64)
            weight = mx.zeros((probe_dim, probe_dim), dtype=mx.float16)
            quantized = quantize_matrix(weight, spec)
            y = linear(mx.zeros((1, probe_dim), dtype=mx.float16), quantized)
            mx.eval(y)
            _QUANT_SUPPORT_CACHE[key] = None
        except Exception as exc:  # noqa: BLE001 - MLX raises varied error types per backend/version.
            _QUANT_SUPPORT_CACHE[key] = f"{type(exc).__name__}: {exc}"
    return _QUANT_SUPPORT_CACHE[key]


def ensure_quantization_supported(spec: MLXQuantizationSpec | None) -> None:
    """Raise :class:`UnsupportedMLXQuantizationError` if ``spec`` cannot run here."""
    if spec is None:
        return
    error = quantization_support_error(spec)
    if error is None:
        return
    import mlx.core as mx

    mlx_version = getattr(mx, "__version__", "unknown")
    raise UnsupportedMLXQuantizationError(
        f"MLX quantization mode '{spec.label}' is not supported by the installed mlx "
        f"({mlx_version}): {error}. Upgrade mlx or pick a supported mode "
        f"(int8 is currently the most reliable quality/memory target).")


def quantize_matrix(weight, spec: MLXQuantizationSpec | None):
    if spec is None:
        return weight
    import mlx.core as mx

    if len(weight.shape) < 2:
        return weight
    q = mx.quantize(weight, group_size=spec.group_size, bits=spec.bits, mode=spec.mode)
    biases = q[2] if len(q) == 3 else None
    eval_args = [q[0], q[1]]
    if biases is not None:
        eval_args.append(biases)
    mx.eval(*eval_args)
    return QuantizedMatrix(
        weight=q[0],
        scales=q[1],
        biases=biases,
        spec=spec,
        dequantized_dtype=weight.dtype,
    )


def linear(x, weight, bias=None):
    import mlx.core as mx

    if isinstance(weight, QuantizedMatrix):
        y = mx.quantized_matmul(
            x,
            weight.weight,
            weight.scales,
            weight.biases,
            transpose=True,
            group_size=weight.spec.group_size,
            bits=weight.spec.bits,
            mode=weight.spec.mode,
        ).astype(x.dtype)
    else:
        y = x @ weight.T
    if bias is not None:
        y = y + bias
    return y


def _use_fast_norm() -> bool:
    """Opt-in to MLX's fused ``mx.fast`` normalization kernels.

    Off by default so the numerically-explicit reference path stays the
    baseline. Set ``FASTVIDEO_MLX_FAST_NORM=1`` to route LayerNorm/RMSNorm
    through single fused Metal kernels (fewer intermediates, less memory
    traffic) and benchmark the speedup.
    """
    import os

    return os.environ.get("FASTVIDEO_MLX_FAST_NORM", "0") == "1"


def layer_norm(x, weight=None, bias=None, eps: float = 1e-6):
    import mlx.core as mx

    if _use_fast_norm():
        # Compute in fp32 (matching the reference below) so downstream dtype
        # and precision are identical across call sites.
        w = weight.astype(mx.float32) if weight is not None else None
        b = bias.astype(mx.float32) if bias is not None else None
        return mx.fast.layer_norm(x.astype(mx.float32), w, b, eps)

    x_float = x.astype(mx.float32)
    mean = mx.mean(x_float, axis=-1, keepdims=True)
    var = mx.mean(mx.square(x_float - mean), axis=-1, keepdims=True)
    y = (x_float - mean) * mx.rsqrt(var + eps)
    if weight is not None:
        y = y * weight
    if bias is not None:
        y = y + bias
    return y


def rms_norm(x, weight, eps: float = 1e-6):
    import mlx.core as mx

    if _use_fast_norm():
        return mx.fast.rms_norm(x, weight, eps)

    orig_dtype = x.dtype
    x_float = x.astype(mx.float32)
    variance = mx.mean(mx.square(x_float), axis=-1, keepdims=True)
    y = x_float * mx.rsqrt(variance + eps)
    return y.astype(orig_dtype) * weight


def apply_rotary_emb(x, cos, sin, *, is_neox_style: bool = False):
    """Apply FastVideo's rotary convention to MLX tensors.

    Args:
        x: [batch, seq, heads, head_dim]
        cos/sin: [seq, head_dim] for Wan's full-dimension rotate-pair style,
          or [seq, head_dim // 2] for traditional RoPE.
    """
    import mlx.core as mx

    head_size = x.shape[-1]
    rope_dim = cos.shape[-1]
    cos = cos[None, :, None, :]
    sin = sin[None, :, None, :]
    x_float = x.astype(mx.float32)

    if rope_dim == head_size:
        x_pairs = x_float.reshape(*x.shape[:-1], -1, 2)
        x_real = x_pairs[..., 0]
        x_imag = x_pairs[..., 1]
        x_rotated = mx.stack([-x_imag, x_real], axis=-1).reshape(*x.shape)
        return (x_float * cos + x_rotated * sin).astype(x.dtype)

    if is_neox_style:
        x1, x2 = mx.split(x_float, 2, axis=-1)
        o1 = x1 * cos - x2 * sin
        o2 = x2 * cos + x1 * sin
        return mx.concatenate([o1, o2], axis=-1).astype(x.dtype)

    x1 = x_float[..., ::2]
    x2 = x_float[..., 1::2]
    o1 = x1 * cos - x2 * sin
    o2 = x2 * cos + x1 * sin
    return mx.stack([o1, o2], axis=-1).reshape(*x.shape).astype(x.dtype)


# tanh-GELU constant as a Python float. A NumPy scalar (``np.sqrt(...)``) here
# makes ``np.float64 * mx.array`` dispatch through NumPy, which evals the traced
# array and breaks mx.compile ("Attempting to eval an array during function
# transformations"). A plain float dispatches through mx and traces cleanly.
_GELU_TANH_COEF = math.sqrt(2.0 / math.pi)


def gelu_tanh(x):
    import mlx.core as mx

    return 0.5 * x * (1.0 + mx.tanh(_GELU_TANH_COEF * (x + 0.044715 * mx.power(x, 3.0))))


def silu(x):
    import mlx.core as mx

    return x * mx.sigmoid(x)


def timestep_embedding(t, dim: int, max_period: int = 10000):
    import mlx.core as mx

    half = dim // 2
    freqs = mx.exp(-math.log(max_period) * mx.arange(0, half, dtype=mx.float32) / half)
    args = t[:, None].astype(mx.float32) * freqs[None]
    embedding = mx.concatenate([mx.cos(args), mx.sin(args)], axis=-1)
    if dim % 2:
        embedding = mx.concatenate([embedding, mx.zeros_like(embedding[:, :1])], axis=-1)
    return embedding


def scale_residual(residual, x, gate):
    return residual + x * gate


def scale_residual_layer_norm_scale_shift(residual, x, gate, shift, scale, weight=None, bias=None, eps: float = 1e-6):
    if isinstance(gate, int):
        assert gate == 1
        residual_output = residual + x
    else:
        residual_output = residual + x * gate
    normalized = layer_norm(residual_output, weight=weight, bias=bias, eps=eps)
    modulated = normalized * (1.0 + scale) + shift
    return modulated, residual_output


class MLXWanT2VCrossAttention:
    def __init__(self, weights: dict[str, "mx.array"], *, dim: int, num_heads: int, eps: float = 1e-6) -> None:
        self.weights = weights
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.eps = eps

    def __call__(self, x, context):
        import mlx.core as mx

        batch = x.shape[0]
        q = linear(x, self.weights["attn2.to_q.weight"], self.weights.get("attn2.to_q.bias"))
        q = rms_norm(q, self.weights["attn2.norm_q.weight"], eps=self.eps).reshape(
            batch, -1, self.num_heads, self.head_dim)

        if context.shape[1] == 0:
            attended = mx.zeros_like(q)
        else:
            k = linear(context, self.weights["attn2.to_k.weight"], self.weights.get("attn2.to_k.bias"))
            k = rms_norm(k, self.weights["attn2.norm_k.weight"], eps=self.eps).reshape(
                batch, -1, self.num_heads, self.head_dim)
            v = linear(context, self.weights["attn2.to_v.weight"], self.weights.get("attn2.to_v.bias")).reshape(
                batch, -1, self.num_heads, self.head_dim)
            attended = mx.fast.scaled_dot_product_attention(
                q.transpose(0, 2, 1, 3),
                k.transpose(0, 2, 1, 3),
                v.transpose(0, 2, 1, 3),
                scale=self.head_dim**-0.5,
            ).transpose(0, 2, 1, 3)

        attended = attended.reshape(batch, -1, self.dim)
        return linear(attended, self.weights["attn2.to_out.weight"], self.weights.get("attn2.to_out.bias"))


class MLXWanTransformerBlock:
    """Dense T2V Wan transformer block for the experimental MLX runtime.

    This mirrors the non-VSA PyTorch block for single-process dense attention.
    Rotary embeddings and sequence-parallel paths are intentionally left out of
    this first parity target.
    """

    def __init__(self, weights: dict[str, "mx.array"], *, dim: int, ffn_dim: int, num_heads: int, eps: float = 1e-6):
        self.weights = weights
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.eps = eps
        self.attn2 = MLXWanT2VCrossAttention(weights, dim=dim, num_heads=num_heads, eps=eps)

    def __call__(self, hidden_states, encoder_hidden_states, temb, freqs_cis=None):
        import mlx.core as mx

        orig_dtype = hidden_states.dtype
        e = self.weights["scale_shift_table"] + temb.astype(mx.float32)
        shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = mx.split(e, 6, axis=1)

        norm_hidden_states = layer_norm(hidden_states.astype(mx.float32), eps=self.eps)
        norm_hidden_states = (norm_hidden_states * (1.0 + scale_msa) + shift_msa).astype(orig_dtype)

        query = linear(norm_hidden_states, self.weights["to_q.weight"], self.weights.get("to_q.bias"))
        key = linear(norm_hidden_states, self.weights["to_k.weight"], self.weights.get("to_k.bias"))
        value = linear(norm_hidden_states, self.weights["to_v.weight"], self.weights.get("to_v.bias"))

        query = rms_norm(query, self.weights["norm_q.weight"], eps=self.eps).reshape(
            hidden_states.shape[0], -1, self.num_heads, self.head_dim)
        key = rms_norm(key, self.weights["norm_k.weight"], eps=self.eps).reshape(
            hidden_states.shape[0], -1, self.num_heads, self.head_dim)
        value = value.reshape(hidden_states.shape[0], -1, self.num_heads, self.head_dim)

        if freqs_cis is not None:
            cos, sin = freqs_cis
            query = apply_rotary_emb(query, cos, sin, is_neox_style=False)
            key = apply_rotary_emb(key, cos, sin, is_neox_style=False)

        # Self-attention only. FASTVIDEO_MLX_WINDOW=0/unset → full SDPA (byte-identical
        # to the historical path). When >0, use chunked symmetric sliding-window
        # attention (see windowed_attention.py). Cross-attn (attn2) stays full.
        # Optional FASTVIDEO_MLX_WINDOW_SINK (default 0) adds global sink tokens.
        q_bh = query.transpose(0, 2, 1, 3)  # (B, H, S, D)
        k_bh = key.transpose(0, 2, 1, 3)
        v_bh = value.transpose(0, 2, 1, 3)
        scale = self.head_dim**-0.5
        window = int(os.environ.get("FASTVIDEO_MLX_WINDOW", "0") or "0")
        if window > 0:
            from fastvideo.mlx_runtime.windowed_attention import windowed_attention

            sink = int(os.environ.get("FASTVIDEO_MLX_WINDOW_SINK", "0") or "0")
            attn_output = windowed_attention(q_bh, k_bh, v_bh, window=window, sink=sink, scale=scale)
        else:
            attn_output = mx.fast.scaled_dot_product_attention(q_bh, k_bh, v_bh, scale=scale)
        attn_output = attn_output.transpose(0, 2, 1, 3)
        attn_output = attn_output.reshape(hidden_states.shape[0], -1, self.dim)
        attn_output = linear(attn_output, self.weights["to_out.weight"], self.weights.get("to_out.bias"))

        norm_hidden_states, hidden_states = scale_residual_layer_norm_scale_shift(
            hidden_states,
            attn_output,
            gate_msa,
            0.0,
            0.0,
            weight=self.weights["self_attn_residual_norm.norm.weight"],
            bias=self.weights["self_attn_residual_norm.norm.bias"],
            eps=self.eps,
        )
        norm_hidden_states = norm_hidden_states.astype(orig_dtype)
        hidden_states = hidden_states.astype(orig_dtype)

        attn_output = self.attn2(norm_hidden_states, encoder_hidden_states)
        norm_hidden_states, hidden_states = scale_residual_layer_norm_scale_shift(
            hidden_states,
            attn_output,
            1,
            c_shift_msa,
            c_scale_msa,
            eps=self.eps,
        )
        norm_hidden_states = norm_hidden_states.astype(orig_dtype)
        hidden_states = hidden_states.astype(orig_dtype)

        ff_output = linear(norm_hidden_states, self.weights["ffn.fc_in.weight"], self.weights.get("ffn.fc_in.bias"))
        ff_output = gelu_tanh(ff_output)
        ff_output = linear(ff_output, self.weights["ffn.fc_out.weight"], self.weights.get("ffn.fc_out.bias"))
        hidden_states = scale_residual(hidden_states, ff_output, c_gate_msa)
        return hidden_states.astype(orig_dtype)


def mlx_block_weights_from_torch(torch_block) -> dict[str, "mx.array"]:
    return {name: torch_to_mx(value) for name, value in torch_block.state_dict().items()}


class MLXWanDiT:
    """Experimental FP16 Wan/FastWan DiT forward path in MLX."""

    def __init__(
        self,
        weights: dict[str, "mx.array"],
        blocks: list[MLXWanTransformerBlock],
        config: dict,
        *,
        compile: bool = False,
    ) -> None:
        import os

        self.weights = weights
        self.blocks = blocks
        self.config = config
        self.num_heads = int(config["num_attention_heads"])
        self.head_dim = int(config["attention_head_dim"])
        self.hidden_size = self.num_heads * self.head_dim
        self.ffn_dim = int(config["ffn_dim"])
        self.in_channels = int(config["in_channels"])
        self.out_channels = int(config["out_channels"])
        self.patch_size = tuple(config["patch_size"])
        self.freq_dim = int(config["freq_dim"])
        # Opt-in graph fusion. With fixed weights and static shapes, the whole
        # denoise-step forward is a pure function of (latents, timestep) -- a
        # good mx.compile target. Off by default so the eager path stays the
        # baseline; enable via constructor or FASTVIDEO_MLX_COMPILE=1 and verify
        # with the benchmark's SSIM ~= 1.0 check.
        self._enable_compile = compile or os.environ.get("FASTVIDEO_MLX_COMPILE", "0") == "1"
        self._compiled_forward = None

    def patch_embed(self, hidden_states):
        batch, channels, frames, height, width = hidden_states.shape
        pt, ph, pw = self.patch_size
        patch_dim = channels * pt * ph * pw
        x = hidden_states.reshape(batch, channels, frames // pt, pt, height // ph, ph, width // pw, pw)
        x = x.transpose(0, 2, 4, 6, 1, 3, 5, 7).reshape(batch, -1, patch_dim)
        return linear(x, self.weights["patch_embedding.weight"], self.weights.get("patch_embedding.bias"))

    def condition(self, timestep, encoder_hidden_states):
        t_freq = timestep_embedding(timestep, self.freq_dim).astype(
            weight_dtype(self.weights["condition_embedder.time_embedder.linear_1.weight"]))
        temb = linear(
            t_freq,
            self.weights["condition_embedder.time_embedder.linear_1.weight"],
            self.weights["condition_embedder.time_embedder.linear_1.bias"],
        )
        temb = silu(temb)
        temb = linear(
            temb,
            self.weights["condition_embedder.time_embedder.linear_2.weight"],
            self.weights["condition_embedder.time_embedder.linear_2.bias"],
        )
        timestep_proj = silu(temb)
        timestep_proj = linear(
            timestep_proj,
            self.weights["condition_embedder.time_proj.weight"],
            self.weights["condition_embedder.time_proj.bias"],
        ).reshape(timestep.shape[0], 6, self.hidden_size)

        encoder_hidden_states = linear(
            encoder_hidden_states,
            self.weights["condition_embedder.text_embedder.linear_1.weight"],
            self.weights["condition_embedder.text_embedder.linear_1.bias"],
        )
        encoder_hidden_states = gelu_tanh(encoder_hidden_states)
        encoder_hidden_states = linear(
            encoder_hidden_states,
            self.weights["condition_embedder.text_embedder.linear_2.weight"],
            self.weights["condition_embedder.text_embedder.linear_2.bias"],
        )
        return temb, timestep_proj, encoder_hidden_states

    def output(self, hidden_states, temb, *, batch: int, frames: int, height: int, width: int):
        pt, ph, pw = self.patch_size
        post_patch_frames = frames // pt
        post_patch_height = height // ph
        post_patch_width = width // pw
        shift, scale = mx_split_two(self.weights["scale_shift_table"] + temb[:, None, :], axis=1)
        hidden_states = layer_norm(hidden_states, eps=float(self.config["eps"])) * (1.0 + scale) + shift
        hidden_states = hidden_states.astype(weight_dtype(self.weights["proj_out.weight"]))
        hidden_states = linear(hidden_states, self.weights["proj_out.weight"], self.weights["proj_out.bias"])
        hidden_states = hidden_states.reshape(
            batch,
            post_patch_frames,
            post_patch_height,
            post_patch_width,
            pt,
            ph,
            pw,
            self.out_channels,
        )
        hidden_states = hidden_states.transpose(0, 7, 1, 4, 2, 5, 3, 6)
        return hidden_states.reshape(batch, self.out_channels, frames, height, width)

    def _forward(self, hidden_states, encoder_hidden_states, timestep, cos, sin):
        """Pure forward used both eagerly and as the mx.compile target.

        ``cos``/``sin`` are passed as separate array args (rather than a tuple)
        so the function traces cleanly under mx.compile.
        """
        batch, _, frames, height, width = hidden_states.shape
        freqs_cis = (cos, sin) if cos is not None else None
        hidden_states = self.patch_embed(hidden_states)
        temb, timestep_proj, encoder_hidden_states = self.condition(timestep, encoder_hidden_states)
        for block in self.blocks:
            hidden_states = block(hidden_states, encoder_hidden_states, timestep_proj, freqs_cis=freqs_cis)
        return self.output(hidden_states, temb, batch=batch, frames=frames, height=height, width=width)

    def __call__(self, hidden_states, encoder_hidden_states, timestep, freqs_cis):
        cos, sin = freqs_cis if freqs_cis is not None else (None, None)
        if self._enable_compile and cos is not None:
            import mlx.core as mx

            if self._compiled_forward is None:
                self._compiled_forward = mx.compile(self._forward)
            try:
                return self._compiled_forward(hidden_states, encoder_hidden_states, timestep, cos, sin)
            except Exception as exc:  # noqa: BLE001 - some quant graphs may not trace; fall back to eager.
                logger.warning("mx.compile forward failed (%s); falling back to eager execution.", exc)
                self._enable_compile = False
                self._compiled_forward = None
        return self._forward(hidden_states, encoder_hidden_states, timestep, cos, sin)


def mx_split_two(x, *, axis: int):
    import mlx.core as mx

    left, right = mx.split(x, 2, axis=axis)
    return left, right


def _load_safetensor_value(handle, name: str):
    return handle.get_tensor(name)


def _load_mx_array_from_safetensor(handle, name: str, dtype):
    """Load a safetensors value and cast before creating the MLX array.

    The FastWan Diffusers checkpoint is fp32. Creating an MLX array first and
    then casting it to fp16 briefly materializes a large fp32 MLX allocation.
    Casting the CPU tensor before crossing into MLX keeps the transient GPU-side
    footprint lower.
    """
    import mlx.core as mx
    import torch

    tensor = handle.get_tensor(name)
    if dtype == mx.float16:
        tensor = tensor.to(torch.float16)
    elif dtype == mx.float32:
        tensor = tensor.to(torch.float32)
    elif dtype == mx.bfloat16:
        # NumPy has no bfloat16, so bridge through fp32 and cast on-device below.
        tensor = tensor.to(torch.float32)
    array = mx.array(tensor.numpy())
    del tensor
    if dtype is not None and array.dtype != dtype:
        array = array.astype(dtype)
    mx.eval(array)
    return array


def _eval_loaded_weight(value) -> None:
    import mlx.core as mx

    if isinstance(value, QuantizedMatrix):
        eval_args = [value.weight, value.scales]
        if value.biases is not None:
            eval_args.append(value.biases)
        mx.eval(*eval_args)
    else:
        mx.eval(value)


def mlx_block_weights_from_diffusers_safetensors(
    checkpoint_path: str | Path,
    *,
    block_index: int = 0,
    quantization: str | MLXQuantizationSpec | None = None,
    dtype=None,
) -> dict[str, "mx.array"]:
    """Load one Diffusers-format Wan block into the MLX dense-block key layout."""
    import mlx.core as mx
    from safetensors import safe_open

    prefix = f"blocks.{block_index}."
    key_map = {
        "scale_shift_table": "scale_shift_table",
        "attn1.to_q.weight": "to_q.weight",
        "attn1.to_q.bias": "to_q.bias",
        "attn1.to_k.weight": "to_k.weight",
        "attn1.to_k.bias": "to_k.bias",
        "attn1.to_v.weight": "to_v.weight",
        "attn1.to_v.bias": "to_v.bias",
        "attn1.to_out.0.weight": "to_out.weight",
        "attn1.to_out.0.bias": "to_out.bias",
        "attn1.norm_q.weight": "norm_q.weight",
        "attn1.norm_k.weight": "norm_k.weight",
        "attn2.to_q.weight": "attn2.to_q.weight",
        "attn2.to_q.bias": "attn2.to_q.bias",
        "attn2.to_k.weight": "attn2.to_k.weight",
        "attn2.to_k.bias": "attn2.to_k.bias",
        "attn2.to_v.weight": "attn2.to_v.weight",
        "attn2.to_v.bias": "attn2.to_v.bias",
        "attn2.to_out.0.weight": "attn2.to_out.weight",
        "attn2.to_out.0.bias": "attn2.to_out.bias",
        "attn2.norm_q.weight": "attn2.norm_q.weight",
        "attn2.norm_k.weight": "attn2.norm_k.weight",
        "ffn.net.0.proj.weight": "ffn.fc_in.weight",
        "ffn.net.0.proj.bias": "ffn.fc_in.bias",
        "ffn.net.2.weight": "ffn.fc_out.weight",
        "ffn.net.2.bias": "ffn.fc_out.bias",
        "norm2.weight": "self_attn_residual_norm.norm.weight",
        "norm2.bias": "self_attn_residual_norm.norm.bias",
    }

    spec = MLXQuantizationSpec.from_name(quantization) if (quantization is None or isinstance(quantization, str)) else quantization
    ensure_quantization_supported(spec)
    matrix_targets = {target for target in key_map.values() if target.endswith(".weight") and "norm" not in target}
    weights = {}
    with safe_open(str(checkpoint_path), framework="pt", device="cpu") as handle:
        available = set(handle.keys())
        for source_name, target_name in key_map.items():
            full = prefix + source_name
            if full not in available:
                # Biases are optional: e.g. Wan2.1-14B has bias-free attention/FFN.
                # The block forward already fetches biases via ``.get(...)``.
                if source_name.endswith(".bias"):
                    continue
                raise KeyError(f"missing required block weight: {full}")
            array = _load_mx_array_from_safetensor(handle, full, dtype)
            loaded = quantize_matrix(array, spec) if target_name in matrix_targets else array
            _eval_loaded_weight(loaded)
            weights[target_name] = loaded
            del array
    return weights


def mlx_dit_from_diffusers_safetensors(
    checkpoint_path: str | Path,
    config_path: str | Path,
    *,
    dtype: str = "fp16",
    num_blocks: int | None = None,
    quantization: str | MLXQuantizationSpec | None = None,
) -> MLXWanDiT:
    import mlx.core as mx
    from safetensors import safe_open

    config = json.loads(Path(config_path).read_text())
    total_blocks = int(config["num_layers"])
    if num_blocks is None:
        num_blocks = total_blocks
    cast_dtype = {"fp16": mx.float16, "bf16": mx.bfloat16, "fp32": mx.float32}[dtype]
    spec = MLXQuantizationSpec.from_name(quantization) if (quantization is None or isinstance(quantization, str)) else quantization
    ensure_quantization_supported(spec)

    top_level_names = [
        "patch_embedding.weight",
        "patch_embedding.bias",
        "condition_embedder.time_embedder.linear_1.weight",
        "condition_embedder.time_embedder.linear_1.bias",
        "condition_embedder.time_embedder.linear_2.weight",
        "condition_embedder.time_embedder.linear_2.bias",
        "condition_embedder.time_proj.weight",
        "condition_embedder.time_proj.bias",
        "condition_embedder.text_embedder.linear_1.weight",
        "condition_embedder.text_embedder.linear_1.bias",
        "condition_embedder.text_embedder.linear_2.weight",
        "condition_embedder.text_embedder.linear_2.bias",
        "scale_shift_table",
        "proj_out.weight",
        "proj_out.bias",
    ]
    weights = {}
    with safe_open(str(checkpoint_path), framework="pt", device="cpu") as handle:
        available = set(handle.keys())
        for name in top_level_names:
            if name not in available:
                if name.endswith(".bias"):
                    continue
                raise KeyError(f"missing required weight: {name}")
            array = _load_mx_array_from_safetensor(handle, name, cast_dtype)
            if name == "patch_embedding.weight":
                array = array.reshape(int(config["num_attention_heads"]) * int(config["attention_head_dim"]), -1)
            if name.endswith(".weight") and name not in {"scale_shift_table"}:
                loaded = quantize_matrix(array, spec)
            else:
                loaded = array
            _eval_loaded_weight(loaded)
            weights[name] = loaded
            del array

    blocks = []
    for block_index in range(num_blocks):
        block_weights = mlx_block_weights_from_diffusers_safetensors(
            checkpoint_path,
            block_index=block_index,
            quantization=spec,
            dtype=cast_dtype,
        )
        block_weights = {
            name: (value if isinstance(value, QuantizedMatrix) else value.astype(cast_dtype))
            for name, value in block_weights.items()
        }
        for value in block_weights.values():
            _eval_loaded_weight(value)
        blocks.append(
            MLXWanTransformerBlock(
                block_weights,
                dim=int(config["num_attention_heads"]) * int(config["attention_head_dim"]),
                ffn_dim=int(config["ffn_dim"]),
                num_heads=int(config["num_attention_heads"]),
                eps=float(config["eps"]),
            )
        )
    return MLXWanDiT(weights, blocks, config)


def torch_block_state_from_diffusers_safetensors(
    checkpoint_path: str | Path,
    *,
    block_index: int = 0,
) -> dict[str, "torch.Tensor"]:
    """Load one Diffusers-format Wan block into FastVideo's dense block keys."""
    from safetensors import safe_open

    prefix = f"blocks.{block_index}."
    key_map = {
        "scale_shift_table": "scale_shift_table",
        "attn1.to_q.weight": "to_q.weight",
        "attn1.to_q.bias": "to_q.bias",
        "attn1.to_k.weight": "to_k.weight",
        "attn1.to_k.bias": "to_k.bias",
        "attn1.to_v.weight": "to_v.weight",
        "attn1.to_v.bias": "to_v.bias",
        "attn1.to_out.0.weight": "to_out.weight",
        "attn1.to_out.0.bias": "to_out.bias",
        "attn1.norm_q.weight": "norm_q.weight",
        "attn1.norm_k.weight": "norm_k.weight",
        "attn2.to_q.weight": "attn2.to_q.weight",
        "attn2.to_q.bias": "attn2.to_q.bias",
        "attn2.to_k.weight": "attn2.to_k.weight",
        "attn2.to_k.bias": "attn2.to_k.bias",
        "attn2.to_v.weight": "attn2.to_v.weight",
        "attn2.to_v.bias": "attn2.to_v.bias",
        "attn2.to_out.0.weight": "attn2.to_out.weight",
        "attn2.to_out.0.bias": "attn2.to_out.bias",
        "attn2.norm_q.weight": "attn2.norm_q.weight",
        "attn2.norm_k.weight": "attn2.norm_k.weight",
        "ffn.net.0.proj.weight": "ffn.fc_in.weight",
        "ffn.net.0.proj.bias": "ffn.fc_in.bias",
        "ffn.net.2.weight": "ffn.fc_out.weight",
        "ffn.net.2.bias": "ffn.fc_out.bias",
        "norm2.weight": "self_attn_residual_norm.norm.weight",
        "norm2.bias": "self_attn_residual_norm.norm.bias",
    }

    state = {}
    with safe_open(str(checkpoint_path), framework="pt", device="cpu") as handle:
        for source_name, target_name in key_map.items():
            state[target_name] = handle.get_tensor(prefix + source_name).float()
    return state
