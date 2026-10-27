# SPDX-License-Identifier: Apache-2.0
# mypy: disable-error-code=no-untyped-call
"""MiniMax-H3 audio VAE (DAC encoder + BigVGAN decoder) for Apple Silicon MLX.

Faithful port of ``fastvideo/models/vaes/minimax_h3_audio.py``:

- **Encoder**: plain-Snake residual units (dilations 1/3/9), strided
  down-sampling blocks, then the attention projection block — causal
  multi-head attention over the latent stream with q/v biases and a forced
  zero k bias, head averaging, average pooling into ``latent_channels``
  streams, output projection, and a GeGLU MLP.
- **Decoder**: 1x1 projection back to the trunk width, BigVGAN with
  weight-normalized ConvTranspose1d upsampling, AMP residual blocks with
  SnakeBeta behind alias-free Kaiser-window up/down sampling, final SnakeBeta
  + conv, and the [-1, 1] clamp.
- Kaiser-window low-pass filters exactly as released; released filter buffers
  are used when present so numerics do not depend on window construction.

Waveforms are float32 in [-1, 1] at 32 kHz. Stereo latents ``(2, 32, N)``
decode independently while preserving channel order and duration. Production
code never imports PyTorch; torch parity references live only in tests.
"""

from __future__ import annotations

import gc
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np

from fastvideo.logger import init_logger

logger = init_logger(__name__)


def _ct(x, *order):
    """Contiguous transpose: some MLX builds silently mis-compute ops on
    strided views, so every layout change materializes."""
    return mx.contiguous(x.transpose(*order))


@dataclass(frozen=True)
class MiniMaxH3AudioVAEConfigView:
    """Architecture constants (defaults mirror the released audio_vae/config.json)."""

    encoder_dim: int = 64
    encoder_rates: tuple[int, ...] = (2, 4, 4, 5, 5)
    latent_dim: int = 2048
    latent_channels: int = 32
    num_attention_heads: int = 8
    decoder_dim: int = 1024
    decoder_rates: tuple[int, ...] = (5, 5, 2, 2, 2, 2, 2)
    decoder_kernel_sizes: tuple[int, ...] = (9, 9, 4, 4, 4, 4, 4)
    resblock_kernel_sizes: tuple[int, ...] = (3, 7, 11)
    resblock_dilation_sizes: tuple[tuple[int, ...], ...] = ((1, 3, 5), (1, 3, 5), (1, 3, 5))
    sampling_rate: int = 32000
    latents_mean: tuple[float, ...] | None = None
    latents_std: tuple[float, ...] | None = None

    @classmethod
    def from_vae_dir(cls, vae_dir: str | Path) -> MiniMaxH3AudioVAEConfigView:
        config = json.loads((Path(vae_dir) / "config.json").read_text())
        return cls(
            encoder_dim=int(config["encoder_dim"]),
            encoder_rates=tuple(int(v) for v in config["encoder_rates"]),
            latent_dim=int(config["latent_dim"]),
            latent_channels=int(config["latent_channels"]),
            num_attention_heads=int(config["num_attention_heads"]),
            decoder_dim=int(config["decoder_dim"]),
            decoder_rates=tuple(int(v) for v in config["decoder_rates"]),
            decoder_kernel_sizes=tuple(int(v) for v in config["decoder_kernel_sizes"]),
            resblock_kernel_sizes=tuple(int(v) for v in config["resblock_kernel_sizes"]),
            resblock_dilation_sizes=tuple(tuple(int(d) for d in group) for group in config["resblock_dilation_sizes"]),
            sampling_rate=int(config["sampling_rate"]),
            latents_mean=tuple(float(v) for v in config["latents_mean"]) if "latents_mean" in config else None,
            latents_std=tuple(float(v) for v in config["latents_std"]) if "latents_std" in config else None,
        )

    @property
    def hop_length(self) -> int:
        return math.prod(self.encoder_rates)


ACTIVATION_RATIO = 2

# ---------------------------------------------------------------------------
# Primitives (torch layout (B, C, L); convs run channels-last internally)
# ---------------------------------------------------------------------------


def _wn_weight(weight_v, weight_g):
    """torch weight_norm (dim=0): w = v * g / per-out-channel ||v||."""
    norm = mx.sqrt(mx.sum(weight_v * weight_v, axis=(1, 2), keepdims=True))
    return weight_v * (weight_g / norm)


def _conv1d(x, weight, bias=None, *, stride: int = 1, padding: int = 0, dilation: int = 1, groups: int = 1):
    """(B, C, L) in/out around channels-last MLX conv1d. weight: (O, I, K)."""
    # mx.contiguous guards against silent mis-computation on strided views.
    y = mx.conv1d(_ct(x, 0, 2, 1),
                  _ct(weight, 0, 2, 1),
                  stride=stride,
                  padding=padding,
                  dilation=dilation,
                  groups=groups)
    if bias is not None:
        y = y + bias
    return _ct(y, 0, 2, 1)


def _conv_transpose1d(x, weight_torch_layout, bias=None, *, stride: int = 1, padding: int = 0):
    """(B, C, L) in/out. weight in torch layout (I, O, K) -> MLX (O, K, I)."""
    y = mx.conv_transpose1d(_ct(x, 0, 2, 1), _ct(weight_torch_layout, 1, 2, 0), stride=stride, padding=padding)
    if bias is not None:
        y = y + bias
    return _ct(y, 0, 2, 1)


def _replicate_pad(x, left: int, right: int):
    """Replicate padding on the last axis of (B, C, L)."""
    if left == 0 and right == 0:
        return x
    pieces = []
    if left > 0:
        pieces.append(mx.repeat(x[:, :, :1], left, axis=-1))
    pieces.append(x)
    if right > 0:
        pieces.append(mx.repeat(x[:, :, -1:], right, axis=-1))
    return mx.concatenate(pieces, axis=-1)


def _gelu_tanh(x):
    return 0.5 * x * (1.0 + mx.tanh(0.7978845608028654 * (x + 0.044715 * x * x * x)))


def _layer_norm(x, weight, bias, eps: float = 1e-5):
    mean = x.mean(axis=-1, keepdims=True)
    var = ((x - mean)**2).mean(axis=-1, keepdims=True)
    return (x - mean) / mx.sqrt(var + eps) * weight + bias


def _linear(x, weight, bias=None):
    y = x @ weight.T
    if bias is not None:
        y = y + bias
    return y


def kaiser_sinc_filter1d(cutoff: float, half_width: float, kernel_size: int) -> np.ndarray:
    """NumPy replica of the released Kaiser-windowed sinc low-pass."""
    half_size = kernel_size // 2
    attenuation = 2.285 * (half_size - 1) * math.pi * (4 * half_width) + 7.95
    if attenuation > 50.0:
        beta = 0.1102 * (attenuation - 8.7)
    elif attenuation >= 21.0:
        beta = 0.5842 * (attenuation - 21)**0.4 + 0.07886 * (attenuation - 21.0)
    else:
        beta = 0.0
    window = np.kaiser(kernel_size, beta).astype(np.float32)  # periodic=False semantics
    if kernel_size % 2 == 0:
        time = np.arange(-half_size, half_size, dtype=np.float32) + 0.5
    else:
        time = np.arange(kernel_size, dtype=np.float32) - half_size
    filter_ = 2 * cutoff * window * np.sinc(2 * cutoff * time)
    filter_ /= filter_.sum()
    return filter_.astype(np.float32)


def _snake(x, alpha):
    """MiniMaxH3AudioSnake1d: x + sin(alpha*x)^2 / (alpha + 1e-9)."""
    return x + mx.sin(alpha * x)**2 / (alpha + 1e-9)


def _snake_beta(x, alpha, beta):
    """SnakeBeta: x + sin(exp(alpha)*x)^2 / (exp(beta) + 1e-9)."""
    a = mx.exp(mx.reshape(alpha, (-1, )))[None, :, None]
    b = mx.exp(mx.reshape(beta, (-1, )))[None, :, None]
    return x + mx.sin(a * x)**2 / (b + 1e-9)


def _low_pass(x, filter_, stride: int):
    """Grouped low-pass with replicate padding. x: (B, C, L); filter: (K,)."""
    channels = x.shape[1]
    kernel_size = filter_.shape[0]
    even = kernel_size % 2 == 0
    pad_left = kernel_size // 2 - int(even)
    pad_right = kernel_size // 2
    padded = _replicate_pad(x, pad_left, pad_right)
    weight = mx.repeat(filter_[None, :, None], channels, axis=0)  # depthwise (C, K, 1), MLX layout
    # Depthwise weights are already in MLX layout; do not route through _conv1d
    # (it would re-transpose them as torch-layout (O, I, K)).
    y = mx.conv1d(_ct(padded, 0, 2, 1), weight, stride=stride, groups=channels)
    return _ct(y, 0, 2, 1)


def _up_sample(x, filter_, ratio: int, kernel_size: int):
    """Kaiser-windowed up-sampling with the official cropping. x: (B, C, L).

    Implemented as zero-stuffing plus depthwise conv1d with the flipped
    filter: identical to ``F.conv_transpose1d`` for symmetric filters, and it
    avoids grouped conv_transpose whose weight layout differs across MLX
    builds.
    """
    pad = kernel_size // ratio - 1
    pad_left = pad * ratio + (kernel_size - ratio) // 2
    pad_right = pad * ratio + (kernel_size - ratio + 1) // 2
    batch, channels, length = x.shape
    padded = _replicate_pad(x, pad, pad)
    stuffed_len = padded.shape[-1] * ratio
    parts = [mx.expand_dims(padded, -1)] + [mx.zeros((batch, channels, padded.shape[-1], 1))] * (ratio - 1)
    stuffed = mx.reshape(mx.concatenate(parts, axis=-1), (batch, channels, stuffed_len))
    # Correlation alignment: zero-pad by (K-1) on both sides of the stuffed signal.
    stuffed = _ct(
        mx.concatenate([
            mx.zeros((batch, channels, kernel_size - 1)),
            stuffed,
            mx.zeros((batch, channels, kernel_size - 1)),
        ],
                       axis=-1), 0, 2, 1)
    weight = mx.repeat(mx.reshape(filter_[::-1], (1, -1, 1)), channels, axis=0)  # depthwise (C, K, 1)
    y = _ct(mx.conv1d(stuffed, weight, stride=1, groups=channels), 0, 2, 1)
    out_len = (padded.shape[-1] - 1) * ratio + kernel_size
    full = ratio * y[..., :out_len]
    end = full.shape[-1] - pad_right if pad_right > 0 else full.shape[-1]
    return full[..., pad_left:end]


def _activation1d(x, alpha, beta, up_filter, down_filter, ratio: int = ACTIVATION_RATIO):
    """Alias-free SnakeBeta activation: up-sample -> SnakeBeta -> low-pass."""
    x = _up_sample(x, up_filter, ratio, up_filter.shape[0])
    x = _snake_beta(x, alpha, beta)
    return _low_pass(x, down_filter, ratio)


# ---------------------------------------------------------------------------
# The MLX H3 audio VAE
# ---------------------------------------------------------------------------


class MLXMiniMaxH3AudioVAE:
    """DAC-style posterior encoder plus BigVGAN waveform decoder."""

    def __init__(self, weights: dict[str, Any], config: MiniMaxH3AudioVAEConfigView, *, include_encoder: bool = True):
        self.weights = weights
        self.config = config
        self.has_encoder = include_encoder
        self.latent_channels = config.latent_channels
        mean = config.latents_mean if config.latents_mean is not None else [0.0] * config.latent_channels
        std = config.latents_std if config.latents_std is not None else [1.0] * config.latent_channels
        self._latents_mean = np.asarray(mean, dtype=np.float32).reshape(1, -1, 1)
        self._latents_std = np.asarray(std, dtype=np.float32).reshape(1, -1, 1)

    # -- normalization --------------------------------------------------

    def normalize_latents(self, latents):
        return (latents - mx.array(self._latents_mean)) / mx.array(self._latents_std)

    def denormalize_latents(self, latents):
        return latents * mx.array(self._latents_std) + mx.array(self._latents_mean)

    def _filter(self, key: str) -> mx.array:
        """Released filter buffer flattened to (K,) (they ship as (1, 1, K))."""
        try:
            return mx.reshape(self.weights[key], (-1, ))
        except KeyError as exc:
            raise KeyError(f"H3 audio VAE is missing released filter buffer '{key}'") from exc

    # -- encoder ----------------------------------------------------------

    def encode(self, waveform):
        """Mono waveform (1, 1, S) -> posterior mean/logvar each (1, C, N)."""
        if not self.has_encoder:
            raise RuntimeError("This MLX H3 audio VAE was loaded without encoder weights.")
        cfg = self.config
        w = self.weights
        samples = waveform.shape[-1]
        right_pad = math.ceil(samples / cfg.hop_length) * cfg.hop_length - samples
        if right_pad > 0:
            waveform = mx.pad(waveform, ((0, 0), (0, 0), (0, right_pad)))

        x = _conv1d(waveform,
                    _wn_weight(w["encoder.block.0.weight_v"], w["encoder.block.0.weight_g"]),
                    w["encoder.block.0.bias"],
                    padding=3)
        dim = cfg.encoder_dim
        for index, stride in enumerate(cfg.encoder_rates, start=1):
            prefix = f"encoder.block.{index}"
            dim *= 2
            # Three residual units with dilations 1, 3, 9 at dim//2 channels.
            for unit, dilation in enumerate((1, 3, 9)):
                p = f"{prefix}.block.{unit}"
                residual = _snake(x, w[f"{p}.block.0.alpha"])
                residual = _conv1d(residual,
                                   _wn_weight(w[f"{p}.block.1.weight_v"], w[f"{p}.block.1.weight_g"]),
                                   w[f"{p}.block.1.bias"],
                                   padding=((7 - 1) * dilation) // 2,
                                   dilation=dilation)
                residual = _snake(residual, w[f"{p}.block.2.alpha"])
                residual = _conv1d(residual, _wn_weight(w[f"{p}.block.3.weight_v"], w[f"{p}.block.3.weight_g"]),
                                   w[f"{p}.block.3.bias"])
                pad = (x.shape[-1] - residual.shape[-1]) // 2
                if pad > 0:
                    x = x[..., pad:-pad]
                x = x + residual
            x = _snake(x, w[f"{prefix}.block.3.alpha"])
            x = _conv1d(x,
                        _wn_weight(w[f"{prefix}.block.4.weight_v"], w[f"{prefix}.block.4.weight_g"]),
                        w[f"{prefix}.block.4.bias"],
                        stride=stride,
                        padding=math.ceil(stride / 2))
        snake_index = len(cfg.encoder_rates) + 1
        conv_index = len(cfg.encoder_rates) + 2
        x = _snake(x, w[f"encoder.block.{snake_index}.alpha"])
        hidden = _conv1d(x,
                         _wn_weight(w[f"encoder.block.{conv_index}.weight_v"],
                                    w[f"encoder.block.{conv_index}.weight_g"]),
                         w[f"encoder.block.{conv_index}.bias"],
                         padding=1)

        projected = _ct(self._attention_projection(_ct(hidden, 0, 2, 1)), 0, 2, 1)
        mean = _conv1d(projected, w["mean_proj.weight"], w["mean_proj.bias"])
        logvar = _conv1d(projected, w["logs_proj.weight"], w["logs_proj.bias"])
        return mean, logvar

    def _attention_projection(self, hidden):
        """MiniMaxH3AudioAttnProjection over (B, S, latent_dim)."""
        cfg = self.config
        w = self.weights
        num_heads = cfg.num_attention_heads
        head_dim = hidden.shape[-1] // num_heads
        batch, seq_len, _ = hidden.shape

        normed = _layer_norm(hidden, w["pre_block.norm1.weight"], w["pre_block.norm1.bias"])
        bias = mx.concatenate([w["pre_block.attn.q_bias"], w["pre_block.attn.zero_k_bias"], w["pre_block.attn.v_bias"]])
        qkv = _linear(normed, w["pre_block.attn.qkv.weight"], bias)
        qkv = qkv.reshape(batch, seq_len, 3, num_heads, head_dim)
        query = _ct(qkv[:, :, 0], 0, 2, 1, 3)  # (B, H, S, Dh)
        key = _ct(qkv[:, :, 1], 0, 2, 1, 3)
        value = _ct(qkv[:, :, 2], 0, 2, 1, 3)

        scores = (query @ _ct(key, 0, 1, 3, 2)) * head_dim**-0.5
        causal_mask = mx.triu(mx.ones((seq_len, seq_len), dtype=mx.bool_), k=1)
        scores = mx.where(causal_mask[None, None], mx.array(-np.inf, dtype=scores.dtype), scores)
        attn_weights = mx.softmax(scores, axis=-1)
        attended = attn_weights @ value  # (B, H, S, Dh)
        attended = _ct(attended, 0, 2, 1, 3).mean(axis=2)  # head mean -> (B, S, Dh)

        out_dim = cfg.latent_channels
        pool_factor = attended.shape[-1] // out_dim
        if attended.shape[-1] % out_dim != 0:
            raise ValueError("Attention head dimension must pool evenly into latent channels.")
        pooled = attended.reshape(batch, seq_len, out_dim, pool_factor).mean(axis=-1)
        attn_out = _linear(pooled, w["pre_block.attn.proj.weight"], w["pre_block.attn.proj.bias"])

        projected = _linear(_layer_norm(hidden, w["pre_block.norm3.weight"], w["pre_block.norm3.bias"]),
                            w["pre_block.proj.weight"], w["pre_block.proj.bias"]) + attn_out
        mlp_in = _layer_norm(projected, w["pre_block.norm2.weight"], w["pre_block.norm2.bias"])
        # GeGluMlp carries its own inner LayerNorm before w0/w1.
        mlp_normed = _layer_norm(mlp_in, w["pre_block.mlp.norm.weight"], w["pre_block.mlp.norm.bias"])
        gate = _gelu_tanh(_linear(mlp_normed, w["pre_block.mlp.w0.weight"], w["pre_block.mlp.w0.bias"]))
        value_mlp = _linear(mlp_normed, w["pre_block.mlp.w1.weight"], w["pre_block.mlp.w1.bias"])
        mlp_out = _linear(gate * value_mlp, w["pre_block.mlp.w2.weight"], w["pre_block.mlp.w2.bias"])
        return projected + mlp_out

    # -- decoder ----------------------------------------------------------

    def decode(self, latents):
        """Latents (B, 32, N) -> waveforms (B, 1, S) clamped to [-1, 1]."""
        cfg = self.config
        w = self.weights
        hidden = _conv1d(latents, w["dec_in_proj.weight"], w["dec_in_proj.bias"])
        hidden = _conv1d(hidden,
                         _wn_weight(w["decoder.conv_pre.weight_v"], w["decoder.conv_pre.weight_g"]),
                         w["decoder.conv_pre.bias"],
                         padding=3)
        num_resblocks = len(cfg.resblock_kernel_sizes)
        for index, (rate, kernel) in enumerate(zip(cfg.decoder_rates, cfg.decoder_kernel_sizes, strict=False)):
            weight = _wn_weight(w[f"decoder.ups.{index}.0.weight_v"], w[f"decoder.ups.{index}.0.weight_g"])
            hidden = _conv_transpose1d(hidden,
                                       weight,
                                       w[f"decoder.ups.{index}.0.bias"],
                                       stride=rate,
                                       padding=(kernel - rate) // 2)
            residual_sum = None
            for j in range(num_resblocks):
                out = self._amp_block(hidden, f"decoder.resblocks.{index * num_resblocks + j}",
                                      cfg.resblock_kernel_sizes[j], cfg.resblock_dilation_sizes[j])
                residual_sum = out if residual_sum is None else residual_sum + out
            if residual_sum is None:
                raise RuntimeError("H3 audio VAE decoder has no residual blocks.")
            hidden = residual_sum / num_resblocks
        alpha = w["decoder.activation_post.act.alpha"]
        beta = w["decoder.activation_post.act.beta"]
        up_filter = self._filter("decoder.activation_post.upsample.filter")
        down_filter = self._filter("decoder.activation_post.downsample.lowpass.filter")
        hidden = _activation1d(hidden, alpha, beta, up_filter, down_filter)
        hidden = _conv1d(hidden,
                         _wn_weight(w["decoder.conv_post.weight_v"], w["decoder.conv_post.weight_g"]),
                         w.get("decoder.conv_post.bias"),
                         padding=3)
        return mx.clip(hidden, -1.0, 1.0)

    def _amp_block(self, x, prefix: str, kernel_size: int, dilations: tuple[int, ...]):
        """MiniMaxH3AudioAMPBlock: per dilation, act -> conv1 -> act -> conv2 + skip."""
        w = self.weights
        for j, dilation in enumerate(dilations):
            pad1 = (kernel_size * dilation - dilation) // 2
            pad2 = (kernel_size - 1) // 2
            up1 = self._filter(f"{prefix}.activations.{2 * j}.upsample.filter")
            down1 = self._filter(f"{prefix}.activations.{2 * j}.downsample.lowpass.filter")
            residual = _activation1d(x, w[f"{prefix}.activations.{2 * j}.act.alpha"],
                                     w[f"{prefix}.activations.{2 * j}.act.beta"], up1, down1)
            residual = _conv1d(residual,
                               _wn_weight(w[f"{prefix}.convs1.{j}.weight_v"], w[f"{prefix}.convs1.{j}.weight_g"]),
                               w[f"{prefix}.convs1.{j}.bias"],
                               padding=pad1,
                               dilation=dilation)
            up2 = self._filter(f"{prefix}.activations.{2 * j + 1}.upsample.filter")
            down2 = self._filter(f"{prefix}.activations.{2 * j + 1}.downsample.lowpass.filter")
            residual = _activation1d(residual, w[f"{prefix}.activations.{2 * j + 1}.act.alpha"],
                                     w[f"{prefix}.activations.{2 * j + 1}.act.beta"], up2, down2)
            residual = _conv1d(residual,
                               _wn_weight(w[f"{prefix}.convs2.{j}.weight_v"], w[f"{prefix}.convs2.{j}.weight_g"]),
                               w[f"{prefix}.convs2.{j}.bias"],
                               padding=pad2)
            x = residual + x
        return x


def mlx_h3_audio_vae_from_file(weights_path: str | Path,
                               *,
                               include_encoder: bool = True,
                               config: MiniMaxH3AudioVAEConfigView | None = None,
                               component_dir: str | Path | None = None) -> MLXMiniMaxH3AudioVAE:
    """Load the released audio VAE from its single safetensors file.

    The released checkpoint is ~605 MB fp32; it is read whole (bounded) and
    kept at release precision.
    """
    weights_path = Path(weights_path)
    if config is None:
        if component_dir is None:
            component_dir = weights_path.parent
        config = MiniMaxH3AudioVAEConfigView.from_vae_dir(component_dir)
    arrays = mx.load(str(weights_path))
    wanted_prefixes: tuple[str, ...] = ("dec_in_proj.", "decoder.", "mean_proj.", "logs_proj.")
    if include_encoder:
        wanted_prefixes += ("encoder.", "pre_block.")
    weights: dict[str, Any] = {}
    for key in arrays:
        if key.startswith(wanted_prefixes):
            weights[key] = arrays[key]
    del arrays
    gc.collect()
    mx.clear_cache()
    required = ["dec_in_proj.weight", "decoder.conv_pre.weight_v", "decoder.conv_post.weight_v"]
    if include_encoder:
        required.extend(
            ("mean_proj.weight", "logs_proj.weight", "pre_block.attn.qkv.weight", "encoder.block.0.weight_v"))
    missing = [key for key in required if key not in weights]
    if missing:
        raise KeyError(f"H3 audio VAE is missing required tensors: {missing}")
    return MLXMiniMaxH3AudioVAE(weights, config, include_encoder=include_encoder)


def mlx_h3_audio_vae_from_dir(component_dir: str | Path,
                              *,
                              include_encoder: bool = True,
                              storage_dtype: str = "fp32") -> MLXMiniMaxH3AudioVAE:
    """Load from the released component directory (audio_vae/)."""
    if storage_dtype != "fp32":
        raise ValueError(f"H3 audio VAE numerics require storage_dtype='fp32', got {storage_dtype!r}.")

    component_dir = Path(component_dir)
    single = component_dir / "diffusion_pytorch_model.safetensors"
    if not single.exists():
        raise FileNotFoundError(f"No audio VAE safetensors under {component_dir}")
    return mlx_h3_audio_vae_from_file(single, include_encoder=include_encoder, component_dir=component_dir)
