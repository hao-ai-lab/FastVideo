# SPDX-License-Identifier: Apache-2.0
"""
LTX-2 transformer implementation
"""

from dataclasses import dataclass, replace
from enum import Enum
import functools
import math
import os
from pathlib import Path
from typing import Any, Optional, Tuple, Callable

import numpy as np

import torch
import torch.nn as nn
from einops import rearrange, repeat

from fastvideo.attention.backends.sdpa import SDPAMetadata
from fastvideo.attention.layer import LocalAttention
from fastvideo.configs.models.dits import LTX2VideoConfig
from fastvideo.forward_context import get_forward_context, set_forward_context
from fastvideo.models.dits.base import CachableDiT
from fastvideo.platforms import AttentionBackendEnum


def get_timestep_embedding(
    timesteps: torch.Tensor,
    embedding_dim: int,
    flip_sin_to_cos: bool = False,
    downscale_freq_shift: float = 1,
    scale: float = 1,
    max_period: int = 10000,
) -> torch.Tensor:
    """Sinusoidal timestep embedding used by LTX-2 AdaLN."""
    if len(timesteps.shape) != 1:
        raise ValueError("Timesteps should be a 1d-array")

    half_dim = embedding_dim // 2
    exponent = -math.log(max_period) * torch.arange(start=0, end=half_dim, dtype=torch.float32, device=timesteps.device)
    exponent = exponent / (half_dim - downscale_freq_shift)

    emb = torch.exp(exponent)
    emb = timesteps[:, None].float() * emb[None, :]
    emb = scale * emb
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)

    if flip_sin_to_cos:
        emb = torch.cat([emb[:, half_dim:], emb[:, :half_dim]], dim=-1)

    if embedding_dim % 2 == 1:
        emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
    return emb


class TimestepEmbedding(torch.nn.Module):
    """Two-layer MLP to project timestep embeddings."""
    def __init__(
        self,
        in_channels: int,
        time_embed_dim: int,
        out_dim: int | None = None,
        post_act_fn: str | None = None,
        cond_proj_dim: int | None = None,
        sample_proj_bias: bool = True,
    ):
        super().__init__()
        self.linear_1 = torch.nn.Linear(in_channels, time_embed_dim, sample_proj_bias)
        self.cond_proj = torch.nn.Linear(cond_proj_dim, in_channels, bias=False) if cond_proj_dim is not None else None
        self.act = torch.nn.SiLU()
        time_embed_dim_out = out_dim if out_dim is not None else time_embed_dim
        self.linear_2 = torch.nn.Linear(time_embed_dim, time_embed_dim_out, sample_proj_bias)
        self.post_act = None if post_act_fn is None else None

    def forward(self, sample: torch.Tensor, condition: torch.Tensor | None = None) -> torch.Tensor:
        if condition is not None:
            sample = sample + self.cond_proj(condition)
        sample = self.linear_1(sample)
        if self.act is not None:
            sample = self.act(sample)
        sample = self.linear_2(sample)
        if self.post_act is not None:
            sample = self.post_act(sample)
        return sample


class Timesteps(torch.nn.Module):
    """Sinusoidal timestep embedding wrapper with scaling knobs."""
    def __init__(self, num_channels: int, flip_sin_to_cos: bool, downscale_freq_shift: float, scale: int = 1):
        super().__init__()
        self.num_channels = num_channels
        self.flip_sin_to_cos = flip_sin_to_cos
        self.downscale_freq_shift = downscale_freq_shift
        self.scale = scale

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        return get_timestep_embedding(
            timesteps,
            self.num_channels,
            flip_sin_to_cos=self.flip_sin_to_cos,
            downscale_freq_shift=self.downscale_freq_shift,
            scale=self.scale,
        )


class PixArtAlphaCombinedTimestepSizeEmbeddings(torch.nn.Module):
    """PixArt-Alpha timestep embedding used by LTX-2 AdaLN."""
    def __init__(self, embedding_dim: int, size_emb_dim: int):
        super().__init__()
        self.outdim = size_emb_dim
        self.time_proj = Timesteps(num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=0)
        self.timestep_embedder = TimestepEmbedding(in_channels=256, time_embed_dim=embedding_dim)

    def forward(self, timestep: torch.Tensor, hidden_dtype: torch.dtype) -> torch.Tensor:
        timesteps_proj = self.time_proj(timestep)
        timesteps_emb = self.timestep_embedder(timesteps_proj.to(dtype=hidden_dtype))
        return timesteps_emb


class AdaLayerNormSingle(torch.nn.Module):
    """AdaLN-single modulation that emits scale/shift/gates for LTX-2."""
    def __init__(self, embedding_dim: int, embedding_coefficient: int = 6):
        super().__init__()
        self.emb = PixArtAlphaCombinedTimestepSizeEmbeddings(
            embedding_dim,
            size_emb_dim=embedding_dim // 3,
        )
        self.silu = torch.nn.SiLU()
        self.linear = torch.nn.Linear(embedding_dim, embedding_coefficient * embedding_dim, bias=True)

    def forward(
        self,
        timestep: torch.Tensor,
        hidden_dtype: Optional[torch.dtype] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        embedded_timestep = self.emb(timestep, hidden_dtype=hidden_dtype)
        return self.linear(self.silu(embedded_timestep)), embedded_timestep


class PixArtAlphaTextProjection(torch.nn.Module):
    """Caption projection MLP used by LTX-2."""
    def __init__(self, in_features: int, hidden_size: int, out_features: int | None = None, act_fn: str = "gelu_tanh"):
        super().__init__()
        if out_features is None:
            out_features = hidden_size
        self.linear_1 = torch.nn.Linear(in_features=in_features, out_features=hidden_size, bias=True)
        if act_fn == "gelu_tanh":
            self.act_1 = torch.nn.GELU(approximate="tanh")
        elif act_fn == "silu":
            self.act_1 = torch.nn.SiLU()
        else:
            raise ValueError(f"Unknown activation function: {act_fn}")
        self.linear_2 = torch.nn.Linear(in_features=hidden_size, out_features=out_features, bias=True)

    def forward(self, caption: torch.Tensor) -> torch.Tensor:
        hidden_states = self.linear_1(caption)
        hidden_states = self.act_1(hidden_states)
        hidden_states = self.linear_2(hidden_states)
        return hidden_states


class GELUApprox(nn.Module):
    """Linear + tanh-approximate GELU used by LTX-2 FFN."""
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.proj = nn.Linear(in_features, out_features)
        self.act = nn.GELU(approximate="tanh")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.proj(x))


class FeedForward(nn.Module):
    """LTX-2 FFN: GELUApprox -> Identity -> Linear."""
    def __init__(self, dim: int, dim_out: int, mult: int = 4) -> None:
        super().__init__()
        inner_dim = int(dim * mult)
        project_in = GELUApprox(dim, inner_dim)
        self.net = nn.Sequential(project_in, nn.Identity(), nn.Linear(inner_dim, dim_out))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class VideoLatentShape(tuple):
    """Helper for (B, C, T, H, W) latent shapes."""
    @property
    def batch(self) -> int:
        return self[0]

    @property
    def channels(self) -> int:
        return self[1]

    @property
    def frames(self) -> int:
        return self[2]

    @property
    def height(self) -> int:
        return self[3]

    @property
    def width(self) -> int:
        return self[4]

    @staticmethod
    def from_torch_shape(shape: torch.Size) -> "VideoLatentShape":
        return VideoLatentShape(shape)

    def to_torch_shape(self) -> torch.Size:
        return torch.Size(self)


class AudioLatentShape(tuple):
    """Helper for (B, C, T, F) audio latent shapes."""

    @property
    def batch(self) -> int:
        return self[0]

    @property
    def channels(self) -> int:
        return self[1]

    @property
    def frames(self) -> int:
        return self[2]

    @property
    def mel_bins(self) -> int:
        return self[3]

    def to_torch_shape(self) -> torch.Size:
        return torch.Size(self)

    @staticmethod
    def from_torch_shape(shape: torch.Size) -> "AudioLatentShape":
        return AudioLatentShape(shape)

    @staticmethod
    def from_duration(
        batch: int,
        duration: float,
        channels: int,
        mel_bins: int,
        sample_rate: int,
        hop_length: int,
        audio_latent_downsample_factor: int,
    ) -> "AudioLatentShape":
        latents_per_second = float(sample_rate) / float(
            hop_length) / float(audio_latent_downsample_factor)
        return AudioLatentShape(
            (batch, channels, round(duration * latents_per_second), mel_bins))


class VideoLatentPatchifier:
    """Patchify/unpatchify latent tokens for LTX-2."""
    def __init__(self, patch_size: int):
        self._patch_size = (1, patch_size, patch_size)

    @property
    def patch_size(self) -> Tuple[int, int, int]:
        return self._patch_size

    def get_token_count(self, tgt_shape: VideoLatentShape) -> int:
        return math.prod(tgt_shape.to_torch_shape()[2:]) // math.prod(self._patch_size)

    def patchify(self, latents: torch.Tensor) -> torch.Tensor:
        return rearrange(
            latents,
            "b c (f p1) (h p2) (w p3) -> b (f h w) (c p1 p2 p3)",
            p1=self._patch_size[0],
            p2=self._patch_size[1],
            p3=self._patch_size[2],
        )

    def unpatchify(self, latents: torch.Tensor, output_shape: VideoLatentShape) -> torch.Tensor:
        patch_grid_frames = output_shape.frames // self._patch_size[0]
        patch_grid_height = output_shape.height // self._patch_size[1]
        patch_grid_width = output_shape.width // self._patch_size[2]
        return rearrange(
            latents,
            "b (f h w) (c p q) -> b c f (h p) (w q)",
            f=patch_grid_frames,
            h=patch_grid_height,
            w=patch_grid_width,
            p=self._patch_size[1],
            q=self._patch_size[2],
        )

    def get_patch_grid_bounds(
        self,
        output_shape: VideoLatentShape,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        frames = output_shape.frames
        height = output_shape.height
        width = output_shape.width
        batch_size = output_shape.batch

        grid_coords = torch.meshgrid(
            torch.arange(start=0, end=frames, step=self._patch_size[0], device=device),
            torch.arange(start=0, end=height, step=self._patch_size[1], device=device),
            torch.arange(start=0, end=width, step=self._patch_size[2], device=device),
            indexing="ij",
        )

        patch_starts = torch.stack(grid_coords, dim=0)
        patch_size_delta = torch.tensor(
            self._patch_size,
            device=patch_starts.device,
            dtype=patch_starts.dtype,
        ).view(3, 1, 1, 1)
        patch_ends = patch_starts + patch_size_delta
        latent_coords = torch.stack((patch_starts, patch_ends), dim=-1)
        latent_coords = repeat(
            latent_coords,
            "c f h w bounds -> b c (f h w) bounds",
            b=batch_size,
            bounds=2,
        )
        return latent_coords


class AudioLatentPatchifier:
    """Patchify/unpatchify audio latents and compute timing bounds."""

    def __init__(
        self,
        patch_size: int,
        sample_rate: int,
        hop_length: int,
        audio_latent_downsample_factor: int,
        is_causal: bool = True,
        shift: int = 0,
    ) -> None:
        self.hop_length = hop_length
        self.sample_rate = sample_rate
        self.audio_latent_downsample_factor = audio_latent_downsample_factor
        self.is_causal = is_causal
        self.shift = shift
        self._patch_size = (1, patch_size, patch_size)

    def get_token_count(self, tgt_shape: AudioLatentShape) -> int:
        return tgt_shape.frames

    def patchify(self, latents: torch.Tensor) -> torch.Tensor:
        return rearrange(latents, "b c t f -> b t (c f)")

    def unpatchify(
        self,
        latents: torch.Tensor,
        output_shape: AudioLatentShape,
    ) -> torch.Tensor:
        return rearrange(
            latents,
            "b t (c f) -> b c t f",
            c=output_shape.channels,
            f=output_shape.mel_bins,
        )

    def get_patch_grid_bounds(
        self,
        output_shape: AudioLatentShape,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        start_timings = self._get_audio_latent_time_in_sec(
            self.shift,
            output_shape.frames + self.shift,
            torch.float32,
            device,
        )
        start_timings = start_timings.unsqueeze(0).expand(output_shape.batch,
                                                          -1).unsqueeze(1)

        end_timings = self._get_audio_latent_time_in_sec(
            self.shift + 1,
            output_shape.frames + self.shift + 1,
            torch.float32,
            device,
        )
        end_timings = end_timings.unsqueeze(0).expand(output_shape.batch,
                                                      -1).unsqueeze(1)

        return torch.stack([start_timings, end_timings], dim=-1)

    def _get_audio_latent_time_in_sec(
        self,
        start_latent: int,
        end_latent: int,
        dtype: torch.dtype,
        device: Optional[torch.device],
    ) -> torch.Tensor:
        resolved_device = device or torch.device("cpu")
        audio_latent_frame = torch.arange(
            start_latent, end_latent, dtype=dtype, device=resolved_device)
        audio_mel_frame = audio_latent_frame * self.audio_latent_downsample_factor
        if self.is_causal:
            causal_offset = 1
            audio_mel_frame = (
                audio_mel_frame + causal_offset -
                self.audio_latent_downsample_factor).clamp(min=0)
        return audio_mel_frame * self.hop_length / self.sample_rate


def _get_pixel_coords(
    latent_coords: torch.Tensor,
    scale_factors: tuple[int, int, int],
    fps: float | None,
    causal_fix: bool = True,
) -> torch.Tensor:
    broadcast_shape = [1] * latent_coords.ndim
    broadcast_shape[1] = -1
    scale_tensor = torch.tensor(
        scale_factors,
        device=latent_coords.device,
        dtype=torch.float32,
    ).view(*broadcast_shape)
    pixel_coords = latent_coords.to(torch.float32) * scale_tensor
    if causal_fix:
        pixel_coords[:, 0, ...] = (
            pixel_coords[:, 0, ...] + 1 - scale_factors[0]).clamp(min=0)
    if fps:
        pixel_coords[:, 0, ...] = pixel_coords[:, 0, ...] / fps
    return pixel_coords


def _to_denoised(
    sample: torch.Tensor,
    velocity: torch.Tensor,
    sigma: torch.Tensor,
    calc_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    if isinstance(sigma, torch.Tensor):
        sigma = sigma.to(calc_dtype)
    while sigma.ndim < sample.ndim:
        sigma = sigma.unsqueeze(-1)
    return (sample.to(calc_dtype) - velocity.to(calc_dtype) * sigma).to(sample.dtype)


def _debug_block_log_line(message: str) -> None:
    if os.getenv("LTX2_PIPELINE_DEBUG_LOG", "0") != "1":
        return
    log_path = os.getenv("LTX2_PIPELINE_DEBUG_PATH", "")
    if not log_path:
        return
    log_dir = os.path.dirname(log_path)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(message + "\n")


def _debug_transformer_args(prefix: str, args: "TransformerArgs | None") -> None:
    if os.getenv("LTX2_PIPELINE_DEBUG_LOG", "0") != "1" or args is None:
        return
    pe_cos, pe_sin = args.positional_embeddings
    cross_cos = None
    cross_sin = None
    if args.cross_positional_embeddings is not None:
        cross_cos, cross_sin = args.cross_positional_embeddings
    mask = args.context_mask
    if mask is None:
        mask_summary = "mask=None"
    else:
        finite = torch.isfinite(mask)
        finite_sum = mask[finite].sum().item() if finite.any() else 0.0
        mask_summary = (
            f"mask_min={mask.min().item():.6f} "
            f"mask_max={mask.max().item():.6f} "
            f"mask_finite_sum={finite_sum:.6f} "
            f"mask_finite_count={finite.sum().item()}"
        )
    _debug_block_log_line(
        f"{prefix}:x_sum={args.x.float().sum().item():.6f} "
        f"context_sum={args.context.float().sum().item():.6f} "
        f"t_sum={args.timesteps.float().sum().item():.6f} "
        f"emb_sum={args.embedded_timestep.float().sum().item():.6f} "
        f"pe_cos_sum={pe_cos.float().sum().item():.6f} "
        f"pe_sin_sum={pe_sin.float().sum().item():.6f} "
        f"cross_pe_cos_sum={(cross_cos.float().sum().item() if cross_cos is not None else 0.0):.6f} "
        f"cross_pe_sin_sum={(cross_sin.float().sum().item() if cross_sin is not None else 0.0):.6f} "
        f"{mask_summary}"
    )


class LTXRopeType(Enum):
    """LTX-2 rotary variants (interleaved vs split)."""
    INTERLEAVED = "interleaved"
    SPLIT = "split"


DEFAULT_LTX2_SCALE_FACTORS = (8, 32, 32)
DEFAULT_LTX2_AUDIO_CHANNELS = 8
DEFAULT_LTX2_AUDIO_MEL_BINS = 16
DEFAULT_LTX2_AUDIO_SAMPLE_RATE = 16000
DEFAULT_LTX2_AUDIO_HOP_LENGTH = 160
DEFAULT_LTX2_AUDIO_DOWNSAMPLE = 4
# The vocoder upsamples from mel spectrograms (16kHz sample rate) to 24kHz audio
DEFAULT_LTX2_VOCODER_OUTPUT_SAMPLE_RATE = 24000


def apply_ltx_rotary_emb(
    input_tensor: torch.Tensor,
    freqs_cis: tuple[torch.Tensor, torch.Tensor],
    rope_type: LTXRopeType = LTXRopeType.INTERLEAVED,
) -> torch.Tensor:
    """Apply LTX-2 rotary embeddings to a tensor."""
    if rope_type == LTXRopeType.INTERLEAVED:
        return _apply_ltx_interleaved_rotary_emb(input_tensor, *freqs_cis)
    if rope_type == LTXRopeType.SPLIT:
        return _apply_ltx_split_rotary_emb(input_tensor, *freqs_cis)
    raise ValueError(f"Invalid rope type: {rope_type}")


def _apply_ltx_interleaved_rotary_emb(
    input_tensor: torch.Tensor, cos_freqs: torch.Tensor, sin_freqs: torch.Tensor
) -> torch.Tensor:
    t_dup = rearrange(input_tensor, "... (d r) -> ... d r", r=2)
    t1, t2 = t_dup.unbind(dim=-1)
    t_dup = torch.stack((-t2, t1), dim=-1)
    input_tensor_rot = rearrange(t_dup, "... d r -> ... (d r)")
    return input_tensor * cos_freqs + input_tensor_rot * sin_freqs


def _apply_ltx_split_rotary_emb(
    input_tensor: torch.Tensor, cos_freqs: torch.Tensor, sin_freqs: torch.Tensor
) -> torch.Tensor:
    needs_reshape = False
    if input_tensor.ndim != 4 and cos_freqs.ndim == 4:
        b, h, t, _ = cos_freqs.shape
        input_tensor = input_tensor.reshape(b, t, h, -1).swapaxes(1, 2)
        needs_reshape = True

    split_input = rearrange(input_tensor, "... (d r) -> ... d r", d=2)
    first_half_input = split_input[..., :1, :]
    second_half_input = split_input[..., 1:, :]

    output = split_input * cos_freqs.unsqueeze(-2)
    first_half_output = output[..., :1, :]
    second_half_output = output[..., 1:, :]

    first_half_output.addcmul_(-sin_freqs.unsqueeze(-2), second_half_input)
    second_half_output.addcmul_(sin_freqs.unsqueeze(-2), first_half_input)

    output = rearrange(output, "... d r -> ... (d r)")
    if needs_reshape:
        output = output.swapaxes(1, 2).reshape(b, t, -1)
    return output


@functools.lru_cache(maxsize=5)
def generate_ltx_freq_grid_np(
    positional_embedding_theta: float, positional_embedding_max_pos_count: int, inner_dim: int
) -> torch.Tensor:
    """Generate LTX-2 rotary frequencies with high-precision numpy."""
    theta = positional_embedding_theta
    start = 1
    end = theta
    n_elem = 2 * positional_embedding_max_pos_count
    pow_indices = np.power(
        theta,
        np.linspace(
            np.log(start) / np.log(theta),
            np.log(end) / np.log(theta),
            inner_dim // n_elem,
            dtype=np.float64,
        ),
    )
    return torch.tensor(pow_indices * math.pi / 2, dtype=torch.float32)


@functools.lru_cache(maxsize=5)
def generate_ltx_freq_grid_pytorch(
    positional_embedding_theta: float, positional_embedding_max_pos_count: int, inner_dim: int
) -> torch.Tensor:
    """Generate LTX-2 rotary frequencies in torch for speed."""
    theta = positional_embedding_theta
    start = 1
    end = theta
    n_elem = 2 * positional_embedding_max_pos_count
    indices = theta ** (
        torch.linspace(
            math.log(start, theta),
            math.log(end, theta),
            inner_dim // n_elem,
            dtype=torch.float32,
        )
    )
    indices = indices.to(dtype=torch.float32)
    return indices * math.pi / 2


def _ltx_get_fractional_positions(indices_grid: torch.Tensor, max_pos: list[int]) -> torch.Tensor:
    n_pos_dims = indices_grid.shape[1]
    if n_pos_dims != len(max_pos):
        raise ValueError(
            f"Number of position dimensions ({n_pos_dims}) must match max_pos length ({len(max_pos)})"
        )
    return torch.stack(
        [indices_grid[:, i] / max_pos[i] for i in range(n_pos_dims)],
        dim=-1,
    )


def _ltx_generate_freqs(
    indices: torch.Tensor, indices_grid: torch.Tensor, max_pos: list[int], use_middle_indices_grid: bool
) -> torch.Tensor:
    if use_middle_indices_grid:
        indices_grid_start, indices_grid_end = indices_grid[..., 0], indices_grid[..., 1]
        indices_grid = (indices_grid_start + indices_grid_end) / 2.0
    elif len(indices_grid.shape) == 4:
        indices_grid = indices_grid[..., 0]

    fractional_positions = _ltx_get_fractional_positions(indices_grid, max_pos)
    indices = indices.to(device=fractional_positions.device)
    freqs = (indices * (fractional_positions.unsqueeze(-1) * 2 - 1)).transpose(-1, -2).flatten(2)
    return freqs


def _ltx_split_freqs_cis(
    freqs: torch.Tensor, pad_size: int, num_attention_heads: int
) -> tuple[torch.Tensor, torch.Tensor]:
    cos_freq = freqs.cos()
    sin_freq = freqs.sin()

    if pad_size != 0:
        cos_padding = torch.ones_like(cos_freq[:, :, :pad_size])
        sin_padding = torch.zeros_like(sin_freq[:, :, :pad_size])
        cos_freq = torch.concatenate([cos_padding, cos_freq], axis=-1)
        sin_freq = torch.concatenate([sin_padding, sin_freq], axis=-1)

    b = cos_freq.shape[0]
    t = cos_freq.shape[1]
    cos_freq = cos_freq.reshape(b, t, num_attention_heads, -1)
    sin_freq = sin_freq.reshape(b, t, num_attention_heads, -1)
    cos_freq = torch.swapaxes(cos_freq, 1, 2)
    sin_freq = torch.swapaxes(sin_freq, 1, 2)
    return cos_freq, sin_freq


def _ltx_interleaved_freqs_cis(freqs: torch.Tensor, pad_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    cos_freq = freqs.cos().repeat_interleave(2, dim=-1)
    sin_freq = freqs.sin().repeat_interleave(2, dim=-1)
    if pad_size != 0:
        cos_padding = torch.ones_like(cos_freq[:, :, :pad_size])
        sin_padding = torch.zeros_like(cos_freq[:, :, :pad_size])
        cos_freq = torch.cat([cos_padding, cos_freq], dim=-1)
        sin_freq = torch.cat([sin_padding, sin_freq], dim=-1)
    return cos_freq, sin_freq


def precompute_ltx_freqs_cis(
    indices_grid: torch.Tensor,
    dim: int,
    out_dtype: torch.dtype,
    theta: float = 10000.0,
    max_pos: list[int] | None = None,
    use_middle_indices_grid: bool = False,
    num_attention_heads: int = 32,
    rope_type: LTXRopeType = LTXRopeType.INTERLEAVED,
    freq_grid_generator: Callable[[float, int, int], torch.Tensor] = generate_ltx_freq_grid_pytorch,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Precompute LTX-2 rotary cos/sin grids for (t, x, y) positions."""
    if max_pos is None:
        max_pos = [20, 2048, 2048]

    indices = freq_grid_generator(theta, indices_grid.shape[1], dim)
    freqs = _ltx_generate_freqs(indices, indices_grid, max_pos, use_middle_indices_grid)

    if rope_type == LTXRopeType.SPLIT:
        expected_freqs = dim // 2
        current_freqs = freqs.shape[-1]
        pad_size = expected_freqs - current_freqs
        cos_freq, sin_freq = _ltx_split_freqs_cis(freqs, pad_size, num_attention_heads)
    else:
        n_elem = 2 * indices_grid.shape[1]
        cos_freq, sin_freq = _ltx_interleaved_freqs_cis(freqs, dim % n_elem)
    return cos_freq.to(out_dtype), sin_freq.to(out_dtype)


@dataclass(frozen=True)
class TransformerArgs:
    """Pack transformer inputs for LTX-2 blocks."""
    x: torch.Tensor
    context: torch.Tensor
    context_mask: torch.Tensor | None
    timesteps: torch.Tensor
    embedded_timestep: torch.Tensor
    positional_embeddings: torch.Tensor
    cross_positional_embeddings: torch.Tensor | None
    cross_scale_shift_timestep: torch.Tensor | None
    cross_gate_timestep: torch.Tensor | None
    enabled: bool


@dataclass(frozen=True)
class Modality:
    """Lightweight modality container for LTX-2 inputs."""
    enabled: bool
    latent: torch.Tensor
    timesteps: torch.Tensor
    positions: torch.Tensor
    context: torch.Tensor
    context_mask: torch.Tensor | None = None


class TransformerArgsPreprocessor:
    """Prepare LTX-2 transformer inputs (patchify, AdaLN, rope)."""
    def __init__(  # noqa: PLR0913
        self,
        patchify_proj: torch.nn.Linear,
        adaln: AdaLayerNormSingle,
        caption_projection: PixArtAlphaTextProjection,
        inner_dim: int,
        max_pos: list[int],
        num_attention_heads: int,
        use_middle_indices_grid: bool,
        timestep_scale_multiplier: int,
        double_precision_rope: bool,
        positional_embedding_theta: float,
        rope_type: LTXRopeType,
    ) -> None:
        self.patchify_proj = patchify_proj
        self.adaln = adaln
        self.caption_projection = caption_projection
        self.inner_dim = inner_dim
        self.max_pos = max_pos
        self.num_attention_heads = num_attention_heads
        self.use_middle_indices_grid = use_middle_indices_grid
        self.timestep_scale_multiplier = timestep_scale_multiplier
        self.double_precision_rope = double_precision_rope
        self.positional_embedding_theta = positional_embedding_theta
        self.rope_type = rope_type

    def _prepare_timestep(
        self, timestep: torch.Tensor, batch_size: int, hidden_dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor]:
        timestep = timestep * self.timestep_scale_multiplier
        timestep, embedded_timestep = self.adaln(timestep.flatten(), hidden_dtype=hidden_dtype)
        timestep = timestep.view(batch_size, -1, timestep.shape[-1])
        embedded_timestep = embedded_timestep.view(batch_size, -1, embedded_timestep.shape[-1])
        return timestep, embedded_timestep

    def _prepare_context(
        self,
        context: torch.Tensor,
        x: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        batch_size = x.shape[0]
        if context.device != x.device:
            context = context.to(x.device)
        if attention_mask is not None and attention_mask.device != x.device:
            attention_mask = attention_mask.to(x.device)
        context = self.caption_projection(context)
        context = context.view(batch_size, -1, x.shape[-1])
        return context, attention_mask

    def _prepare_attention_mask(self, attention_mask: torch.Tensor | None, x_dtype: torch.dtype) -> torch.Tensor | None:
        if attention_mask is None or torch.is_floating_point(attention_mask):
            return attention_mask
        return (attention_mask - 1).to(x_dtype).reshape(
            (attention_mask.shape[0], 1, -1, attention_mask.shape[-1])
        ) * torch.finfo(x_dtype).max

    def _prepare_positional_embeddings(
        self,
        positions: torch.Tensor,
        inner_dim: int,
        max_pos: list[int],
        use_middle_indices_grid: bool,
        num_attention_heads: int,
        x_dtype: torch.dtype,
    ) -> torch.Tensor:
        if self.double_precision_rope:
            freq_grid_generator = generate_ltx_freq_grid_np
        else:
            freq_grid_generator = generate_ltx_freq_grid_pytorch
        return precompute_ltx_freqs_cis(
            positions,
            dim=inner_dim,
            out_dtype=x_dtype,
            theta=self.positional_embedding_theta,
            max_pos=max_pos,
            use_middle_indices_grid=use_middle_indices_grid,
            num_attention_heads=num_attention_heads,
            rope_type=self.rope_type,
            freq_grid_generator=freq_grid_generator,
        )

    def prepare(self, modality: Modality) -> TransformerArgs:
        x = self.patchify_proj(modality.latent)
        timestep, embedded_timestep = self._prepare_timestep(modality.timesteps, x.shape[0], modality.latent.dtype)
        context, attention_mask = self._prepare_context(modality.context, x, modality.context_mask)
        attention_mask = self._prepare_attention_mask(attention_mask, modality.latent.dtype)
        pe = self._prepare_positional_embeddings(
            positions=modality.positions,
            inner_dim=self.inner_dim,
            max_pos=self.max_pos,
            use_middle_indices_grid=self.use_middle_indices_grid,
            num_attention_heads=self.num_attention_heads,
            x_dtype=modality.latent.dtype,
        )
        return TransformerArgs(
            x=x,
            context=context,
            context_mask=attention_mask,
            timesteps=timestep,
            embedded_timestep=embedded_timestep,
            positional_embeddings=pe,
            cross_positional_embeddings=None,
            cross_scale_shift_timestep=None,
            cross_gate_timestep=None,
            enabled=modality.enabled,
        )


class MultiModalTransformerArgsPreprocessor:
    """Prepare transformer args for audio/video cross-attention paths."""

    def __init__(  # noqa: PLR0913
        self,
        patchify_proj: torch.nn.Linear,
        adaln: AdaLayerNormSingle,
        caption_projection: PixArtAlphaTextProjection,
        cross_scale_shift_adaln: AdaLayerNormSingle,
        cross_gate_adaln: AdaLayerNormSingle,
        inner_dim: int,
        max_pos: list[int],
        num_attention_heads: int,
        cross_pe_max_pos: int,
        use_middle_indices_grid: bool,
        audio_cross_attention_dim: int,
        timestep_scale_multiplier: int,
        double_precision_rope: bool,
        positional_embedding_theta: float,
        rope_type: LTXRopeType,
        av_ca_timestep_scale_multiplier: int,
    ) -> None:
        self.simple_preprocessor = TransformerArgsPreprocessor(
            patchify_proj=patchify_proj,
            adaln=adaln,
            caption_projection=caption_projection,
            inner_dim=inner_dim,
            max_pos=max_pos,
            num_attention_heads=num_attention_heads,
            use_middle_indices_grid=use_middle_indices_grid,
            timestep_scale_multiplier=timestep_scale_multiplier,
            double_precision_rope=double_precision_rope,
            positional_embedding_theta=positional_embedding_theta,
            rope_type=rope_type,
        )
        self.cross_scale_shift_adaln = cross_scale_shift_adaln
        self.cross_gate_adaln = cross_gate_adaln
        self.cross_pe_max_pos = cross_pe_max_pos
        self.audio_cross_attention_dim = audio_cross_attention_dim
        self.av_ca_timestep_scale_multiplier = av_ca_timestep_scale_multiplier

    def prepare(self, modality: Modality) -> TransformerArgs:
        transformer_args = self.simple_preprocessor.prepare(modality)
        cross_pe = self.simple_preprocessor._prepare_positional_embeddings(
            positions=modality.positions[:, 0:1, :],
            inner_dim=self.audio_cross_attention_dim,
            max_pos=[self.cross_pe_max_pos],
            use_middle_indices_grid=True,
            num_attention_heads=self.simple_preprocessor.num_attention_heads,
            x_dtype=modality.latent.dtype,
        )

        cross_scale_shift_timestep, cross_gate_timestep = self._prepare_cross_attention_timestep(
            timestep=modality.timesteps,
            timestep_scale_multiplier=self.simple_preprocessor.timestep_scale_multiplier,
            batch_size=transformer_args.x.shape[0],
            hidden_dtype=modality.latent.dtype,
        )

        return replace(
            transformer_args,
            cross_positional_embeddings=cross_pe,
            cross_scale_shift_timestep=cross_scale_shift_timestep,
            cross_gate_timestep=cross_gate_timestep,
        )

    def _prepare_cross_attention_timestep(
        self,
        timestep: torch.Tensor,
        timestep_scale_multiplier: int,
        batch_size: int,
        hidden_dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        timestep = timestep * timestep_scale_multiplier

        av_ca_factor = self.av_ca_timestep_scale_multiplier / timestep_scale_multiplier

        scale_shift_timestep, _ = self.cross_scale_shift_adaln(
            timestep.flatten(),
            hidden_dtype=hidden_dtype,
        )
        scale_shift_timestep = scale_shift_timestep.view(batch_size, -1, scale_shift_timestep.shape[-1])
        gate_noise_timestep, _ = self.cross_gate_adaln(
            timestep.flatten() * av_ca_factor,
            hidden_dtype=hidden_dtype,
        )
        gate_noise_timestep = gate_noise_timestep.view(batch_size, -1, gate_noise_timestep.shape[-1])

        return scale_shift_timestep, gate_noise_timestep


@dataclass
class TransformerConfig:
    """Attention/FFN dims for LTX-2 transformer blocks."""
    dim: int
    heads: int
    d_head: int
    context_dim: int


class LTXSelfAttention(nn.Module):
    """LTX-2 attention block with RMSNorm + FastVideo LocalAttention."""
    def __init__(
        self,
        query_dim: int,
        context_dim: int | None,
        heads: int,
        dim_head: int,
        norm_eps: float,
        rope_type: LTXRopeType,
        supported_attention_backends: tuple[AttentionBackendEnum, ...],
    ) -> None:
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = query_dim if context_dim is None else context_dim

        self.heads = heads
        self.dim_head = dim_head
        self.rope_type = rope_type

        self.q_norm = torch.nn.RMSNorm(inner_dim, eps=norm_eps)
        self.k_norm = torch.nn.RMSNorm(inner_dim, eps=norm_eps)
        self.to_q = nn.Linear(query_dim, inner_dim, bias=True)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=True)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=True)
        self.to_out = nn.Sequential(nn.Linear(inner_dim, query_dim, bias=True), nn.Identity())

        self.attn = LocalAttention(
            num_heads=heads,
            head_size=dim_head,
            dropout_rate=0,
            softmax_scale=None,
            causal=False,
            supported_attention_backends=supported_attention_backends,
        )
        self.attn_masked = LocalAttention(
            num_heads=heads,
            head_size=dim_head,
            dropout_rate=0,
            softmax_scale=None,
            causal=False,
            supported_attention_backends=(AttentionBackendEnum.TORCH_SDPA,),
        )

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        pe: tuple[torch.Tensor, torch.Tensor] | None = None,
        k_pe: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        q = self.to_q(x)
        context = x if context is None else context
        k = self.to_k(context)
        v = self.to_v(context)

        q = self.q_norm(q)
        k = self.k_norm(k)

        if pe is not None:
            q = apply_ltx_rotary_emb(q, pe, self.rope_type)
            k = apply_ltx_rotary_emb(k, pe if k_pe is None else k_pe, self.rope_type)

        b, q_len, _ = q.shape
        k_len = k.shape[1]
        q = q.view(b, q_len, self.heads, self.dim_head)
        k = k.view(b, k_len, self.heads, self.dim_head)
        v = v.view(b, k_len, self.heads, self.dim_head)
        if mask is not None:
            if mask.ndim == 2:
                mask = mask.unsqueeze(0)
            if mask.ndim == 3:
                mask = mask.unsqueeze(1)
            try:
                forward_ctx = get_forward_context()
                current_timestep = forward_ctx.current_timestep
                forward_batch = forward_ctx.forward_batch
            except AssertionError:
                current_timestep = 0
                forward_batch = None
            attn_metadata = SDPAMetadata(
                current_timestep=current_timestep,
                attn_mask=mask,
            )
            with set_forward_context(
                current_timestep=current_timestep,
                attn_metadata=attn_metadata,
                forward_batch=forward_batch,
            ):
                out = self.attn_masked(q, k, v)
        else:
            out = self.attn(q, k, v)
        out = out.reshape(b, q_len, -1)
        return self.to_out(out)


class BasicAVTransformerBlock(torch.nn.Module):
    """LTX-2 transformer block (audio/video + cross attention + FFN)."""

    def __init__(
        self,
        idx: int,
        video: TransformerConfig | None = None,
        audio: TransformerConfig | None = None,
        rope_type: LTXRopeType = LTXRopeType.INTERLEAVED,
        norm_eps: float = 1e-6,
    ):
        super().__init__()
        self.idx = idx

        if video is not None:
            self.attn1 = LTXSelfAttention(
                query_dim=video.dim,
                context_dim=None,
                heads=video.heads,
                dim_head=video.d_head,
                norm_eps=norm_eps,
                rope_type=rope_type,
                supported_attention_backends=(AttentionBackendEnum.FLASH_ATTN, AttentionBackendEnum.TORCH_SDPA),
            )
            self.attn2 = LTXSelfAttention(
                query_dim=video.dim,
                context_dim=video.context_dim,
                heads=video.heads,
                dim_head=video.d_head,
                norm_eps=norm_eps,
                rope_type=rope_type,
                supported_attention_backends=(AttentionBackendEnum.FLASH_ATTN, AttentionBackendEnum.TORCH_SDPA),
            )
            self.ff = FeedForward(video.dim, dim_out=video.dim)
            self.scale_shift_table = torch.nn.Parameter(torch.empty(6, video.dim))

        if audio is not None:
            self.audio_attn1 = LTXSelfAttention(
                query_dim=audio.dim,
                context_dim=None,
                heads=audio.heads,
                dim_head=audio.d_head,
                norm_eps=norm_eps,
                rope_type=rope_type,
                supported_attention_backends=(AttentionBackendEnum.FLASH_ATTN, AttentionBackendEnum.TORCH_SDPA),
            )
            self.audio_attn2 = LTXSelfAttention(
                query_dim=audio.dim,
                context_dim=audio.context_dim,
                heads=audio.heads,
                dim_head=audio.d_head,
                norm_eps=norm_eps,
                rope_type=rope_type,
                supported_attention_backends=(AttentionBackendEnum.FLASH_ATTN, AttentionBackendEnum.TORCH_SDPA),
            )
            self.audio_ff = FeedForward(audio.dim, dim_out=audio.dim)
            self.audio_scale_shift_table = torch.nn.Parameter(torch.empty(6, audio.dim))

        if audio is not None and video is not None:
            self.audio_to_video_attn = LTXSelfAttention(
                query_dim=video.dim,
                context_dim=audio.dim,
                heads=audio.heads,
                dim_head=audio.d_head,
                norm_eps=norm_eps,
                rope_type=rope_type,
                supported_attention_backends=(AttentionBackendEnum.FLASH_ATTN, AttentionBackendEnum.TORCH_SDPA),
            )
            self.video_to_audio_attn = LTXSelfAttention(
                query_dim=audio.dim,
                context_dim=video.dim,
                heads=audio.heads,
                dim_head=audio.d_head,
                norm_eps=norm_eps,
                rope_type=rope_type,
                supported_attention_backends=(AttentionBackendEnum.FLASH_ATTN, AttentionBackendEnum.TORCH_SDPA),
            )
            self.scale_shift_table_a2v_ca_audio = torch.nn.Parameter(torch.empty(5, audio.dim))
            self.scale_shift_table_a2v_ca_video = torch.nn.Parameter(torch.empty(5, video.dim))

        self.norm_eps = norm_eps

    def get_ada_values(
        self, scale_shift_table: torch.Tensor, batch_size: int, timestep: torch.Tensor, indices: slice
    ) -> tuple[torch.Tensor, ...]:
        num_ada_params = scale_shift_table.shape[0]
        ada_values = (
            scale_shift_table[indices].unsqueeze(0).unsqueeze(0).to(device=timestep.device, dtype=timestep.dtype)
            + timestep.reshape(batch_size, timestep.shape[1], num_ada_params, -1)[:, :, indices, :]
        ).unbind(dim=2)
        return ada_values

    def get_av_ca_ada_values(
        self,
        scale_shift_table: torch.Tensor,
        batch_size: int,
        scale_shift_timestep: torch.Tensor,
        gate_timestep: torch.Tensor,
        num_scale_shift_values: int = 4,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        scale_shift_ada_values = self.get_ada_values(
            scale_shift_table[:num_scale_shift_values, :], batch_size, scale_shift_timestep, slice(None, None)
        )
        gate_ada_values = self.get_ada_values(
            scale_shift_table[num_scale_shift_values:, :], batch_size, gate_timestep, slice(None, None)
        )

        scale_shift_chunks = [t.squeeze(2) for t in scale_shift_ada_values]
        gate_ada_values = [t.squeeze(2) for t in gate_ada_values]

        return (*scale_shift_chunks, *gate_ada_values)

    def forward(
        self,
        video: TransformerArgs | None,
        audio: TransformerArgs | None,
    ) -> tuple[TransformerArgs | None, TransformerArgs | None]:
        vx = video.x if video is not None else None
        ax = audio.x if audio is not None else None

        run_vx = video is not None and video.enabled and vx.numel() > 0
        run_ax = audio is not None and audio.enabled and ax.numel() > 0
        run_a2v = run_vx and run_ax
        run_v2a = run_ax and run_vx

        if run_vx:
            vshift_msa, vscale_msa, vgate_msa = self.get_ada_values(
                self.scale_shift_table, vx.shape[0], video.timesteps, slice(0, 3)
            )
            norm_vx = torch.nn.functional.rms_norm(vx, (vx.shape[-1],), eps=self.norm_eps) * (1 + vscale_msa) + vshift_msa
            vx = vx + self.attn1(norm_vx, pe=video.positional_embeddings) * vgate_msa
            vx = vx + self.attn2(
                torch.nn.functional.rms_norm(vx, (vx.shape[-1],), eps=self.norm_eps),
                context=video.context,
                mask=video.context_mask,
            )

        if run_ax:
            ashift_msa, ascale_msa, agate_msa = self.get_ada_values(
                self.audio_scale_shift_table, ax.shape[0], audio.timesteps, slice(0, 3)
            )
            norm_ax = torch.nn.functional.rms_norm(ax, (ax.shape[-1],), eps=self.norm_eps) * (1 + ascale_msa) + ashift_msa
            ax = ax + self.audio_attn1(norm_ax, pe=audio.positional_embeddings) * agate_msa
            ax = ax + self.audio_attn2(
                torch.nn.functional.rms_norm(ax, (ax.shape[-1],), eps=self.norm_eps),
                context=audio.context,
                mask=audio.context_mask,
            )

        if run_a2v or run_v2a:
            vx_norm3 = torch.nn.functional.rms_norm(vx, (vx.shape[-1],), eps=self.norm_eps)
            ax_norm3 = torch.nn.functional.rms_norm(ax, (ax.shape[-1],), eps=self.norm_eps)

            (
                scale_ca_audio_hidden_states_a2v,
                shift_ca_audio_hidden_states_a2v,
                scale_ca_audio_hidden_states_v2a,
                shift_ca_audio_hidden_states_v2a,
                gate_out_v2a,
            ) = self.get_av_ca_ada_values(
                self.scale_shift_table_a2v_ca_audio,
                ax.shape[0],
                audio.cross_scale_shift_timestep,
                audio.cross_gate_timestep,
            )

            (
                scale_ca_video_hidden_states_a2v,
                shift_ca_video_hidden_states_a2v,
                scale_ca_video_hidden_states_v2a,
                shift_ca_video_hidden_states_v2a,
                gate_out_a2v,
            ) = self.get_av_ca_ada_values(
                self.scale_shift_table_a2v_ca_video,
                vx.shape[0],
                video.cross_scale_shift_timestep,
                video.cross_gate_timestep,
            )

            if run_a2v:
                vx_scaled = vx_norm3 * (1 + scale_ca_video_hidden_states_a2v) + shift_ca_video_hidden_states_a2v
                ax_scaled = ax_norm3 * (1 + scale_ca_audio_hidden_states_a2v) + shift_ca_audio_hidden_states_a2v
                vx = vx + (
                    self.audio_to_video_attn(
                        vx_scaled,
                        context=ax_scaled,
                        pe=video.cross_positional_embeddings,
                        k_pe=audio.cross_positional_embeddings,
                    )
                    * gate_out_a2v
                )

            if run_v2a:
                ax_scaled = ax_norm3 * (1 + scale_ca_audio_hidden_states_v2a) + shift_ca_audio_hidden_states_v2a
                vx_scaled = vx_norm3 * (1 + scale_ca_video_hidden_states_v2a) + shift_ca_video_hidden_states_v2a
                ax = ax + (
                    self.video_to_audio_attn(
                        ax_scaled,
                        context=vx_scaled,
                        pe=audio.cross_positional_embeddings,
                        k_pe=video.cross_positional_embeddings,
                    )
                    * gate_out_v2a
                )

        if run_vx:
            vshift_mlp, vscale_mlp, vgate_mlp = self.get_ada_values(
                self.scale_shift_table, vx.shape[0], video.timesteps, slice(3, None)
            )
            vx_scaled = torch.nn.functional.rms_norm(vx, (vx.shape[-1],), eps=self.norm_eps) * (1 + vscale_mlp) + vshift_mlp
            vx = vx + self.ff(vx_scaled) * vgate_mlp

        if run_ax:
            ashift_mlp, ascale_mlp, agate_mlp = self.get_ada_values(
                self.audio_scale_shift_table, ax.shape[0], audio.timesteps, slice(3, None)
            )
            ax_scaled = torch.nn.functional.rms_norm(ax, (ax.shape[-1],), eps=self.norm_eps) * (1 + ascale_mlp) + ashift_mlp
            ax = ax + self.audio_ff(ax_scaled) * agate_mlp

        if os.getenv("LTX2_PIPELINE_DEBUG_LOG", "0") == "1":
            video_sum = vx.float().sum().item() if vx is not None else 0.0
            audio_sum = ax.float().sum().item() if ax is not None else 0.0
            _debug_block_log_line(
                f"fastvideo:block={self.idx}:video_sum={video_sum:.6f} "
                f"audio_sum={audio_sum:.6f}"
            )

        return (
            replace(video, x=vx) if video is not None else None,
            replace(audio, x=ax) if audio is not None else None,
        )


class LTXModelType(Enum):
    """Model type flags for LTX-2."""
    AudioVideo = "ltx av model"
    VideoOnly = "ltx video only model"
    AudioOnly = "ltx audio only model"

    def is_video_enabled(self) -> bool:
        return self in (LTXModelType.AudioVideo, LTXModelType.VideoOnly)

    def is_audio_enabled(self) -> bool:
        return self in (LTXModelType.AudioVideo, LTXModelType.AudioOnly)


class LTXModel(torch.nn.Module):
    """Core LTX-2 transformer stack for audio/video latents."""

    def __init__(  # noqa: PLR0913
        self,
        *,
        model_type: LTXModelType = LTXModelType.AudioVideo,
        num_attention_heads: int = 32,
        attention_head_dim: int = 128,
        in_channels: int = 128,
        out_channels: int = 128,
        num_layers: int = 48,
        cross_attention_dim: int = 4096,
        norm_eps: float = 1e-06,
        caption_channels: int = 3840,
        positional_embedding_theta: float = 10000.0,
        positional_embedding_max_pos: list[int] | None = None,
        timestep_scale_multiplier: int = 1000,
        use_middle_indices_grid: bool = True,
        audio_num_attention_heads: int = 32,
        audio_attention_head_dim: int = 64,
        audio_in_channels: int = 128,
        audio_out_channels: int = 128,
        audio_cross_attention_dim: int = 2048,
        audio_positional_embedding_max_pos: list[int] | None = None,
        av_ca_timestep_scale_multiplier: int = 1,
        rope_type: LTXRopeType = LTXRopeType.INTERLEAVED,
        double_precision_rope: bool = False,
    ):
        super().__init__()
        self._enable_gradient_checkpointing = False
        self.use_middle_indices_grid = use_middle_indices_grid
        self.rope_type = rope_type
        self.double_precision_rope = double_precision_rope
        self.timestep_scale_multiplier = timestep_scale_multiplier
        self.positional_embedding_theta = positional_embedding_theta
        self.model_type = model_type
        cross_pe_max_pos = None

        if model_type.is_video_enabled():
            if positional_embedding_max_pos is None:
                positional_embedding_max_pos = [20, 2048, 2048]
            self.positional_embedding_max_pos = positional_embedding_max_pos
            self.num_attention_heads = num_attention_heads
            self.inner_dim = num_attention_heads * attention_head_dim
            self._init_video(
                in_channels=in_channels,
                out_channels=out_channels,
                caption_channels=caption_channels,
                norm_eps=norm_eps,
            )

        if model_type.is_audio_enabled():
            if audio_positional_embedding_max_pos is None:
                audio_positional_embedding_max_pos = [20]
            self.audio_positional_embedding_max_pos = audio_positional_embedding_max_pos
            self.audio_num_attention_heads = audio_num_attention_heads
            self.audio_inner_dim = self.audio_num_attention_heads * audio_attention_head_dim
            self._init_audio(
                in_channels=audio_in_channels,
                out_channels=audio_out_channels,
                caption_channels=caption_channels,
                norm_eps=norm_eps,
            )

        if model_type.is_video_enabled() and model_type.is_audio_enabled():
            cross_pe_max_pos = max(self.positional_embedding_max_pos[0], self.audio_positional_embedding_max_pos[0])
            self.av_ca_timestep_scale_multiplier = av_ca_timestep_scale_multiplier
            self.audio_cross_attention_dim = audio_cross_attention_dim
            self._init_audio_video(num_scale_shift_values=4)

        self._init_preprocessors(cross_pe_max_pos)
        self._init_transformer_blocks(
            num_layers=num_layers,
            attention_head_dim=attention_head_dim if model_type.is_video_enabled() else 0,
            cross_attention_dim=cross_attention_dim,
            audio_attention_head_dim=audio_attention_head_dim if model_type.is_audio_enabled() else 0,
            audio_cross_attention_dim=audio_cross_attention_dim,
            norm_eps=norm_eps,
        )

    def _init_video(
        self,
        in_channels: int,
        out_channels: int,
        caption_channels: int,
        norm_eps: float,
    ) -> None:
        self.patchify_proj = torch.nn.Linear(in_channels, self.inner_dim, bias=True)
        self.adaln_single = AdaLayerNormSingle(self.inner_dim)
        self.caption_projection = PixArtAlphaTextProjection(
            in_features=caption_channels,
            hidden_size=self.inner_dim,
        )
        self.scale_shift_table = torch.nn.Parameter(torch.empty(2, self.inner_dim))
        self.norm_out = torch.nn.LayerNorm(self.inner_dim, elementwise_affine=False, eps=norm_eps)
        self.proj_out = torch.nn.Linear(self.inner_dim, out_channels)

    def _init_audio(
        self,
        in_channels: int,
        out_channels: int,
        caption_channels: int,
        norm_eps: float,
    ) -> None:
        self.audio_patchify_proj = torch.nn.Linear(in_channels, self.audio_inner_dim, bias=True)
        self.audio_adaln_single = AdaLayerNormSingle(self.audio_inner_dim)
        self.audio_caption_projection = PixArtAlphaTextProjection(
            in_features=caption_channels,
            hidden_size=self.audio_inner_dim,
        )
        self.audio_scale_shift_table = torch.nn.Parameter(torch.empty(2, self.audio_inner_dim))
        self.audio_norm_out = torch.nn.LayerNorm(self.audio_inner_dim, elementwise_affine=False, eps=norm_eps)
        self.audio_proj_out = torch.nn.Linear(self.audio_inner_dim, out_channels)

    def _init_audio_video(self, num_scale_shift_values: int) -> None:
        self.av_ca_video_scale_shift_adaln_single = AdaLayerNormSingle(
            self.inner_dim,
            embedding_coefficient=num_scale_shift_values,
        )
        self.av_ca_audio_scale_shift_adaln_single = AdaLayerNormSingle(
            self.audio_inner_dim,
            embedding_coefficient=num_scale_shift_values,
        )
        self.av_ca_a2v_gate_adaln_single = AdaLayerNormSingle(
            self.inner_dim,
            embedding_coefficient=1,
        )
        self.av_ca_v2a_gate_adaln_single = AdaLayerNormSingle(
            self.audio_inner_dim,
            embedding_coefficient=1,
        )

    def _init_preprocessors(self, cross_pe_max_pos: int | None = None) -> None:
        if self.model_type.is_video_enabled() and self.model_type.is_audio_enabled():
            self.video_args_preprocessor = MultiModalTransformerArgsPreprocessor(
                patchify_proj=self.patchify_proj,
                adaln=self.adaln_single,
                caption_projection=self.caption_projection,
                cross_scale_shift_adaln=self.av_ca_video_scale_shift_adaln_single,
                cross_gate_adaln=self.av_ca_a2v_gate_adaln_single,
                inner_dim=self.inner_dim,
                max_pos=self.positional_embedding_max_pos,
                num_attention_heads=self.num_attention_heads,
                cross_pe_max_pos=cross_pe_max_pos,
                use_middle_indices_grid=self.use_middle_indices_grid,
                audio_cross_attention_dim=self.audio_cross_attention_dim,
                timestep_scale_multiplier=self.timestep_scale_multiplier,
                double_precision_rope=self.double_precision_rope,
                positional_embedding_theta=self.positional_embedding_theta,
                rope_type=self.rope_type,
                av_ca_timestep_scale_multiplier=self.av_ca_timestep_scale_multiplier,
            )
            self.audio_args_preprocessor = MultiModalTransformerArgsPreprocessor(
                patchify_proj=self.audio_patchify_proj,
                adaln=self.audio_adaln_single,
                caption_projection=self.audio_caption_projection,
                cross_scale_shift_adaln=self.av_ca_audio_scale_shift_adaln_single,
                cross_gate_adaln=self.av_ca_v2a_gate_adaln_single,
                inner_dim=self.audio_inner_dim,
                max_pos=self.audio_positional_embedding_max_pos,
                num_attention_heads=self.audio_num_attention_heads,
                cross_pe_max_pos=cross_pe_max_pos,
                use_middle_indices_grid=self.use_middle_indices_grid,
                audio_cross_attention_dim=self.audio_cross_attention_dim,
                timestep_scale_multiplier=self.timestep_scale_multiplier,
                double_precision_rope=self.double_precision_rope,
                positional_embedding_theta=self.positional_embedding_theta,
                rope_type=self.rope_type,
                av_ca_timestep_scale_multiplier=self.av_ca_timestep_scale_multiplier,
            )
        elif self.model_type.is_video_enabled():
            self.video_args_preprocessor = TransformerArgsPreprocessor(
                patchify_proj=self.patchify_proj,
                adaln=self.adaln_single,
                caption_projection=self.caption_projection,
                inner_dim=self.inner_dim,
                max_pos=self.positional_embedding_max_pos,
                num_attention_heads=self.num_attention_heads,
                use_middle_indices_grid=self.use_middle_indices_grid,
                timestep_scale_multiplier=self.timestep_scale_multiplier,
                double_precision_rope=self.double_precision_rope,
                positional_embedding_theta=self.positional_embedding_theta,
                rope_type=self.rope_type,
            )
        elif self.model_type.is_audio_enabled():
            self.audio_args_preprocessor = TransformerArgsPreprocessor(
                patchify_proj=self.audio_patchify_proj,
                adaln=self.audio_adaln_single,
                caption_projection=self.audio_caption_projection,
                inner_dim=self.audio_inner_dim,
                max_pos=self.audio_positional_embedding_max_pos,
                num_attention_heads=self.audio_num_attention_heads,
                use_middle_indices_grid=self.use_middle_indices_grid,
                timestep_scale_multiplier=self.timestep_scale_multiplier,
                double_precision_rope=self.double_precision_rope,
                positional_embedding_theta=self.positional_embedding_theta,
                rope_type=self.rope_type,
            )

    def _init_transformer_blocks(
        self,
        num_layers: int,
        attention_head_dim: int,
        cross_attention_dim: int,
        audio_attention_head_dim: int,
        audio_cross_attention_dim: int,
        norm_eps: float,
    ) -> None:
        video_config = (
            TransformerConfig(
                dim=self.inner_dim,
                heads=self.num_attention_heads,
                d_head=attention_head_dim,
                context_dim=cross_attention_dim,
            )
            if self.model_type.is_video_enabled()
            else None
        )
        audio_config = (
            TransformerConfig(
                dim=self.audio_inner_dim,
                heads=self.audio_num_attention_heads,
                d_head=audio_attention_head_dim,
                context_dim=audio_cross_attention_dim,
            )
            if self.model_type.is_audio_enabled()
            else None
        )
        self.transformer_blocks = torch.nn.ModuleList(
            [
                BasicAVTransformerBlock(
                    idx=idx,
                    video=video_config,
                    audio=audio_config,
                    rope_type=self.rope_type,
                    norm_eps=norm_eps,
                )
                for idx in range(num_layers)
            ]
        )

    def _process_transformer_blocks(
        self,
        video: TransformerArgs | None,
        audio: TransformerArgs | None,
    ) -> tuple[TransformerArgs | None, TransformerArgs | None]:
        for block in self.transformer_blocks:
            video, audio = block(video=video, audio=audio)
        return video, audio

    def _process_output(
        self,
        scale_shift_table: torch.Tensor,
        norm_out: torch.nn.LayerNorm,
        proj_out: torch.nn.Linear,
        x: torch.Tensor,
        embedded_timestep: torch.Tensor,
    ) -> torch.Tensor:
        scale_shift_values = (
            scale_shift_table[None, None].to(device=x.device, dtype=x.dtype) + embedded_timestep[:, :, None]
        )
        shift, scale = scale_shift_values[:, :, 0], scale_shift_values[:, :, 1]
        x = norm_out(x)
        x = x * (1 + scale) + shift
        x = proj_out(x)
        return x

    def forward(
        self,
        video: Modality | None,
        audio: Modality | None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        if os.getenv("LTX2_PIPELINE_DEBUG_LOG", "0") == "1":
            _debug_block_log_line(
                "fastvideo:patchify_proj"
                f":video_w_sum={self.patchify_proj.weight.float().sum().item():.6f} "
                f"video_b_sum={self.patchify_proj.bias.float().sum().item():.6f} "
                f"audio_w_sum={self.audio_patchify_proj.weight.float().sum().item():.6f} "
                f"audio_b_sum={self.audio_patchify_proj.bias.float().sum().item():.6f}"
            )
        if not self.model_type.is_video_enabled() and video is not None:
            raise ValueError("Video is not enabled for this model")
        if not self.model_type.is_audio_enabled() and audio is not None:
            raise ValueError("Audio is not enabled for this model")

        video_args = self.video_args_preprocessor.prepare(video) if video is not None else None
        audio_args = self.audio_args_preprocessor.prepare(audio) if audio is not None else None
        _debug_transformer_args("fastvideo:prep_video", video_args)
        _debug_transformer_args("fastvideo:prep_audio", audio_args)
        video_out, audio_out = self._process_transformer_blocks(video_args, audio_args)

        vx = (
            self._process_output(
                self.scale_shift_table, self.norm_out, self.proj_out, video_out.x, video_out.embedded_timestep
            )
            if video_out is not None
            else None
        )
        ax = (
            self._process_output(
                self.audio_scale_shift_table,
                self.audio_norm_out,
                self.audio_proj_out,
                audio_out.x,
                audio_out.embedded_timestep,
            )
            if audio_out is not None
            else None
        )
        return vx, ax


class LTX2Transformer3DModel(CachableDiT):
    """
    LTX-2 transformer using native FastVideo LTX-2 modules.
    """

    param_names_mapping = LTX2VideoConfig().param_names_mapping
    reverse_param_names_mapping = LTX2VideoConfig().reverse_param_names_mapping
    lora_param_names_mapping = LTX2VideoConfig().lora_param_names_mapping

    def __init__(self, config: LTX2VideoConfig, hf_config: dict[str, Any]):
        super().__init__(config=config, hf_config=hf_config)

        arch = config.arch_config

        model_type = LTXModelType.AudioVideo
        self.model = LTXModel(
            model_type=model_type,
            num_attention_heads=arch.num_attention_heads,
            attention_head_dim=arch.attention_head_dim,
            in_channels=arch.in_channels,
            out_channels=arch.out_channels,
            num_layers=arch.num_layers,
            cross_attention_dim=arch.cross_attention_dim,
            norm_eps=arch.norm_eps,
            caption_channels=arch.caption_channels,
            positional_embedding_theta=arch.positional_embedding_theta,
            positional_embedding_max_pos=arch.positional_embedding_max_pos,
            timestep_scale_multiplier=arch.timestep_scale_multiplier,
            use_middle_indices_grid=arch.use_middle_indices_grid,
            rope_type=LTXRopeType(arch.rope_type),
            double_precision_rope=arch.double_precision_rope,
            audio_num_attention_heads=arch.audio_num_attention_heads,
            audio_attention_head_dim=arch.audio_attention_head_dim,
            audio_in_channels=arch.audio_in_channels,
            audio_out_channels=arch.audio_out_channels,
            audio_cross_attention_dim=arch.audio_cross_attention_dim,
            audio_positional_embedding_max_pos=arch.audio_positional_embedding_max_pos,
            av_ca_timestep_scale_multiplier=arch.av_ca_timestep_scale_multiplier,
        )

        self.patchifier = VideoLatentPatchifier(patch_size=arch.patch_size[1])
        self.audio_patchifier = AudioLatentPatchifier(
            patch_size=DEFAULT_LTX2_AUDIO_MEL_BINS,
            sample_rate=DEFAULT_LTX2_AUDIO_SAMPLE_RATE,
            hop_length=DEFAULT_LTX2_AUDIO_HOP_LENGTH,
            audio_latent_downsample_factor=DEFAULT_LTX2_AUDIO_DOWNSAMPLE,
            is_causal=True,
            shift=0,
        )

        self.hidden_size = arch.num_attention_heads * arch.attention_head_dim
        self.num_attention_heads = arch.num_attention_heads
        self.num_channels_latents = arch.num_channels_latents

        if os.getenv("LTX2_DEBUG_DETAIL", "0") == "1":
            detail_path = os.getenv("LTX2_PIPELINE_DEBUG_DETAIL_PATH", "")
            if detail_path:
                self._attach_debug_detail_hooks(detail_path)

    def _attach_debug_detail_hooks(self, log_path: str) -> None:
        path = Path(log_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            path.unlink()

        def _format_sum(tensor: torch.Tensor | None) -> str:
            if tensor is None:
                return "None"
            return f"{tensor.float().sum().item():.6f}"

        def _hook_factory(block_idx: int, name: str):
            def _hook(_module, _inputs, outputs):  # noqa: ANN001
                if isinstance(outputs, tuple):
                    out = outputs[0]
                else:
                    out = outputs
                out_sum = _format_sum(out if torch.is_tensor(out) else None)
                with path.open("a", encoding="utf-8") as f:
                    f.write(f"fastvideo:{block_idx}:{name}:out_sum={out_sum}\n")
            return _hook

        for block in self.model.transformer_blocks:
            idx = block.idx
            for name in (
                "attn1",
                "attn2",
                "ff",
                "audio_attn1",
                "audio_attn2",
                "audio_ff",
                "audio_to_video_attn",
                "video_to_audio_attn",
            ):
                if hasattr(block, name):
                    getattr(block, name).register_forward_hook(
                        _hook_factory(idx, name))

        def _output_hook(label: str):
            def _hook(_module, _inputs, outputs):  # noqa: ANN001
                out = outputs[0] if isinstance(outputs, tuple) else outputs
                out_sum = _format_sum(out if torch.is_tensor(out) else None)
                with path.open("a", encoding="utf-8") as f:
                    f.write(f"fastvideo:output:{label}:out_sum={out_sum}\n")
            return _hook

        if hasattr(self.model, "proj_out"):
            self.model.proj_out.register_forward_hook(_output_hook("proj_out"))
        if hasattr(self.model, "audio_proj_out"):
            self.model.audio_proj_out.register_forward_hook(
                _output_hook("audio_proj_out"))

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | list[torch.Tensor],
        timestep: torch.LongTensor,
        encoder_hidden_states_image: torch.Tensor | list[torch.Tensor] | None = None,
        encoder_attention_mask: torch.Tensor | None = None,
        guidance=None,
        audio_hidden_states: torch.Tensor | None = None,
        audio_encoder_hidden_states: torch.Tensor | None = None,
        audio_timestep: torch.Tensor | None = None,
        audio_encoder_attention_mask: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        if isinstance(encoder_hidden_states, list):
            encoder_hidden_states = encoder_hidden_states[0]

        video_shape = VideoLatentShape.from_torch_shape(hidden_states.shape)
        positions = self.patchifier.get_patch_grid_bounds(
            video_shape, device=hidden_states.device
        )
        fps = None
        try:
            forward_ctx = get_forward_context()
        except AssertionError:
            forward_ctx = None
        if forward_ctx is not None and forward_ctx.forward_batch is not None:
            fps_value = forward_ctx.forward_batch.fps
            if isinstance(fps_value, list):
                fps_value = fps_value[0] if fps_value else None
            if fps_value is not None:
                fps = float(fps_value)
        positions = _get_pixel_coords(
            positions,
            DEFAULT_LTX2_SCALE_FACTORS,
            fps=fps,
            causal_fix=True,
        ).to(hidden_states.dtype)
        latents = self.patchifier.patchify(hidden_states)
        video_modality = Modality(
            enabled=True,
            latent=latents,
            timesteps=timestep,
            positions=positions,
            context=encoder_hidden_states,
            context_mask=encoder_attention_mask,
        )
        if os.getenv("LTX2_PIPELINE_DEBUG_LOG", "0") == "1":
            video_head = latents.flatten()[:8].float().tolist()
            video_flat = latents.float().flatten()
            video_checksum = (video_flat * torch.arange(video_flat.numel(), device=video_flat.device)).sum().item()
            _debug_block_log_line(
                "fastvideo:modality_video"
                f":latent_sum={latents.float().sum().item():.6f} "
                f"latent_shape={tuple(latents.shape)} "
                f"positions_sum={positions.float().sum().item():.6f} "
                f"positions_shape={tuple(positions.shape)} "
                f"latent_head={video_head} "
                f"latent_checksum={video_checksum:.6f}"
            )
        audio_modality = None
        audio_shape = None
        if audio_hidden_states is not None and audio_encoder_hidden_states is not None and audio_timestep is not None:
            audio_shape = AudioLatentShape.from_torch_shape(
                audio_hidden_states.shape)
            audio_positions = self.audio_patchifier.get_patch_grid_bounds(
                audio_shape, device=audio_hidden_states.device)
            audio_latents = self.audio_patchifier.patchify(
                audio_hidden_states)
            audio_modality = Modality(
                enabled=True,
                latent=audio_latents,
                timesteps=audio_timestep,
                positions=audio_positions,
                context=audio_encoder_hidden_states,
                context_mask=audio_encoder_attention_mask,
            )
            if os.getenv("LTX2_PIPELINE_DEBUG_LOG", "0") == "1":
                audio_head = audio_latents.flatten()[:8].float().tolist()
                audio_flat = audio_latents.float().flatten()
                audio_checksum = (audio_flat * torch.arange(audio_flat.numel(), device=audio_flat.device)).sum().item()
                _debug_block_log_line(
                    "fastvideo:modality_audio"
                    f":latent_sum={audio_latents.float().sum().item():.6f} "
                    f"latent_shape={tuple(audio_latents.shape)} "
                    f"positions_sum={audio_positions.float().sum().item():.6f} "
                    f"positions_shape={tuple(audio_positions.shape)} "
                    f"latent_head={audio_head} "
                    f"latent_checksum={audio_checksum:.6f}"
                )
        video_out, audio_out = self.model(
            video=video_modality,
            audio=audio_modality,
        )
        if video_out is not None and video_modality is not None:
            video_out = _to_denoised(
                video_modality.latent,
                video_out,
                video_modality.timesteps,
            )
        if audio_out is not None and audio_modality is not None:
            audio_out = _to_denoised(
                audio_modality.latent,
                audio_out,
                audio_modality.timesteps,
            )
        video_out = self.patchifier.unpatchify(
            video_out, output_shape=video_shape)
        if audio_out is None or audio_shape is None:
            return video_out
        audio_out = self.audio_patchifier.unpatchify(
            audio_out, output_shape=audio_shape)
        return video_out, audio_out
