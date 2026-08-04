# Copyright 2025 MiniMax authors and HuggingFace Team
# SPDX-License-Identifier: Apache-2.0

"""Native MiniMax H3 waveform autoencoder."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm

from fastvideo.configs.models.vaes.minimax_h3_audio import MiniMaxH3AudioVAEConfig
from fastvideo.layers.linear import ReplicatedLinear
from fastvideo.logger import init_logger

logger = init_logger(__name__)


class MiniMaxH3AudioDiagonalGaussianDistribution:
    """Diagonal Gaussian parameterized by mean and log standard deviation."""

    def __init__(self, mean: torch.Tensor, logs: torch.Tensor):
        self.mean = mean
        self.logs = logs
        self.std = torch.exp(logs)

    def mode(self) -> torch.Tensor:
        return self.mean

    def sample(self, generator: torch.Generator | None = None) -> torch.Tensor:
        noise_device = self.mean.device
        if generator is not None and generator.device.type == "cpu":
            noise_device = torch.device("cpu")
        noise = torch.randn(
            self.mean.shape,
            generator=generator,
            device=noise_device,
            dtype=self.mean.dtype,
        ).to(self.mean.device)
        return self.mean + self.std * noise


@dataclass
class MiniMaxH3AudioEncoderOutput:
    latent_dist: MiniMaxH3AudioDiagonalGaussianDistribution


@dataclass
class MiniMaxH3AudioDecoderOutput:
    sample: torch.Tensor


def _wn_conv1d(*args, **kwargs) -> nn.Module:
    return weight_norm(nn.Conv1d(*args, **kwargs))


def _linear(layer: ReplicatedLinear, hidden_states: torch.Tensor) -> torch.Tensor:
    return layer(hidden_states)[0]


def _module_dtype(module: nn.Module) -> torch.dtype:
    return next(module.parameters()).dtype


def kaiser_sinc_filter1d(cutoff: float, half_width: float, kernel_size: int) -> torch.Tensor:
    """Build the persistent low-pass filter used by alias-free activations."""

    half_size = kernel_size // 2
    attenuation = 2.285 * (half_size - 1) * math.pi * (4 * half_width) + 7.95
    if attenuation > 50.0:
        beta = 0.1102 * (attenuation - 8.7)
    elif attenuation >= 21.0:
        beta = 0.5842 * (attenuation - 21) ** 0.4 + 0.07886 * (attenuation - 21.0)
    else:
        beta = 0.0

    window = torch.kaiser_window(kernel_size, beta=beta, periodic=False, dtype=torch.float32)
    if kernel_size % 2 == 0:
        time = torch.arange(-half_size, half_size, dtype=torch.float32) + 0.5
    else:
        time = torch.arange(kernel_size, dtype=torch.float32) - half_size

    filter_ = 2 * cutoff * window * torch.sinc(2 * cutoff * time)
    filter_ /= filter_.sum()
    return filter_.view(1, 1, kernel_size)


class MiniMaxH3AudioSnake1d(nn.Module):

    def __init__(self, channels: int):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1, channels, 1))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return hidden_states + (self.alpha + 1e-9).reciprocal() * torch.sin(self.alpha * hidden_states).pow(2)


class MiniMaxH3AudioSnakeBeta(nn.Module):

    def __init__(self, channels: int):
        super().__init__()
        self.alpha = nn.Parameter(torch.zeros(channels))
        self.beta = nn.Parameter(torch.zeros(channels))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        alpha = torch.exp(self.alpha.unsqueeze(0).unsqueeze(-1))
        beta = torch.exp(self.beta.unsqueeze(0).unsqueeze(-1))
        return hidden_states + (beta + 1e-9).reciprocal() * torch.sin(alpha * hidden_states).pow(2)


class MiniMaxH3AudioLowPassFilter1d(nn.Module):

    def __init__(self, cutoff: float, half_width: float, stride: int, kernel_size: int):
        super().__init__()
        even = kernel_size % 2 == 0
        self.pad_left = kernel_size // 2 - int(even)
        self.pad_right = kernel_size // 2
        self.stride = stride
        self.register_buffer("filter", kaiser_sinc_filter1d(cutoff, half_width, kernel_size))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        num_channels = hidden_states.shape[1]
        hidden_states = F.pad(hidden_states, (self.pad_left, self.pad_right), mode="replicate")
        return F.conv1d(
            hidden_states,
            self.filter.expand(num_channels, -1, -1),
            stride=self.stride,
            groups=num_channels,
        )


class MiniMaxH3AudioUpSample1d(nn.Module):

    def __init__(self, ratio: int, kernel_size: int):
        super().__init__()
        self.ratio = ratio
        self.stride = ratio
        self.pad = kernel_size // ratio - 1
        self.pad_left = self.pad * self.stride + (kernel_size - self.stride) // 2
        self.pad_right = self.pad * self.stride + (kernel_size - self.stride + 1) // 2
        self.register_buffer(
            "filter",
            kaiser_sinc_filter1d(cutoff=0.5 / ratio, half_width=0.6 / ratio, kernel_size=kernel_size),
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        num_channels = hidden_states.shape[1]
        hidden_states = F.pad(hidden_states, (self.pad, self.pad), mode="replicate")
        hidden_states = self.ratio * F.conv_transpose1d(
            hidden_states,
            self.filter.expand(num_channels, -1, -1),
            stride=self.stride,
            groups=num_channels,
        )
        return hidden_states[..., self.pad_left:-self.pad_right]


class MiniMaxH3AudioDownSample1d(nn.Module):

    def __init__(self, ratio: int, kernel_size: int):
        super().__init__()
        self.lowpass = MiniMaxH3AudioLowPassFilter1d(
            cutoff=0.5 / ratio,
            half_width=0.6 / ratio,
            stride=ratio,
            kernel_size=kernel_size,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.lowpass(hidden_states)


class MiniMaxH3AudioActivation1d(nn.Module):

    def __init__(self, activation: nn.Module, ratio: int = 2, kernel_size: int = 12):
        super().__init__()
        self.act = activation
        self.upsample = MiniMaxH3AudioUpSample1d(ratio, kernel_size)
        self.downsample = MiniMaxH3AudioDownSample1d(ratio, kernel_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.upsample(hidden_states)
        hidden_states = self.act(hidden_states)
        return self.downsample(hidden_states)


class MiniMaxH3AudioResidualUnit(nn.Module):

    def __init__(self, dim: int, dilation: int):
        super().__init__()
        self.block = nn.Sequential(
            MiniMaxH3AudioSnake1d(dim),
            _wn_conv1d(dim, dim, kernel_size=7, dilation=dilation, padding=((7 - 1) * dilation) // 2),
            MiniMaxH3AudioSnake1d(dim),
            _wn_conv1d(dim, dim, kernel_size=1),
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        residual = self.block(hidden_states)
        pad = (hidden_states.shape[-1] - residual.shape[-1]) // 2
        if pad > 0:
            hidden_states = hidden_states[..., pad:-pad]
        return hidden_states + residual


class MiniMaxH3AudioEncoderBlock(nn.Module):

    def __init__(self, dim: int, stride: int):
        super().__init__()
        self.block = nn.Sequential(
            MiniMaxH3AudioResidualUnit(dim // 2, dilation=1),
            MiniMaxH3AudioResidualUnit(dim // 2, dilation=3),
            MiniMaxH3AudioResidualUnit(dim // 2, dilation=9),
            MiniMaxH3AudioSnake1d(dim // 2),
            _wn_conv1d(
                dim // 2,
                dim,
                kernel_size=2 * stride,
                stride=stride,
                padding=math.ceil(stride / 2),
            ),
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.block(hidden_states)


class MiniMaxH3AudioEncoder(nn.Module):

    def __init__(self, d_model: int, strides: tuple[int, ...], d_latent: int):
        super().__init__()
        block: list[nn.Module] = [_wn_conv1d(1, d_model, kernel_size=7, padding=3)]
        for stride in strides:
            d_model *= 2
            block.append(MiniMaxH3AudioEncoderBlock(d_model, stride=stride))
        block += [
            MiniMaxH3AudioSnake1d(d_model),
            _wn_conv1d(d_model, d_latent, kernel_size=3, padding=1),
        ]
        self.block = nn.Sequential(*block)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.block(hidden_states)


class MiniMaxH3AudioGeGluMlp(nn.Module):

    def __init__(self, in_features: int, hidden_features: int):
        super().__init__()
        self.norm = nn.LayerNorm(in_features)
        self.act = nn.GELU(approximate="tanh")
        self.w0 = ReplicatedLinear(in_features, hidden_features, params_dtype=torch.float32)
        self.w1 = ReplicatedLinear(in_features, hidden_features, params_dtype=torch.float32)
        self.w2 = ReplicatedLinear(hidden_features, in_features, params_dtype=torch.float32)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.norm(hidden_states)
        hidden_states = self.act(_linear(self.w0, hidden_states)) * _linear(self.w1, hidden_states)
        return _linear(self.w2, hidden_states)


class MiniMaxH3AudioCausalAttention(nn.Module):
    """Causal projection from the encoder trunk width to the latent width."""

    def __init__(self, in_dim: int, out_dim: int, num_heads: int):
        super().__init__()
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.head_dim = in_dim // num_heads
        self.qkv = ReplicatedLinear(in_dim, in_dim * 3, bias=False, params_dtype=torch.float32)
        self.q_bias = nn.Parameter(torch.zeros(in_dim))
        self.v_bias = nn.Parameter(torch.zeros(in_dim))
        self.register_buffer("zero_k_bias", torch.zeros(in_dim))
        self.proj = ReplicatedLinear(out_dim, out_dim, params_dtype=torch.float32)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape
        qkv = F.linear(
            hidden_states,
            self.qkv.weight,
            torch.cat((self.q_bias, self.zero_k_bias, self.v_bias)),
        )
        query, key, value = (
            qkv.reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim).permute(2, 0, 1, 3, 4).unbind(0)
        )

        # H3's projection is an unusual causal flat stream, not a VAE spatial-attention block.
        hidden_states = F.scaled_dot_product_attention(
            query.transpose(1, 2),
            key.transpose(1, 2),
            value.transpose(1, 2),
            dropout_p=0.0,
            is_causal=True,
        ).transpose(1, 2)
        hidden_states = torch.mean(hidden_states, dim=2)
        hidden_states = F.adaptive_avg_pool1d(hidden_states, self.out_dim)
        return _linear(self.proj, hidden_states)


class MiniMaxH3AudioAttnProjection(nn.Module):

    def __init__(self, in_dim: int, out_dim: int, num_heads: int, mlp_ratio: int = 2):
        super().__init__()
        self.norm1 = nn.LayerNorm(in_dim)
        self.attn = MiniMaxH3AudioCausalAttention(in_dim, out_dim, num_heads)
        self.proj = ReplicatedLinear(in_dim, out_dim, params_dtype=torch.float32)
        self.norm3 = nn.LayerNorm(in_dim)
        self.norm2 = nn.LayerNorm(out_dim)
        self.mlp = MiniMaxH3AudioGeGluMlp(in_features=out_dim, hidden_features=out_dim * mlp_ratio)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = _linear(self.proj, self.norm3(hidden_states)) + self.attn(self.norm1(hidden_states))
        return hidden_states + self.mlp(self.norm2(hidden_states))


class MiniMaxH3AudioAMPBlock(nn.Module):

    def __init__(self, channels: int, kernel_size: int, dilation: tuple[int, ...]):
        super().__init__()
        self.convs1 = nn.ModuleList([
            _wn_conv1d(channels, channels, kernel_size, dilation=d, padding=(kernel_size * d - d) // 2)
            for d in dilation
        ])
        self.convs2 = nn.ModuleList([
            _wn_conv1d(channels, channels, kernel_size, dilation=1, padding=(kernel_size - 1) // 2)
            for _ in dilation
        ])
        self.activations = nn.ModuleList([
            MiniMaxH3AudioActivation1d(activation=MiniMaxH3AudioSnakeBeta(channels))
            for _ in range(2 * len(dilation))
        ])

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        acts1, acts2 = self.activations[::2], self.activations[1::2]
        for conv1, conv2, act1, act2 in zip(self.convs1, self.convs2, acts1, acts2):
            residual = conv1(act1(hidden_states))
            residual = conv2(act2(residual))
            hidden_states = residual + hidden_states
        return hidden_states


class MiniMaxH3AudioBigVGANDecoder(nn.Module):

    def __init__(
        self,
        in_channels: int,
        upsample_initial_channel: int,
        upsample_rates: tuple[int, ...],
        upsample_kernel_sizes: tuple[int, ...],
        resblock_kernel_sizes: tuple[int, ...],
        resblock_dilation_sizes: tuple[tuple[int, ...], ...],
    ):
        super().__init__()
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)
        self.conv_pre = _wn_conv1d(in_channels, upsample_initial_channel, 7, 1, padding=3)

        self.ups = nn.ModuleList()
        for i, (rate, kernel) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            self.ups.append(nn.ModuleList([
                weight_norm(nn.ConvTranspose1d(
                    upsample_initial_channel // (2**i),
                    upsample_initial_channel // (2 ** (i + 1)),
                    kernel,
                    rate,
                    padding=(kernel - rate) // 2,
                ))
            ]))

        self.resblocks = nn.ModuleList()
        for i in range(self.num_upsamples):
            channels = upsample_initial_channel // (2 ** (i + 1))
            for kernel, dilation in zip(resblock_kernel_sizes, resblock_dilation_sizes):
                self.resblocks.append(MiniMaxH3AudioAMPBlock(channels, kernel, tuple(dilation)))

        self.activation_post = MiniMaxH3AudioActivation1d(activation=MiniMaxH3AudioSnakeBeta(channels))
        self.conv_post = _wn_conv1d(channels, 1, 7, 1, padding=3, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.conv_pre(hidden_states)
        for i in range(self.num_upsamples):
            hidden_states = self.ups[i][0](hidden_states)
            residual = None
            for j in range(self.num_kernels):
                block = self.resblocks[i * self.num_kernels + j](hidden_states)
                residual = block if residual is None else residual + block
            hidden_states = residual / self.num_kernels

        hidden_states = self.activation_post(hidden_states)
        hidden_states = self.conv_post(hidden_states)
        return torch.clamp(hidden_states, min=-1.0, max=1.0)


class MiniMaxH3AudioVAE(nn.Module):
    """DAC encoder plus BigVGAN decoder for mono 32 kHz waveforms."""

    def __init__(self, config: MiniMaxH3AudioVAEConfig):
        super().__init__()
        self.config = config
        self.fastvideo_config = config
        arch = config.arch_config

        encoder_rates = tuple(int(rate) for rate in arch.encoder_rates)
        decoder_rates = tuple(int(rate) for rate in arch.decoder_rates)
        self.hop_length = math.prod(encoder_rates)
        self.sampling_rate = int(arch.sampling_rate)
        self.latent_channels = int(arch.latent_channels)
        self.audio_channels = 1
        latents_mean = arch.latents_mean if arch.latents_mean is not None else [0.0] * self.latent_channels
        latents_std = arch.latents_std if arch.latents_std is not None else [1.0] * self.latent_channels
        self.register_buffer(
            "latents_mean",
            torch.tensor(latents_mean, dtype=torch.float32).view(1, -1, 1),
            persistent=False,
        )
        self.register_buffer(
            "latents_std",
            torch.tensor(latents_std, dtype=torch.float32).view(1, -1, 1),
            persistent=False,
        )

        if math.prod(decoder_rates) != self.hop_length:
            raise ValueError(
                f"`decoder_rates` must upsample by the encoder hop length {self.hop_length}, got "
                f"{math.prod(decoder_rates)}."
            )
        if arch.latent_dim % arch.latent_channels != 0:
            raise ValueError(
                f"`latent_dim` ({arch.latent_dim}) must be a multiple of `latent_channels` "
                f"({arch.latent_channels})."
            )

        self.encoder = MiniMaxH3AudioEncoder(
            d_model=arch.encoder_dim,
            strides=encoder_rates,
            d_latent=arch.latent_dim,
        )
        self.pre_block = MiniMaxH3AudioAttnProjection(
            arch.latent_dim,
            arch.latent_channels,
            num_heads=arch.num_attention_heads,
        )
        self.mean_proj = nn.Conv1d(arch.latent_channels, arch.latent_channels, 1)
        self.logs_proj = nn.Conv1d(arch.latent_channels, arch.latent_channels, 1)

        self.dec_in_proj = nn.Conv1d(arch.latent_channels, arch.latent_dim, 1)
        self.decoder = MiniMaxH3AudioBigVGANDecoder(
            in_channels=arch.latent_dim,
            upsample_initial_channel=arch.decoder_dim,
            upsample_rates=decoder_rates,
            upsample_kernel_sizes=tuple(int(kernel) for kernel in arch.decoder_kernel_sizes),
            resblock_kernel_sizes=tuple(int(kernel) for kernel in arch.resblock_kernel_sizes),
            resblock_dilation_sizes=tuple(
                tuple(int(dilation) for dilation in group) for group in arch.resblock_dilation_sizes
            ),
        )

        # The H3 checkpoint and waveform numerics require this component to stay FP32.
        self.float()

    def encode(
        self,
        sample: torch.Tensor,
        return_dict: bool = True,
    ) -> MiniMaxH3AudioEncoderOutput | tuple[MiniMaxH3AudioDiagonalGaussianDistribution]:
        if sample.ndim != 3 or sample.shape[1] != 1:
            raise ValueError(f"`sample` must have shape [batch_size, 1, samples], got {tuple(sample.shape)}.")

        right_pad = math.ceil(sample.shape[-1] / self.hop_length) * self.hop_length - sample.shape[-1]
        if right_pad > 0:
            sample = F.pad(sample, (0, right_pad))

        encoder_dtype = _module_dtype(self.encoder)
        hidden_states = self.encoder(sample.to(encoder_dtype))
        hidden_states = self.pre_block(hidden_states.transpose(1, 2)).transpose(1, 2)
        mean = self.mean_proj(hidden_states)
        logs = self.logs_proj(hidden_states)
        if encoder_dtype != torch.float32:
            mean = mean.float()
            logs = logs.float()

        posterior = MiniMaxH3AudioDiagonalGaussianDistribution(mean, logs)
        if not return_dict:
            return (posterior,)
        return MiniMaxH3AudioEncoderOutput(latent_dist=posterior)

    def normalize_latents(self, latents: torch.Tensor) -> torch.Tensor:
        return (latents - self.latents_mean) / self.latents_std

    def denormalize_latents(self, latents: torch.Tensor) -> torch.Tensor:
        return latents * self.latents_std + self.latents_mean

    def decode(
        self,
        latents: torch.Tensor,
        return_dict: bool = True,
    ) -> MiniMaxH3AudioDecoderOutput | tuple[torch.Tensor]:
        if latents.ndim != 3:
            raise ValueError(
                f"`latents` must have shape [batch_size, latent_channels, num_frames], got {tuple(latents.shape)}."
            )

        decoder_dtype = _module_dtype(self.decoder)
        decoded = self.decoder(self.dec_in_proj(latents.to(decoder_dtype)))
        if decoder_dtype != torch.float32:
            decoded = decoded.float()

        if not return_dict:
            return (decoded,)
        return MiniMaxH3AudioDecoderOutput(sample=decoded)

    def forward(
        self,
        sample: torch.Tensor,
        sample_posterior: bool = False,
        return_dict: bool = True,
        generator: torch.Generator | None = None,
    ) -> MiniMaxH3AudioDecoderOutput | tuple[torch.Tensor]:
        posterior = self.encode(sample).latent_dist
        latents = posterior.sample(generator=generator) if sample_posterior else posterior.mode()
        return self.decode(latents, return_dict=return_dict)


# Keep the Diffusers checkpoint class name loadable through FastVideo's registry.
AutoencoderKLMiniMaxH3Audio = MiniMaxH3AudioVAE
EntryClass = [MiniMaxH3AudioVAE, AutoencoderKLMiniMaxH3Audio]
