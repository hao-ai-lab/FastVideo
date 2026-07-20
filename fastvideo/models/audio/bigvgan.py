# SPDX-License-Identifier: MIT
"""Reusable native BigVGAN-v2 vocoder.

Adapted from NVIDIA BigVGAN-v2 and its alias-free activation implementation.
The CUDA activation kernel is intentionally excluded; FastVideo uses the
portable PyTorch path for deterministic loading and parity.
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Conv1d, ConvTranspose1d
from torch.nn.utils.parametrizations import weight_norm
from torch.nn.utils.parametrize import remove_parametrizations


class AttrDict(dict):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.__dict__ = self


def get_padding(kernel_size: int, dilation: int = 1) -> int:
    return int((kernel_size * dilation - dilation) / 2)


def init_weights(module: nn.Module, mean: float = 0.0, std: float = 0.01) -> None:
    if "Conv" in module.__class__.__name__:
        module.weight.data.normal_(mean, std)


def kaiser_sinc_filter1d(cutoff: float, half_width: float, kernel_size: int) -> torch.Tensor:
    even = kernel_size % 2 == 0
    half_size = kernel_size // 2
    delta_f = 4 * half_width
    amplitude = 2.285 * (half_size - 1) * math.pi * delta_f + 7.95
    if amplitude > 50.0:
        beta = 0.1102 * (amplitude - 8.7)
    elif amplitude >= 21.0:
        beta = 0.5842 * (amplitude - 21) ** 0.4 + 0.07886 * (amplitude - 21.0)
    else:
        beta = 0.0
    window = torch.kaiser_window(kernel_size, beta=beta, periodic=False)
    time = torch.arange(-half_size, half_size) + 0.5 if even else torch.arange(kernel_size) - half_size
    if cutoff == 0:
        kernel = torch.zeros_like(time)
    else:
        kernel = 2 * cutoff * window * torch.sinc(2 * cutoff * time)
        kernel /= kernel.sum()
    return kernel.view(1, 1, kernel_size)


class LowPassFilter1d(nn.Module):
    def __init__(
        self,
        cutoff: float = 0.5,
        half_width: float = 0.6,
        stride: int = 1,
        padding: bool = True,
        padding_mode: str = "replicate",
        kernel_size: int = 12,
    ) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.even = kernel_size % 2 == 0
        self.pad_left = kernel_size // 2 - int(self.even)
        self.pad_right = kernel_size // 2
        self.stride = stride
        self.padding = padding
        self.padding_mode = padding_mode
        self.register_buffer("filter", kaiser_sinc_filter1d(cutoff, half_width, kernel_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, channels, _ = x.shape
        if self.padding:
            x = F.pad(x, (self.pad_left, self.pad_right), mode=self.padding_mode)
        return F.conv1d(x, self.filter.expand(channels, -1, -1), stride=self.stride, groups=channels)


class UpSample1d(nn.Module):
    def __init__(self, ratio: int = 2, kernel_size: int | None = None) -> None:
        super().__init__()
        self.ratio = ratio
        self.kernel_size = int(6 * ratio // 2) * 2 if kernel_size is None else kernel_size
        self.stride = ratio
        self.pad = self.kernel_size // ratio - 1
        self.pad_left = self.pad * self.stride + (self.kernel_size - self.stride) // 2
        self.pad_right = self.pad * self.stride + (self.kernel_size - self.stride + 1) // 2
        self.register_buffer(
            "filter", kaiser_sinc_filter1d(cutoff=0.5 / ratio, half_width=0.6 / ratio, kernel_size=self.kernel_size)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, channels, _ = x.shape
        x = F.pad(x, (self.pad, self.pad), mode="replicate")
        x = self.ratio * F.conv_transpose1d(
            x, self.filter.expand(channels, -1, -1), stride=self.stride, groups=channels
        )
        return x[..., self.pad_left : -self.pad_right]


class DownSample1d(nn.Module):
    def __init__(self, ratio: int = 2, kernel_size: int | None = None) -> None:
        super().__init__()
        self.ratio = ratio
        self.kernel_size = int(6 * ratio // 2) * 2 if kernel_size is None else kernel_size
        self.lowpass = LowPassFilter1d(
            cutoff=0.5 / ratio, half_width=0.6 / ratio, stride=ratio, kernel_size=self.kernel_size
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lowpass(x)


class Activation1d(nn.Module):
    def __init__(
        self,
        activation: nn.Module,
        up_ratio: int = 2,
        down_ratio: int = 2,
        up_kernel_size: int = 12,
        down_kernel_size: int = 12,
    ) -> None:
        super().__init__()
        self.up_ratio = up_ratio
        self.down_ratio = down_ratio
        self.act = activation
        self.upsample = UpSample1d(up_ratio, up_kernel_size)
        self.downsample = DownSample1d(down_ratio, down_kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.downsample(self.act(self.upsample(x)))


class Snake(nn.Module):
    def __init__(
        self, in_features: int, alpha: float = 1.0, alpha_trainable: bool = True, alpha_logscale: bool = False
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.alpha_logscale = alpha_logscale
        initial = torch.zeros(in_features) * alpha if alpha_logscale else torch.ones(in_features) * alpha
        self.alpha = nn.Parameter(initial, requires_grad=alpha_trainable)
        self.no_div_by_zero = 1e-9

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        alpha = self.alpha.unsqueeze(0).unsqueeze(-1)
        if self.alpha_logscale:
            alpha = torch.exp(alpha)
        return x + (1.0 / (alpha + self.no_div_by_zero)) * torch.pow(
            torch.sin(x * alpha), 2)


class SnakeBeta(nn.Module):
    def __init__(
        self, in_features: int, alpha: float = 1.0, alpha_trainable: bool = True, alpha_logscale: bool = False
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.alpha_logscale = alpha_logscale
        initial = torch.zeros(in_features) * alpha if alpha_logscale else torch.ones(in_features) * alpha
        self.alpha = nn.Parameter(initial.clone(), requires_grad=alpha_trainable)
        self.beta = nn.Parameter(initial.clone(), requires_grad=alpha_trainable)
        self.no_div_by_zero = 1e-9

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        alpha = self.alpha.unsqueeze(0).unsqueeze(-1)
        beta = self.beta.unsqueeze(0).unsqueeze(-1)
        if self.alpha_logscale:
            alpha = torch.exp(alpha)
            beta = torch.exp(beta)
        return x + (1.0 / (beta + self.no_div_by_zero)) * torch.pow(
            torch.sin(x * alpha), 2)


def _activation(name: str, channels: int, logscale: bool) -> Activation1d:
    if name == "snake":
        activation = Snake(channels, alpha_logscale=logscale)
    elif name == "snakebeta":
        activation = SnakeBeta(channels, alpha_logscale=logscale)
    else:
        raise ValueError(f"Unsupported BigVGAN activation: {name}")
    return Activation1d(activation)


class AMPBlock1(nn.Module):
    def __init__(
        self,
        config: AttrDict,
        channels: int,
        kernel_size: int = 3,
        dilation: tuple[int, ...] = (1, 3, 5),
        activation: str = "snake",
    ) -> None:
        super().__init__()
        self.convs1 = nn.ModuleList(
            [
                weight_norm(
                    Conv1d(
                        channels, channels, kernel_size, stride=1, dilation=rate, padding=get_padding(kernel_size, rate)
                    )
                )
                for rate in dilation
            ]
        )
        self.convs1.apply(init_weights)
        self.convs2 = nn.ModuleList(
            [
                weight_norm(
                    Conv1d(channels, channels, kernel_size, stride=1, dilation=1, padding=get_padding(kernel_size, 1))
                )
                for _ in dilation
            ]
        )
        self.convs2.apply(init_weights)
        self.activations = nn.ModuleList(
            [
                _activation(activation, channels, config.snake_logscale)
                for _ in range(len(self.convs1) + len(self.convs2))
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        acts1, acts2 = self.activations[::2], self.activations[1::2]
        for conv1, conv2, act1, act2 in zip(self.convs1, self.convs2, acts1, acts2, strict=True):
            residual = conv2(act2(conv1(act1(x))))
            x = residual + x
        return x

    def remove_weight_norm(self) -> None:
        for layer in self.convs1:
            remove_parametrizations(layer, "weight")
        for layer in self.convs2:
            remove_parametrizations(layer, "weight")


class AMPBlock2(nn.Module):
    def __init__(
        self,
        config: AttrDict,
        channels: int,
        kernel_size: int = 3,
        dilation: tuple[int, ...] = (1, 3, 5),
        activation: str = "snake",
    ) -> None:
        super().__init__()
        self.convs = nn.ModuleList(
            [
                weight_norm(
                    Conv1d(
                        channels, channels, kernel_size, stride=1, dilation=rate, padding=get_padding(kernel_size, rate)
                    )
                )
                for rate in dilation
            ]
        )
        self.convs.apply(init_weights)
        self.activations = nn.ModuleList([_activation(activation, channels, config.snake_logscale) for _ in self.convs])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for conv, activation in zip(self.convs, self.activations, strict=True):
            x = conv(activation(x)) + x
        return x

    def remove_weight_norm(self) -> None:
        for layer in self.convs:
            remove_parametrizations(layer, "weight")


class BigVGANV2(nn.Module):
    """BigVGAN-v2 generator compatible with NVIDIA checkpoint keys."""

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        config = dict(config)
        config.pop("_class_name", None)
        config.pop("architectures", None)
        weight_norm_removed = bool(config.pop("weight_norm_removed", False))
        self.config = AttrDict(config)
        if self.config.get("use_cuda_kernel", False):
            raise ValueError("FastVideo BigVGANV2 supports only the portable PyTorch path")
        self.config["use_cuda_kernel"] = False
        self.num_kernels = len(self.config.resblock_kernel_sizes)
        self.num_upsamples = len(self.config.upsample_rates)
        self.conv_pre = weight_norm(Conv1d(self.config.num_mels, self.config.upsample_initial_channel, 7, 1, padding=3))
        if self.config.resblock == "1":
            block_class = AMPBlock1
        elif self.config.resblock == "2":
            block_class = AMPBlock2
        else:
            raise ValueError(f"Unsupported BigVGAN resblock: {self.config.resblock}")

        self.ups = nn.ModuleList()
        for index, (rate, kernel) in enumerate(
            zip(self.config.upsample_rates, self.config.upsample_kernel_sizes, strict=True)
        ):
            self.ups.append(
                nn.ModuleList(
                    [
                        weight_norm(
                            ConvTranspose1d(
                                self.config.upsample_initial_channel // (2**index),
                                self.config.upsample_initial_channel // (2 ** (index + 1)),
                                kernel,
                                rate,
                                padding=(kernel - rate) // 2,
                            )
                        )
                    ]
                )
            )

        self.resblocks = nn.ModuleList()
        for index in range(len(self.ups)):
            channels = self.config.upsample_initial_channel // (2 ** (index + 1))
            for kernel, dilation in zip(
                self.config.resblock_kernel_sizes, self.config.resblock_dilation_sizes, strict=True
            ):
                self.resblocks.append(
                    block_class(self.config, channels, kernel, tuple(dilation), activation=self.config.activation)
                )

        channels = self.config.upsample_initial_channel // (2 ** len(self.ups))
        self.activation_post = _activation(self.config.activation, channels, self.config.snake_logscale)
        self.use_bias_at_final = self.config.get("use_bias_at_final", True)
        self.conv_post = weight_norm(Conv1d(channels, 1, 7, 1, padding=3, bias=self.use_bias_at_final))
        for upsampler in self.ups:
            upsampler.apply(init_weights)
        self.conv_post.apply(init_weights)
        self.use_tanh_at_final = self.config.get("use_tanh_at_final", True)
        if weight_norm_removed:
            self.remove_weight_norm()

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        hidden = self.conv_pre(mel)
        for index in range(self.num_upsamples):
            for upsampler in self.ups[index]:
                hidden = upsampler(hidden)
            accumulated = None
            for kernel in range(self.num_kernels):
                block_output = self.resblocks[
                    index * self.num_kernels + kernel](hidden)
                if accumulated is None:
                    accumulated = block_output
                else:
                    # Preserve BigVGAN's published sequential accumulation
                    # order. Tree reduction drifts through later nonlinear
                    # upsampling stages with the full checkpoint.
                    accumulated += block_output
            assert accumulated is not None
            hidden = accumulated / self.num_kernels
        hidden = self.conv_post(self.activation_post(hidden))
        if self.use_tanh_at_final:
            return torch.tanh(hidden)
        return torch.clamp(hidden, min=-1.0, max=1.0)

    def remove_weight_norm(self) -> None:
        try:
            for upsamplers in self.ups:
                for upsampler in upsamplers:
                    remove_parametrizations(upsampler, "weight")
            for block in self.resblocks:
                block.remove_weight_norm()
            remove_parametrizations(self.conv_pre, "weight")
            remove_parametrizations(self.conv_post, "weight")
        except ValueError:
            # Idempotent for pipeline setup and converted checkpoints.
            return


EntryClass = BigVGANV2
