# SPDX-License-Identifier: Apache-2.0
# Minimal blocks for Stable Audio DiT + VAE (from stable-audio-tools models/blocks.py).
import math

import torch
from torch import nn


def _snake_beta(x: torch.Tensor, alpha: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
    return x + (1.0 / (beta + 1e-9)) * torch.pow(torch.sin(x * alpha), 2)


class SnakeBeta(nn.Module):
    """From stable-audio-tools blocks (BigVGAN-style)."""

    def __init__(
        self,
        in_features: int,
        alpha: float = 1.0,
        alpha_trainable: bool = True,
        alpha_logscale: bool = True,
    ):
        super().__init__()
        self.in_features = in_features
        self.alpha_logscale = alpha_logscale
        if alpha_logscale:
            self.alpha = nn.Parameter(torch.zeros(in_features) * alpha)
            self.beta = nn.Parameter(torch.zeros(in_features) * alpha)
        else:
            self.alpha = nn.Parameter(torch.ones(in_features) * alpha)
            self.beta = nn.Parameter(torch.ones(in_features) * alpha)
        self.alpha.requires_grad = alpha_trainable
        self.beta.requires_grad = alpha_trainable

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        alpha = self.alpha.unsqueeze(0).unsqueeze(-1)
        beta = self.beta.unsqueeze(0).unsqueeze(-1)
        if self.alpha_logscale:
            alpha = torch.exp(alpha)
            beta = torch.exp(beta)
        return _snake_beta(x, alpha, beta)


class FourierFeatures(nn.Module):
    def __init__(self, in_features: int, out_features: int, std: float = 1.0):
        super().__init__()
        assert out_features % 2 == 0
        self.weight = nn.Parameter(
            torch.randn([out_features // 2, in_features]) * std
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        f = 2 * math.pi * input @ self.weight.T
        return torch.cat([f.cos(), f.sin()], dim=-1)
