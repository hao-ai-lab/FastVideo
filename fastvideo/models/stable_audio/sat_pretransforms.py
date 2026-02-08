# SPDX-License-Identifier: Apache-2.0
# AutoencoderPretransform for Stable Audio (in-repo, no clone).
from __future__ import annotations

import torch
from torch import nn


class Pretransform(nn.Module):
    def __init__(
        self,
        enable_grad: bool,
        io_channels: int,
        is_discrete: bool,
    ):
        super().__init__()
        self.is_discrete = is_discrete
        self.io_channels = io_channels
        self.encoded_channels = None
        self.downsampling_ratio = None
        self.enable_grad = enable_grad

    def encode(self, x: torch.Tensor, **kwargs: object) -> torch.Tensor:
        raise NotImplementedError

    def decode(self, z: torch.Tensor, **kwargs: object) -> torch.Tensor:
        raise NotImplementedError


class AutoencoderPretransform(Pretransform):
    def __init__(
        self,
        model: nn.Module,
        scale: float = 1.0,
        model_half: bool = False,
        iterate_batch: bool = False,
        chunked: bool = False,
    ):
        super().__init__(
            enable_grad=False,
            io_channels=model.io_channels,
            is_discrete=(
                getattr(model, "bottleneck", None) is not None
                and getattr(model.bottleneck, "is_discrete", False)
            ),
        )
        self.model = model
        self.model.requires_grad_(False).eval()
        self.scale = scale
        self.downsampling_ratio = model.downsampling_ratio
        self.io_channels = model.io_channels
        self.sample_rate = getattr(model, "sample_rate", 44100)
        self.model_half = model_half
        self.iterate_batch = iterate_batch
        self.encoded_channels = model.latent_dim
        self.chunked = chunked
        self.num_quantizers = (
            getattr(model.bottleneck, "num_quantizers", None)
            if getattr(model, "bottleneck", None) is not None
            else None
        )
        self.codebook_size = (
            getattr(model.bottleneck, "codebook_size", None)
            if getattr(model, "bottleneck", None) is not None
            else None
        )
        if self.model_half:
            self.model.half()

    def encode(self, x: torch.Tensor, **kwargs: object) -> torch.Tensor:
        if self.model_half:
            x = x.half()
        encoded = self.model.encode_audio(
            x, chunked=self.chunked, iterate_batch=self.iterate_batch, **kwargs
        )
        if self.model_half:
            encoded = encoded.float()
        return encoded / self.scale

    def decode(self, z: torch.Tensor, **kwargs: object) -> torch.Tensor:
        z = z * self.scale
        if self.model_half:
            z = z.half()
        decoded = self.model.decode_audio(
            z, chunked=self.chunked, iterate_batch=self.iterate_batch, **kwargs
        )
        if self.model_half:
            decoded = decoded.float()
        return decoded

    def load_state_dict(self, state_dict: dict, strict: bool = True) -> None:
        self.model.load_state_dict(state_dict, strict=strict)
