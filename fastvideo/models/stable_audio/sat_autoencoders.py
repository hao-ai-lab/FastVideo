# SPDX-License-Identifier: Apache-2.0
# Oobleck VAE + AudioAutoencoder for Stable Audio pretransform (in-repo, no clone).
from __future__ import annotations

import math
from typing import Any, Dict

import torch
from torch import nn
from torch.nn.utils import weight_norm

from fastvideo.models.stable_audio.sat_blocks import SnakeBeta


def WNConv1d(*args, **kwargs):
    return weight_norm(nn.Conv1d(*args, **kwargs))


def WNConvTranspose1d(*args, **kwargs):
    return weight_norm(nn.ConvTranspose1d(*args, **kwargs))


def _checkpoint(function, *args, **kwargs):
    kwargs.setdefault("use_reentrant", False)
    return torch.utils.checkpoint.checkpoint(function, *args, **kwargs)


def get_activation(
    activation: str, antialias: bool = False, channels: int | None = None
) -> nn.Module:
    if activation == "elu":
        act = nn.ELU()
    elif activation == "snake":
        act = SnakeBeta(channels)
    elif activation == "none":
        act = nn.Identity()
    else:
        raise ValueError(f"Unknown activation {activation}")
    if antialias:
        act = nn.Identity()  # skip alias_free_torch for in-repo
    return act


class ResidualUnit(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dilation: int,
        use_snake: bool = False,
        antialias_activation: bool = False,
    ):
        super().__init__()
        self.dilation = dilation
        padding = (dilation * (7 - 1)) // 2
        self.layers = nn.Sequential(
            get_activation(
                "snake" if use_snake else "elu",
                antialias=antialias_activation,
                channels=out_channels,
            ),
            WNConv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=7,
                dilation=dilation,
                padding=padding,
            ),
            get_activation(
                "snake" if use_snake else "elu",
                antialias=antialias_activation,
                channels=out_channels,
            ),
            WNConv1d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=1,
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = x
        if self.training:
            x = _checkpoint(self.layers, x)
        else:
            x = self.layers(x)
        return x + res


class EncoderBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int,
        use_snake: bool = False,
        antialias_activation: bool = False,
    ):
        super().__init__()
        self.layers = nn.Sequential(
            ResidualUnit(
                in_channels, in_channels, 1, use_snake=use_snake
            ),
            ResidualUnit(
                in_channels, in_channels, 3, use_snake=use_snake
            ),
            ResidualUnit(
                in_channels, in_channels, 9, use_snake=use_snake
            ),
            get_activation(
                "snake" if use_snake else "elu",
                antialias=antialias_activation,
                channels=in_channels,
            ),
            WNConv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=2 * stride,
                stride=stride,
                padding=math.ceil(stride / 2),
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class DecoderBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int,
        use_snake: bool = False,
        antialias_activation: bool = False,
        use_nearest_upsample: bool = False,
    ):
        super().__init__()
        if use_nearest_upsample:
            upsample_layer = nn.Sequential(
                nn.Upsample(scale_factor=stride, mode="nearest"),
                WNConv1d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=2 * stride,
                    stride=1,
                    bias=False,
                    padding="same",
                ),
            )
        else:
            upsample_layer = WNConvTranspose1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=2 * stride,
                stride=stride,
                padding=math.ceil(stride / 2),
            )
        self.layers = nn.Sequential(
            get_activation(
                "snake" if use_snake else "elu",
                antialias=antialias_activation,
                channels=in_channels,
            ),
            upsample_layer,
            ResidualUnit(out_channels, out_channels, 1, use_snake=use_snake),
            ResidualUnit(out_channels, out_channels, 3, use_snake=use_snake),
            ResidualUnit(out_channels, out_channels, 9, use_snake=use_snake),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class OobleckEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int = 2,
        channels: int = 128,
        latent_dim: int = 32,
        c_mults: list = (1, 2, 4, 8),
        strides: list = (2, 4, 8, 8),
        use_snake: bool = False,
        antialias_activation: bool = False,
    ):
        super().__init__()
        self.in_channels = in_channels
        c_mults = [1] + list(c_mults)
        self.depth = len(c_mults)
        layers = [
            WNConv1d(
                in_channels=in_channels,
                out_channels=c_mults[0] * channels,
                kernel_size=7,
                padding=3,
            )
        ]
        for i in range(self.depth - 1):
            layers.append(
                EncoderBlock(
                    c_mults[i] * channels,
                    c_mults[i + 1] * channels,
                    strides[i],
                    use_snake=use_snake,
                )
            )
        layers += [
            get_activation(
                "snake" if use_snake else "elu",
                antialias=antialias_activation,
                channels=c_mults[-1] * channels,
            ),
            WNConv1d(
                in_channels=c_mults[-1] * channels,
                out_channels=latent_dim,
                kernel_size=3,
                padding=1,
            ),
        ]
        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class OobleckDecoder(nn.Module):
    def __init__(
        self,
        out_channels: int = 2,
        channels: int = 128,
        latent_dim: int = 32,
        c_mults: list = (1, 2, 4, 8),
        strides: list = (2, 4, 8, 8),
        use_snake: bool = False,
        antialias_activation: bool = False,
        use_nearest_upsample: bool = False,
        final_tanh: bool = True,
    ):
        super().__init__()
        self.out_channels = out_channels
        c_mults = [1] + list(c_mults)
        self.depth = len(c_mults)
        layers = [
            WNConv1d(
                in_channels=latent_dim,
                out_channels=c_mults[-1] * channels,
                kernel_size=7,
                padding=3,
            ),
        ]
        for i in range(self.depth - 1, 0, -1):
            layers.append(
                DecoderBlock(
                    c_mults[i] * channels,
                    c_mults[i - 1] * channels,
                    strides[i - 1],
                    use_snake=use_snake,
                    antialias_activation=antialias_activation,
                    use_nearest_upsample=use_nearest_upsample,
                )
            )
        layers += [
            get_activation(
                "snake" if use_snake else "elu",
                antialias=antialias_activation,
                channels=c_mults[0] * channels,
            ),
            WNConv1d(
                in_channels=c_mults[0] * channels,
                out_channels=out_channels,
                kernel_size=7,
                padding=3,
                bias=False,
            ),
            nn.Tanh() if final_tanh else nn.Identity(),
        ]
        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class AudioAutoencoder(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        latent_dim: int,
        downsampling_ratio: int,
        sample_rate: int,
        io_channels: int = 2,
        bottleneck: Any = None,
        pretransform: Any = None,
        in_channels: int | None = None,
        out_channels: int | None = None,
        soft_clip: bool = False,
    ):
        super().__init__()
        self.downsampling_ratio = downsampling_ratio
        self.sample_rate = sample_rate
        self.latent_dim = latent_dim
        self.io_channels = io_channels
        self.in_channels = io_channels
        self.out_channels = io_channels
        self.min_length = downsampling_ratio
        if in_channels is not None:
            self.in_channels = in_channels
        if out_channels is not None:
            self.out_channels = out_channels
        self.bottleneck = bottleneck
        self.encoder = encoder
        self.decoder = decoder
        self.pretransform = pretransform
        self.soft_clip = soft_clip
        self.is_discrete = (
            bottleneck is not None and getattr(bottleneck, "is_discrete", False)
        )

    def encode(
        self,
        audio: torch.Tensor,
        skip_bottleneck: bool = False,
        return_info: bool = False,
        skip_pretransform: bool = False,
        iterate_batch: bool = False,
        **kwargs: Any,
    ) -> torch.Tensor | tuple:
        if self.pretransform is not None and not skip_pretransform:
            with torch.no_grad():
                audio = self.pretransform.encode(audio)
        if self.encoder is not None:
            latents = self.encoder(audio)
        else:
            latents = audio
        if self.bottleneck is not None and not skip_bottleneck:
            latents = self.bottleneck.encode(latents, **kwargs)
        if return_info:
            return latents, {}
        return latents

    def decode(
        self,
        latents: torch.Tensor,
        skip_bottleneck: bool = False,
        iterate_batch: bool = False,
        **kwargs: Any,
    ) -> torch.Tensor:
        if self.bottleneck is not None and not skip_bottleneck:
            latents = self.bottleneck.decode(latents)
        decoded = self.decoder(latents, **kwargs)
        if self.pretransform is not None:
            with torch.no_grad():
                decoded = self.pretransform.decode(decoded)
        if self.soft_clip:
            decoded = torch.tanh(decoded)
        return decoded

    def encode_audio(
        self,
        audio: torch.Tensor,
        chunked: bool = False,
        overlap: int = 32,
        chunk_size: int = 128,
        **kwargs: Any,
    ) -> torch.Tensor:
        if not chunked:
            return self.encode(audio, **kwargs)
        samples_per_latent = int(self.downsampling_ratio)
        total_size = audio.shape[2]
        batch_size = audio.shape[0]
        chunk_size_samp = chunk_size * samples_per_latent
        overlap_samp = overlap * samples_per_latent
        hop_size = chunk_size_samp - overlap_samp
        chunks_list = []
        i = 0
        for i in range(0, total_size - chunk_size_samp + 1, hop_size):
            chunks_list.append(audio[:, :, i : i + chunk_size_samp])
        if i + chunk_size_samp < total_size:
            chunks_list.append(audio[:, :, -chunk_size_samp:])
        chunks = torch.stack(chunks_list)
        num_chunks = chunks.shape[0]
        y_size = total_size // samples_per_latent
        y_final = torch.zeros(
            (batch_size, self.latent_dim, y_size),
            dtype=chunks.dtype,
            device=audio.device,
        )
        for j in range(num_chunks):
            y_chunk = self.encode(chunks[j])
            if j == num_chunks - 1:
                t_end = y_size
                t_start = t_end - y_chunk.shape[2]
            else:
                t_start = j * hop_size // samples_per_latent
                t_end = t_start + chunk_size_samp // samples_per_latent
            ol = overlap_samp // samples_per_latent // 2
            chunk_start = 0
            chunk_end = y_chunk.shape[2]
            if j > 0:
                t_start += ol
                chunk_start += ol
            if j < num_chunks - 1:
                t_end -= ol
                chunk_end -= ol
            y_final[:, :, t_start:t_end] = y_chunk[:, :, chunk_start:chunk_end]
        return y_final

    def decode_audio(
        self,
        latents: torch.Tensor,
        chunked: bool = False,
        overlap: int = 32,
        chunk_size: int = 128,
        **kwargs: Any,
    ) -> torch.Tensor:
        if not chunked:
            return self.decode(latents, **kwargs)
        hop_size = chunk_size - overlap
        total_size = latents.shape[2]
        batch_size = latents.shape[0]
        chunks_list = []
        i = 0
        for i in range(0, total_size - chunk_size + 1, hop_size):
            chunks_list.append(latents[:, :, i : i + chunk_size])
        if i + chunk_size < total_size:
            chunks_list.append(latents[:, :, -chunk_size:])
        chunks = torch.stack(chunks_list)
        num_chunks = chunks.shape[0]
        samples_per_latent = int(self.downsampling_ratio)
        y_size = total_size * samples_per_latent
        y_final = torch.zeros(
            (batch_size, self.out_channels, y_size),
            dtype=chunks.dtype,
            device=latents.device,
        )
        for j in range(num_chunks):
            y_chunk = self.decode(chunks[j])
            if j == num_chunks - 1:
                t_end = y_size
                t_start = t_end - y_chunk.shape[2]
            else:
                t_start = j * hop_size * samples_per_latent
                t_end = t_start + chunk_size * samples_per_latent
            ol = (overlap // 2) * samples_per_latent
            chunk_start = 0
            chunk_end = y_chunk.shape[2]
            if j > 0:
                t_start += ol
                chunk_start += ol
            if j < num_chunks - 1:
                t_end -= ol
                chunk_end -= ol
            y_final[:, :, t_start:t_end] = y_chunk[:, :, chunk_start:chunk_end]
        return y_final


def create_encoder_from_config(
    encoder_config: Dict[str, Any],
    latent_dim_override: int | None = None,
) -> nn.Module:
    encoder_type = encoder_config.get("type")
    if encoder_type != "oobleck":
        raise ValueError(
            f"Only oobleck encoder is supported in-repo; got {encoder_type}"
        )
    config = dict(encoder_config["config"])
    if latent_dim_override is not None:
        config["latent_dim"] = latent_dim_override
    encoder = OobleckEncoder(**config)
    if not encoder_config.get("requires_grad", True):
        for p in encoder.parameters():
            p.requires_grad = False
    return encoder


def create_decoder_from_config(
    decoder_config: Dict[str, Any],
    latent_dim_override: int | None = None,
) -> nn.Module:
    decoder_type = decoder_config.get("type")
    if decoder_type != "oobleck":
        raise ValueError(
            f"Only oobleck decoder is supported in-repo; got {decoder_type}"
        )
    config = dict(decoder_config["config"])
    if latent_dim_override is not None:
        config["latent_dim"] = latent_dim_override
    decoder = OobleckDecoder(**config)
    if not decoder_config.get("requires_grad", True):
        for p in decoder.parameters():
            p.requires_grad = False
    return decoder


def _vae_sample(mean: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    stdev = torch.nn.functional.softplus(scale) + 1e-4
    return torch.randn_like(mean, device=mean.device, dtype=mean.dtype) * stdev + mean


class VAEBottleneck(nn.Module):
    """Splits encoder output (2*latent_dim) into mean and scale, samples to latent_dim."""

    def __init__(self) -> None:
        super().__init__()

    def encode(
        self, x: torch.Tensor, return_info: bool = False, **kwargs: Any
    ) -> torch.Tensor | tuple:
        mean, scale = x.chunk(2, dim=1)
        out = _vae_sample(mean, scale)
        if return_info:
            return out, {}
        return out

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        return x


def _create_bottleneck_from_config(bottleneck_config: Dict[str, Any] | None) -> nn.Module | None:
    if bottleneck_config is None:
        return None
    bt_type = bottleneck_config.get("type")
    if bt_type == "vae":
        return VAEBottleneck()
    raise ValueError(f"Only bottleneck type 'vae' is supported in-repo; got {bt_type}")


def create_autoencoder_from_config(config: Dict[str, Any]) -> AudioAutoencoder:
    ae_config = config["model"]
    latent_dim = ae_config["latent_dim"]
    encoder = create_encoder_from_config(ae_config["encoder"])
    decoder = create_decoder_from_config(ae_config["decoder"])
    bottleneck = _create_bottleneck_from_config(ae_config.get("bottleneck"))
    downsampling_ratio = ae_config["downsampling_ratio"]
    io_channels = ae_config["io_channels"]
    sample_rate = config["sample_rate"]
    in_channels = ae_config.get("in_channels")
    out_channels = ae_config.get("out_channels")
    soft_clip = ae_config["decoder"].get("soft_clip", False)
    return AudioAutoencoder(
        encoder=encoder,
        decoder=decoder,
        io_channels=io_channels,
        latent_dim=latent_dim,
        downsampling_ratio=downsampling_ratio,
        sample_rate=sample_rate,
        bottleneck=bottleneck,
        pretransform=None,
        in_channels=in_channels,
        out_channels=out_channels,
        soft_clip=soft_clip,
    )
