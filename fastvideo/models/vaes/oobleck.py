# SPDX-License-Identifier: Apache-2.0
"""Stable Audio Open 1.0 "Oobleck" VAE.

5-stage Conv1d autoencoder with Snake activations + diagonal-Gaussian
bottleneck. Loads `stabilityai/stable-audio-open-1.0/vae/` weights
directly via `OobleckVAE.from_pretrained(...)`.

    vae = OobleckVAE.from_pretrained("stabilityai/stable-audio-open-1.0", subfolder="vae")
    waveform = vae.decode(latent)            # (B, audio_channels, samples)
    latent   = vae.encode(waveform).sample()  # or .mode()
"""
from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils import weight_norm

from fastvideo.logger import init_logger

logger = init_logger(__name__)


class Snake1d(nn.Module):
    """A 1D Snake activation with learnable per-channel alpha/beta."""

    def __init__(self, hidden_dim: int, logscale: bool = True):
        super().__init__()
        self.alpha = nn.Parameter(torch.zeros(1, hidden_dim, 1))
        self.beta = nn.Parameter(torch.zeros(1, hidden_dim, 1))
        self.alpha.requires_grad = True
        self.beta.requires_grad = True
        self.logscale = logscale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        alpha = self.alpha if not self.logscale else torch.exp(self.alpha)
        beta = self.beta if not self.logscale else torch.exp(self.beta)
        x = x.reshape(shape[0], shape[1], -1)
        x = x + (beta + 1e-9).reciprocal() * torch.sin(alpha * x).pow(2)
        return x.reshape(shape)


class OobleckResidualUnit(nn.Module):
    def __init__(self, dimension: int = 16, dilation: int = 1):
        super().__init__()
        pad = ((7 - 1) * dilation) // 2
        self.snake1 = Snake1d(dimension)
        self.conv1 = weight_norm(nn.Conv1d(
            dimension, dimension, kernel_size=7, dilation=dilation, padding=pad,
        ))
        self.snake2 = Snake1d(dimension)
        self.conv2 = weight_norm(nn.Conv1d(dimension, dimension, kernel_size=1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(self.snake1(x))
        out = self.conv2(self.snake2(out))
        pad = (x.shape[-1] - out.shape[-1]) // 2
        if pad > 0:
            x = x[..., pad:-pad]
        return x + out


class OobleckEncoderBlock(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, stride: int = 1):
        super().__init__()
        self.res_unit1 = OobleckResidualUnit(input_dim, dilation=1)
        self.res_unit2 = OobleckResidualUnit(input_dim, dilation=3)
        self.res_unit3 = OobleckResidualUnit(input_dim, dilation=9)
        self.snake1 = Snake1d(input_dim)
        self.conv1 = weight_norm(nn.Conv1d(
            input_dim, output_dim,
            kernel_size=2 * stride, stride=stride,
            padding=math.ceil(stride / 2),
        ))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.res_unit1(x)
        x = self.res_unit2(x)
        x = self.snake1(self.res_unit3(x))
        return self.conv1(x)


class OobleckDecoderBlock(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, stride: int = 1):
        super().__init__()
        self.snake1 = Snake1d(input_dim)
        self.conv_t1 = weight_norm(nn.ConvTranspose1d(
            input_dim, output_dim,
            kernel_size=2 * stride, stride=stride,
            padding=math.ceil(stride / 2),
        ))
        self.res_unit1 = OobleckResidualUnit(output_dim, dilation=1)
        self.res_unit2 = OobleckResidualUnit(output_dim, dilation=3)
        self.res_unit3 = OobleckResidualUnit(output_dim, dilation=9)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.snake1(x)
        x = self.conv_t1(x)
        x = self.res_unit1(x)
        x = self.res_unit2(x)
        return self.res_unit3(x)


class OobleckDiagonalGaussianDistribution:
    """Diagonal-Gaussian VAE posterior with `softplus(scale) + 1e-4` std."""

    def __init__(self, parameters: torch.Tensor, deterministic: bool = False):
        self.parameters = parameters
        self.mean, self.scale = parameters.chunk(2, dim=1)
        self.std = nn.functional.softplus(self.scale) + 1e-4
        self.var = self.std * self.std
        self.logvar = torch.log(self.var)
        self.deterministic = deterministic

    def sample(self, generator: torch.Generator | None = None) -> torch.Tensor:
        noise = torch.randn(
            self.mean.shape, generator=generator,
            device=self.parameters.device, dtype=self.parameters.dtype,
        )
        return self.mean + self.std * noise

    def mode(self) -> torch.Tensor:
        return self.mean


@dataclass
class OobleckDecoderOutput:
    sample: torch.Tensor


class OobleckEncoder(nn.Module):
    def __init__(
        self,
        encoder_hidden_size: int,
        audio_channels: int,
        downsampling_ratios: list[int],
        channel_multiples: list[int],
    ):
        super().__init__()
        strides = downsampling_ratios
        channel_multiples = [1] + list(channel_multiples)
        self.conv1 = weight_norm(nn.Conv1d(
            audio_channels, encoder_hidden_size, kernel_size=7, padding=3,
        ))
        self.block = nn.ModuleList([
            OobleckEncoderBlock(
                input_dim=encoder_hidden_size * channel_multiples[i],
                output_dim=encoder_hidden_size * channel_multiples[i + 1],
                stride=s,
            )
            for i, s in enumerate(strides)
        ])
        d_model = encoder_hidden_size * channel_multiples[-1]
        self.snake1 = Snake1d(d_model)
        self.conv2 = weight_norm(nn.Conv1d(
            d_model, encoder_hidden_size, kernel_size=3, padding=1,
        ))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        for m in self.block:
            x = m(x)
        x = self.snake1(x)
        return self.conv2(x)


class OobleckDecoder(nn.Module):
    def __init__(
        self,
        channels: int,
        input_channels: int,
        audio_channels: int,
        upsampling_ratios: list[int],
        channel_multiples: list[int],
    ):
        super().__init__()
        strides = upsampling_ratios
        channel_multiples = [1] + list(channel_multiples)
        self.conv1 = weight_norm(nn.Conv1d(
            input_channels, channels * channel_multiples[-1],
            kernel_size=7, padding=3,
        ))
        self.block = nn.ModuleList([
            OobleckDecoderBlock(
                input_dim=channels * channel_multiples[len(strides) - i],
                output_dim=channels * channel_multiples[len(strides) - i - 1],
                stride=s,
            )
            for i, s in enumerate(strides)
        ])
        self.snake1 = Snake1d(channels)
        self.conv2 = weight_norm(nn.Conv1d(
            channels, audio_channels, kernel_size=7, padding=3, bias=False,
        ))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        for layer in self.block:
            x = layer(x)
        x = self.snake1(x)
        return self.conv2(x)


# ---------------------------------------------------------------------------
# Top-level VAE
# ---------------------------------------------------------------------------


class OobleckVAE(nn.Module):
    """Stable Audio Open 1.0 VAE.

    Constructed either from an `OobleckVAEConfig` (the standard
    `VAELoader` path) or from explicit kwargs (back-compat for tests
    and `from_pretrained` callers).
    """

    def __init__(
        self,
        config=None,  # type: OobleckVAEConfig | None
        *,
        encoder_hidden_size: int = 128,
        downsampling_ratios: list[int] | None = None,
        channel_multiples: list[int] | None = None,
        decoder_channels: int = 128,
        decoder_input_channels: int = 64,
        audio_channels: int = 2,
        sampling_rate: int = 44100,
    ):
        super().__init__()
        if config is not None:
            arch = config.arch_config
            encoder_hidden_size = arch.encoder_hidden_size
            downsampling_ratios = list(arch.downsampling_ratios)
            channel_multiples = list(arch.channel_multiples)
            decoder_channels = arch.decoder_channels
            decoder_input_channels = arch.decoder_input_channels
            audio_channels = arch.audio_channels
            sampling_rate = arch.sampling_rate
        if downsampling_ratios is None:
            downsampling_ratios = [2, 4, 4, 8, 8]
        if channel_multiples is None:
            channel_multiples = [1, 2, 4, 8, 16]
        self.encoder_hidden_size = encoder_hidden_size
        self.downsampling_ratios = downsampling_ratios
        self.decoder_channels = decoder_channels
        self.upsampling_ratios = list(reversed(downsampling_ratios))
        self.hop_length = int(np.prod(downsampling_ratios))
        self.sampling_rate = sampling_rate
        self.audio_channels = audio_channels
        self.decoder_input_channels = decoder_input_channels

        self.encoder = OobleckEncoder(
            encoder_hidden_size=encoder_hidden_size,
            audio_channels=audio_channels,
            downsampling_ratios=downsampling_ratios,
            channel_multiples=channel_multiples,
        )
        self.decoder = OobleckDecoder(
            channels=decoder_channels,
            input_channels=decoder_input_channels,
            audio_channels=audio_channels,
            upsampling_ratios=self.upsampling_ratios,
            channel_multiples=channel_multiples,
        )

    def encode(
        self, x: torch.Tensor,
    ) -> OobleckDiagonalGaussianDistribution:
        return OobleckDiagonalGaussianDistribution(self.encoder(x))

    def decode(self, z: torch.Tensor) -> OobleckDecoderOutput:
        return OobleckDecoderOutput(sample=self.decoder(z))

    def forward(
        self, sample: torch.Tensor, sample_posterior: bool = False,
    ) -> OobleckDecoderOutput:
        posterior = self.encode(sample)
        z = posterior.sample() if sample_posterior else posterior.mode()
        return self.decode(z)

    # -------------------------------------------------------------------
    # Loader
    # -------------------------------------------------------------------
    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        *,
        subfolder: str | None = None,
        torch_dtype: torch.dtype | None = None,
    ) -> "OobleckVAE":
        """Instantiate and load weights from a Stable Audio VAE dir.

        `model_path` may be:
          * a HF repo id (e.g. `stabilityai/stable-audio-open-1.0`),
          * a local directory containing `config.json` + safetensors,
          * a local directory whose `subfolder="vae"` holds those files.

        For gated repos, the HF token is read from `HF_TOKEN` /
        `HUGGINGFACE_HUB_TOKEN` / `HF_API_KEY` (see `resolve_hf_token`).
        """
        import inspect
        from safetensors.torch import load_file

        from fastvideo.utils import resolve_hf_token

        # Resolve to a local directory.
        if os.path.isdir(model_path):
            root = model_path
        else:
            from huggingface_hub import snapshot_download
            allow = ["vae/*"] if subfolder else ["*"]
            root = snapshot_download(
                repo_id=model_path, token=resolve_hf_token(), allow_patterns=allow,
            )
        if subfolder:
            root = os.path.join(root, subfolder)
        if not os.path.isdir(root):
            raise FileNotFoundError(f"Not a directory: {root}")

        cfg_path = os.path.join(root, "config.json")
        if not os.path.isfile(cfg_path):
            raise FileNotFoundError(
                f"Expected config.json at {cfg_path}. If using a HF repo, "
                f"pass subfolder='vae'."
            )
        with open(cfg_path) as f:
            cfg = json.load(f)
        cfg_fields = {k: v for k, v in cfg.items() if not k.startswith("_")}
        # Diffusers configs commonly carry extra fields (`scaling_factor`,
        # `_diffusers_version`, ...) the bare `OobleckVAE` ctor doesn't accept.
        init_params = inspect.signature(cls.__init__).parameters
        cfg_fields = {k: v for k, v in cfg_fields.items() if k in init_params}

        model = cls(**cfg_fields)

        weights_path = os.path.join(root, "diffusion_pytorch_model.safetensors")
        if not os.path.isfile(weights_path):
            # Allow `model.safetensors` as a fallback.
            alt = os.path.join(root, "model.safetensors")
            if os.path.isfile(alt):
                weights_path = alt
            else:
                raise FileNotFoundError(
                    f"No safetensors weights under {root}. Expected "
                    f"diffusion_pytorch_model.safetensors."
                )
        state = load_file(weights_path)
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing:
            raise RuntimeError(
                f"OobleckVAE missing {len(missing)} keys from {weights_path}: "
                f"{missing[:5]}"
            )
        if unexpected:
            # Non-critical: some checkpoints embed the VAE inside a larger
            # container (e.g. `pretransform.model.*`). Log the count so
            # genuine loader regressions don't go unnoticed.
            logger.debug(
                "OobleckVAE: ignored %d unexpected keys from %s "
                "(first 3: %s)", len(unexpected), weights_path, unexpected[:3],
            )

        if torch_dtype is not None:
            model = model.to(dtype=torch_dtype)
        model.eval()
        return model


EntryClass = OobleckVAE
