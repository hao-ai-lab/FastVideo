# SPDX-License-Identifier: Apache-2.0
"""Optional, approximate MiniMax H3 tiny decoder for CUDA/CPU PyTorch.

Architecture and temporal mapping adapted from madebyollin/taehv at
62f7591f59dfbb4c3c02b7a621d180a9eeaba26c (MIT, Ollin Boer Bohan).
This decoder consumes normalized diffusion latents; it does not use the full
H3 VAE's latent mean/std or pixel denormalization.
"""

from __future__ import annotations

import hashlib
import tempfile
import urllib.request
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from fastvideo.logger import init_logger

logger = init_logger(__name__)

TAEH3_REVISION = "62f7591f59dfbb4c3c02b7a621d180a9eeaba26c"
TAEH3_URL = f"https://raw.githubusercontent.com/madebyollin/taehv/{TAEH3_REVISION}/safetensors/taeh3.safetensors"
TAEH3_SHA256 = "4fd022bfcab08772fe0536b17ea1a3bbb5625be11e397868d1c5d891863d4c13"

_EXPECTED_SHAPES: dict[str, tuple[int, ...]] = {
    "decoder.1.weight": (256, 24, 3, 3),
    "decoder.1.bias": (256, ),
    "decoder.7.conv.weight": (256, 256, 1, 1),
    "decoder.8.weight": (128, 256, 3, 3),
    "decoder.13.conv.weight": (256, 128, 1, 1),
    "decoder.14.weight": (64, 128, 3, 3),
    "decoder.19.conv.weight": (128, 64, 1, 1),
    "decoder.20.weight": (64, 64, 3, 3),
    "decoder.22.weight": (12, 64, 3, 3),
    "decoder.22.bias": (12, ),
}
for _index, _channels in ((3, 256), (4, 256), (5, 256), (9, 128), (10, 128), (11, 128), (15, 64), (16, 64), (17, 64)):
    for _layer in (0, 2, 4):
        _prefix = f"decoder.{_index}.conv.{_layer}"
        _EXPECTED_SHAPES[f"{_prefix}.weight"] = (_channels, _channels * (2 if _layer == 0 else 1), 3, 3)
        _EXPECTED_SHAPES[f"{_prefix}.bias"] = (_channels, )


def ensure_taeh3_checkpoint(checkpoint_path: str | Path | None = None) -> Path:
    """Fetch only pinned weights, atomically; never download executable code."""
    if checkpoint_path is not None:
        path = Path(checkpoint_path).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"TAEH3 checkpoint not found: {path}")
        if path.suffix != ".safetensors":
            raise ValueError("The TAEH3 decoder requires a .safetensors checkpoint.")
        return path
    path = Path.home() / ".cache/fastvideo/taehv/taeh3.safetensors"

    def verify(candidate: Path) -> None:
        hasher = hashlib.sha256()
        with candidate.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                hasher.update(chunk)
        digest = hasher.hexdigest()
        if digest != TAEH3_SHA256:
            raise RuntimeError(f"TAEH3 checkpoint failed SHA-256 verification: {candidate}")

    if path.exists():
        verify(path)
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".safetensors", delete=False) as temporary_file:
        temporary = Path(temporary_file.name)
    try:
        with urllib.request.urlopen(TAEH3_URL, timeout=60) as response, temporary.open("wb") as handle:
            while chunk := response.read(1 << 20):
                handle.write(chunk)
        verify(temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    logger.info("Cached TAEH3 checkpoint at %s", path)
    return path


class TorchTAEH3Decoder:
    """Decode H3 NTCHW latents with bounded temporal feature memory."""

    def __init__(self, checkpoint_path: str | Path, *, dtype: torch.dtype = torch.float32) -> None:
        from safetensors.torch import load_file

        raw = load_file(str(checkpoint_path))
        actual = {key for key in raw if key.startswith("decoder.")}
        if actual != set(_EXPECTED_SHAPES):
            raise ValueError(f"TAEH3 decoder keys mismatch: missing={set(_EXPECTED_SHAPES) - actual}, "
                             f"unexpected={actual - set(_EXPECTED_SHAPES)}")
        self.dtype = dtype
        self.weights: dict[str, torch.Tensor] = {}
        for key, shape in _EXPECTED_SHAPES.items():
            value = raw[key]
            if tuple(value.shape) != shape:
                raise ValueError(f"TAEH3 weight {key} has shape {tuple(value.shape)}, expected {shape}")
            self.weights[key] = value.detach().to(dtype=dtype).contiguous()

    def to(self, device: torch.device) -> "TorchTAEH3Decoder":
        self.weights = {key: value.to(device=device, non_blocking=True) for key, value in self.weights.items()}
        return self

    def _conv(self, x: torch.Tensor, name: str) -> torch.Tensor:
        weight = self.weights[f"{name}.weight"]
        bias = self.weights.get(f"{name}.bias")
        padding = weight.shape[-1] // 2
        return F.conv2d(x, weight, bias, padding=padding)

    def _chunk(self, x: torch.Tensor, memory: dict[int, torch.Tensor]) -> torch.Tensor:
        n, t, c, h, w = x.shape
        x = F.relu(self._conv(torch.tanh(x.reshape(n * t, c, h, w) / 3.0) * 3.0, "decoder.1"))
        for indices, grow, projection, stride in (((3, 4, 5), 7, 8, 1), ((9, 10, 11), 13, 14, 2), ((15, 16, 17), 19, 20,
                                                                                                   2)):
            for index in indices:
                nt, c, h, w = x.shape
                sequence = x.reshape(n, -1, c, h, w)
                previous = memory.get(index)
                if previous is None:
                    previous = torch.zeros_like(sequence[:, :1])
                past = torch.cat([previous, sequence[:, :-1]], dim=1).reshape_as(x)
                memory[index] = sequence[:, -1:].contiguous()
                y = torch.cat([x, past], dim=1)
                for layer in (0, 2, 4):
                    y = self._conv(y, f"decoder.{index}.conv.{layer}")
                    if layer != 4:
                        y = F.relu(y)
                x = F.relu(x + y)
            nt, c_pre, h, w = x.shape
            x = F.interpolate(x, scale_factor=2, mode="nearest")
            x = self._conv(x, f"decoder.{grow}.conv")
            x = x.reshape(nt, stride, c_pre, h * 2, w * 2).reshape(nt * stride, c_pre, h * 2, w * 2)
            x = self._conv(x, f"decoder.{projection}")
        x = self._conv(F.relu(x), "decoder.22")
        x = F.pixel_shuffle(x, 2).clamp(0, 1)
        nt, c, h, w = x.shape
        return x.reshape(n, -1, c, h, w)

    def decode_ntchw(self, latents: torch.Tensor, *, chunk_size: int = 5) -> torch.Tensor:
        """Return NTCHW RGB in [0, 1] for H3's valid 5*k-3 latent lengths."""
        if latents.ndim != 5 or latents.shape[2] != 24 or min(latents.shape) <= 0:
            raise ValueError(f"Expected nonempty NTCHW H3 latents with 24 channels, got {tuple(latents.shape)}")
        if latents.shape[1] % 5 != 2:
            raise ValueError("H3 latent time must be 5*k-3, for example 2, 7, or 37.")
        if chunk_size < 1:
            raise ValueError("TAEH3 chunk_size must be positive.")
        x = latents.to(dtype=self.dtype)
        memory: dict[int, torch.Tensor] = {}
        frames: list[torch.Tensor] = []
        for start in range(0, x.shape[1], chunk_size):
            decoded = self._chunk(x[:, start:start + chunk_size], memory)
            keep = [i for i in range(decoded.shape[1]) if (start * 4 + i) % 20 >= 3]
            frames.append(decoded[:, keep])
        return torch.cat(frames, dim=1)


_DECODER_CACHE: dict[tuple[str, str, str], TorchTAEH3Decoder] = {}


def decode_ncthw_latents_taeh3(
    latents: torch.Tensor,
    *,
    device: torch.device,
    checkpoint_path: str | Path | None = None,
    chunk_size: int = 5,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Decode normalized NCTHW diffusion latents into NCTHW RGB in [0, 1]."""
    if latents.ndim != 5:
        raise ValueError(f"Expected NCTHW latents, got {tuple(latents.shape)}")
    checkpoint = ensure_taeh3_checkpoint(checkpoint_path)
    cache_key = (str(checkpoint), str(device), str(dtype))
    decoder = _DECODER_CACHE.get(cache_key)
    if decoder is None:
        decoder = TorchTAEH3Decoder(checkpoint, dtype=dtype).to(device)
        _DECODER_CACHE[cache_key] = decoder
    ntchw = latents.to(device=device, dtype=dtype).permute(0, 2, 1, 3, 4).contiguous()
    rgb = decoder.decode_ntchw(ntchw, chunk_size=chunk_size)
    return rgb.permute(0, 2, 1, 3, 4).contiguous()


def taeh3_decoded_pixel_shape(latent_shape: tuple[int, ...] | torch.Size) -> tuple[int, int, int, int, int]:
    """Return NCTHW pixel shape for H3 TAEH3 (16x spatial, drop 3 of every 20 raw frames)."""
    if len(latent_shape) != 5:
        raise ValueError(f"MiniMax-H3 latents must be five-dimensional, got shape {tuple(latent_shape)}.")
    batch, channels, latent_frames, latent_height, latent_width = map(int, latent_shape)
    if channels != 24:
        raise ValueError(f"TAEH3 latents must have 24 channels, got {channels}.")
    if latent_frames % 5 != 2:
        raise ValueError(f"H3 latent time must be 5*k-3, got {latent_frames}.")
    raw_frames = latent_frames * 4
    kept = sum(1 for index in range(raw_frames) if index % 20 >= 3)
    return (batch, 3, kept, latent_height * 16, latent_width * 16)
