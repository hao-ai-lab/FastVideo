# SPDX-License-Identifier: Apache-2.0
"""Optional, approximate MiniMax H3 tiny decoder, executed entirely in MLX.

Architecture and temporal mapping adapted from madebyollin/taehv at
62f7591f59dfbb4c3c02b7a621d180a9eeaba26c (MIT, Ollin Boer Bohan).
See fastvideo/third_party/taehv/LICENSE. Weights remain in the upstream format.
This decoder consumes normalized diffusion latents; it does not use the full
H3 VAE's latent mean/std or pixel denormalization.
"""

from __future__ import annotations

import hashlib
import tempfile
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np

TAEH3_REVISION = "62f7591f59dfbb4c3c02b7a621d180a9eeaba26c"
TAEH3_URL = f"https://raw.githubusercontent.com/madebyollin/taehv/{TAEH3_REVISION}/safetensors/taeh3.safetensors"
TAEH3_SHA256 = "4fd022bfcab08772fe0536b17ea1a3bbb5625be11e397868d1c5d891863d4c13"


def ensure_taeh3_checkpoint(checkpoint_path: str | Path | None = None) -> Path:
    """Fetch only pinned weights, atomically; never download executable code.

    Explicit local safetensors paths may contain custom trained weights.
    Managed cache entries must match the upstream SHA-256 digest.
    """
    if checkpoint_path is not None:
        path = Path(checkpoint_path).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"TAEH3 checkpoint not found: {path}")
        if path.suffix != ".safetensors":
            raise ValueError("The MLX TAEH3 decoder requires a .safetensors checkpoint.")
        return path
    path = Path.home() / ".cache/fastvideo/taehv/taeh3.safetensors"

    def verify(candidate: Path) -> None:
        with candidate.open("rb") as handle:
            hasher = hashlib.sha256()
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
    return path


class MLXTAEH3Decoder:
    """Decode H3 NTCHW latents with bounded temporal feature memory.

    Chunks carry the last input of every MemBlock into the next chunk.
    Chunking limits activation memory without resetting the causal state.
    The full H3 VAE remains the quality reference, not a numerical oracle for
    this independently trained tiny decoder.
    """

    def __init__(self, checkpoint_path: str | Path, *, dtype: str = "fp32") -> None:
        import mlx.core as mx

        if dtype not in ("fp32", "fp16", "bf16"):
            raise ValueError(f"Unsupported TAEH3 dtype: {dtype}")
        self.dtype = {"fp32": mx.float32, "fp16": mx.float16, "bf16": mx.bfloat16}[dtype]
        raw = mx.load(str(checkpoint_path))
        expected: dict[str, tuple[int, ...]] = {
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
        for index, channels in ((3, 256), (4, 256), (5, 256), (9, 128), (10, 128), (11, 128), (15, 64), (16, 64), (17,
                                                                                                                   64)):
            for layer in (0, 2, 4):
                prefix = f"decoder.{index}.conv.{layer}"
                expected[f"{prefix}.weight"] = (channels, channels * (2 if layer == 0 else 1), 3, 3)
                expected[f"{prefix}.bias"] = (channels, )
        actual = {key for key in raw if key.startswith("decoder.")}
        if actual != set(expected):
            raise ValueError(f"TAEH3 decoder keys mismatch: missing={set(expected) - actual}, "
                             f"unexpected={actual - set(expected)}")
        self.weights = {}
        for key, shape in expected.items():
            value = raw[key]
            if tuple(value.shape) != shape:
                raise ValueError(f"TAEH3 weight {key} has shape {value.shape}, expected {shape}")
            # Keep convolutions and features channels-last throughout execution.
            value = value.transpose(0, 2, 3, 1) if value.ndim == 4 else value
            self.weights[key] = mx.contiguous(value.astype(self.dtype))
        mx.eval(*self.weights.values())

    def _conv(self, x: Any, name: str) -> Any:
        import mlx.core as mx

        weight = self.weights[f"{name}.weight"]
        y = mx.conv2d(x, weight, padding=weight.shape[1] // 2)
        bias = self.weights.get(f"{name}.bias")
        return y if bias is None else y + bias

    def _chunk(self, x: Any, memory: dict[int, Any]) -> Any:
        import mlx.core as mx

        n, t, h, w, c = x.shape
        x = mx.maximum(self._conv(mx.tanh(x.reshape(n * t, h, w, c) / 3.0) * 3.0, "decoder.1"), 0)
        for indices, grow, projection, stride in (((3, 4, 5), 7, 8, 1), ((9, 10, 11), 13, 14, 2), ((15, 16, 17), 19, 20,
                                                                                                   2)):
            for index in indices:
                _, h, w, c = x.shape
                sequence = x.reshape(n, -1, h, w, c)
                previous = memory.get(index)
                if previous is None:
                    previous = mx.zeros_like(sequence[:, :1])
                past = mx.concatenate([previous, sequence[:, :-1]], axis=1).reshape(x.shape)
                # Preserve the final feature frame for the next execution chunk.
                memory[index] = mx.contiguous(sequence[:, -1:])
                y = mx.concatenate([x, past], axis=-1)
                for layer in (0, 2, 4):
                    y = self._conv(y, f"decoder.{index}.conv.{layer}")
                    if layer != 4:
                        y = mx.maximum(y, 0)
                x = mx.maximum(x + y, 0)
            nt, h, w, c = x.shape
            x = mx.broadcast_to(x.reshape(nt, h, 1, w, 1, c), (nt, h, 2, w, 2, c))
            x = x.reshape(nt, h * 2, w * 2, c)
            x = self._conv(x, f"decoder.{grow}.conv")
            # Torch TGrow splits its channel-major output into temporal frames.
            x = x.reshape(nt, h * 2, w * 2, stride, c).transpose(0, 3, 1, 2, 4)
            x = self._conv(x.reshape(nt * stride, h * 2, w * 2, c), f"decoder.{projection}")
        x = self._conv(mx.maximum(x, 0), "decoder.22")
        nt, h, w, _ = x.shape
        # Torch pixel_shuffle stores channels as (RGB, patch_y, patch_x).
        x = x.reshape(nt, h, w, 3, 2, 2).transpose(0, 1, 4, 2, 5, 3)
        return mx.clip(x.reshape(n, -1, h * 2, w * 2, 3), 0, 1)

    def decode_ntchw(self, latents: Any, *, chunk_size: int = 5) -> Any:
        """Return NTCHW RGB in [0, 1] for H3's valid 5*k-3 latent lengths."""
        import mlx.core as mx

        if latents.ndim != 5 or latents.shape[2] != 24 or min(latents.shape) <= 0:
            raise ValueError(f"Expected nonempty NTCHW H3 latents with 24 channels, got {latents.shape}")
        if latents.shape[1] % 5 != 2:
            raise ValueError("H3 latent time must be 5*k-3, for example 2, 7, or 37.")
        if chunk_size < 1:
            raise ValueError("TAEH3 chunk_size must be positive.")
        x = latents.astype(self.dtype).transpose(0, 1, 3, 4, 2)
        memory: dict[int, Any] = {}
        frames = []
        for start in range(0, x.shape[1], chunk_size):
            decoded = self._chunk(x[:, start:start + chunk_size], memory)
            # Remove three raw frames from each five-latent (20-frame) group.
            keep = [i for i in range(decoded.shape[1]) if (start * 4 + i) % 20 >= 3]
            decoded = decoded[:, mx.array(keep, dtype=mx.int32)]
            mx.eval(decoded, *memory.values())
            frames.append(decoded)
        return mx.concatenate(frames, axis=1).transpose(0, 1, 4, 2, 3)


def decode_latents_taeh3_mlx(latents: np.ndarray,
                             *,
                             checkpoint_path: str | Path | None = None,
                             dtype: str = "fp32",
                             chunk_size: int = 5) -> np.ndarray:
    """Decode normalized NCTHW diffusion latents into NTHWC float RGB."""
    import mlx.core as mx

    if latents.ndim != 5:
        raise ValueError(f"Expected NCTHW latents, got {latents.shape}")
    checkpoint = ensure_taeh3_checkpoint(checkpoint_path)
    decoder = MLXTAEH3Decoder(checkpoint, dtype=dtype)
    output = decoder.decode_ntchw(mx.array(latents.transpose(0, 2, 1, 3, 4)), chunk_size=chunk_size)
    mx.eval(output)
    return np.asarray(output.astype(mx.float32)).transpose(0, 1, 3, 4, 2)
