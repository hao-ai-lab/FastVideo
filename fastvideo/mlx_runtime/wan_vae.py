# SPDX-License-Identifier: Apache-2.0
"""Wan VAE decode helpers for Apple Silicon MLX inference.

Two decode backends:

1. **TAEHV (primary / fast)** — Tiny AutoEncoder (madebyollin/taehv). Fully
   MLX-native Conv2d path. ``taew2_1.pth`` for Wan2.1 (z_dim=16),
   ``taew2_2.pth`` for Wan2.2 5B (z_dim=48, patch_size=2). Expected decode
   wall-clock ~seconds vs ~minutes for the full 3D VAE on MPS.

2. **Full AutoencoderKLWan (reference / quality)** — denormalize with
   ``latents_mean`` / ``latents_std`` then torch decode (MPS preferred). Used
   for parity gates and when TAEHV is unavailable. A pure-MLX 3D-conv port of
   the residual Wan2.2 decoder is left as follow-up (causal feat-cache +
   residual up blocks are large); TAEHV covers the product latency path.

Diffusion latents from the DiT are **not** mean/std-normalized for TAEHV
(matching ``taehv_decode.py``); full VAE decode **does** denormalize first
(matching ``mlx_wan_prompt_to_video.decode_latents_to_video``).
"""

from __future__ import annotations

import hashlib
import json
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np

GIB = 1024**3

TAEW2_1_URL = "https://raw.githubusercontent.com/madebyollin/taehv/main/taew2_1.pth"
TAEW2_2_URL = "https://raw.githubusercontent.com/madebyollin/taehv/main/taew2_2.pth"
# Validated 2026-07-02 / 2026-07-09 against upstream madebyollin/taehv.
TAEW2_1_SHA256 = "d26151e76cdc2c9424bef988de874b33d9a53f30ef3060cd556c429c469c797e"
# Leave empty to skip pin for taew2_2 until a long-lived pin is recorded (still
# downloads; user-supplied paths are never hash-checked).
TAEW2_2_SHA256 = ""

DecodeBackend = Literal["taehv", "taehv-torch", "wan-vae"]


def _cache_dir() -> Path:
    return Path.home() / ".cache" / "fastvideo" / "taehv"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_taehv_checkpoint(*, z_dim: int, checkpoint_path: Path | None = None) -> Path:
    """Return a TAEHV weight file for the given latent channel count."""
    if checkpoint_path is not None:
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"TAEHV checkpoint not found: {checkpoint_path}")
        return checkpoint_path
    if z_dim == 16:
        name, url, expect = "taew2_1.pth", TAEW2_1_URL, TAEW2_1_SHA256
    elif z_dim == 48:
        name, url, expect = "taew2_2.pth", TAEW2_2_URL, TAEW2_2_SHA256
    else:
        raise ValueError(f"No TAEHV checkpoint mapped for z_dim={z_dim} (supported: 16, 48)")
    path = _cache_dir() / name
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading {url} -> {path}")
        urllib.request.urlretrieve(url, path)  # noqa: S310 - public pinned artifact.
    if expect:
        actual = _sha256(path)
        if actual != expect:
            raise RuntimeError(f"TAEHV {path} sha256 mismatch (expected {expect}, got {actual})")
    return path


@dataclass(frozen=True)
class WanVAEConfigView:
    """Minimal config fields needed for denormalize + spatial scale."""

    z_dim: int
    latents_mean: tuple[float, ...]
    latents_std: tuple[float, ...]
    scale_factor_spatial: int = 8
    scale_factor_temporal: int = 4
    patch_size: int | None = None
    vae_dir: Path | None = None

    @classmethod
    def from_vae_dir(cls, vae_dir: Path) -> WanVAEConfigView:
        cfg = json.loads((vae_dir / "config.json").read_text())
        return cls(
            z_dim=int(cfg["z_dim"]),
            latents_mean=tuple(float(x) for x in cfg["latents_mean"]),
            latents_std=tuple(float(x) for x in cfg["latents_std"]),
            scale_factor_spatial=int(cfg.get("scale_factor_spatial", 8)),
            scale_factor_temporal=int(cfg.get("scale_factor_temporal", 4)),
            patch_size=cfg.get("patch_size"),
            vae_dir=vae_dir,
        )


def denormalize_latents_np(latents: np.ndarray, config: WanVAEConfigView) -> np.ndarray:
    """Apply ``z / (1/std) + mean`` ⇔ ``z * std + mean`` (diffusers Wan path).

    Matches ``mlx_wan_prompt_to_video``::

        latents = latents / latents_std + latents_mean
        # where latents_std was stored as 1/std in the tensor view
    """
    mean = np.asarray(config.latents_mean, dtype=np.float32).reshape(1, -1, 1, 1, 1)
    std = np.asarray(config.latents_std, dtype=np.float32).reshape(1, -1, 1, 1, 1)
    return latents.astype(np.float32) * std + mean


# ---------------------------------------------------------------------------
# MLX TAEHV decoder (Conv2d stack — primary fully-MLX product path)
# ---------------------------------------------------------------------------


def _mlx_conv2d(x: Any, weight: Any, bias: Any, *, stride: int = 1) -> Any:
    import mlx.core as mx

    # x: NCHW, weight: OIHW
    y = mx.conv2d(x.transpose(0, 2, 3, 1), weight.transpose(0, 2, 3, 1), stride=stride, padding=1)
    y = y.transpose(0, 3, 1, 2)
    if bias is not None:
        y = y + bias.reshape(1, -1, 1, 1)
    return y


def _mlx_conv2d_1x1(x: Any, weight: Any, bias: Any = None) -> Any:
    import mlx.core as mx

    y = mx.conv2d(x.transpose(0, 2, 3, 1), weight.transpose(0, 2, 3, 1), stride=1, padding=0)
    y = y.transpose(0, 3, 1, 2)
    if bias is not None:
        y = y + bias.reshape(1, -1, 1, 1)
    return y


def _load_torch_state(path: Path) -> dict[str, np.ndarray]:
    import torch

    sd = torch.load(path, map_location="cpu", weights_only=True)
    return {k: v.detach().float().cpu().numpy() for k, v in sd.items()}


class MLXTAEHVDecoder:
    """Minimal MLX port of TAEHV ``decoder`` (parallel-over-time MemBlocks)."""

    def __init__(self, checkpoint_path: Path) -> None:
        import mlx.core as mx

        self.checkpoint_path = Path(checkpoint_path)
        name = self.checkpoint_path.name
        self.patch_size = 2 if "taew2_2" in name else 1
        self.latent_channels = 48 if "taew2_2" in name else 16
        self.image_channels = 3
        self.frames_to_trim = 3  # TGrow strides (1,2,2) → 2**2 - 1 for w2.1/w2.2 defaults

        sd = _load_torch_state(self.checkpoint_path)
        # Patch TGrow kernels like upstream TAEHV.patch_tgrow_layers.
        self.weights = {k: mx.array(v) for k, v in sd.items()}
        self._n_f = [256, 128, 64, 64]

    def decode_ntchw(self, latents_ntchw: Any) -> Any:
        """Decode ``[N, T, C, H, W]`` diffusion latents → ``[N, T_out, 3, H_p, W_p]`` in ~[0,1]."""
        import mlx.core as mx

        x = latents_ntchw
        n, t, c, h, w = x.shape
        if c != self.latent_channels:
            raise ValueError(f"expected C={self.latent_channels}, got {c}")
        x = x.reshape(n * t, c, h, w)
        x = self._run_decoder_parallel(x, n=n)
        # Pixel-shuffle if patch_size > 1: (NT, 3*p*p, H, W) -> (NT, 3, H*p, W*p)
        if self.patch_size > 1:
            p = self.patch_size
            nt, c_out, hh, ww = x.shape
            x = x.reshape(nt, self.image_channels, p, p, hh, ww)
            x = x.transpose(0, 1, 4, 2, 5, 3).reshape(nt, self.image_channels, hh * p, ww * p)
        _, c_out, hh, ww = x.shape
        t_out = x.shape[0] // n
        x = x.reshape(n, t_out, c_out, hh, ww)
        if self.frames_to_trim > 0 and t_out > self.frames_to_trim:
            x = x[:, self.frames_to_trim:]
        return mx.clip(x, 0.0, 1.0)

    def _run_decoder_parallel(self, x: Any, *, n: int) -> Any:
        """Apply decoder Sequential with MemBlock temporal memory (parallel mode)."""
        import mlx.core as mx

        w = self.weights

        def memblock(base: int, xx: Any, past: Any) -> Any:
            cat = mx.concatenate([xx, past], axis=1)
            h = _mlx_conv2d(cat, w[f"decoder.{base}.conv.0.weight"], w.get(f"decoder.{base}.conv.0.bias"))
            h = mx.maximum(h, 0.0)
            h = _mlx_conv2d(h, w[f"decoder.{base}.conv.2.weight"], w.get(f"decoder.{base}.conv.2.bias"))
            h = mx.maximum(h, 0.0)
            h = _mlx_conv2d(h, w[f"decoder.{base}.conv.4.weight"], w.get(f"decoder.{base}.conv.4.bias"))
            skip_key = f"decoder.{base}.skip.weight"
            skip = _mlx_conv2d_1x1(xx, w[skip_key], None) if skip_key in w else xx
            return mx.maximum(h + skip, 0.0)

        def upsample2(xx: Any) -> Any:
            nt, c, h, ww = xx.shape
            xx = xx.reshape(nt, c, h, 1, ww, 1)
            xx = mx.broadcast_to(xx, (nt, c, h, 2, ww, 2))
            return xx.reshape(nt, c, h * 2, ww * 2)

        def tgrow(base: int, xx: Any, stride: int) -> Any:
            wt = w[f"decoder.{base}.conv.weight"]
            out_ch = int(xx.shape[1]) * stride
            if int(wt.shape[0]) > out_ch:
                wt = wt[-out_ch:]
            y = _mlx_conv2d_1x1(xx, wt, None)
            if stride == 1:
                return y
            # TGrow.forward: (NT, C*stride, H, W) -> (NT*stride, C, H, W)
            nt, c, h, ww = y.shape
            c_in = c // stride
            y = y.reshape(nt, stride, c_in, h, ww).transpose(0, 1, 2, 3, 4)
            return y.reshape(nt * stride, c_in, h, ww)

        def mem_past(xx: Any) -> Any:
            nt, c, h, ww = xx.shape
            t_cur = nt // n
            x_ = xx.reshape(n, t_cur, c, h, ww)
            # pad one zero frame at t=0, align past[t] = x[t-1]
            past = mx.concatenate([mx.zeros_like(x_[:, :1]), x_[:, :-1]], axis=1)
            return past.reshape(nt, c, h, ww)

        # 0 Clamp, 1 conv, 2 ReLU
        x = mx.tanh(x / 3.0) * 3.0
        x = _mlx_conv2d(x, w["decoder.1.weight"], w.get("decoder.1.bias"))
        x = mx.maximum(x, 0.0)
        for mem_idx in (3, 4, 5):
            x = memblock(mem_idx, x, mem_past(x))
        x = upsample2(x)
        x = tgrow(7, x, 1)
        x = _mlx_conv2d(x, w["decoder.8.weight"], w.get("decoder.8.bias"))
        for mem_idx in (9, 10, 11):
            x = memblock(mem_idx, x, mem_past(x))
        x = upsample2(x)
        x = tgrow(13, x, 2)
        x = _mlx_conv2d(x, w["decoder.14.weight"], w.get("decoder.14.bias"))
        for mem_idx in (15, 16, 17):
            x = memblock(mem_idx, x, mem_past(x))
        x = upsample2(x)
        x = tgrow(19, x, 2)
        x = _mlx_conv2d(x, w["decoder.20.weight"], w.get("decoder.20.bias"))
        x = mx.maximum(x, 0.0)
        x = _mlx_conv2d(x, w["decoder.22.weight"], w.get("decoder.22.bias"))
        return x


def decode_latents_taehv_mlx(
    latents_np: np.ndarray,
    *,
    z_dim: int | None = None,
    checkpoint_path: Path | None = None,
) -> np.ndarray:
    """Decode DiT latents ``[B,C,T,H,W]`` with MLX TAEHV → pixels ``[B,T,H,W,3]`` in [0,1]."""
    import mlx.core as mx

    if latents_np.ndim != 5:
        raise ValueError(f"expected [B,C,T,H,W], got {latents_np.shape}")
    c = latents_np.shape[1]
    z = z_dim if z_dim is not None else c
    ckpt = ensure_taehv_checkpoint(z_dim=z, checkpoint_path=checkpoint_path)
    dec = MLXTAEHVDecoder(ckpt)
    # NTCHW
    x = mx.array(latents_np.transpose(0, 2, 1, 3, 4).astype(np.float32))
    out = dec.decode_ntchw(x)  # N T C H W
    mx.eval(out)
    arr = np.array(out)
    # B T H W C
    return arr.transpose(0, 1, 3, 4, 2)


def decode_latents_wan_vae_torch(
    latents_np: np.ndarray,
    *,
    vae_dir: Path,
    device: str = "auto",
    dtype_name: str = "fp16",
) -> np.ndarray:
    """Full AutoencoderKLWan decode on torch (MPS/CPU) with mean/std denormalize.

    Returns pixels ``[B, T, H, W, 3]`` float in [0, 1].
    """
    import torch
    from diffusers import AutoencoderKLWan
    from diffusers.video_processor import VideoProcessor

    if device == "auto":
        device = "mps" if torch.backends.mps.is_available() else "cpu"
    dtype = torch.float16 if dtype_name == "fp16" and device == "mps" else torch.float32
    config = WanVAEConfigView.from_vae_dir(vae_dir)
    vae = AutoencoderKLWan.from_pretrained(vae_dir, torch_dtype=dtype, local_files_only=True).to(device)
    vae.eval()
    latents = torch.from_numpy(latents_np.astype(np.float32)).to(device=device, dtype=dtype)
    mean = torch.tensor(config.latents_mean, device=device, dtype=dtype).view(1, -1, 1, 1, 1)
    inv_std = (1.0 / torch.tensor(config.latents_std, device=device, dtype=dtype)).view(1, -1, 1, 1, 1)
    latents = latents / inv_std + mean  # matches prompt_to_video path
    with torch.no_grad():
        video = vae.decode(latents, return_dict=False)[0]
    video = VideoProcessor(vae_scale_factor=config.scale_factor_spatial).postprocess_video(video, output_type="np")
    return video  # [B, T, H, W, 3]


def decode_latents_to_video(
    latents_np: np.ndarray,
    output_path: Path,
    *,
    fps: int = 16,
    backend: DecodeBackend = "taehv",
    vae_dir: Path | None = None,
    z_dim: int | None = None,
    taehv_checkpoint: Path | None = None,
    torch_device: str = "auto",
) -> dict[str, Any]:
    """Decode and write MP4. Returns metrics dict including ``decode_s``."""
    import time

    from diffusers.utils import export_to_video

    t0 = time.perf_counter()
    if backend in ("taehv", "taehv-torch"):
        c = latents_np.shape[1] if z_dim is None else z_dim
        if backend == "taehv":
            video = decode_latents_taehv_mlx(latents_np, z_dim=c, checkpoint_path=taehv_checkpoint)
        else:
            # torch TAEHV (regression / parity reference)
            import torch
            from fastvideo.third_party.taehv import TAEHV

            ckpt = ensure_taehv_checkpoint(z_dim=c, checkpoint_path=taehv_checkpoint)
            if torch_device == "auto":
                torch_device = "mps" if torch.backends.mps.is_available() else "cpu"
            dtype = torch.float16 if torch_device == "mps" else torch.float32
            model = TAEHV(str(ckpt)).to(device=torch_device, dtype=dtype).eval()
            lat = torch.from_numpy(latents_np).to(device=torch_device, dtype=dtype)
            with torch.no_grad():
                out = model.decode_video(lat.transpose(1, 2), parallel=True, show_progress_bar=False)
            video = out[0].permute(0, 2, 3, 1).float().cpu().numpy()[None, ...]
            # out is NTCHW -> need BTHWC; decode_video returns NTCHW for batch
            if video.ndim == 4:
                video = video[None]
    elif backend == "wan-vae":
        if vae_dir is None:
            raise ValueError("vae_dir required for wan-vae backend")
        video = decode_latents_wan_vae_torch(latents_np, vae_dir=vae_dir, device=torch_device)
    else:
        raise ValueError(f"unknown backend {backend}")

    decode_s = time.perf_counter() - t0
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # export_to_video expects list/array of frames HxWxC
    frames = video[0]
    frames = np.clip(frames, 0.0, 1.0)
    export_to_video(frames, str(output_path), fps=fps)
    return {
        "decode_s": decode_s,
        "backend": backend,
        "output_path": str(output_path),
        "num_frames": int(frames.shape[0]),
        "resolution": f"{frames.shape[2]}x{frames.shape[1]}" if frames.ndim == 4 else None,
    }
