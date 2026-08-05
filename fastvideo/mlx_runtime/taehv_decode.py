# SPDX-License-Identifier: Apache-2.0
"""Optional TAEHV decode helpers for Apple Silicon FastWan experiments.

The TAEHV module itself is vendored at ``fastvideo/third_party/taehv`` (MIT,
madebyollin/taehv), so no source code is downloaded or executed at runtime.
Only the ``taew2_1.pth`` checkpoint is fetched on demand, and its sha256 is
verified before use.
"""

from __future__ import annotations

import hashlib
import importlib.util
import urllib.request
from pathlib import Path

import numpy as np

TAEW2_1_CHECKPOINT_URL = "https://raw.githubusercontent.com/madebyollin/taehv/main/taew2_1.pth"
# sha256 of the upstream taew2_1.pth this module was validated against
# (fetched 2026-07-02). If upstream publishes a new checkpoint, revalidate the
# decode path and update this pin.
TAEW2_1_CHECKPOINT_SHA256 = "d26151e76cdc2c9424bef988de874b33d9a53f30ef3060cd556c429c469c797e"
# Wan2.2 5B (z_dim=48) — see madebyollin/taehv taew2_2.pth; prefer
# ``fastvideo.mlx_runtime.wan_vae.ensure_taehv_checkpoint(z_dim=48)`` for new code.
TAEW2_2_CHECKPOINT_URL = "https://raw.githubusercontent.com/madebyollin/taehv/main/taew2_2.pth"


def _default_cache_dir() -> Path:
    return Path.home() / ".cache" / "fastvideo" / "taehv"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_checkpoint(path: Path) -> None:
    actual = _sha256(path)
    if actual != TAEW2_1_CHECKPOINT_SHA256:
        raise RuntimeError(f"TAEHV checkpoint at {path} failed sha256 verification "
                           f"(expected {TAEW2_1_CHECKPOINT_SHA256}, got {actual}). "
                           "Delete the file to re-download it, or pass --taehv-checkpoint-path "
                           "pointing at a checkpoint you trust.")


def ensure_taew2_1_checkpoint(checkpoint_path: Path | None = None) -> Path:
    """Return a verified ``taew2_1.pth``, downloading into the cache if needed.

    A caller-supplied ``checkpoint_path`` is treated as trusted user input and
    is not hash-checked (it may legitimately be a newer or retrained decoder);
    only the checkpoint this module downloads itself is pinned.
    """
    if checkpoint_path is not None:
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"TAEHV checkpoint not found: {checkpoint_path}")
        return checkpoint_path

    checkpoint_path = _default_cache_dir() / "taew2_1.pth"
    if not checkpoint_path.exists():
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading {TAEW2_1_CHECKPOINT_URL} -> {checkpoint_path}")
        urllib.request.urlretrieve(TAEW2_1_CHECKPOINT_URL,
                                   checkpoint_path)  # noqa: S310 - pinned public artifact, hash-verified below.
    _verify_checkpoint(checkpoint_path)
    return checkpoint_path


def _load_taehv_class(source_path: Path | None):
    if source_path is None:
        from fastvideo.third_party.taehv import TAEHV

        return TAEHV
    # Explicit local override for experimenting with a modified TAEHV; this is
    # a user-supplied file on disk, never something this module downloads.
    spec = importlib.util.spec_from_file_location("fastvideo_external_taehv", source_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load TAEHV source from {source_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.TAEHV


def decode_latents_to_video_taehv(
    *,
    latents_np: np.ndarray,
    output_path: Path,
    fps: int,
    device,
    dtype,
    parallel: bool,
    source_path: Path | None = None,
    checkpoint_path: Path | None = None,
) -> None:
    """Decode Wan/FastWan diffusion latents with TAEW2.1 and export MP4.

    TAEHV's Wan wrapper expects the diffusion latents directly, without applying
    the standard Wan VAE's `latents_mean` / `latents_std` shift.
    """
    import torch
    from diffusers.utils import export_to_video

    checkpoint_path = ensure_taew2_1_checkpoint(checkpoint_path)
    TAEHV = _load_taehv_class(source_path)
    taehv = TAEHV(str(checkpoint_path)).to(device=device, dtype=dtype)
    taehv.eval()

    latents = torch.from_numpy(latents_np).to(device=device, dtype=dtype)
    with torch.no_grad():
        video_ntchw = taehv.decode_video(
            latents.transpose(1, 2),
            parallel=parallel,
            show_progress_bar=False,
        )
    video = video_ntchw.transpose(1, 2)
    video_np = video[0].permute(1, 2, 3, 0).float().cpu().numpy()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    export_to_video(video_np, str(output_path), fps=fps)
