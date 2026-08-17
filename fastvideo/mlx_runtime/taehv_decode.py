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
TAEW2_2_CHECKPOINT_URL = ("https://raw.githubusercontent.com/madebyollin/taehv/"
                          "563f40bdc820ed86bcad72ea515ee48f06bd22ec/taew2_2.pth")


def _default_cache_dir() -> Path:
    """Return the default directory used to cache TAEHV checkpoints.

    Returns:
        Path: The TAEHV checkpoint cache directory under the user's home directory.
    """
    return Path.home() / ".cache" / "fastvideo" / "taehv"


def _sha256(path: Path) -> str:
    """
    Compute the SHA-256 digest of a file.

    Parameters:
        path (Path): The file whose contents are hashed.

    Returns:
        str: The file's SHA-256 digest in hexadecimal form.
    """
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_checkpoint(path: Path) -> None:
    """Verify that a TAEW2.1 checkpoint matches the expected SHA-256 digest.

    Parameters:
        path (Path): Path to the checkpoint file.

    Raises:
        RuntimeError: If the checkpoint digest does not match the expected value.
    """
    actual = _sha256(path)
    if actual != TAEW2_1_CHECKPOINT_SHA256:
        raise RuntimeError(f"TAEHV checkpoint at {path} failed sha256 verification "
                           f"(expected {TAEW2_1_CHECKPOINT_SHA256}, got {actual}). "
                           "Delete the file to re-download it, or pass --taehv-checkpoint-path "
                           "pointing at a checkpoint you trust.")


def ensure_taew2_1_checkpoint(checkpoint_path: Path | None = None) -> Path:
    """
    Ensure the TAEW2.1 checkpoint is available locally.

    A caller-provided path is treated as trusted and is only checked for existence.
    The module-managed cached checkpoint is verified against the pinned SHA-256 digest
    after downloading or before reuse.

    Parameters:
        checkpoint_path (Path | None): Optional path to a caller-provided checkpoint.

    Returns:
        Path: The available checkpoint path.

    Raises:
        FileNotFoundError: If a caller-provided checkpoint does not exist.
        RuntimeError: If a module-managed checkpoint fails verification.
    """
    if checkpoint_path is not None:
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"TAEHV checkpoint not found: {checkpoint_path}")
        return checkpoint_path

    checkpoint_path = _default_cache_dir() / "taew2_1.pth"
    if not checkpoint_path.exists():
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading {TAEW2_1_CHECKPOINT_URL} -> {checkpoint_path}")
        import socket
        import tempfile
        # Download to a temporary file, verify, then atomically rename.
        with tempfile.NamedTemporaryFile(
                mode="wb",
                dir=checkpoint_path.parent,
                prefix=".tmp_taew2_1_",
                suffix=".pth",
                delete=False,
        ) as tmp_file:
            tmp_path = Path(tmp_file.name)
            try:
                old_timeout = socket.getdefaulttimeout()
                socket.setdefaulttimeout(300)
                try:
                    urllib.request.urlretrieve(
                        TAEW2_1_CHECKPOINT_URL,
                        tmp_path,  # noqa: S310 - pinned public artifact, hash-verified below.
                    )
                finally:
                    socket.setdefaulttimeout(old_timeout)
                _verify_checkpoint(tmp_path)
                tmp_path.replace(checkpoint_path)
            except Exception:
                tmp_path.unlink(missing_ok=True)
                raise
    else:
        _verify_checkpoint(checkpoint_path)
    return checkpoint_path


def _load_taehv_class(source_path: Path | None):
    """Load the TAEHV class from the vendored implementation or a local source override.

    Parameters:
        source_path (Path | None): Path to a local Python file defining `TAEHV`; `None` selects the vendored implementation.

    Returns:
        The loaded `TAEHV` class.

    Raises:
        RuntimeError: If the specified source cannot be loaded.
    """
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
