# SPDX-License-Identifier: Apache-2.0
"""Small MLX RIFE wrapper for frame interpolation experiments.

The backend is the Apple-Silicon-native ``rife-mlx`` package, using the
``mlx-community/RIFE-4.25`` weights. Frames are HWC RGB ``uint8`` arrays.
"""

from __future__ import annotations

from collections.abc import Iterable
from functools import lru_cache

import numpy as np
from huggingface_hub.utils import LocalEntryNotFoundError


class RIFEBackendError(RuntimeError):
    """Raised when the MLX RIFE backend cannot be loaded or run."""


class RIFEWeightsUnavailableError(RIFEBackendError):
    """Raised when uncached RIFE weights cannot be downloaded."""


def aligned_keyframe_count(target_frames: int, factor: int, temporal_compression: int = 4) -> int:
    """Return the smallest VAE-aligned keyframe count that RIFE can expand to the target."""
    if target_frames < 1:
        raise ValueError(f"target_frames must be >= 1, got {target_frames}")
    if factor < 1:
        raise ValueError(f"factor must be >= 1, got {factor}")
    if temporal_compression < 1:
        raise ValueError(f"temporal_compression must be >= 1, got {temporal_compression}")
    required_intervals = (target_frames - 1 + factor - 1) // factor
    aligned_intervals = ((required_intervals + temporal_compression - 1) // temporal_compression * temporal_compression)
    return aligned_intervals + 1


def _require_hwc_rgb(frame: np.ndarray, index: int) -> np.ndarray:
    array = np.asarray(frame)
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError(f"frame {index} must have shape HxWx3, got {array.shape}")
    if array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(array)


@lru_cache(maxsize=2)
def load_model(version: str = "4.25", weights_dir: str | None = None):
    """Load the MLX-native RIFE model.

    ``weights_dir`` is passed through to ``build_model`` in the vendored ``rife_mlx``.
    When it is ``None``, the package downloads/uses the Hugging Face
    ``mlx-community/RIFE-4.25`` snapshot.
    """
    try:
        from fastvideo.third_party.rife_mlx.utils.weights import build_model
    except ImportError:
        # Fall back to a separately installed upstream package, for anyone who
        # already has one in the environment.
        try:
            from rife_mlx.utils.weights import build_model
        except ImportError as exc:
            raise RIFEBackendError("MLX RIFE backend is unavailable. It ships vendored under "
                                   "fastvideo/third_party/rife_mlx, so this usually means MLX "
                                   "itself is missing: install with `uv pip install -e '.[mlx]'`.") from exc

    try:
        return build_model(version, weights_dir=weights_dir)
    except LocalEntryNotFoundError as exc:
        raise RIFEWeightsUnavailableError(f"MLX RIFE {version} weights are unavailable: {exc}") from exc
    except Exception as exc:  # noqa: BLE001 - preserve exact backend failure.
        raise RIFEBackendError(f"Failed to load MLX RIFE {version}: {exc}") from exc


def interpolate_pair(
    frame_a: np.ndarray,
    frame_b: np.ndarray,
    timestep: float = 0.5,
    *,
    model=None,
    scale: float = 1.0,
) -> np.ndarray:
    """Interpolate one RGB frame between two input RGB frames."""
    if not 0.0 < timestep < 1.0:
        raise ValueError(f"timestep must be inside (0, 1), got {timestep}")
    img0 = _require_hwc_rgb(frame_a, 0)
    img1 = _require_hwc_rgb(frame_b, 1)
    if img0.shape != img1.shape:
        raise ValueError(f"frame shapes must match, got {img0.shape} and {img1.shape}")

    if model is None:
        model = load_model()
    try:
        try:
            from fastvideo.third_party.rife_mlx.pipeline_mlx import interpolate_pair as _interpolate_pair
        except ImportError:
            from rife_mlx.pipeline_mlx import interpolate_pair as _interpolate_pair

        return _interpolate_pair(model, img0, img1, timestep=timestep, scale=scale)
    except Exception as exc:  # noqa: BLE001 - preserve exact backend failure.
        raise RIFEBackendError(f"MLX RIFE interpolation failed at timestep={timestep}: {exc}") from exc


def interpolate(
    frames: list[np.ndarray] | Iterable[np.ndarray],
    factor: int = 2,
    *,
    model=None,
    scale: float = 1.0,
) -> list[np.ndarray]:
    """Return an Nx interpolated frame list.

    For ``len(frames)=41`` and ``factor=2``, the output length is 81:
    ``(41 - 1) * 2 + 1``. Original keyframes are preserved in order and RIFE
    fills ``factor - 1`` intermediate timesteps between each adjacent pair.
    """
    frame_list = [_require_hwc_rgb(frame, idx) for idx, frame in enumerate(frames)]
    if factor < 1:
        raise ValueError(f"factor must be >= 1, got {factor}")
    if len(frame_list) < 2 or factor == 1:
        return [frame.copy() for frame in frame_list]

    first_shape = frame_list[0].shape
    for idx, frame in enumerate(frame_list[1:], start=1):
        if frame.shape != first_shape:
            raise ValueError(f"all frames must have the same shape; frame 0={first_shape}, frame {idx}={frame.shape}")

    if model is None:
        model = load_model()

    out: list[np.ndarray] = []
    for left, right in zip(frame_list[:-1], frame_list[1:], strict=True):
        out.append(left)
        for step in range(1, factor):
            out.append(interpolate_pair(left, right, step / factor, model=model, scale=scale))
    out.append(frame_list[-1])
    return out
