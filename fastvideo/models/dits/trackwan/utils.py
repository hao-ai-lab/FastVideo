# SPDX-License-Identifier: Apache-2.0
"""Helpers for WanTrack inference demos."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch


def create_track_presets(
    num_frames: int,
    num_tracks: int = 8,
    seed: int = 0,
) -> dict[str, torch.Tensor]:
    """Build simple linear tracks in normalized ``[0, 1]`` coordinates.

    Returns unbatched tensors:
    ``track_points`` ``[T, N, 2]``, ``track_visibility`` ``[T, N]``,
    ``track_ids`` ``[N]``.
    """
    if num_frames < 1 or num_tracks < 1:
        raise ValueError("num_frames and num_tracks must be positive")

    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    starts = torch.rand(num_tracks, 2, generator=generator)
    ends = torch.rand(num_tracks, 2, generator=generator)
    alphas = torch.linspace(0.0, 1.0, num_frames).view(num_frames, 1, 1)
    points = starts.unsqueeze(0) * (1.0 - alphas) + ends.unsqueeze(0) * alphas
    visibility = torch.ones(num_frames, num_tracks, dtype=torch.float32)
    track_ids = torch.arange(num_tracks, dtype=torch.long)
    return {
        "track_points": points.to(torch.float32),
        "track_visibility": visibility,
        "track_ids": track_ids,
    }


def _as_tensor(value: Any, *, dtype: torch.dtype | None = None) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        tensor = value.detach().cpu()
    else:
        tensor = torch.as_tensor(np.asarray(value))
    if dtype is not None:
        tensor = tensor.to(dtype=dtype)
    return tensor


def _ensure_batch(track_points: torch.Tensor, track_visibility: torch.Tensor,
                  track_ids: torch.Tensor | None) -> dict[str, torch.Tensor]:
    """Normalize shapes to ``[B, T, N, 2]`` / ``[B, T, N]`` / ``[B, N]``."""
    points = _as_tensor(track_points, dtype=torch.float32)
    visibility = _as_tensor(track_visibility, dtype=torch.float32)

    if points.ndim == 3:
        points = points.unsqueeze(0)
    if visibility.ndim == 2:
        visibility = visibility.unsqueeze(0)
    if points.ndim != 4 or points.shape[-1] != 2:
        raise ValueError(f"track_points must be [B,T,N,2] or [T,N,2], got {tuple(points.shape)}")
    if visibility.shape != points.shape[:-1]:
        raise ValueError("track_visibility shape must match track_points[:-1], got "
                         f"{tuple(visibility.shape)} vs {tuple(points.shape)}")

    batch_size, _t, num_tracks = points.shape[:3]
    if track_ids is None:
        ids = torch.arange(num_tracks, dtype=torch.long).unsqueeze(0).expand(batch_size, -1).contiguous()
    else:
        ids = _as_tensor(track_ids, dtype=torch.long)
        if ids.ndim == 1:
            ids = ids.unsqueeze(0)
        if ids.shape != (batch_size, num_tracks):
            raise ValueError(f"track_ids must be [B,N] or [N], got {tuple(ids.shape)}")

    return {
        "track_points": points,
        "track_visibility": visibility,
        "track_ids": ids,
    }


def _load_array_file(path: Path) -> Any:
    suffix = path.suffix.lower()
    if suffix in {".pt", ".pth"}:
        return torch.load(path, map_location="cpu", weights_only=False)
    if suffix == ".npz":
        return np.load(path, allow_pickle=True)
    if suffix == ".npy":
        return np.load(path)
    raise ValueError(f"Unsupported track file type: {path} (use .pt/.pth/.npz/.npy)")


def load_tracks(
    *,
    tracks_path: str | Path | None = None,
    track_points_path: str | Path | None = None,
    track_visibility_path: str | Path | None = None,
    track_ids_path: str | Path | None = None,
    num_frames: int | None = None,
    num_tracks: int = 8,
    seed: int = 0,
) -> dict[str, torch.Tensor]:
    """Load track tensors from disk, or synthesize a demo preset.

    Accepted inputs:
    - ``tracks_path``: ``.pt``/``.npz`` dict with ``track_points`` +
      ``track_visibility`` (optional ``track_ids``), or a directory containing
      ``track_points.*`` / ``track_visibility.*`` / ``track_ids.*``
    - or the three individual file paths
    - if nothing is provided, ``create_track_presets`` is used
    """
    if tracks_path is not None:
        path = Path(tracks_path)
        if path.is_dir():
            def _find(stem: str) -> Path | None:
                for suffix in (".pt", ".pth", ".npy", ".npz"):
                    candidate = path / f"{stem}{suffix}"
                    if candidate.exists():
                        return candidate
                return None

            points_file = _find("track_points")
            vis_file = _find("track_visibility")
            ids_file = _find("track_ids")
            if points_file is None or vis_file is None:
                raise FileNotFoundError(
                    f"{path} must contain track_points.(pt|npy|npz) and "
                    "track_visibility.(pt|npy|npz)")
            return load_tracks(
                track_points_path=points_file,
                track_visibility_path=vis_file,
                track_ids_path=ids_file,
                num_frames=num_frames,
            )

        payload = _load_array_file(path)
        if isinstance(payload, np.lib.npyio.NpzFile):
            payload = {key: payload[key] for key in payload.files}
        if not isinstance(payload, dict):
            raise ValueError(f"{path} must contain a dict with track_points/track_visibility")
        points = payload.get("track_points")
        visibility = payload.get("track_visibility")
        if points is None or visibility is None:
            raise ValueError(f"{path} missing required keys track_points/track_visibility")
        tracks = _ensure_batch(points, visibility, payload.get("track_ids"))
    elif track_points_path is not None and track_visibility_path is not None:
        points = _load_array_file(Path(track_points_path))
        visibility = _load_array_file(Path(track_visibility_path))
        if isinstance(points, dict):
            points = points["track_points"] if "track_points" in points else next(iter(points.values()))
        if isinstance(visibility, dict):
            visibility = (visibility["track_visibility"]
                          if "track_visibility" in visibility else next(iter(visibility.values())))
        ids = None
        if track_ids_path is not None:
            ids = _load_array_file(Path(track_ids_path))
            if isinstance(ids, dict):
                ids = ids["track_ids"] if "track_ids" in ids else next(iter(ids.values()))
        tracks = _ensure_batch(points, visibility, ids)
    elif track_points_path is None and track_visibility_path is None and track_ids_path is None:
        if num_frames is None:
            raise ValueError("num_frames is required when synthesizing demo tracks")
        raw = create_track_presets(num_frames, num_tracks=num_tracks, seed=seed)
        tracks = _ensure_batch(raw["track_points"], raw["track_visibility"], raw["track_ids"])
    else:
        raise ValueError("Provide --tracks, or both --track-points and --track-visibility")

    if num_frames is not None:
        tracks["track_points"] = tracks["track_points"][:, :num_frames]
        tracks["track_visibility"] = tracks["track_visibility"][:, :num_frames]
    return tracks
