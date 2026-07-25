# SPDX-License-Identifier: Apache-2.0
"""Thread-safe, future-only handle control for causal WanTrack sessions."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import math
import threading
from typing import Any

import numpy as np


@dataclass(frozen=True, slots=True)
class HandleState:
    handle_id: str
    anchor: tuple[float, float]
    position: tuple[float, float]


@dataclass(frozen=True, slots=True)
class AppliedControl:
    revision: int
    radius: float
    active_handle_ids: tuple[str, ...]
    tracks: np.ndarray
    visibility: np.ndarray


def make_normalized_grid(size: int = 50) -> np.ndarray:
    if size <= 1:
        raise ValueError("grid size must be greater than one")
    axis = np.linspace(0.0, 1.0, size, dtype=np.float32)
    x, y = np.meshgrid(axis, axis, indexing="xy")
    return np.stack([x, y], axis=-1).reshape(-1, 2)


def cosine_radius_weights(
    points: np.ndarray,
    center: tuple[float, float],
    radius: float,
) -> np.ndarray:
    if not 0.0 < radius <= math.sqrt(2.0):
        raise ValueError("radius must be in (0, sqrt(2)]")
    distance = np.linalg.norm(
        np.asarray(points, dtype=np.float32) - np.asarray(center, dtype=np.float32),
        axis=-1,
    )
    normalized = np.clip(distance / float(radius), 0.0, 1.0)
    weights = 0.5 * (1.0 + np.cos(np.pi * normalized))
    weights[distance >= radius] = 0.0
    return weights.astype(np.float32, copy=False)


def deform_grid(
    grid: np.ndarray,
    handles: tuple[HandleState, ...],
    radius: float,
) -> np.ndarray:
    """Blend overlapping handle displacements without amplifying overlap."""
    grid = np.asarray(grid, dtype=np.float32)
    if not handles:
        return grid.copy()
    weighted_displacement = np.zeros_like(grid)
    total_weight = np.zeros((grid.shape[0], 1), dtype=np.float32)
    for handle in handles:
        weights = cosine_radius_weights(grid, handle.anchor, radius)[:, None]
        displacement = (np.asarray(handle.position, dtype=np.float32) - np.asarray(handle.anchor, dtype=np.float32))
        weighted_displacement += weights * displacement
        total_weight += weights
    # A lone handle retains its cosine falloff. Overlap is normalized once the
    # combined influence exceeds one, avoiding motion amplification.
    blended = weighted_displacement / np.maximum(1.0, total_weight)
    return np.clip(grid + blended, 0.0, 1.0).astype(np.float32, copy=False)


class StableGridController:
    """Own stable grid identities and boundary-applied control revisions."""

    def __init__(
        self,
        handles: list[dict[str, Any]],
        *,
        radius: float,
        grid_size: int = 50,
    ) -> None:
        if not handles:
            raise ValueError("at least one handle is required")
        self.grid = make_normalized_grid(grid_size)
        self.radius = self._validate_radius(radius)
        self._handles: dict[str, HandleState] = {}
        for raw in handles:
            state = self._coerce_handle(raw)
            if state.handle_id in self._handles:
                raise ValueError(f"duplicate handle id: {state.handle_id!r}")
            self._handles[state.handle_id] = state
        self._accepted_revision = 0
        self._applied_revision = 0
        self._pending: list[dict[str, Any]] = []
        self._lock = threading.RLock()

    @staticmethod
    def _validate_radius(radius: float) -> float:
        value = float(radius)
        if not 0.0 < value <= math.sqrt(2.0):
            raise ValueError("radius must be in (0, sqrt(2)]")
        return value

    @staticmethod
    def _coerce_point(raw: dict[str, Any]) -> tuple[float, float]:
        point = (float(raw["x"]), float(raw["y"]))
        if not all(math.isfinite(v) and 0.0 <= v <= 1.0 for v in point):
            raise ValueError("handle coordinates must be finite and normalized")
        return point

    @classmethod
    def _coerce_handle(cls, raw: dict[str, Any]) -> HandleState:
        handle_id = str(raw.get("id", "")).strip()
        if not handle_id:
            raise ValueError("handle id must be non-empty")
        point = cls._coerce_point(raw)
        return HandleState(handle_id, point, point)

    @property
    def accepted_revision(self) -> int:
        with self._lock:
            return self._accepted_revision

    @property
    def applied_revision(self) -> int:
        with self._lock:
            return self._applied_revision

    @property
    def active_handle_ids(self) -> tuple[str, ...]:
        with self._lock:
            return tuple(sorted(self._handles))

    def queue_revision(
        self,
        revision: int,
        *,
        samples: list[dict[str, Any]] | None = None,
        add: list[dict[str, Any]] | None = None,
        remove: list[str] | None = None,
        handles: list[dict[str, Any]] | None = None,
        radius: float | None = None,
        received_at_ms: float | None = None,
    ) -> bool:
        """Queue a monotonic revision; stale revisions are ignored."""
        revision = int(revision)
        with self._lock:
            if revision <= self._accepted_revision:
                return False
            payload = {
                "revision": revision,
                "samples": deepcopy(samples or []),
                "add": deepcopy(add or []),
                "remove": [str(item) for item in (remove or [])],
                "handles": deepcopy(handles),
                "radius": (None if radius is None else self._validate_radius(radius)),
                "received_at_ms": (None if received_at_ms is None else float(received_at_ms)),
            }
            self._pending.append(payload)
            self._accepted_revision = revision
            return True

    def render_constant(self, frame_count: int) -> AppliedControl:
        if frame_count <= 0:
            raise ValueError("frame_count must be positive")
        with self._lock:
            frame = deform_grid(
                self.grid,
                tuple(self._handles.values()),
                self.radius,
            )
            tracks = np.repeat(frame[None], frame_count, axis=0)
            visibility = np.ones(tracks.shape[:2], dtype=np.float32)
            return AppliedControl(
                revision=self._applied_revision,
                radius=self.radius,
                active_handle_ids=tuple(sorted(self._handles)),
                tracks=tracks,
                visibility=visibility,
            )

    def apply_pending(
        self,
        frame_count: int,
        *,
        interval_start_ms: float,
        interval_end_ms: float,
    ) -> AppliedControl:
        """Atomically apply queued lifecycle state and resample the interval."""
        if frame_count <= 0:
            raise ValueError("frame_count must be positive")
        if interval_end_ms < interval_start_ms:
            raise ValueError("control interval end precedes its start")
        with self._lock:
            if not self._pending:
                return self.render_constant(frame_count)
            pending = self._pending
            self._pending = []

            old_handles = dict(self._handles)
            new_handles = dict(self._handles)
            sample_map: dict[str, list[tuple[float, tuple[float, float]]]] = {}
            latest_radius = self.radius
            for payload in pending:
                if payload["radius"] is not None:
                    latest_radius = payload["radius"]
                explicit_handles = payload["handles"]
                if explicit_handles is not None:
                    desired: dict[str, HandleState] = {}
                    for raw in explicit_handles:
                        candidate = self._coerce_handle(raw)
                        previous = new_handles.get(candidate.handle_id)
                        if previous is not None:
                            candidate = HandleState(
                                candidate.handle_id,
                                previous.anchor,
                                previous.position,
                            )
                        desired[candidate.handle_id] = candidate
                    new_handles = desired
                for handle_id in payload["remove"]:
                    new_handles.pop(handle_id, None)
                for raw in payload["add"]:
                    candidate = self._coerce_handle(raw)
                    new_handles[candidate.handle_id] = candidate
                for raw in payload["samples"]:
                    handle_id = str(raw.get("id", "")).strip()
                    if not handle_id:
                        raise ValueError("control sample id must be non-empty")
                    point = self._coerce_point(raw)
                    timestamp = raw.get("timestamp_ms", raw.get("time_ms"))
                    if timestamp is None:
                        timestamp = payload["received_at_ms"]
                    if timestamp is None:
                        timestamp = interval_end_ms
                    sample_map.setdefault(handle_id, []).append((float(timestamp), point))

            frame_times = np.linspace(
                interval_start_ms,
                interval_end_ms,
                frame_count,
                dtype=np.float64,
            )
            paths: dict[str, np.ndarray] = {}
            for handle_id, state in new_handles.items():
                prior = old_handles.get(handle_id, state)
                samples = sorted(sample_map.get(handle_id, []), key=lambda item: item[0])
                times = [float(interval_start_ms)]
                points = [prior.position]
                for timestamp, point in samples:
                    timestamp = min(
                        max(timestamp, interval_start_ms),
                        interval_end_ms,
                    )
                    if timestamp == times[-1]:
                        points[-1] = point
                    else:
                        times.append(timestamp)
                        points.append(point)
                if times[-1] < interval_end_ms:
                    times.append(float(interval_end_ms))
                    points.append(points[-1])
                x = np.interp(frame_times, times, [point[0] for point in points])
                y = np.interp(frame_times, times, [point[1] for point in points])
                paths[handle_id] = np.stack([x, y], axis=-1).astype(np.float32)
                final_position = (
                    float(paths[handle_id][-1, 0]),
                    float(paths[handle_id][-1, 1]),
                )
                new_handles[handle_id] = HandleState(
                    handle_id,
                    state.anchor,
                    final_position,
                )

            tracks = np.empty(
                (frame_count, self.grid.shape[0], 2),
                dtype=np.float32,
            )
            for frame_index in range(frame_count):
                frame_handles = tuple(
                    HandleState(
                        handle_id,
                        state.anchor,
                        (
                            float(paths[handle_id][frame_index, 0]),
                            float(paths[handle_id][frame_index, 1]),
                        ),
                    ) for handle_id, state in sorted(new_handles.items()))
                tracks[frame_index] = deform_grid(
                    self.grid,
                    frame_handles,
                    latest_radius,
                )

            self._handles = new_handles
            self.radius = latest_radius
            self._applied_revision = pending[-1]["revision"]
            visibility = np.ones(tracks.shape[:2], dtype=np.float32)
            return AppliedControl(
                revision=self._applied_revision,
                radius=self.radius,
                active_handle_ids=tuple(sorted(self._handles)),
                tracks=tracks,
                visibility=visibility,
            )
