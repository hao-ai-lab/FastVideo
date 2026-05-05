# SPDX-License-Identifier: Apache-2.0
"""Health, readiness, and status routes for streaming servers."""
from __future__ import annotations

from collections.abc import Callable, Mapping
from datetime import datetime, timezone
from typing import Any, Protocol, TypeGuard, TypeAlias

from fastapi import APIRouter

from fastvideo.entrypoints.streaming.gpu_pool import PoolHealth

SERVICE_NAME = "ltx2-streaming-backend"


class _StatusPool(Protocol):

    def get_status(self) -> Mapping[str, Any]:
        ...


class _HealthPool(Protocol):

    def health(self) -> PoolHealth:
        ...


_PoolLike: TypeAlias = _StatusPool | _HealthPool
PoolRef: TypeAlias = _PoolLike | Callable[[], _PoolLike | None] | None


def build_health_router(pool: PoolRef = None) -> APIRouter:
    """Build a router exposing streaming liveness/readiness endpoints.

    ``pool`` may be either a concrete pool or a zero-argument callable that
    returns the current pool. The callable form lets product servers keep their
    own lifespan-managed runtime singleton without adding public global state.
    """
    router = APIRouter()

    @router.get("/health")
    @router.get("/healthz")
    async def get_healthz() -> dict[str, Any]:
        """Liveness probe for process-level health."""
        return {
            "status": "ok",
            "service": SERVICE_NAME,
            "ts": _utc_now_iso(),
        }

    @router.get("/readyz")
    async def get_readyz() -> dict[str, Any]:
        """Readiness probe for router/load-balancer health checks."""
        status_payload = await get_pool_status(pool)
        ready_workers = _ready_worker_count(status_payload)
        return {
            "status": "ready" if ready_workers > 0 else "warming",
            "service": SERVICE_NAME,
            "ready_gpu_workers": ready_workers,
            "total_gpus": _as_int(status_payload.get("total_gpus")),
            "available_gpus": _as_int(status_payload.get("available_gpus")),
            "warmup_successful_gpus": _as_int(status_payload.get("warmup_successful_gpus")),
            "warmup_failed_gpus": _as_int(status_payload.get("warmup_failed_gpus")),
            "queue_size": _as_int(status_payload.get("queue_size")),
            "ts": _utc_now_iso(),
        }

    @router.get("/status")
    async def get_status() -> dict[str, Any]:
        """Get the current status of the GPU pool."""
        return await get_pool_status(pool)

    return router


async def get_pool_status(pool: PoolRef = None) -> dict[str, Any]:
    """Return the generic GPU pool status payload used by ``/status``."""
    resolved = _resolve_pool(pool)
    if resolved is None:
        return _zero_pool_status()
    if _has_get_status(resolved):
        return dict(resolved.get_status())
    if _has_health(resolved):
        return _status_from_health(resolved.health())
    return _zero_pool_status()


def _resolve_pool(pool: PoolRef) -> _PoolLike | None:
    if callable(pool):
        return pool()
    return pool


def _has_get_status(pool: _PoolLike) -> TypeGuard[_StatusPool]:
    return callable(getattr(pool, "get_status", None))


def _has_health(pool: _PoolLike) -> TypeGuard[_HealthPool]:
    return callable(getattr(pool, "health", None))


def _zero_pool_status() -> dict[str, Any]:
    return {
        "total_gpus": 0,
        "available_gpus": 0,
        "queue_size": 0,
        "warmup_enabled": False,
        "warmup_successful_gpus": 0,
        "warmup_failed_gpus": 0,
        "gpu_status": {},
    }


def _status_from_health(health: PoolHealth) -> dict[str, Any]:
    total_workers = max(0, int(health.total_workers))
    available_workers = max(0, min(int(health.available_workers), total_workers))
    active_sessions = max(0, int(health.active_sessions))
    gpu_status: dict[int, dict[str, Any]] = {}
    remaining_active_sessions = active_sessions
    for gpu_id in range(total_workers):
        is_available = gpu_id < available_workers
        client_count = 0 if is_available or remaining_active_sessions <= 0 else 1
        remaining_active_sessions -= client_count
        gpu_status[gpu_id] = {
            "ready": True,
            "available": is_available,
            "client_count": client_count,
            "current_model_id": None,
            "process_alive": True,
            "warmup_enabled": False,
            "warmup_success": True,
            "warmup_error": None,
            "warmup_timings": {},
        }
    return {
        "total_gpus": total_workers,
        "available_gpus": available_workers,
        "queue_size": max(0, int(health.queued_sessions)),
        "warmup_enabled": False,
        "warmup_successful_gpus": total_workers,
        "warmup_failed_gpus": 0,
        "gpu_status": gpu_status,
    }


def _ready_worker_count(status_payload: Mapping[str, Any]) -> int:
    gpu_status = status_payload.get("gpu_status")
    if isinstance(gpu_status, Mapping):
        ready_workers = 0
        for value in gpu_status.values():
            if not isinstance(value, Mapping):
                continue
            ready = bool(value.get("ready"))
            process_alive = bool(value.get("process_alive", True))
            if ready and process_alive:
                ready_workers += 1
        if ready_workers > 0:
            return ready_workers
    successful = _as_int(status_payload.get("warmup_successful_gpus"))
    if successful > 0:
        return successful
    total = _as_int(status_payload.get("total_gpus"))
    failed = _as_int(status_payload.get("warmup_failed_gpus"))
    if total > 0 and failed < total:
        return total - failed
    return 0


def _as_int(value: Any) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    return 0


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


__all__ = [
    "SERVICE_NAME",
    "build_health_router",
    "get_pool_status",
]
