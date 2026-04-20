# SPDX-License-Identifier: Apache-2.0
"""Replica registry + health-check loop.

The registry tracks the set of known backend replicas and their live
health. The router consults it for "pick a backend for this session"
decisions and a background task updates it from periodic HTTP probes.

State machine per replica::

    HEALTHY ──(N consecutive failures)──▶ UNHEALTHY
       ▲                                     │
       └──────(M consecutive successes)──────┘

Where N = :attr:`RouterConfig.failure_threshold` and
M = :attr:`RouterConfig.recovery_threshold`.
"""
from __future__ import annotations

import asyncio
import enum
import time
from dataclasses import dataclass, field
from typing import Any

from fastvideo.entrypoints.streaming.router.config import (
    ReplicaEndpoint,
    RouterConfig,
)
from fastvideo.logger import init_logger

HttpProbe = Any
"""Structural alias for health-probe callables. Concrete signature is
``async def __call__(url: str, *, timeout: float) -> tuple[float,
str | None]``; typing.Callable cannot express keyword-only parameters,
so duck-typing is the pragmatic compromise."""

logger = init_logger(__name__)


class ReplicaStatus(enum.Enum):
    UNKNOWN = "unknown"
    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"


@dataclass
class ReplicaHealth:
    status: ReplicaStatus = ReplicaStatus.UNKNOWN
    last_ok_at: float | None = None
    last_failure_at: float | None = None
    consecutive_failures: int = 0
    consecutive_successes: int = 0
    last_latency_ms: float | None = None


@dataclass
class Replica:
    endpoint: ReplicaEndpoint
    health: ReplicaHealth = field(default_factory=ReplicaHealth)

    @property
    def url(self) -> str:
        return self.endpoint.url

    @property
    def primary(self) -> bool:
        return self.endpoint.primary

    @property
    def is_healthy(self) -> bool:
        return self.health.status is ReplicaStatus.HEALTHY


class ReplicaRegistry:
    """Stateful map of replica URL → :class:`Replica`.

    Selection favors primary replicas when healthy; otherwise the first
    healthy non-primary is returned. When none are healthy, the
    registry returns ``None`` so the router can reject incoming
    sessions with ``gpu_unavailable``.
    """

    def __init__(self, replicas: list[ReplicaEndpoint]) -> None:
        if not replicas:
            raise ValueError("ReplicaRegistry requires at least one replica")
        self._replicas: dict[str, Replica] = {endpoint.url: Replica(endpoint=endpoint) for endpoint in replicas}
        self._lock = asyncio.Lock()

    def all(self) -> list[Replica]:
        return list(self._replicas.values())

    def get(self, url: str) -> Replica | None:
        return self._replicas.get(url)

    def primaries(self) -> list[Replica]:
        return [r for r in self._replicas.values() if r.primary]

    def select(self) -> Replica | None:
        """Pick the best healthy replica.

        Priority order:

        1. Any healthy primary.
        2. Any healthy non-primary (round-robin via insertion order).
        3. ``None`` when nothing is healthy.
        """
        healthy_primaries = [r for r in self._replicas.values() if r.primary and r.is_healthy]
        if healthy_primaries:
            return healthy_primaries[0]
        healthy = [r for r in self._replicas.values() if r.is_healthy]
        if healthy:
            return healthy[0]
        return None

    async def record_success(
        self,
        replica: Replica,
        *,
        recovery_threshold: int,
        latency_ms: float,
    ) -> None:
        async with self._lock:
            h = replica.health
            h.last_ok_at = time.time()
            h.last_latency_ms = latency_ms
            h.consecutive_failures = 0
            h.consecutive_successes += 1
            if (h.status is not ReplicaStatus.HEALTHY and h.consecutive_successes >= recovery_threshold):
                logger.info("router: replica %s marked HEALTHY after %d successes", replica.url,
                            h.consecutive_successes)
                h.status = ReplicaStatus.HEALTHY
                h.consecutive_successes = 0

    async def record_failure(
        self,
        replica: Replica,
        *,
        failure_threshold: int,
        reason: str,
    ) -> None:
        async with self._lock:
            h = replica.health
            h.last_failure_at = time.time()
            h.consecutive_successes = 0
            h.consecutive_failures += 1
            if (h.status is not ReplicaStatus.UNHEALTHY and h.consecutive_failures >= failure_threshold):
                logger.warning("router: replica %s marked UNHEALTHY after %d failures: %s", replica.url,
                               h.consecutive_failures, reason)
                h.status = ReplicaStatus.UNHEALTHY


async def run_health_check_loop(
    registry: ReplicaRegistry,
    config: RouterConfig,
    *,
    stop_event: asyncio.Event,
    http_get: HttpProbe | None = None,
) -> None:
    """Poll each replica's health endpoint on a fixed interval.

    ``http_get`` is pluggable so unit tests can inject a deterministic
    probe without hitting the network. The default uses ``httpx``.
    """
    if http_get is None:
        http_get = _default_http_probe
    while not stop_event.is_set():
        for replica in registry.all():
            status_ms, error = await http_get(
                replica.url + config.health_check_path,
                timeout=config.health_check_timeout_seconds,
            )
            if error is None:
                await registry.record_success(
                    replica,
                    recovery_threshold=config.recovery_threshold,
                    latency_ms=status_ms,
                )
            else:
                await registry.record_failure(
                    replica,
                    failure_threshold=config.failure_threshold,
                    reason=error,
                )
        try:
            await asyncio.wait_for(
                stop_event.wait(),
                timeout=config.health_check_interval_seconds,
            )
        except asyncio.TimeoutError:
            continue


async def _default_http_probe(
    url: str,
    *,
    timeout: float,
) -> tuple[float, str | None]:
    try:
        import httpx
    except ImportError:  # pragma: no cover - optional extra
        return 0.0, "httpx not installed; router health checks disabled"
    start = time.perf_counter()
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.get(url)
    except Exception as exc:
        return 0.0, f"{type(exc).__name__}: {exc}"
    latency_ms = (time.perf_counter() - start) * 1000.0
    if response.status_code >= 400:
        return latency_ms, f"HTTP {response.status_code}"
    return latency_ms, None


__all__ = [
    "HttpProbe",
    "Replica",
    "ReplicaHealth",
    "ReplicaRegistry",
    "ReplicaStatus",
    "run_health_check_loop",
]
