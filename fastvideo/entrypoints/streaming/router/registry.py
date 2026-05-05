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
import contextlib
import enum
import time
from collections.abc import AsyncIterator, Awaitable, Callable
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

        1. The first healthy primary (insertion order).
        2. The first healthy non-primary (insertion order).
        3. ``None`` when nothing is healthy.

        This MVP picks the first match within each tier; it does NOT
        load-balance across multiple healthy replicas of the same tier.
        Round-robin and weighted distribution are deferred until a real
        N-way active deployment exists.
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
            # State machine: UNKNOWN -> HEALTHY is immediate; only the
            # UNHEALTHY -> HEALTHY transition is gated by recovery_threshold.
            if h.status is ReplicaStatus.UNKNOWN:
                logger.info("router: replica %s initial probe ok, marking HEALTHY", replica.url)
                h.status = ReplicaStatus.HEALTHY
                h.consecutive_successes = 0
            elif (h.status is ReplicaStatus.UNHEALTHY and h.consecutive_successes >= recovery_threshold):
                logger.info("router: replica %s recovered to HEALTHY after %d successes", replica.url,
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
    """Poll all replicas' health endpoints in parallel on a fixed interval.

    ``http_get`` is pluggable so unit tests can inject a deterministic
    probe without hitting the network. The default builds a single
    ``httpx.AsyncClient`` shared across the loop's lifetime so the
    common case (steady polling against a stable replica set) reuses
    TCP/TLS connections instead of paying handshake cost per probe.

    Probes within one polling cycle run concurrently via ``asyncio.gather``
    so a slow replica doesn't push the cycle past
    ``health_check_interval_seconds``.
    """
    if http_get is not None:
        await _run_loop(registry, config, stop_event, http_get)
        return
    async with _build_default_probe(config) as probe:
        await _run_loop(registry, config, stop_event, probe)


async def _run_loop(
    registry: ReplicaRegistry,
    config: RouterConfig,
    stop_event: asyncio.Event,
    http_get: Callable[..., Awaitable[tuple[float, str | None]]],
) -> None:
    while not stop_event.is_set():
        replicas = registry.all()
        results = await asyncio.gather(
            *[
                http_get(replica.url + config.health_check_path, timeout=config.health_check_timeout_seconds)
                for replica in replicas
            ],
            return_exceptions=True,
        )
        for replica, result in zip(replicas, results, strict=True):
            if isinstance(result, BaseException):
                await registry.record_failure(
                    replica,
                    failure_threshold=config.failure_threshold,
                    reason=f"{type(result).__name__}: {result}",
                )
                continue
            status_ms, error = result
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


@contextlib.asynccontextmanager
async def _build_default_probe(
    config: RouterConfig, ) -> AsyncIterator[Callable[..., Awaitable[tuple[float, str | None]]]]:
    try:
        import httpx
    except ImportError as exc:  # pragma: no cover - optional extra
        raise RuntimeError("router health checks require httpx; install with "
                           "`pip install fastvideo[streaming]` or `pip install httpx`") from exc

    async with httpx.AsyncClient(timeout=config.health_check_timeout_seconds) as client:

        async def probe(url: str, *, timeout: float) -> tuple[float, str | None]:
            start = time.perf_counter()
            try:
                response = await client.get(url, timeout=timeout)
            except Exception as exc:
                return 0.0, f"{type(exc).__name__}: {exc}"
            latency_ms = (time.perf_counter() - start) * 1000.0
            if response.status_code >= 400:
                return latency_ms, f"HTTP {response.status_code}"
            return latency_ms, None

        yield probe


__all__ = [
    "HttpProbe",
    "Replica",
    "ReplicaHealth",
    "ReplicaRegistry",
    "ReplicaStatus",
    "run_health_check_loop",
]
