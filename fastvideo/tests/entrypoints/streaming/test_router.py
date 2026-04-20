# SPDX-License-Identifier: Apache-2.0
"""Router tests — registry semantics + health loop behavior.

Avoids real WebSocket proxying (the bridge requires the ``websockets``
package and two running uvicorn processes); test_server.py already
covers the end-to-end WS protocol against a direct backend.
"""
from __future__ import annotations

import asyncio

import pytest

from fastvideo.entrypoints.streaming.router.config import (
    ReplicaEndpoint,
    RouterConfig,
)
from fastvideo.entrypoints.streaming.router.registry import (
    ReplicaRegistry,
    ReplicaStatus,
    run_health_check_loop,
)


def _registry(
    *,
    num_primary: int = 1,
    num_secondary: int = 1,
) -> ReplicaRegistry:
    replicas = [
        ReplicaEndpoint(
            url=f"http://primary-{i}:8000",
            primary=True,
        )
        for i in range(num_primary)
    ] + [
        ReplicaEndpoint(
            url=f"http://secondary-{i}:8000",
            primary=False,
        )
        for i in range(num_secondary)
    ]
    return ReplicaRegistry(replicas)


class TestReplicaRegistry:

    def test_requires_replicas(self):
        with pytest.raises(ValueError):
            ReplicaRegistry([])

    def test_select_none_when_all_unknown(self):
        assert _registry().select() is None

    def test_select_prefers_healthy_primary(self):
        reg = _registry()

        async def promote():
            primary = reg.primaries()[0]
            await reg.record_success(
                primary, recovery_threshold=1, latency_ms=1.0)
            # Also mark secondary healthy — primary should still win.
            sec = next(r for r in reg.all() if not r.primary)
            await reg.record_success(
                sec, recovery_threshold=1, latency_ms=1.0)
            return reg.select()

        pick = asyncio.run(promote())
        assert pick is not None
        assert pick.primary

    def test_falls_back_to_secondary_when_primary_unhealthy(self):
        reg = _registry()

        async def run():
            primary = reg.primaries()[0]
            sec = next(r for r in reg.all() if not r.primary)
            # Fail primary past threshold, succeed secondary.
            for _ in range(3):
                await reg.record_failure(
                    primary, failure_threshold=3, reason="mock")
            await reg.record_success(
                sec, recovery_threshold=1, latency_ms=1.0)
            return reg.select()

        pick = asyncio.run(run())
        assert pick is not None
        assert not pick.primary

    def test_failure_threshold_transitions_to_unhealthy(self):
        reg = _registry()
        primary = reg.primaries()[0]

        async def run():
            for _ in range(2):
                await reg.record_failure(
                    primary, failure_threshold=3, reason="x")
            assert primary.health.status is not ReplicaStatus.UNHEALTHY
            await reg.record_failure(
                primary, failure_threshold=3, reason="x")
            return primary

        result = asyncio.run(run())
        assert result.health.status is ReplicaStatus.UNHEALTHY
        assert result.health.consecutive_failures == 3

    def test_recovery_threshold_returns_to_healthy(self):
        reg = _registry()
        primary = reg.primaries()[0]

        async def run():
            for _ in range(3):
                await reg.record_failure(
                    primary, failure_threshold=3, reason="x")
            assert primary.health.status is ReplicaStatus.UNHEALTHY
            for _ in range(2):
                await reg.record_success(
                    primary, recovery_threshold=2, latency_ms=5.0)
            return primary

        result = asyncio.run(run())
        assert result.health.status is ReplicaStatus.HEALTHY
        assert result.health.last_latency_ms == 5.0

    def test_record_success_resets_failure_counter(self):
        reg = _registry()
        primary = reg.primaries()[0]

        async def run():
            await reg.record_failure(
                primary, failure_threshold=10, reason="x")
            await reg.record_success(
                primary, recovery_threshold=1, latency_ms=1.0)
            return primary

        result = asyncio.run(run())
        assert result.health.consecutive_failures == 0


class TestHealthCheckLoop:

    def test_loop_transitions_replicas_on_probe_results(self):
        config = RouterConfig(
            replicas=[
                ReplicaEndpoint(url="http://a", primary=True),
                ReplicaEndpoint(url="http://b"),
            ],
            health_check_interval_seconds=0.01,
            failure_threshold=1,
            recovery_threshold=1,
        )
        reg = ReplicaRegistry(config.replicas)
        stop_event = asyncio.Event()

        calls: list[str] = []

        async def probe(url, *, timeout):
            calls.append(url)
            if "http://a" in url:
                return 1.0, None
            return 0.0, "mock failure"

        async def run() -> None:
            task = asyncio.create_task(run_health_check_loop(
                registry=reg, config=config, stop_event=stop_event,
                http_get=probe,
            ))
            await asyncio.sleep(0.05)
            stop_event.set()
            await task

        asyncio.run(run())
        a = reg.get("http://a")
        b = reg.get("http://b")
        assert a is not None
        assert b is not None
        assert a.health.status is ReplicaStatus.HEALTHY
        assert b.health.status is ReplicaStatus.UNHEALTHY
        assert any("/health" in c for c in calls)


class TestRouterApp:

    def test_status_endpoint_lists_replicas(self):
        from starlette.testclient import TestClient

        from fastvideo.entrypoints.streaming.router.main import build_router_app

        config = RouterConfig(
            replicas=[
                ReplicaEndpoint(url="http://a", primary=True),
                ReplicaEndpoint(url="http://b"),
            ],
            health_check_interval_seconds=60,  # don't actually poll
        )
        reg = ReplicaRegistry(config.replicas)
        app = build_router_app(config, registry=reg)
        client = TestClient(app)
        response = client.get("/status")
        body = response.json()
        urls = {r["url"] for r in body["replicas"]}
        assert urls == {"http://a", "http://b"}
        # Initial status is UNKNOWN.
        assert all(r["status"] == "unknown" for r in body["replicas"])

    def test_ws_rejects_when_no_healthy_replica(self):
        from starlette.testclient import TestClient

        from fastvideo.entrypoints.streaming.router.main import build_router_app

        config = RouterConfig(
            replicas=[ReplicaEndpoint(url="http://a", primary=True)],
            health_check_interval_seconds=60,
        )
        reg = ReplicaRegistry(config.replicas)
        app = build_router_app(config, registry=reg)
        client = TestClient(app)
        with client.websocket_connect("/v1/stream") as ws:
            err = ws.receive_json()
            assert err["type"] == "error"
            assert err["code"] == "gpu_unavailable"
