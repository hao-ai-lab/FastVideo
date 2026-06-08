# SPDX-License-Identifier: Apache-2.0
"""Health/readiness/status route contracts for streaming servers."""
from __future__ import annotations

import asyncio
from typing import Any

import pytest

pytest.importorskip("starlette")
from fastapi import FastAPI  # noqa: E402
from starlette.testclient import TestClient  # noqa: E402

from fastvideo.api.schema import (  # noqa: E402
    GeneratorConfig,
    GenerationRequest,
    SamplingConfig,
    ServeConfig,
    StreamingConfig,
)
from fastvideo.entrypoints.streaming.gpu_pool import (  # noqa: E402
    GpuPool,
    PoolAssignment,
    PoolHealth,
)
from fastvideo.entrypoints.streaming.health import (  # noqa: E402
    build_health_router,
    get_pool_status,
)
from fastvideo.entrypoints.streaming.server import build_app  # noqa: E402


_STATUS_KEYS = {
    "total_gpus",
    "available_gpus",
    "queue_size",
    "warmup_enabled",
    "warmup_successful_gpus",
    "warmup_failed_gpus",
    "gpu_status",
}


class _FakePool(GpuPool):

    def __init__(self, status: dict[str, Any] | None = None) -> None:
        self._status: dict[str, Any] = status or _fake_status()

    def get_status(self) -> dict[str, Any]:
        return self._status

    def health(self) -> PoolHealth:
        return PoolHealth(
            total_workers=self._status["total_gpus"],
            available_workers=self._status["available_gpus"],
            active_sessions=0,
            queued_sessions=self._status["queue_size"],
        )

    async def acquire(self, session_id: str, *, timeout: float | None = None) -> PoolAssignment:
        raise AssertionError("health tests should not acquire a worker")

    async def run(self, session_id: str, request: GenerationRequest) -> Any:
        raise AssertionError("health tests should not run generation")

    async def release(self, session_id: str) -> None:
        pass

    async def shutdown(self) -> None:
        pass


def _fake_status() -> dict[str, Any]:
    return {
        "total_gpus": 2,
        "available_gpus": 1,
        "queue_size": 3,
        "warmup_enabled": True,
        "warmup_successful_gpus": 1,
        "warmup_failed_gpus": 1,
        "gpu_status": {
            0: {
                "ready": True,
                "available": True,
                "client_count": 0,
                "current_model_id": "ltx2",
                "process_alive": True,
                "warmup_enabled": True,
                "warmup_success": True,
                "warmup_error": None,
                "warmup_timings": {"load": 1.2},
            },
            1: {
                "ready": False,
                "available": False,
                "client_count": 0,
                "current_model_id": None,
                "process_alive": False,
                "warmup_enabled": True,
                "warmup_success": False,
                "warmup_error": "boot failed",
                "warmup_timings": {},
            },
        },
    }


def _build_health_client(pool=None) -> TestClient:
    app = FastAPI()
    app.include_router(build_health_router(pool))
    return TestClient(app)


def _build_serve_config() -> ServeConfig:
    return ServeConfig(
        generator=GeneratorConfig(model_path="/models/fake"),
        default_request=GenerationRequest(
            sampling=SamplingConfig(
                num_frames=12,
                height=64,
                width=64,
                fps=12,
                num_inference_steps=1,
            ),
        ),
        streaming=StreamingConfig(
            session_timeout_seconds=60,
            generation_segment_cap=2,
        ),
    )


class TestHealthRouterWithoutPool:

    def test_healthz_stays_live_without_pool(self):
        client = _build_health_client(pool=None)
        response = client.get("/healthz")
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "ok"
        assert body["service"] == "ltx2-streaming-backend"
        assert isinstance(body["ts"], str)

    def test_readyz_reports_warming_zeroed_pool(self):
        client = _build_health_client(pool=None)
        response = client.get("/readyz")
        assert response.status_code == 200
        body = response.json()
        assert body == {
            "status": "warming",
            "service": "ltx2-streaming-backend",
            "ready_gpu_workers": 0,
            "total_gpus": 0,
            "available_gpus": 0,
            "warmup_successful_gpus": 0,
            "warmup_failed_gpus": 0,
            "queue_size": 0,
            "ts": body["ts"],
        }
        assert isinstance(body["ts"], str)

    def test_status_returns_zeroed_pool_shape(self):
        client = _build_health_client(pool=None)
        response = client.get("/status")
        assert response.status_code == 200
        body = response.json()
        assert set(body) == _STATUS_KEYS
        assert body["total_gpus"] == 0
        assert body["available_gpus"] == 0
        assert body["queue_size"] == 0
        assert body["gpu_status"] == {}


class TestHealthRouterWithPool:

    def test_all_routes_return_expected_shapes(self):
        client = _build_health_client(pool=_FakePool())

        health = client.get("/health").json()
        assert health["status"] == "ok"
        assert health["service"] == "ltx2-streaming-backend"

        ready = client.get("/readyz").json()
        assert ready["status"] == "ready"
        assert ready["ready_gpu_workers"] == 1
        assert ready["total_gpus"] == 2
        assert ready["available_gpus"] == 1
        assert ready["warmup_successful_gpus"] == 1
        assert ready["warmup_failed_gpus"] == 1
        assert ready["queue_size"] == 3

        status = client.get("/status").json()
        assert set(status) == _STATUS_KEYS
        assert status["total_gpus"] == 2
        assert status["available_gpus"] == 1
        assert status["queue_size"] == 3
        assert "0" in status["gpu_status"]
        assert status["gpu_status"]["0"]["ready"] is True

    def test_pool_can_be_resolved_lazily(self):
        pool = _FakePool()
        client = _build_health_client(pool=lambda: pool)
        response = client.get("/status")
        assert response.status_code == 200
        assert response.json()["total_gpus"] == 2

    def test_helper_returns_same_pool_status_shape(self):
        status = asyncio.run(get_pool_status(_FakePool()))
        assert set(status) == _STATUS_KEYS
        assert status["gpu_status"][0]["current_model_id"] == "ltx2"


class TestBuildAppHealthMount:

    def test_build_app_keeps_legacy_health_and_mounts_public_routes(self):
        app = build_app(_build_serve_config(), pool=_FakePool())
        client = TestClient(app)

        legacy_health = client.get("/health")
        assert legacy_health.status_code == 200
        assert legacy_health.json() == {
            "status": "ok",
            "sessions": 0,
            "stream_mode": "av_fmp4",
        }

        healthz = client.get("/healthz")
        assert healthz.status_code == 200
        assert healthz.json()["service"] == "ltx2-streaming-backend"

        readyz = client.get("/readyz")
        assert readyz.status_code == 200
        assert readyz.json()["status"] == "ready"

        status = client.get("/status")
        assert status.status_code == 200
        assert status.json()["total_gpus"] == 2
