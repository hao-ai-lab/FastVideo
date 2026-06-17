# SPDX-License-Identifier: Apache-2.0
"""FastAPI app for the local performance dashboard."""

from __future__ import annotations

import os
import threading
from datetime import datetime, timezone
from typing import Any

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from fastvideo.tests.performance import hf_store

from .service import build_latest_summary, build_trends, filter_records

DEFAULT_TRACKING_ROOT = "/tmp/fastvideo-perf-dashboard"
DEFAULT_DAYS = 90
FRONTEND_DIST = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "performance_dashboard", "frontend", "dist"))


class PerformanceDataStore:

    def __init__(self, tracking_root: str | None = None) -> None:
        self.tracking_root = tracking_root or os.environ.get("PERFORMANCE_TRACKING_ROOT", DEFAULT_TRACKING_ROOT)
        self._lock = threading.Lock()
        self.last_sync_at: str | None = None
        self.last_sync_error: str | None = None

    @property
    def repo_id(self) -> str:
        return hf_store.HF_REPO_ID

    def sync(self) -> dict[str, Any]:
        with self._lock:
            try:
                local_dir = hf_store.sync_from_hf(self.tracking_root, reuse_existing=False)
                self.last_sync_at = datetime.now(timezone.utc).isoformat()
                self.last_sync_error = None
                return {
                    "ok": True,
                    "repo_id": self.repo_id,
                    "tracking_root": local_dir,
                    "last_sync_at": self.last_sync_at,
                    "last_sync_error": None,
                }
            except Exception as exc:
                self.last_sync_at = datetime.now(timezone.utc).isoformat()
                self.last_sync_error = str(exc)
                return {
                    "ok": False,
                    "repo_id": self.repo_id,
                    "tracking_root": self.tracking_root,
                    "last_sync_at": self.last_sync_at,
                    "last_sync_error": self.last_sync_error,
                }

    def ensure_synced(self) -> None:
        if self.last_sync_at is not None:
            return
        with self._lock:
            if self.last_sync_at is None:
                hf_store.sync_from_hf(self.tracking_root, reuse_existing=True)
                self.last_sync_at = datetime.now(timezone.utc).isoformat()

    def load_records(self, *, days: int | None = None, successful_only: bool = False) -> list[dict[str, Any]]:
        self.ensure_synced()
        return hf_store.load_records(self.tracking_root, days=days, successful_only=successful_only)

    def health(self) -> dict[str, Any]:
        return {
            "ok": self.last_sync_error is None,
            "repo_id": self.repo_id,
            "tracking_root": self.tracking_root,
            "last_sync_at": self.last_sync_at,
            "last_sync_error": self.last_sync_error,
        }


def create_app(store: PerformanceDataStore | None = None) -> FastAPI:
    data_store = store or PerformanceDataStore()
    app = FastAPI(title="FastVideo Performance Dashboard")
    app.state.performance_store = data_store

    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://localhost:5173",
            "http://127.0.0.1:5173",
            "http://localhost:3000",
            "http://127.0.0.1:3000",
        ],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/api/performance/health")
    def health() -> dict[str, Any]:
        return data_store.health()

    @app.post("/api/performance/refresh")
    def refresh() -> dict[str, Any]:
        return data_store.sync()

    @app.get("/api/performance/records")
    def records(
        days: int = Query(DEFAULT_DAYS, ge=1, le=3650),
        model_id: str | None = None,
        gpu_type: str | None = None,
        success: bool | None = None,
    ) -> dict[str, Any]:
        loaded = data_store.load_records(days=days)
        filtered = filter_records(loaded, model_id=model_id, gpu_type=gpu_type, success=success)
        return {
            "records": filtered,
            "count": len(filtered),
            "filters": {
                "days": days,
                "model_id": model_id,
                "gpu_type": gpu_type,
                "success": success,
            },
            "sync": data_store.health(),
        }

    @app.get("/api/performance/summary")
    def summary(
        days: int = Query(DEFAULT_DAYS, ge=1, le=3650),
        model_id: str | None = None,
        gpu_type: str | None = None,
    ) -> dict[str, Any]:
        # Latest status should be stable when users change the trend window.
        # Use all cached records for latest/baseline computation; the ``days``
        # query is kept only so the frontend can share filter state across
        # endpoints without affecting the summary semantics.
        loaded = data_store.load_records(days=None)
        filtered = filter_records(loaded, model_id=model_id, gpu_type=gpu_type)
        rows = build_latest_summary(filtered, max_regression=float(os.environ.get("PERF_MAX_REGRESSION", "0.05")))
        return {
            "rows": rows,
            "count": len(rows),
            "status_counts": {
                "pass": sum(1 for row in rows if row["status"] == "pass"),
                "fail": sum(1 for row in rows if row["status"] == "fail"),
            },
            "filters": {
                "days": None,
                "trend_window_days": days,
                "model_id": model_id,
                "gpu_type": gpu_type,
            },
            "sync": data_store.health(),
        }

    @app.get("/api/performance/trends")
    def trends(
        days: int = Query(DEFAULT_DAYS, ge=1, le=3650),
        model_id: str | None = None,
        gpu_type: str | None = None,
    ) -> dict[str, Any]:
        loaded = data_store.load_records(days=days)
        filtered = filter_records(loaded, model_id=model_id, gpu_type=gpu_type)
        groups = build_trends(filtered)
        return {
            "groups": groups,
            "count": len(groups),
            "filters": {
                "days": days,
                "model_id": model_id,
                "gpu_type": gpu_type,
            },
            "sync": data_store.health(),
        }

    assets_dir = os.path.join(FRONTEND_DIST, "assets")
    index_file = os.path.join(FRONTEND_DIST, "index.html")
    if os.path.isdir(assets_dir) and os.path.isfile(index_file):
        app.mount("/assets", StaticFiles(directory=assets_dir), name="performance-dashboard-assets")

        @app.get("/{full_path:path}", include_in_schema=False)
        def frontend(full_path: str) -> FileResponse:
            return FileResponse(index_file)

    return app


app = create_app()
