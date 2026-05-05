"""Health, readiness, and status probes."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException

import runtime
from utils import _utc_now_iso

router = APIRouter()


def _gpu_pool_ready_snapshot() -> tuple[bool, str, int]:
    if runtime.gpu_pool is None:
        return False, "GPU pool not initialized.", 0

    slots = getattr(runtime.gpu_pool, "slots", None)
    if not isinstance(slots, dict) or not slots:
        return False, "GPU pool has no slots.", 0

    ready_slots = 0
    for slot in slots.values():
        process = getattr(slot, "process", None)
        ready = bool(getattr(slot, "ready", False))
        process_alive = bool(process and process.is_alive())
        if ready and process_alive:
            ready_slots += 1

    if ready_slots <= 0:
        return False, "No ready GPU worker processes.", 0
    return True, "", ready_slots


@router.get("/healthz")
async def get_healthz():
    """Liveness probe for process-level health."""
    return {
        "status": "ok",
        "service": "ltx2-streaming-backend",
        "ts": _utc_now_iso(),
    }


@router.get("/readyz")
async def get_readyz():
    """Readiness probe for router/load-balancer health checks."""
    if runtime.prompt_enhancer is None:
        raise HTTPException(
            status_code=503,
            detail="Prompt enhancer not initialized.",
        )

    ready, reason, ready_slots = _gpu_pool_ready_snapshot()
    if not ready:
        raise HTTPException(status_code=503, detail=reason)

    status_payload = runtime.gpu_pool.get_status() if runtime.gpu_pool is not None else {}
    return {
        "status": "ready",
        "service": "ltx2-streaming-backend",
        "ready_gpu_workers": ready_slots,
        "total_gpus": status_payload.get("total_gpus", 0),
        "available_gpus": status_payload.get("available_gpus", 0),
        "warmup_successful_gpus": status_payload.get("warmup_successful_gpus", 0),
        "warmup_failed_gpus": status_payload.get("warmup_failed_gpus", 0),
        "queue_size": status_payload.get("queue_size", 0),
        "ts": _utc_now_iso(),
    }


@router.get("/status")
async def get_status():
    """Get the current status of the GPU pool."""
    if runtime.gpu_pool is None:
        return {"error": "GPU pool not initialized"}
    return runtime.gpu_pool.get_status()


@router.get("/internal/monitor/sessions")
async def get_internal_monitor_sessions():
    """Internal monitor payload for router-level replica session dashboards."""
    if runtime.gpu_pool is None:
        raise HTTPException(status_code=503, detail="GPU pool not initialized.")
    status_payload = runtime.gpu_pool.get_status()
    max_available_sessions = status_payload.get("total_gpus")
    if not isinstance(max_available_sessions, int) or max_available_sessions < 0:
        max_available_sessions = 0
    prompt_provider_success_counts: dict[str, int] = {}
    if runtime.prompt_enhancer is not None:
        prompt_provider_success_counts = (runtime.prompt_enhancer.get_provider_success_counts())
    return {
        "service": "ltx2-streaming-backend",
        "pending_sessions": len(runtime.gpu_pool.waiting_list),
        "max_available_sessions": max_available_sessions,
        "prompt_provider_success_counts": prompt_provider_success_counts,
        "ts": _utc_now_iso(),
    }
