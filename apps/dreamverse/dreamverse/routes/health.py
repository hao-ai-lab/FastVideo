"""Dreamverse-specific monitor routes."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException

import dreamverse.runtime as runtime
from dreamverse.utils import _utc_now_iso

router = APIRouter()


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
        prompt_provider_success_counts = runtime.prompt_enhancer.get_provider_success_counts()
    return {
        "service": "ltx2-streaming-backend",
        "pending_sessions": len(runtime.gpu_pool.waiting_list),
        "max_available_sessions": max_available_sessions,
        "prompt_provider_success_counts": prompt_provider_success_counts,
        "ts": _utc_now_iso(),
    }
