# SPDX-License-Identifier: Apache-2.0
"""``GET /v1/streamv2/health``: the server is serving.

The generator is created before the server accepts requests, so answering at
all means the server is ready to generate.
"""
from __future__ import annotations

from fastapi import APIRouter

router = APIRouter()


@router.get("/v1/streamv2/health")
async def get_health() -> dict[str, str]:
    return {"status": "ready"}
