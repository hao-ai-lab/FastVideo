# SPDX-License-Identifier: Apache-2.0
"""Multi-replica load balancer + WebSocket proxy for the streaming server.

Sits in front of one-or-more streaming-server replicas and forwards
WebSocket sessions to a healthy primary, with failover to secondaries.
Kept in-repo under ``fastvideo/entrypoints/streaming/router/`` per the
PR plan's default; the alternative (separate package) is an open
question deferred to review.
"""
from fastvideo.entrypoints.streaming.router.registry import (
    Replica,
    ReplicaHealth,
    ReplicaRegistry,
    ReplicaStatus,
)
from fastvideo.entrypoints.streaming.router.config import RouterConfig
from fastvideo.entrypoints.streaming.router.main import build_router_app, run_router

__all__ = [
    "Replica",
    "ReplicaHealth",
    "ReplicaRegistry",
    "ReplicaStatus",
    "RouterConfig",
    "build_router_app",
    "run_router",
]
