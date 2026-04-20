# SPDX-License-Identifier: Apache-2.0
"""Typed router configuration."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ReplicaEndpoint:
    """One backend replica the router can route to."""

    url: str
    """HTTP base URL, e.g. ``http://host:8000``. WebSocket URL is
    derived automatically by replacing the scheme."""
    name: str | None = None
    primary: bool = False
    """``True`` = prefer this replica over others in steady state."""
    weight: float = 1.0


@dataclass
class RouterConfig:
    """Typed router config loaded from a YAML file.

    Example::

        router:
          host: 0.0.0.0
          port: 9000
          replicas:
            - url: http://streamer-a:8000
              primary: true
            - url: http://streamer-b:8000
          health_check:
            path: /health
            interval_seconds: 5
            failure_threshold: 3
    """

    host: str = "0.0.0.0"
    port: int = 9000
    replicas: list[ReplicaEndpoint] = field(default_factory=list)
    health_check_path: str = "/health"
    health_check_interval_seconds: float = 5.0
    health_check_timeout_seconds: float = 2.0
    failure_threshold: int = 3
    recovery_threshold: int = 2


__all__ = ["ReplicaEndpoint", "RouterConfig"]
