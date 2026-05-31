# SPDX-License-Identifier: Apache-2.0
"""Typed router configuration."""
from __future__ import annotations

from dataclasses import dataclass, field
from urllib.parse import urlparse


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

    Validation runs in ``__post_init__``: empty replicas, non-positive
    intervals/timeouts, thresholds < 1, non-http(s) URLs, and more than
    one primary all raise ``ValueError`` so misconfigurations surface at
    load time rather than as confusing runtime failures.
    """

    host: str = "0.0.0.0"
    port: int = 9000
    replicas: list[ReplicaEndpoint] = field(default_factory=list)
    health_check_path: str = "/health"
    health_check_interval_seconds: float = 5.0
    health_check_timeout_seconds: float = 2.0
    failure_threshold: int = 3
    recovery_threshold: int = 2

    def __post_init__(self) -> None:
        if not self.replicas:
            raise ValueError("RouterConfig.replicas must list at least one replica")
        if self.health_check_interval_seconds <= 0:
            raise ValueError(f"health_check_interval_seconds must be > 0, got {self.health_check_interval_seconds}")
        if self.health_check_timeout_seconds <= 0:
            raise ValueError(f"health_check_timeout_seconds must be > 0, got {self.health_check_timeout_seconds}")
        if self.failure_threshold < 1:
            raise ValueError(f"failure_threshold must be >= 1, got {self.failure_threshold}")
        if self.recovery_threshold < 1:
            raise ValueError(f"recovery_threshold must be >= 1, got {self.recovery_threshold}")
        seen_urls: set[str] = set()
        for replica in self.replicas:
            if not replica.url.startswith(("http://", "https://")):
                raise ValueError(f"ReplicaEndpoint.url must start with http:// or https://, got {replica.url!r}")
            parsed = urlparse(replica.url)
            if parsed.path not in ("", "/"):
                raise ValueError(f"ReplicaEndpoint.url must be a base host[:port] URL without a path; "
                                 f"got {replica.url!r} with path {parsed.path!r}. The router appends "
                                 "`/health` and `/v1/stream` itself.")
            if parsed.query or parsed.fragment:
                raise ValueError(f"ReplicaEndpoint.url must not include query/fragment; got {replica.url!r}")
            if replica.url in seen_urls:
                raise ValueError(f"Duplicate ReplicaEndpoint.url {replica.url!r}; "
                                 "router selection keys by URL so duplicates would silently collapse")
            seen_urls.add(replica.url)
        primaries = sum(1 for r in self.replicas if r.primary)
        if primaries > 1:
            raise ValueError(f"RouterConfig allows at most one primary replica; got {primaries}. "
                             "Multi-primary load distribution is deferred — promote one replica to "
                             "primary and treat the rest as secondaries.")


__all__ = ["ReplicaEndpoint", "RouterConfig"]
