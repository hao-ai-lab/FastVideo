"""TransferManifest — typed description of a cross-pool tensor/artifact move.

Transfers are scheduled WorkUnits, not side effects: every cross-pool edge produces a manifest the
TransferScheduler accounts for (a ``transfer`` WorkUnit kind), and the connector moves the payload.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


def payload_nbytes(value: Any) -> int:
    if hasattr(value, "nbytes"):
        return int(value.nbytes)
    if isinstance(value, dict):
        return sum(payload_nbytes(v) for v in value.values())
    if isinstance(value, list | tuple):
        return sum(payload_nbytes(v) for v in value)
    return 0


@dataclass
class TransferManifest:
    keys: tuple[str, ...]
    producer_id: str
    consumer_id: str
    src_location: str = "in_proc"  # "in_proc" | "shm" | "nccl" | "nixl" | "disk"
    dst_location: str = "in_proc"
    lifetime: str = "request"  # "request" | "session" | "persistent"
    cache_key: Any = None
    priority: int = 0
    nbytes: int = 0
