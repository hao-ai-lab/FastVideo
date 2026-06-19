"""Connectors + flow control.

Two distinct mechanisms are modeled:
  * ``chunk_ready`` readiness signal (readiness-set parking);
  * credit-based flow control (a bounded in-flight window).

Backends are pluggable: ``InProcConnector`` (zero-copy reference handoff — the single-node default)
and ``ShmFakeConnector`` (copies on put to model a serialization boundary). A ``KVConnector`` exposes
the cache-bearing protocol shape (scheduler-side query/alloc/finish + worker-side save/load) that
NIXL/LMCache/Mooncake/KVBM would implement.
"""
from __future__ import annotations

import copy
from typing import Any, Protocol, runtime_checkable

from v2.transport.manifest import TransferManifest, payload_nbytes


class CreditWindow:
    """Credit-based flow control: a bounded number of in-flight transfers."""

    def __init__(self, capacity: int = 8):
        self.capacity = capacity
        self.available = capacity

    def acquire(self, n: int = 1) -> bool:
        if self.available >= n:
            self.available -= n
            return True
        return False

    def release(self, n: int = 1) -> None:
        self.available = min(self.capacity, self.available + n)


@runtime_checkable
class Connector(Protocol):
    name: str

    def put(self, key: str, value: Any, manifest: TransferManifest | None = None) -> None:
        ...

    def chunk_ready(self, key: str) -> bool:
        ...

    def get(self, key: str) -> Any:
        ...

    def take(self, key: str) -> Any:
        ...

    def acquire_credit(self, n: int = 1) -> bool:
        ...

    def release_credit(self, n: int = 1) -> None:
        ...


class InProcConnector:
    """Zero-copy in-process handoff with readiness + credits (the single-node default)."""

    def __init__(self, name: str = "inproc", credit_capacity: int = 8):
        self.name = name
        self._store: dict[str, Any] = {}
        self._ready: set[str] = set()
        self.credits = CreditWindow(credit_capacity)
        self.transfers = 0
        self.bytes_moved = 0
        self.parked = 0  # how many get()s waited on a not-yet-ready key (readiness parking)

    def _materialize(self, value: Any) -> Any:
        return value  # in-proc: pass by reference (zero overhead)

    def put(self, key: str, value: Any, manifest: TransferManifest | None = None) -> None:
        self._store[key] = self._materialize(value)
        self._ready.add(key)
        self.transfers += 1
        self.bytes_moved += (manifest.nbytes if manifest and manifest.nbytes else payload_nbytes(value))

    def chunk_ready(self, key: str) -> bool:
        return key in self._ready

    def get(self, key: str) -> Any:
        if key not in self._ready:
            self.parked += 1
            return None
        return self._store.get(key)

    def take(self, key: str) -> Any:
        v = self.get(key)
        if key in self._ready:
            self._store.pop(key, None)
            self._ready.discard(key)
        return v

    def acquire_credit(self, n: int = 1) -> bool:
        return self.credits.acquire(n)

    def release_credit(self, n: int = 1) -> None:
        self.credits.release(n)


class ShmFakeConnector(InProcConnector):
    """Models a shared-memory/RDMA boundary: copies the payload on put (a real transfer, not a ref)."""

    def __init__(self, name: str = "shm", credit_capacity: int = 8):
        super().__init__(name, credit_capacity)

    def _materialize(self, value: Any) -> Any:
        if hasattr(value, "copy"):
            return value.copy()
        return copy.deepcopy(value)


@runtime_checkable
class KVConnector(Protocol):
    """Cache-bearing edge protocol: scheduler-side query/alloc/finish + worker-side save/load —
    the shape NIXL/LMCache/Mooncake/KVBM implement."""

    def query(self, key: str) -> bool:
        ...

    def alloc(self, key: str, nbytes: int) -> bool:
        ...

    def finish(self, key: str) -> None:
        ...

    def save(self, key: str, value: Any) -> None:
        ...

    def load(self, key: str) -> Any:
        ...


class InProcKVConnector:

    def __init__(self, capacity_bytes: int = 1 << 30):
        self.capacity_bytes = capacity_bytes
        self.used = 0
        self._store: dict[str, Any] = {}
        self._reserved: dict[str, int] = {}

    def query(self, key: str) -> bool:
        return key in self._store

    def alloc(self, key: str, nbytes: int) -> bool:
        if self.used + nbytes > self.capacity_bytes:
            return False
        self._reserved[key] = nbytes
        self.used += nbytes
        return True

    def finish(self, key: str) -> None:
        self.used -= self._reserved.pop(key, 0)
        self._store.pop(key, None)

    def save(self, key: str, value: Any) -> None:
        self._store[key] = value

    def load(self, key: str) -> Any:
        return self._store.get(key)


_BACKENDS = {"in_proc": InProcConnector, "shm": ShmFakeConnector}


def make_connector(kind: str = "in_proc", **kw) -> Connector:
    if kind not in _BACKENDS:
        raise KeyError(f"unknown connector backend {kind!r} (have {list(_BACKENDS)})")
    return _BACKENDS[kind](**kw)
