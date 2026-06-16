"""Per-class cache pools (design_v3 §7.2).

> There is **no single unified block pool**, because a unified pool requires uniform
> bytes-per-block and our cache classes differ by 150–500× in natural granularity ...
> Each class gets a statically budgeted pool behind one ``CacheHandle``.

Mini-fastvideo implements four classes (the §7.2 minimal set):
  * ``FeatureCache``   — content-hash keyed, partitioned by adapter+weights (text/vision encoders)
  * ``ResidualCache``  — cache-dit residuals, scoped per request AND per CFG branch
  * ``SlabKVCache``    — chunk-KV slabs (self-forcing / world models); training mode disables recycle
  * ``PagedKVCache``   — paged text-KV stub for ar_decode (phase-2 omni); minority case, lazy

KV is the minority case — a pure bidirectional deployment (Wan/LTX T2V) allocates none of it.
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any

from .keys import CacheKey, CachePolicy


def _nbytes(value: Any) -> int:
    if hasattr(value, "nbytes"):
        return int(value.nbytes)
    return 0


class _Pool:
    def __init__(self, policy: CachePolicy):
        self.policy = policy
        self.used_bytes = 0
        self.hits = 0
        self.misses = 0


class FeatureCache(_Pool):
    """content-hash keyed, reference-counted-ish FIFO/LRU, budget-aware (§7.2).

    Partitioned by ``adapter_versions``/``weights_version`` through the CacheKey, so two
    workflows sharing a prompt but differing in te-LoRA stack never serve stale embeddings.
    """

    def __init__(self, policy: CachePolicy):
        super().__init__(policy)
        self._store: "OrderedDict[str, tuple[Any, int, CacheKey]]" = OrderedDict()

    def get(self, key: CacheKey) -> Any | None:
        if not self.policy.reuse_across_requests:
            return None
        h = key.hash
        if h in self._store:
            self.hits += 1
            self._store.move_to_end(h)       # LRU
            return self._store[h][0]
        self.misses += 1
        return None

    def put(self, key: CacheKey, value: Any) -> None:
        nb = _nbytes(value)
        h = key.hash
        if h in self._store:
            self.used_bytes -= self._store[h][1]
        self._store[h] = (value, nb, key)
        self._store.move_to_end(h)
        self.used_bytes += nb
        self._evict()

    def _evict(self) -> None:
        while self.used_bytes > self.policy.max_bytes and self._store:
            _h, (_v, nb, _k) = self._store.popitem(last=False)  # FIFO/LRU oldest
            self.used_bytes -= nb

    def invalidate_weights(self, version: str) -> None:
        """RL update_weights bumps weight epoch → drop entries from older epochs (wholesale)."""
        drop = [h for h, (_v, _nb, k) in self._store.items() if k.weights_version != version]
        for h in drop:
            _v, nb, _k = self._store.pop(h)
            self.used_bytes -= nb

    def invalidate_components(self, components: set[str]) -> None:
        """Drop only entries produced by the changed components (design_v3 §7.1 partition-not-flush):
        a transformer-only weight sync must NOT evict text-encoder embeddings."""
        drop = [h for h, (_v, _nb, k) in self._store.items() if k.component_id in components]
        for h in drop:
            _v, nb, _k = self._store.pop(h)
            self.used_bytes -= nb


class ResidualCache(_Pool):
    """cache-dit residual store, scoped per ``LoopState`` AND per CFG branch (§5.1, §11).

    Keyed by (namespace, branch, name) where namespace is the request/loop id. This is the
    structural fix for the module-global residual state that corrupts cache-dit forks under
    concurrency: two interleaved requests have disjoint namespaces.
    """

    def __init__(self, policy: CachePolicy):
        super().__init__(policy)
        self._store: dict[tuple[str, str, str], Any] = {}

    def put(self, namespace: str, branch: str, name: str, value: Any) -> None:
        self._store[(namespace, branch, name)] = value

    def get(self, namespace: str, branch: str, name: str) -> Any | None:
        v = self._store.get((namespace, branch, name))
        if v is not None:
            self.hits += 1
        else:
            self.misses += 1
        return v

    def clear_namespace(self, namespace: str) -> None:
        for k in [k for k in self._store if k[0] == namespace]:
            del self._store[k]


@dataclass
class Slab:
    chunk_index: int
    k: Any
    v: Any


class SlabKVCache(_Pool):
    """Chunk-KV slabs for causal/world-model rollout (§7.2).

    ``training_mode`` disables mid-rollout recycling and keeps grad-aware index snapshots
    so activation-checkpoint recompute doesn't double-advance the cache (self-forcing).
    """

    def __init__(self, policy: CachePolicy):
        super().__init__(policy)
        self._store: dict[str, list[Slab]] = {}
        self.window = max(1, policy.per_component.get("window", 1 << 30))
        self.training_mode = policy.training_mode_disables_recycle

    def append(self, namespace: str, slab: Slab) -> None:
        slabs = self._store.setdefault(namespace, [])
        slabs.append(slab)
        self.used_bytes += _nbytes(slab.k) + _nbytes(slab.v)
        if not self.training_mode and len(slabs) > self.window:
            dropped = slabs.pop(0)  # sliding-window recycle (inference only)
            self.used_bytes -= _nbytes(dropped.k) + _nbytes(dropped.v)

    def get(self, namespace: str) -> list[Slab]:
        return self._store.get(namespace, [])

    def clear_namespace(self, namespace: str) -> None:
        for slab in self._store.pop(namespace, []):
            self.used_bytes -= _nbytes(slab.k) + _nbytes(slab.v)


class PagedKVCache(_Pool):
    """Paged text-KV stub for ar_decode (phase-2 omni). Minority case, materialized lazily."""

    def __init__(self, policy: CachePolicy):
        super().__init__(policy)
        self.total_blocks = max(1, policy.max_bytes // max(policy.block_bytes, 1))
        self.free = self.total_blocks
        self._alloc: dict[str, int] = {}

    def allocate(self, namespace: str, n_blocks: int) -> bool:
        if n_blocks > self.free:
            return False
        self._alloc[namespace] = self._alloc.get(namespace, 0) + n_blocks
        self.free -= n_blocks
        return True

    def free_namespace(self, namespace: str) -> None:
        self.free += self._alloc.pop(namespace, 0)


_CLASS_REGISTRY = {
    "feature": FeatureCache,
    "residual": ResidualCache,
    "slab_kv": SlabKVCache,
    "paged_kv": PagedKVCache,
}


def make_pool(policy: CachePolicy) -> _Pool:
    cls = _CLASS_REGISTRY.get(policy.class_name)
    if cls is None:
        raise KeyError(f"unknown cache class {policy.class_name!r} (have {list(_CLASS_REGISTRY)})")
    return cls(policy)
