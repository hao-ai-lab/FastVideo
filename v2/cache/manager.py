"""CacheManager — per-class pools with static budgets, behind one handle.

Static partitioning makes cross-class fragmentation impossible (jumbo slab traffic cannot
strand text-KV pages and vice versa). Each class gets a budget carved at init from the card's
CacheContracts. ``invalidate_weights`` implements the wholesale RL weight-epoch bump.
"""
from __future__ import annotations

from typing import Any

from v2.cache.classes import _Pool, make_pool
from v2.cache.keys import CachePolicy


class CacheManager:

    def __init__(self, policies: list[CachePolicy] | None = None):
        self._pools: dict[str, _Pool] = {}
        for p in (policies or []):
            self._pools[p.class_name] = make_pool(p)

    @classmethod
    def from_card(cls, card) -> CacheManager:
        """Build the per-class pools a card declares (KV pools materialize only if declared)."""
        policies = []
        for cc in card.caches.values():
            policies.append(
                CachePolicy(
                    class_name=cc.cache_class,
                    max_bytes=cc.max_bytes,
                    block_bytes=cc.block_bytes,
                    eviction=cc.eviction,
                    reuse_across_requests=cc.reuse_across_requests,
                    per_component=dict(cc.per_component),
                    training_mode_disables_recycle=cc.training_mode_disables_recycle,
                ))
        return cls(policies)

    def pool(self, class_name: str) -> _Pool:
        if class_name not in self._pools:
            raise KeyError(f"no cache pool for class {class_name!r}; card did not declare it")
        return self._pools[class_name]

    def has(self, class_name: str) -> bool:
        return class_name in self._pools

    def invalidate_weights(self, version: str) -> None:
        for pool in self._pools.values():
            if hasattr(pool, "invalidate_weights"):
                pool.invalidate_weights(version)

    def invalidate_components(self, components) -> None:
        """Component-scoped invalidation: only drop caches for changed components."""
        comps = set(components)
        for pool in self._pools.values():
            if hasattr(pool, "invalidate_components"):
                pool.invalidate_components(comps)

    def clear_namespace(self, namespace: str) -> None:
        """Release a finished request's per-request caches (residual/slab)."""
        for pool in self._pools.values():
            if hasattr(pool, "clear_namespace"):
                pool.clear_namespace(namespace)
            if hasattr(pool, "free_namespace"):
                pool.free_namespace(namespace)

    def stats(self) -> dict[str, Any]:
        return {
            name: {
                "used_bytes": p.used_bytes,
                "hits": p.hits,
                "misses": p.misses
            }
            for name, p in self._pools.items()
        }
