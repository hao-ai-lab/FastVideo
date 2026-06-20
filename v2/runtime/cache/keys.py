"""CacheKey — cache correctness is a contract.

If a field can change output semantics, it is in the key (incorrect reuse is worse than no reuse).
The serving hazard this kills: a request that shares a prompt but differs in te-LoRA stack must not
serve stale embeddings — so the key is *partitioned* by ``adapter_versions``, not flushed. An RL
``update_weights`` bumps ``weights_version`` and invalidates wholesale.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any


def content_hash(obj: Any) -> str:
    """Stable content hash for feature-cache keys (text/vision embeddings).

    Lets K rollout samples of one prompt reuse a single text encode.
    """
    h = hashlib.sha256()
    if isinstance(obj, str):
        h.update(obj.encode("utf-8"))
    elif isinstance(obj, bytes | bytearray):
        h.update(obj)
    elif hasattr(obj, "tobytes") and hasattr(obj, "shape"):  # numpy / torch tensor
        h.update(str(getattr(obj, "shape", "")).encode())
        h.update(str(getattr(obj, "dtype", "")).encode())
        try:
            h.update(obj.tobytes())
        except Exception:
            h.update(repr(obj).encode())
    else:
        h.update(repr(obj).encode())
    return h.hexdigest()[:32]


@dataclass(frozen=True)
class CacheKey:
    model_id: str
    component_id: str
    loop_id: str | None = None
    weights_version: str = "v0"
    adapter_versions: tuple[tuple[str, str], ...] = ()  # sorted (adapter_id, version) pairs
    precision: str = "float32"
    parallel_plan_hash: str = ""
    shape_sig: str = ""
    layout_sig: str = ""
    scheduler_sig: str | None = None
    guidance_sig: str | None = None
    seed: int | None = None
    input_hashes: tuple[tuple[str, str], ...] = ()
    step_index: int | None = None
    contract_version: str = "v0"

    @property
    def hash(self) -> str:
        return hashlib.sha256(repr(self).encode()).hexdigest()[:24]

    def partition_field(self) -> tuple:
        """Fields that *partition* (not flush) a feature cache: adapters + weights."""
        return (self.weights_version, self.adapter_versions)

    @staticmethod
    def adapters(d: dict[str, str] | None) -> tuple[tuple[str, str], ...]:
        return tuple(sorted((d or {}).items()))

    @staticmethod
    def hashes(d: dict[str, str] | None) -> tuple[tuple[str, str], ...]:
        return tuple(sorted((d or {}).items()))


@dataclass
class CachePolicy:
    """Runtime config for one cache class pool."""
    class_name: str  # "feature" | "residual" | "slab_kv" | "paged_kv"
    max_bytes: int = 1 << 30
    block_bytes: int = 1 << 16
    eviction: str = "lru"  # "lru" | "fifo" | "none"
    reuse_across_requests: bool = True
    per_component: dict[str, int] = field(default_factory=dict)
    training_mode_disables_recycle: bool = False
