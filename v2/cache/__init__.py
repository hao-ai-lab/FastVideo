"""Cache plane — correct by key, per-class pools (design_v3 §7)."""
from __future__ import annotations

from .classes import FeatureCache, PagedKVCache, ResidualCache, Slab, SlabKVCache, make_pool
from .keys import CacheKey, CachePolicy, content_hash
from .manager import CacheManager

__all__ = ["CacheKey", "CachePolicy", "content_hash", "CacheManager",
           "FeatureCache", "ResidualCache", "SlabKVCache", "PagedKVCache", "Slab", "make_pool"]
