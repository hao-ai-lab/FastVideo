"""Per-class cache pools: feature partition/invalidation, slab-KV window/training."""
from __future__ import annotations

import numpy as np

from v2.runtime.cache import CacheKey, CacheManager, CachePolicy
from v2.runtime.cache.classes import FeatureCache, Slab, SlabKVCache
from v2.recipes.wan_causal import build_wan_causal_card


def test_feature_cache_reuse_and_adapter_partition():
    fc = FeatureCache(CachePolicy("feature", max_bytes=1 << 20))
    k1 = CacheKey(model_id="m", component_id="te", weights_version="v0", adapter_versions=(("l", "1"),))
    v = np.ones((4,), dtype="float32")
    assert fc.get(k1) is None                 # miss
    fc.put(k1, v)
    assert np.array_equal(fc.get(k1), v)       # hit
    k2 = CacheKey(model_id="m", component_id="te", weights_version="v0", adapter_versions=(("l", "2"),))
    assert fc.get(k2) is None                  # different te-LoRA stack => partitioned, no stale serve


def test_feature_cache_weight_invalidation_is_wholesale():
    fc = FeatureCache(CachePolicy("feature", max_bytes=1 << 20))
    k = CacheKey(model_id="m", component_id="te", weights_version="v0")
    fc.put(k, np.ones((2,), dtype="float32"))
    fc.invalidate_weights("v1")                # RL update_weights bump
    assert fc.get(k) is None


def test_feature_cache_budget_eviction():
    fc = FeatureCache(CachePolicy("feature", max_bytes=8))   # 8 bytes = one (2,)-float32
    for i in range(5):
        fc.put(CacheKey(model_id="m", component_id="c", step_index=i),
               np.zeros((2,), dtype="float32"))
    assert fc.used_bytes <= 8


def test_slab_kv_inference_window_vs_training_mode():
    inf = SlabKVCache(CachePolicy("slab_kv", reuse_across_requests=False, per_component={"window": 2}))
    for i in range(4):
        inf.append("r", Slab(i, np.zeros((3,), dtype="float32"), None))
    assert len(inf.get("r")) == 2              # sliding window recycles (inference)

    tr = SlabKVCache(CachePolicy("slab_kv", training_mode_disables_recycle=True,
                                 per_component={"window": 2}))
    for i in range(4):
        tr.append("r", Slab(i, np.zeros((3,), dtype="float32"), None))
    assert len(tr.get("r")) == 4               # no mid-rollout recycle in training mode (self-forcing)
    assert tr.used_bytes > 0
    tr.clear_namespace("r")
    assert tr.used_bytes == 0


def test_cache_manager_materializes_only_declared_classes():
    cm = CacheManager.from_card(build_wan_causal_card())
    assert cm.has("feature") and cm.has("slab_kv")
    # a pure bidirectional card declares no KV class
    from v2.recipes.wan21 import build_wan21_card
    cm2 = CacheManager.from_card(build_wan21_card())
    assert cm2.has("feature") and not cm2.has("slab_kv")    # KV is the minority case
