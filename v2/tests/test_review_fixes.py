"""Regression tests for the adversarial-review findings (untested paths now covered).

Locks in: fail-fast on infeasible/deadlocked admission (no busy-spin), feature-key partitioning by
adapter stack + precision, component-scoped weight invalidation (transformer sync keeps the
text-encoder cache), and output comparison flagging symmetric-empty output.
"""
from __future__ import annotations

import numpy as np

from v2.runtime.cache import CacheManager
from v2.core.card import load_card
from v2.runtime.memory import MemoryManager
from v2.recipes import build_default_engine
from v2.recipes.common import cached_text_encode
from v2.recipes.wan21 import build_wan21_card
from v2.core.parity import compare_outputs
from v2.core.request import DiffusionParams, Output, TaskType, VideoArtifact, make_request
from v2.runtime import AdmissionController, AdmissionInfeasible, Engine


def test_infeasible_reservation_fails_fast_not_busy_spin():
    eng = build_default_engine(Engine(admission=AdmissionController(MemoryManager(total_bytes=16))))
    req = make_request(TaskType.T2V, "wan2.1-1.3b", "x", diffusion=DiffusionParams(num_steps=3, seed=1))
    try:
        eng.run(req)
        assert False, "expected AdmissionInfeasible (resident need >> 16 bytes)"
    except AdmissionInfeasible:
        pass


def test_feature_key_partitions_by_adapter_stack():
    card = build_wan21_card()
    inst = load_card(card, cache_manager=CacheManager.from_card(card))
    inst.adapter_versions = {"te_lora": "1"}
    cached_text_encode(inst, "p")                   # miss + store (adapter stack 1)
    h = inst.caches.stats()["feature"]["hits"]
    cached_text_encode(inst, "p")                   # same stack + text ⇒ hit
    assert inst.caches.stats()["feature"]["hits"] == h + 1
    inst.adapter_versions = {"te_lora": "2"}         # swap te-LoRA stack on the resident instance
    m = inst.caches.stats()["feature"]["misses"]
    cached_text_encode(inst, "p")                   # different stack ⇒ MISS, never a stale hit
    assert inst.caches.stats()["feature"]["misses"] == m + 1


def test_transformer_weight_reload_keeps_text_encoder_cache():
    card = build_wan21_card()
    inst = load_card(card, cache_manager=CacheManager.from_card(card))
    cached_text_encode(inst, "p")
    assert inst.caches.stats()["feature"]["used_bytes"] > 0
    inst.set_weights_version("w1", components=["transformer"])   # transformer-only reload
    h = inst.caches.stats()["feature"]["hits"]
    cached_text_encode(inst, "p")                   # text-encoder embed survived ⇒ hit (not flushed)
    assert inst.caches.stats()["feature"]["hits"] == h + 1


def test_compare_outputs_flags_symmetric_empty_output():
    empty = {"r": Output(request_id="r",
                         artifacts={"video": VideoArtifact(name="video", frames=None)})}
    divs = compare_outputs(empty, empty)            # identical BUT empty — must not pass vacuously
    assert divs, "both-empty output must be flagged, not treated as parity"
