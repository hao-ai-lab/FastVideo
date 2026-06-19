"""Regression tests for the adversarial-review findings (untested paths now covered).

Locks in: fail-fast on infeasible/deadlocked admission (no busy-spin), refundable compute budget
(concurrency gate, not lifetime cap), feature-key partitioning by adapter stack + precision,
component-scoped weight invalidation (transformer sync keeps the text-encoder cache), and the
interleave gate flagging symmetric-empty output.
"""
from __future__ import annotations

import numpy as np

from v2._enums import WorkUnitKind
from v2.cache import CacheManager
from v2.card import load_card
from v2.loop.contracts import ResourceRequest, ShapeSignature, WorkPlan
from v2.memory import MemoryManager
from v2.recipes import build_default_engine
from v2.recipes.common import cached_text_encode
from v2.recipes.wan21 import build_wan21_card
from v2.parity import compare_outputs
from v2.request import DiffusionParams, Output, TaskType, VideoArtifact, make_request
from v2.runtime import AdmissionController, AdmissionInfeasible, Engine


def test_infeasible_reservation_fails_fast_not_busy_spin():
    eng = build_default_engine(Engine(admission=AdmissionController(MemoryManager(total_bytes=16))))
    req = make_request(TaskType.T2V, "wan2.1-1.3b", "x", diffusion=DiffusionParams(num_steps=3, seed=1))
    try:
        eng.run(req)
        assert False, "expected AdmissionInfeasible (resident need >> 16 bytes)"
    except AdmissionInfeasible:
        pass


def test_compute_budget_is_a_refundable_concurrency_gate():
    ac = AdmissionController(compute_budget_seconds=1.0)
    plan = WorkPlan(loop_id="l", instance_id="i", kind=WorkUnitKind.DIFFUSION_STEP,
                    shape_sig=ShapeSignature(WorkUnitKind.DIFFUSION_STEP),
                    resources=ResourceRequest(compute_seconds=0.6))
    t1 = ac.admit_step(plan)
    assert t1 is not None and abs(ac.compute_spent - 0.6) < 1e-9
    assert ac.admit_step(plan) is None              # 0.6+0.6 > 1.0 ⇒ deferred (concurrency gate)
    ac.release(t1)
    assert ac.compute_spent == 0.0                  # refunded — NOT a permanent lifetime cap
    assert ac.admit_step(plan) is not None          # fits again after the refund


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


def test_transformer_weight_sync_keeps_text_encoder_cache():
    card = build_wan21_card()
    inst = load_card(card, cache_manager=CacheManager.from_card(card))
    cached_text_encode(inst, "p")
    assert inst.caches.stats()["feature"]["used_bytes"] > 0
    inst.set_weights_version("w1", components=["transformer"])   # RL-style transformer-only sync
    h = inst.caches.stats()["feature"]["hits"]
    cached_text_encode(inst, "p")                   # text-encoder embed survived ⇒ hit (not flushed)
    assert inst.caches.stats()["feature"]["hits"] == h + 1


def test_interleave_gate_flags_symmetric_empty_output():
    empty = {"r": Output(request_id="r",
                         artifacts={"video": VideoArtifact(name="video", frames=None)})}
    divs = compare_outputs(empty, empty)            # identical BUT empty — must not pass vacuously
    assert divs, "both-empty output must be flagged, not treated as parity"
