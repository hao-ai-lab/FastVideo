"""Card / recipe / parallelism / cache-key contracts (design_v3 §4, §7.1, §8, §9.2)."""
from __future__ import annotations

import numpy as np

from v2._enums import ConsistencyLevel, LoopKind, WorkUnitKind
from v2.card import (
    CardValidationError,
    CostModel,
    LoopSpec,
    ModelCard,
    RecipeSpec,
)
from v2.cache.keys import CacheKey, content_hash
from v2.models.wan21 import build_wan21_card
from v2.models.wan_causal import build_wan_causal_card
from v2.parallel import ParallelPlan, ParallelValidationError, validate_plan


def test_wan_card_validates():
    card = build_wan21_card()
    assert card.validate() is card
    assert card.loops_sharing("transformer") == ["diffusion_denoise"]
    assert card.recipe.assumes_loop in card.loops          # the (recipe, runtime) binding


def test_recipe_assumes_missing_loop_is_rejected():
    bad = ModelCard(model_id="bad", family="t", recipe=RecipeSpec(assumes_loop="ghost"))
    try:
        bad.validate()
        assert False, "expected CardValidationError"
    except CardValidationError as e:
        assert "assumes_loop" in str(e)


def test_loop_requires_cost_model_and_declared_cache():
    # a loop referencing an undeclared cache class fails validation
    card = ModelCard(
        model_id="x", family="x",
        loops={"d": LoopSpec(loop_id="d", kind=LoopKind.DIFFUSION_DENOISE,
                             work_unit_kind=WorkUnitKind.DIFFUSION_STEP,
                             step_cost_model=CostModel(kind=WorkUnitKind.DIFFUSION_STEP),
                             cache_policy=["nonexistent"])})
    try:
        card.validate()
        assert False
    except CardValidationError as e:
        assert "cache class" in str(e)


def test_pp_patch_invalid_for_causal_card():
    causal = build_wan_causal_card()
    try:
        validate_plan(ParallelPlan(axes={"pp_patch": 2}), card=causal)
        assert False, "pp_patch on a causal card must be rejected"
    except ParallelValidationError as e:
        assert "pp_patch" in str(e)


def test_cfgp_le_2_and_ownership_conflict():
    try:
        validate_plan(ParallelPlan(axes={"cfgp": 4}))
        assert False
    except ParallelValidationError:
        pass
    # cfgp group + batched CFG policy is a build error
    try:
        validate_plan(ParallelPlan(axes={"cfgp": 2}), cfg_policy_batched=True)
        assert False
    except ParallelValidationError as e:
        assert "ownership conflict" in str(e)


def test_world_size_product_validation():
    plan = ParallelPlan(axes={"sp": 2, "cfgp": 2})       # world = 4
    assert plan.world_size() == 4
    validate_plan(plan, world_size=4)
    try:
        validate_plan(plan, world_size=8)
        assert False
    except ParallelValidationError:
        pass


def test_cache_key_partitions_by_adapter_and_weights():
    # same prompt, different te-LoRA stack => different key (partitioned, not flushed)
    base = dict(model_id="m", component_id="text_encoder",
                input_hashes=(("text", content_hash("a prompt")),))
    k1 = CacheKey(weights_version="v0", adapter_versions=(("te_lora", "1"),), **base)
    k2 = CacheKey(weights_version="v0", adapter_versions=(("te_lora", "2"),), **base)
    k3 = CacheKey(weights_version="v1", adapter_versions=(("te_lora", "1"),), **base)
    assert k1.hash != k2.hash       # different adapters => no stale serve
    assert k1.hash != k3.hash       # different weights => no stale serve
    assert k1.hash == CacheKey(weights_version="v0", adapter_versions=(("te_lora", "1"),), **base).hash


def test_content_hash_stable_for_same_text():
    assert content_hash("hello") == content_hash("hello")
    assert content_hash("hello") != content_hash("world")
    a = np.zeros((2, 2), dtype="float32")
    assert content_hash(a) == content_hash(np.zeros((2, 2), dtype="float32"))
