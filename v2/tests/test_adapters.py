"""The adapter plane — per-request LoRA / ControlNet over one base (design_v3 §7.1, §9.19).

Many adapters served over ONE resident base; a request picks which to apply (`DiffusionParams.adapters`).
The `adapter_versions` cache-key field existed but the serving capability didn't. These assert:
per-request selection changes output, multi-LoRA composes, ControlNet conditions on a control image,
mixed-adapter requests interleave without smearing, hot-swap changes generation, and the cache key
partitions by the active adapter stack (adapted ≠ base, and version A ≠ version B).
"""
from __future__ import annotations

import numpy as np

from v2.cache import CacheManager
from v2.cache.keys import CacheKey
from v2.card import load_card
from v2.models.adapters import build_adapter_card, build_adapter_program
from v2.parity import assert_interleave_parity
from v2.request import DiffusionParams, TaskType, make_request
from v2.request.modalpart import ImagePart, TextPart
from v2.runtime import Engine


def _engine():
    card = build_adapter_card()
    inst = load_card(card, cache_manager=CacheManager.from_card(card))
    eng = Engine()
    eng.register(card.model_id, inst, build_adapter_program())
    return eng, inst


def _gen(eng, adapters=(), control=None, seed=3):
    inputs = (TextPart("a portrait"),) + ((ImagePart(pixels=control),) if control is not None else ())
    req = make_request(TaskType.T2V, "wan-adapters", "a portrait", inputs=inputs,
                       diffusion=DiffusionParams(num_steps=6, seed=seed, adapters=adapters))
    return np.asarray(eng.run(req).artifacts["video"].frames)


def test_per_request_adapter_changes_output():
    eng, _ = _engine()
    base = _gen(eng)
    anime = _gen(eng, ("lora_anime",))
    realistic = _gen(eng, ("lora_realistic",))
    assert not np.array_equal(base, anime)
    assert not np.array_equal(anime, realistic)
    assert np.array_equal(anime, _gen(eng, ("lora_anime",)))     # deterministic


def test_multi_lora_composes():
    eng, _ = _engine()
    a = _gen(eng, ("lora_anime",))
    b = _gen(eng, ("lora_realistic",))
    ab = _gen(eng, ("lora_anime", "lora_realistic"))
    assert not np.array_equal(ab, a) and not np.array_equal(ab, b)


def test_controlnet_conditions_on_the_control_image():
    eng, _ = _engine()
    c_hi = _gen(eng, ("control_pose",), control=np.full((3, 1, 4, 6), 0.5, dtype="float32"))
    c_lo = _gen(eng, ("control_pose",), control=np.full((3, 1, 4, 6), -0.5, dtype="float32"))
    assert not np.array_equal(c_hi, c_lo)                        # different control ⇒ different video


def test_mixed_adapter_requests_interleave_without_smearing():
    """One base, different adapters per request — interleaving is bit-identical to serial (the active
    adapter set lives in the request/LoopState, never global)."""
    eng, _ = _engine()
    reqs = [make_request(TaskType.T2V, "wan-adapters", "a portrait",
                         diffusion=DiffusionParams(num_steps=4, seed=s, adapters=ad))
            for s, ad in [(1, ()), (2, ("lora_anime",)), (3, ("lora_realistic",))]]
    assert not assert_interleave_parity(eng, reqs)


def test_hot_swap_changes_generation():
    eng, inst = _engine()
    before = _gen(eng, ("lora_anime",))
    inst.component("lora_anime").update(seed=999)               # swap the adapter's weights
    assert not np.array_equal(before, _gen(eng, ("lora_anime",)))


def test_cache_key_partitions_by_adapter_stack():
    base = CacheKey(model_id="m", component_id="transformer", weights_version="w1",
                    adapter_versions=(), precision="float32", input_hashes=())
    a = CacheKey(model_id="m", component_id="transformer", weights_version="w1",
                 adapter_versions=CacheKey.adapters({"lora_anime": "v1"}), precision="float32", input_hashes=())
    b = CacheKey(model_id="m", component_id="transformer", weights_version="w1",
                 adapter_versions=CacheKey.adapters({"lora_anime": "v2"}), precision="float32", input_hashes=())
    assert a != base                                            # adapted ≠ base (no stale reuse)
    assert a != b                                               # adapter version A ≠ version B (hot-swap)
