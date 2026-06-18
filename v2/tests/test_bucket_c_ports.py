"""Bucket-C ports: every net-new architecture ported into v2 resolves through the registry AND runs
end-to-end on the CPU toy backend, emitting the correct modality artifact.

Data-driven from ``v2.registry._BUCKET_C`` (+ the cosmos2 reference), so adding a port to that table
automatically extends this regression guard. This exercises the PUBLIC run path (registry.resolve ->
build card+program -> load_card -> Engine.run -> output artifact) for each port, not just the loop in
isolation — catching capability/program/slot wiring breakage. GPU load/run is BRINGUP (gated/large
weights), so this stays on the toy backend.
"""
from __future__ import annotations

import pytest

from v2 import registry
from v2._enums import Capability
from v2.cache import CacheManager
from v2.card import load_card
from v2.request import DiffusionParams, TaskType, make_request
from v2.runtime import Engine

# A generation capability -> the request task that drives it (first match wins, in this priority order).
_CAP_TASK = [
    (Capability.TEXT_TO_VIDEO, TaskType.T2V),
    (Capability.TEXT_TO_IMAGE, TaskType.T2I),
    (Capability.TEXT_TO_VIDEO_SOUND, TaskType.T2VS),
    (Capability.IMAGE_TO_VIDEO, TaskType.I2V),
]

# (primary hf id, package) for each registered net-new port — cosmos2 (the reference) + the _BUCKET_C table.
_PORTS = [("nvidia/Cosmos-Predict2-2B-Video2World", "cosmos2")] + \
         [(row[0][0], row[1]) for row in registry._BUCKET_C]


def _task_for(card) -> TaskType:
    for cap, task in _CAP_TASK:
        if card.capabilities.has(cap):
            return task
    return TaskType.T2V


@pytest.mark.parametrize("hf_id,package", _PORTS, ids=[p for _h, p in _PORTS])
def test_bucket_c_port_resolves_and_runs_on_toy(hf_id, package):
    build_card, build_program = registry.resolve(hf_id)          # PRIMARY: exact-id resolution
    card = build_card()
    assert card.model_id and card.family
    assert card.sampling_defaults.num_steps > 0

    eng = Engine()
    inst = load_card(card, cache_manager=CacheManager.from_card(card))
    eng.register(card.model_id, inst, build_program())
    out = eng.run(make_request(_task_for(card), card.model_id, "a test prompt",
                               diffusion=DiffusionParams(num_steps=3, seed=1)))
    # exactly one of video/image/audio is produced (modality-correct), plus the latents
    media = [k for k in ("video", "image", "audio") if k in out.artifacts]
    assert len(media) == 1, f"{package}: expected one media artifact, got {list(out.artifacts)}"
    assert "latents" in out.artifacts


def test_every_bucket_c_hf_id_resolves_by_exact_id():
    for hf_ids, *_rest in registry._BUCKET_C:
        for hid in hf_ids:
            build_card, _bp = registry.resolve(hid)
            assert build_card().model_id, f"{hid} resolved to a card with no model_id"


def test_bucket_c_architecture_fallback_resolves():
    seen = set()
    for _hf, pkg, _cb, _pb, cls in registry._BUCKET_C:
        if not cls or cls in seen:
            continue
        seen.add(cls)
        build_card, _bp = registry.select_by_architecture({"transformer_cls": cls, "pipeline": None})
        assert build_card().family, f"arch fallback for {cls} produced no family"
