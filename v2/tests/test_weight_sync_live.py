"""Live weight-sync under in-flight serving — the RL flywheel's hardest correctness.

Collocated RL serves rollouts and receives weight updates on the same instance. The hazard: if new
weights are swapped in *while a denoise loop is mid-flight*, that request's trajectory becomes a
half-and-half of two policies — its captured log-probs describe a rollout that never happened, and
training silently corrupts. These tests prove the lifecycle (freeze → drain → transfer → publish →
resume) makes it correct:

  * **Hazard:** a mid-flight swap *does* corrupt the in-flight request (shown, as the thing to avoid).
  * **Lifecycle:** draining the in-flight request first (it finishes on its START weights) leaves it
    bit-identical to the no-sync baseline; a request admitted *after* the sync reflects the new weights.
  * **Per-component versioning:** the sync bumps only the transformer's version + invalidates only its
    caches — the frozen text-encoder's feature cache survives (so a shared prompt still reuses).
"""
from __future__ import annotations

import numpy as np

from v2._enums import ExecutionProfile
from v2.loop.driver import LoopRunner
from v2.platform.backends.toy import ToyDiT
from v2.recipes.common import cached_text_encode, text_encode_node_fn
from v2.recipes.wan21 import build_wan21_card
from v2.request import DiffusionParams, TaskType, make_request
from v2.request.streams import Stream
from v2.runtime import Engine
from v2.runtime.context import RuntimeLoopContext
from v2.training.methods.base import new_instance
from v2.training.weight_sync import WeightSyncController

_HUB = Engine()           # borrow its observer/interceptor hubs for the loop context


def _drive_denoise(inst, prompt="a fox", seed=1, *, steps=8, swap_at=None, new_dit=None):
    """Drive a denoise loop step-by-step; optionally swap the resident transformer at step ``swap_at``
    (the mid-flight hazard). Returns the final latent."""
    req = make_request(TaskType.T2V, inst.card.model_id, prompt,
                       diffusion=DiffusionParams(num_steps=steps, seed=seed))
    slots: dict = {}
    text_encode_node_fn(inst, slots, req, None)
    ctx = RuntimeLoopContext(inst, observers=_HUB.observers, interceptors=_HUB.interceptors,
                             slots=slots, stream=Stream(req.request_id), cancel_scope=None,
                             profile=ExecutionProfile.SERVE, metrics={}, request_id=req.request_id)
    runner = LoopRunner(inst.loop("diffusion_denoise"), ctx, req, inst)
    n = 0
    while not runner.done:
        if swap_at is not None and n == swap_at and new_dit is not None:
            inst.component("transformer").copy_from(new_dit)       # MID-FLIGHT swap
        runner.step()
        n += 1
    return np.asarray(runner.result.outputs["latents"])


def _fresh():
    return new_instance(build_wan21_card())


def test_baseline_then_mid_flight_swap_corrupts():
    """The hazard the lifecycle exists to prevent: swapping weights mid-denoise changes the output."""
    baseline = _drive_denoise(_fresh())
    corrupted = _drive_denoise(_fresh(), swap_at=3, new_dit=ToyDiT(seed=999))
    assert not np.array_equal(corrupted, baseline)                 # a half-and-half rollout


def test_drain_then_sync_does_not_corrupt_in_flight():
    """The lifecycle: freeze admission, let the in-flight request DRAIN (finish on its start weights),
    then sync. The in-flight output is bit-identical to baseline; a post-sync request uses new weights."""
    baseline = _drive_denoise(_fresh())
    inst = _fresh()
    ctrl = WeightSyncController(inst)
    ctrl.freeze()                                                  # no new admissions during the sync
    assert not ctrl.can_admit()
    in_flight = _drive_denoise(inst)                               # drains on OLD weights
    assert np.array_equal(in_flight, baseline)                     # uncorrupted — finished on start weights
    ctrl.sync(ToyDiT(seed=999))                                    # transfer + publish, then resume
    assert ctrl.can_admit() and ctrl.synced == 1
    post = _drive_denoise(inst)                                    # admitted after the sync
    assert not np.array_equal(post, baseline)                      # reflects the NEW weights


def test_sync_versions_and_invalidates_only_the_synced_component():
    inst = _fresh()
    cached_text_encode(inst, "a shared prompt")                    # warm the text-encoder feature cache
    hits0 = inst.caches.stats()["feature"]["hits"]
    WeightSyncController(inst).sync(ToyDiT(seed=7))                 # transformer-only sync
    assert inst.version_of("transformer") != "v0"                  # synced component bumped
    assert inst.version_of("text_encoder") == "v0"                 # frozen encoder untouched
    cached_text_encode(inst, "a shared prompt")                    # same prompt after the sync
    assert inst.caches.stats()["feature"]["hits"] == hits0 + 1     # text cache SURVIVED the transformer sync
