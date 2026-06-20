"""Self-forcing — causal video distillation (extends DMD2): train against your own KV-cached causal
rollout.

The student rollout is the SHARED ``chunk_rollout`` loop (with the slab-KV cache) — exactly the loop
the engine streams at serve time — so the causal student is distilled under the runtime it will be
served on. The loss is DMD2's distribution-matching loss applied to the causal rollout latents. (The
demonstrated structure is the causal rollout + KV + DMD loss; this mini reuses the shared flow-match
sampler in place of the repo's dedicated SelfForcingFlowMatchScheduler.)
"""
from __future__ import annotations

import numpy as np

from v2.core.enums import ConsistencyLevel
from v2.core.request import DiffusionParams, TaskType, make_request
from v2.training.methods.base import new_instance
from v2.training.methods.dmd2 import DMD2Method


class SelfForcingMethod(DMD2Method):
    name = "self_forcing"
    consistency = ConsistencyLevel.C1
    student_loop_id = "chunk_rollout"  # the SHARED causal loop (KV-cached)

    def _rollout_sample(self, prompt: str, seed: int) -> np.ndarray:
        req = make_request(TaskType.T2V, self.student.card.model_id, prompt, diffusion=DiffusionParams(seed=seed))
        res = self._rollout(req)  # drives chunk_rollout + slab-KV
        latents = res.outputs["latents"]
        return np.asarray(latents, dtype="float32")


def build_self_forcing(causal_card, **kw) -> SelfForcingMethod:
    """Build a self-forcing method over a Wan-causal card (chunk_rollout loop)."""
    return SelfForcingMethod(new_instance(causal_card), new_instance(causal_card), new_instance(causal_card), **kw)
