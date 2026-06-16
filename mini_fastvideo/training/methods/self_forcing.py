"""Self-forcing — causal video distillation (design_v3 §10; repo: self_forcing.py extends DMD2).

> Self-forcing is "train against your own KV-cached causal rollout."

The student rollout is the SHARED ``chunk_rollout`` loop (with the slab-KV cache) — exactly the
loop the engine streams at serve time — so the causal student is distilled under the runtime it
will be served on (the (recipe, runtime) pair, §2.1). The distribution-matching loss is DMD2's,
applied to the causal rollout latents. (The repo uses a dedicated SelfForcingFlowMatchScheduler;
the mini reuses the shared flow-match sampler — the structure being demonstrated is the causal
rollout + KV + DMD loss.)
"""
from __future__ import annotations

import numpy as np

from ..._enums import ConsistencyLevel
from ...request import DiffusionParams, TaskType, make_request
from .base import new_instance
from .dmd2 import DMD2Method


class SelfForcingMethod(DMD2Method):
    name = "self_forcing"
    consistency = ConsistencyLevel.C1
    student_loop_id = "chunk_rollout"                    # the SHARED causal loop (KV-cached)

    def _rollout_sample(self, prompt: str, seed: int) -> np.ndarray:
        req = make_request(TaskType.T2V, self.student.card.model_id, prompt,
                           diffusion=DiffusionParams(seed=seed))
        res = self._rollout(req)                         # drives chunk_rollout + slab-KV
        latents = res.outputs["latents"]
        return np.asarray(latents, dtype="float32")


def build_self_forcing(causal_card, **kw) -> SelfForcingMethod:
    """Build a self-forcing method over a Wan-causal card (chunk_rollout loop)."""
    return SelfForcingMethod(new_instance(causal_card), new_instance(causal_card),
                             new_instance(causal_card), **kw)
