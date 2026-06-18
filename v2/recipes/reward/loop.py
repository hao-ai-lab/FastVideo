"""RewardLoop — a served reward model scoring samples as ``REWARD_BATCH`` WorkUnits (design_v3 §10).

The reward in RL is computed by a *model* (PickScore/CLIP/a VLM verifier), not a heuristic. This loop
makes that a first-class part of the runtime: it reads a batch of K rollout media from a slot and emits
``REWARD_BATCH`` work units (chunked) that score them through the engine — so reward compute is admitted,
priced, and interleavable exactly like a denoise step or an AR token. A GRPO group's K samples score in
one (or a few) batched units — the homogeneous-batch win the scheduler likes.
"""
from __future__ import annotations

from v2._enums import WorkUnitKind
from v2.loop.contracts import (
    Done,
    LoopResult,
    LoopState,
    ResourceRequest,
    ShapeSignature,
    StepResult,
    WorkPlan,
)


class RewardLoop:
    def __init__(self, *, loop_id, scorer_id="scorer", cost, media_slot="media", batch=4):
        self.loop_id = loop_id
        self.scorer_id = scorer_id
        self.cost = cost
        self.media_slot = media_slot
        self.batch = batch

    def init(self, req, model, ctx) -> LoopState:
        media = list(ctx.slots.get(self.media_slot) or [])
        st = LoopState(loop_id=self.loop_id, instance_id=model.card.model_id,
                       request_id=req.request_id, profile=ctx.profile)
        st.scratch["batches"] = [media[i:i + self.batch] for i in range(0, len(media), self.batch)] or [[]]
        st.scratch["scores"] = []
        return st

    def next(self, st: LoopState):
        i = st.step_idx
        batches = st.scratch["batches"]
        if i >= len(batches):
            return Done()
        chunk, sid = batches[i], self.scorer_id

        def run(model, override=None):
            scorer = model.component(sid)
            return StepResult(output={"scores": [float(scorer.score(m)) for m in chunk], "i": i})

        return WorkPlan(
            loop_id=self.loop_id, instance_id=st.instance_id, kind=WorkUnitKind.REWARD_BATCH,
            shape_sig=ShapeSignature(WorkUnitKind.REWARD_BATCH, dims=(len(chunk),)),
            resources=ResourceRequest(compute_seconds=self.cost.predict(len(chunk)),
                                      resident_bytes=64 * max(1, len(chunk)), peak_activation_bytes=64),
            payload={"reward_batch": i}, run=run, label=f"reward.batch{i}")

    def advance(self, st: LoopState, result: StepResult) -> LoopState:
        st.scratch["scores"].extend(result.output["scores"])
        st.step_idx += 1
        return st

    def finalize(self, st: LoopState) -> LoopResult:
        return LoopResult(outputs={"rewards": list(st.scratch["scores"])},
                          metrics={"reward_batches": float(st.step_idx)})
