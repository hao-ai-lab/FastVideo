"""SpeculativeARLoop — draft-verify (speculative) decoding (design_v3 §2.2; §9.16).

The AR-side analog of content-adaptive compute (§9.12): a cheap **draft** model proposes K tokens, the
**target** model verifies them, and the loop accepts the longest matching prefix plus one target
correction — a *variable* number of tokens per round (a ragged loop the model owns). Two properties:

  * **Exact.** Every accepted token is one the target would have produced greedily, and the correction
    is the target's own token — so the emitted sequence is *identical* to the target's standalone greedy
    decode. The speedup is free (same output, fewer sequential target steps).
  * **Latency win.** The target's per-round forwards are one batched verify step (here counted as one
    ``verify_round``); standalone decoding would take one *sequential* target step per token. So
    ``verify_rounds`` (the expensive model's latency steps) is well below the token count.

It binds TWO components (``draft`` + ``target``) on one resident instance — two models co-scheduled in
one request — and every round is an ``AR_TOKEN`` WorkUnit.
"""
from __future__ import annotations

from ..._enums import ExecutionProfile, WorkUnitKind
from ...loop.contracts import (
    Done,
    LoopResult,
    LoopState,
    ResourceRequest,
    ShapeSignature,
    StepResult,
    WorkPlan,
)
from ...request.streams import StreamChunk


class SpeculativeARLoop:
    def __init__(self, *, loop_id, draft_id="draft", target_id="target", cost, spec_len=4, max_tokens=12):
        self.loop_id = loop_id
        self.draft_id = draft_id
        self.target_id = target_id
        self.cost = cost
        self.spec_len = spec_len           # K: candidate tokens drafted per round
        self.max_tokens = max_tokens

    def init(self, req, model, ctx) -> LoopState:
        seed = req.sampling.seed if req.sampling.seed is not None else 0
        st = LoopState(loop_id=self.loop_id, instance_id=model.card.model_id,
                       request_id=req.request_id, profile=ctx.profile, seed=seed)
        st.scratch["tokens"] = list(ctx.slots.get("prompt_tokens") or [1])   # prefill
        st.scratch["generated"] = []
        st.scratch["max_tokens"] = min(req.sampling.max_tokens, self.max_tokens)
        st.scratch.update(verify_rounds=0, target_forwards=0, draft_forwards=0, accepted=0, drafted=0)
        return st

    def next(self, st: LoopState):
        if len(st.scratch["generated"]) >= st.scratch["max_tokens"]:
            return Done()
        ctx_tokens = list(st.scratch["tokens"])
        k, did, tid = self.spec_len, self.draft_id, self.target_id

        def run(model, override=None):
            draft, target = model.component(did), model.component(tid)
            # 1) DRAFT proposes k candidate tokens autoregressively (cheap)
            cand, c = [], list(ctx_tokens)
            for _ in range(k):
                t = int(draft.ar_forward(c)); cand.append(t); c.append(t)
            # 2) TARGET verifies (one batched step): accept the matching prefix, then 1 correction
            accepted, tf = [], 0
            for j in range(k):
                tj = int(target.ar_forward(ctx_tokens + cand[:j])); tf += 1
                if tj == cand[j]:
                    accepted.append(cand[j])               # draft agreed with target → accept (free)
                else:
                    accepted.append(tj)                    # mismatch → take the target's token, stop
                    break
            else:
                accepted.append(int(target.ar_forward(ctx_tokens + cand))); tf += 1   # all k accepted + 1
            return StepResult(output={"accepted": accepted, "target_forwards": tf, "drafted": k})

        pos = len(st.scratch["generated"])
        emits = [StreamChunk(stream_id=st.request_id, modality="text", seq=pos, preview=True)]
        return WorkPlan(
            loop_id=self.loop_id, instance_id=st.instance_id, kind=WorkUnitKind.AR_TOKEN,
            shape_sig=ShapeSignature(WorkUnitKind.AR_TOKEN, dims=(len(ctx_tokens), k)),
            resources=ResourceRequest(compute_seconds=self.cost.predict(len(ctx_tokens) + k),
                                      resident_bytes=8 * len(ctx_tokens), peak_activation_bytes=8 * k),
            payload={"round": st.scratch["verify_rounds"]}, run=run, emits=emits,
            label=f"spec.round{st.scratch['verify_rounds']}")

    def advance(self, st: LoopState, result: StepResult) -> LoopState:
        room = st.scratch["max_tokens"] - len(st.scratch["generated"])
        accepted = list(result.output["accepted"])[:room]          # don't overshoot max_tokens
        st.scratch["tokens"].extend(accepted)
        st.scratch["generated"].extend(accepted)
        st.scratch["verify_rounds"] += 1
        st.scratch["target_forwards"] += int(result.output["target_forwards"])
        st.scratch["drafted"] += int(result.output["drafted"])
        st.scratch["accepted"] += len(accepted)
        st.step_idx += 1
        if st.profile == ExecutionProfile.ROLLOUT:
            st.trajectory.append({"round": st.step_idx, "accepted": len(accepted)})
        return st

    def finalize(self, st: LoopState) -> LoopResult:
        toks = st.scratch["generated"]
        rounds = max(1, st.scratch["verify_rounds"])
        return LoopResult(
            outputs={"tokens": toks, "text": "tok:" + ",".join(map(str, toks))},
            metrics={"tokens": float(len(toks)), "verify_rounds": float(st.scratch["verify_rounds"]),
                     "target_forwards": float(st.scratch["target_forwards"]),
                     "draft_forwards": float(st.scratch["drafted"]),
                     "tokens_per_round": float(len(toks)) / rounds},
            behavior=st.trajectory or None)
