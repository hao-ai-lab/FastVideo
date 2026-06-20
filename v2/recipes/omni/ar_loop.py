"""ARDecodeLoop — token decode until EOS/max_tokens.

The autoregressive pathway: ``init`` prefills the prompt tokens (and allocates paged text-KV);
``next`` emits one ``ar_token`` WorkUnit (kernel-free); ``advance`` appends the sampled token and
streams it; ``done`` on EOS/max_tokens. It binds the SAME ``transformer`` component as the omni
model's ``diffusion_denoise`` loop — one resident MoT instance running both loop types. All
per-request state (the token list, paged-KV namespace) lives in ``LoopState``.
"""
from __future__ import annotations

from v2.core.enums import ExecutionProfile, WorkUnitKind
from v2.core.loop.contracts import (
    Done,
    LoopResult,
    LoopState,
    ResourceRequest,
    ShapeSignature,
    StepResult,
    WorkPlan,
)
from v2.core.request.streams import StreamChunk


class ARDecodeLoop:

    def __init__(self, *, loop_id, transformer_id="transformer", max_tokens=8, prompt_slot="prompt_tokens"):
        self.loop_id = loop_id
        self.transformer_id = transformer_id
        self.max_tokens = max_tokens
        self.prompt_slot = prompt_slot  # which slot holds the prefill (chained AR loops differ)

    def init(self, req, model, ctx) -> LoopState:
        seed = req.sampling.seed if req.sampling.seed is not None else 0
        st = LoopState(loop_id=self.loop_id,
                       instance_id=model.card.model_id,
                       request_id=req.request_id,
                       profile=ctx.profile,
                       seed=seed)
        prompt_tokens = ctx.slots.get(self.prompt_slot) or [1]
        st.scratch["tokens"] = list(prompt_tokens)  # prefill (folded into init for the toy)
        st.scratch["generated"] = []
        st.scratch["max_tokens"] = min(req.sampling.max_tokens, self.max_tokens)
        caches = getattr(model, "caches", None)
        if caches is not None and caches.has("paged_kv"):  # reasoner paged text-KV (minority case)
            caches.pool("paged_kv").allocate(req.request_id, n_blocks=1)
        return st

    def next(self, st: LoopState):
        if len(st.scratch["generated"]) >= st.scratch["max_tokens"]:
            return Done()
        tokens = list(st.scratch["tokens"])
        tid = self.transformer_id

        def run(model, override=None):
            nxt = model.component(tid).ar_forward(tokens)  # und pathway of the shared MoT weights
            return StepResult(output={"token": int(nxt)})

        pos = len(st.scratch["generated"])
        emits = [StreamChunk(stream_id=st.request_id, modality="text", seq=pos, preview=True)]
        return WorkPlan(loop_id=self.loop_id,
                        instance_id=st.instance_id,
                        kind=WorkUnitKind.AR_TOKEN,
                        shape_sig=ShapeSignature(WorkUnitKind.AR_TOKEN, dims=(len(tokens), )),
                        resources=ResourceRequest(
                                                  resident_bytes=8 * len(tokens),
                                                  peak_activation_bytes=8),
                        payload={
                            "branch": "ar",
                            "pos": pos
                        },
                        run=run,
                        emits=emits,
                        label=f"ar.tok{pos}")

    def advance(self, st: LoopState, result: StepResult) -> LoopState:
        tok = int(result.output["token"])
        st.scratch["tokens"].append(tok)
        st.scratch["generated"].append(tok)
        st.step_idx += 1
        if st.profile == ExecutionProfile.ROLLOUT:
            st.trajectory.append({"token": tok, "pos": st.step_idx})
        return st

    def finalize(self, st: LoopState) -> LoopResult:
        toks = st.scratch["generated"]
        return LoopResult(outputs={
            "tokens": toks,
            "text": "tok:" + ",".join(map(str, toks))
        },
                          metrics={"tokens": float(len(toks))},
                          behavior=st.trajectory or None)
