"""VocoderLoop — streaming code2wav (``LoopKind.AUDIO_DECODE``; Qwen-Omni stage 2).

The third loop type in the thinker→talker→vocoder cascade. It consumes the talker's speech-codec
tokens in fixed-size chunks; each chunk is one ``AUDIO_CHUNK`` WorkUnit that synthesizes and streams
a waveform segment. Unlike vllm-omni's opaque code2wav stage, every chunk here is a runtime-visible,
step-scheduled, cancellable WorkUnit on the same driven-loop contract used for denoise steps and AR
tokens. All per-request state lives in ``LoopState``.
"""
from __future__ import annotations

import numpy as np

from v2._enums import ExecutionProfile, WorkUnitKind
from v2.loop.contracts import (
    Done,
    LoopResult,
    LoopState,
    ResourceRequest,
    ShapeSignature,
    StepResult,
    WorkPlan,
)
from v2.request.streams import StreamChunk


class VocoderLoop:

    def __init__(self, *, loop_id, vocoder_id="vocoder", cost, chunk_tokens=2, speech_slot="speech_tokens"):
        self.loop_id = loop_id
        self.vocoder_id = vocoder_id
        self.cost = cost
        self.chunk_tokens = chunk_tokens
        self.speech_slot = speech_slot

    def init(self, req, model, ctx) -> LoopState:
        toks = list(ctx.slots.get(self.speech_slot) or [1])
        st = LoopState(loop_id=self.loop_id,
                       instance_id=model.card.model_id,
                       request_id=req.request_id,
                       profile=ctx.profile)
        st.scratch["tokens"] = toks
        st.scratch["chunks"] = [toks[i:i + self.chunk_tokens] for i in range(0, len(toks), self.chunk_tokens)] or [[1]]
        st.scratch["wave"] = []
        st.scratch["stream_audio"] = bool(req.outputs.stream.get("audio"))
        return st

    def next(self, st: LoopState):
        i = st.step_idx
        chunks = st.scratch["chunks"]
        if i >= len(chunks):
            return Done()
        chunk = chunks[i]
        vid = self.vocoder_id

        def run(model, override=None):
            seg = model.component(vid).synthesize(chunk)  # codec tokens → waveform segment
            return StepResult(output={"segment": np.asarray(seg, dtype="float32")})

        emits = []
        if st.scratch.get("stream_audio"):
            emits.append(StreamChunk(stream_id=st.request_id, modality="audio", seq=i, preview=False))
        return WorkPlan(loop_id=self.loop_id,
                        instance_id=st.instance_id,
                        kind=WorkUnitKind.AUDIO_CHUNK,
                        shape_sig=ShapeSignature(WorkUnitKind.AUDIO_CHUNK, dims=(len(chunk), )),
                        resources=ResourceRequest(compute_seconds=self.cost.predict(len(chunk)),
                                                  resident_bytes=8 * len(st.scratch["tokens"]),
                                                  peak_activation_bytes=64),
                        payload={"chunk": i},
                        run=run,
                        emits=emits,
                        label=f"vocoder.chunk{i}")

    def advance(self, st: LoopState, result: StepResult) -> LoopState:
        st.scratch["wave"].append(np.asarray(result.output["segment"], dtype="float32"))
        st.step_idx += 1
        if st.profile == ExecutionProfile.ROLLOUT:
            st.trajectory.append({"chunk": st.step_idx, "samples": int(result.output["segment"].size)})
        return st

    def finalize(self, st: LoopState) -> LoopResult:
        wave = np.concatenate(st.scratch["wave"]) if st.scratch["wave"] else np.zeros(0, dtype="float32")
        return LoopResult(outputs={"samples": wave.astype("float32")},
                          metrics={
                              "audio_chunks": float(st.step_idx),
                              "audio_samples": float(wave.size)
                          },
                          behavior=st.trajectory or None)
