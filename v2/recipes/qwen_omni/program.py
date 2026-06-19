"""Qwen-Omni serve program (vllm-omni ``qwen2_5_omni`` pipeline, made step-visible):

    tokenize → thinker_decode → emit_text
             → thinker→talker (full payload: tokens + hidden state)
             → talker_decode
             → talker→vocoder (speech tokens)
             → vocoder (audio_decode) → emit_audio

The hand-off nodes are the model-native analogue of vllm-omni's ``custom_process_input_func``
(``thinker2talker_full_payload``, ``talker2code2wav_full_payload``), but here they are explicit slot
writes between three driven loops, not opaque stage-boundary callbacks. The talker conditions on the
thinker's hidden state (``reasoner_embed``), not only its tokens — the "full payload" path; the
streaming/sync path would pass tokens only. Final outputs: ``text`` (thinker) and ``audio`` (vocoder).
"""
from __future__ import annotations

import numpy as np

from v2.program import ComponentNode, ModelLoopNode, Program, ProgramKind
from v2.recipes.omni import emit_text_node, tokenize_node


def thinker_to_talker_node(instance, slots, request, ctx) -> None:
    """Full-payload hand-off: condition the talker on the thinker's TOKENS *and* its hidden state.
    (vllm-omni ``thinker2talker_full_payload``; the streaming variant ``thinker2talker_token_only``
    would write only the tokens.)"""
    out = slots.get("thinker_out", {})
    toks = out.get("tokens", []) if isinstance(out, dict) else []
    hidden = instance.component("thinker").reasoner_embed(toks or [1])  # thinker hidden state
    h = int(abs(float(np.sum(hidden))) * 1000.0) % 251 + 1  # one hidden-derived token
    slots["talker_prompt_tokens"] = list(toks) + [h]


def talker_to_vocoder_node(instance, slots, request, ctx) -> None:
    """Speech codec tokens → the vocoder's input slot (vllm-omni ``talker2code2wav``)."""
    out = slots.get("talker_out", {})
    slots["speech_tokens"] = out.get("tokens", []) if isinstance(out, dict) else []


def emit_audio_node(loop_slot: str, out_slot: str = "audio"):

    def fn(instance, slots, request, ctx) -> None:
        out = slots.get(loop_slot, {})
        slots[out_slot] = out.get("samples") if isinstance(out, dict) else out

    return fn


def build_qwen_omni_program() -> Program:
    return Program(
        program_id="qwen_omni.tts",
        kind=ProgramKind.INLINE,
        nodes=[
            ComponentNode("tokenize", fn=tokenize_node, writes=("prompt_tokens", )),
            ModelLoopNode("thinker",
                          loop_id="thinker_decode",
                          output_slot="thinker_out",
                          reads=("prompt_tokens", ),
                          writes=("thinker_out", )),
            ComponentNode("emit_text",
                          fn=emit_text_node("thinker_out", "text"),
                          reads=("thinker_out", ),
                          writes=("text", )),
            ComponentNode("thinker_to_talker",
                          fn=thinker_to_talker_node,
                          reads=("thinker_out", ),
                          writes=("talker_prompt_tokens", )),
            ModelLoopNode("talker",
                          loop_id="talker_decode",
                          output_slot="talker_out",
                          reads=("talker_prompt_tokens", ),
                          writes=("talker_out", )),
            ComponentNode("talker_to_vocoder",
                          fn=talker_to_vocoder_node,
                          reads=("talker_out", ),
                          writes=("speech_tokens", )),
            ModelLoopNode("vocoder",
                          loop_id="vocoder",
                          output_slot="vocoder_out",
                          reads=("speech_tokens", ),
                          writes=("vocoder_out", )),
            ComponentNode("emit_audio",
                          fn=emit_audio_node("vocoder_out", "audio"),
                          reads=("vocoder_out", ),
                          writes=("audio", )),
        ],
        output_artifacts={
            "text": "text",
            "audio": "audio",
            "speech_tokens": "talker_out"
        },
    ).validate()
