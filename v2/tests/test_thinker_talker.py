"""Qwen-Omni thinker→talker→vocoder — the third weight-sharing topology (design_v3 §4; vllm-omni
``qwen2_5_omni`` pipeline).

Asserts the Card/Loop/Program vocabulary holds for a THREE-expert, THREE-loop cascade — the topology
neither MoT (one shared module, two loops) nor joint-RL (two disjoint experts, two loops) covers:

  * three disjoint experts (thinker, talker, vocoder) on three loop *types* (ar_decode, ar_decode,
    audio_decode) in one request — every token and vocoder chunk a runtime-visible WorkUnit;
  * cascaded conditioning — the talker reads the thinker's tokens+hidden state, the vocoder the
    talker's speech tokens (the model-native form of vllm-omni's ``custom_process_input_func``);
  * streaming codec→waveform via AUDIO_CHUNK WorkUnits;
  * the three-loop program still passes the interleave parity gate.
"""
from __future__ import annotations

import numpy as np

from v2._enums import LoopKind, WorkUnitKind
from v2.card import load_card
from v2.cache import CacheManager
from v2.recipes import build_qwen_omni_card, build_qwen_omni_program
from v2.recipes.qwen_omni.program import thinker_to_talker_node
from v2.parity import assert_interleave_parity
from v2.request import OutputSpec, TaskType, make_request
from v2.runtime import Engine


def _engine():
    card = build_qwen_omni_card()
    inst = load_card(card, cache_manager=CacheManager.from_card(card))
    eng = Engine()
    eng.register(card.model_id, inst, build_qwen_omni_program())
    return eng, inst


def _run(eng, prompt="hello world", *, stream_audio=False):
    out = OutputSpec(stream={"audio": True}) if stream_audio else OutputSpec()
    return eng.run(make_request(TaskType.T2A, "qwen-omni-tts", prompt, outputs=out))


# --- card / serve shape ----------------------------------------------------------- #

def test_qwen_omni_card_validates_and_serves_text_and_audio():
    eng, _ = _engine()
    res = _run(eng, "Explain audio generation in 15 words.")
    assert res.artifacts["text"].text.startswith("tok:")          # thinker text
    aud = res.artifacts["audio"]
    samples = np.asarray(aud.samples)
    assert samples.ndim == 1 and samples.size > 0                 # vocoder waveform
    assert aud.sample_rate == 44100                               # AudioArtifact, first-class rate
    assert float(np.abs(samples).max()) <= 1.0                    # bounded waveform


def test_three_disjoint_experts_three_loop_types():
    """The third topology: three loops bind three DISTINCT components (no sharing) — vs Cosmos3 where
    two loops bind the SAME MoT module."""
    card = build_qwen_omni_card()
    kinds = {lid: spec.kind for lid, spec in card.loops.items()}
    assert kinds == {"thinker_decode": LoopKind.AR_DECODE,
                     "talker_decode": LoopKind.AR_DECODE,
                     "vocoder": LoopKind.AUDIO_DECODE}
    assert card.loops["vocoder"].work_unit_kind == WorkUnitKind.AUDIO_CHUNK
    bound = [card.loops[l].shared_weight_components for l in ("thinker_decode", "talker_decode", "vocoder")]
    assert bound == [["thinker"], ["talker"], ["vocoder"]]        # three disjoint singletons
    _, inst = _engine()
    experts = {id(inst.component(c)) for c in ("thinker", "talker", "vocoder")}
    assert len(experts) == 3                                      # genuinely separate weights


# --- cascade conditioning (the hand-offs are real) -------------------------------- #

def test_thinker_to_talker_passes_full_payload():
    """The talker prefill = the thinker's tokens PLUS a hidden-state-derived token (vllm-omni's
    ``thinker2talker_full_payload``)."""
    _, inst = _engine()
    slots = {"thinker_out": {"tokens": [11, 22, 33]}}
    thinker_to_talker_node(inst, slots, None, None)
    pre = slots["talker_prompt_tokens"]
    assert pre[:3] == [11, 22, 33]                                # thinker tokens carried forward
    assert len(pre) == 4                                          # + one hidden-state-derived token


def test_cascade_propagates_thinker_through_to_audio():
    """Different prompt → different thinker tokens → different talker speech → different waveform.
    Proves the three loops are conditioned, not independent."""
    eng_a, _ = _engine()
    eng_b, _ = _engine()
    a = _run(eng_a, "alpha")
    b = _run(eng_b, "beta gamma delta")
    assert a.artifacts["text"].text != b.artifacts["text"].text
    assert not np.array_equal(np.asarray(a.artifacts["audio"].samples),
                              np.asarray(b.artifacts["audio"].samples))


# --- vocoder as runtime-visible AUDIO_CHUNK WorkUnits + streaming ------------------ #

def test_vocoder_streams_audio_chunks():
    eng_off, _ = _engine()
    eng_on, _ = _engine()
    off = _run(eng_off, stream_audio=False)
    on = _run(eng_on, stream_audio=True)
    n_chunks = on.metrics["audio_chunks"]
    assert n_chunks >= 1
    # the audio-streaming delta is exactly the vocoder's per-chunk emits (AR text emits are common)
    assert on.metrics["stream_chunks"] - off.metrics["stream_chunks"] == n_chunks


# --- the three-loop program still passes the core parity gate --------------------- #

def test_qwen_omni_interleave_parity():
    eng, _ = _engine()
    reqs = [make_request(TaskType.T2A, "qwen-omni-tts", p) for p in ("alpha", "beta", "alpha")]
    divs = assert_interleave_parity(eng, reqs)
    assert not divs, f"thinker→talker→vocoder failed interleave parity: {divs}"
