"""Stable Audio Open T2A program: conditioning -> stable_audio_denoise (v-prediction DPM++) -> audio_decode.

The SA specifics (the conditioner's (cross_attn_cond, global_embed) packing; the v-prediction sampler)
live in the conditioning node, the ``StableAudioDenoiseLoop``, and the torch adapters, so the node graph
stays a clean 3-node line.

Deltas vs the video programs:
  * The first node is ``conditioning`` (NOT the generic ``text_encode``): it runs the SA multi-conditioner
    over (prompt, seconds_start, seconds_total) and packs the triple into the loop's flat ``text_embeds``
    payload (so the loop->dit call stays modality-agnostic). It degrades to the generic text encoder on
    the CPU toy (which exposes only ``.encode(text)``), so the loop still runs end-to-end.
  * The last node is ``audio_decode`` (OobleckVAE.decode + slice [start,end] seconds) writing an ``audio``
    artifact (NOT ``video``). BRINGUP: the v2 output-artifact surface is video-shaped, so the ``audio``
    artifact declared here needs an audio-aware saver/output path to consume it. The slot value is the
    decoded waveform numpy array.

Audio duration knobs: ``DiffusionParams`` has no ``audio_start_in_s``/``audio_end_in_s`` (the v2 request
surface is video-shaped). They are read from ``request.node_params['conditioning']`` (the per-node
override surface) with the card's defaults as fallback. A first-class audio request field is TODO (BRINGUP).
"""
from __future__ import annotations

from typing import Any

from v2.program import ComponentNode, ModelLoopNode, Program, ProgramKind

# Default audio window (SA-1.0 short-clip defaults; overridable per request via node_params).
_DEFAULT_AUDIO_START_S = 0.0
_DEFAULT_AUDIO_END_S = 10.0


def _audio_window(request: Any) -> tuple[float, float]:
    """Resolve (start_s, end_s) from the per-node override surface (no audio fields on DiffusionParams)."""
    ov = request.node_override("conditioning") if hasattr(request, "node_override") else {}
    start = float(ov.get("audio_start_in_s", _DEFAULT_AUDIO_START_S))
    end = float(ov.get("audio_end_in_s", _DEFAULT_AUDIO_END_S))
    return start, end


def _encode_one(component: Any, prompt: str, start: float, end: float) -> Any:
    """Encode one prompt through the SA conditioner (``encode(prompt, seconds_start, seconds_total)``),
    falling back to the generic text encoder's ``encode(text)`` on the CPU toy (so the loop still runs)."""
    try:
        return component.encode(prompt or "", start, end)  # SA conditioner: packs (cross, global)
    except TypeError:
        return component.encode(prompt or "")  # ToyTextEncoder / generic T5: single-arg


def _conditioning(instance, slots, request, ctx) -> None:
    # The SA conditioner is declared as the ``text_encoder`` component (kind reuse — see card.py / the
    # adapter docstring). On a GPU box this is the SA multi-conditioner; on CPU it is the ToyTextEncoder.
    comp = instance.component("text_encoder")
    start, end = _audio_window(request)
    slots["text_embeds"] = _encode_one(comp, request.prompt(), start, end)
    slots["neg_text_embeds"] = _encode_one(comp, request.diffusion.negative_prompt, start, end)
    slots["audio_start_in_s"] = start
    slots["audio_end_in_s"] = end


def _audio_decode(instance, slots, request, ctx) -> None:
    # OobleckVAE.decode -> waveform [channels, samples]; slice to [start,end] seconds. The toy VAE returns
    # its own (video-ish) decode but the slot is still named ``audio`` — the artifact is the waveform.
    vae = instance.component("vae")
    waveform = vae.decode(slots["denoise_out"]["latents"])
    sr = int(getattr(vae, "sampling_rate", 44100))
    start = float(slots.get("audio_start_in_s", _DEFAULT_AUDIO_START_S))
    end = float(slots.get("audio_end_in_s", _DEFAULT_AUDIO_END_S))
    if getattr(waveform, "ndim", 0) >= 1 and waveform.shape[-1] >= int(end * sr) > 0:
        waveform = waveform[..., int(start * sr):int(end * sr)]  # slice to the requested window
    slots["audio"] = waveform
    slots["audio_sample_rate"] = sr


def build_stable_audio_program() -> Program:
    return Program(
        program_id="stable_audio.t2a.inline",
        kind=ProgramKind.INLINE,
        nodes=[
            ComponentNode("conditioning",
                          fn=_conditioning,
                          writes=("text_embeds", "neg_text_embeds", "audio_start_in_s", "audio_end_in_s")),
            ModelLoopNode("denoise",
                          loop_id="stable_audio_denoise",
                          output_slot="denoise_out",
                          reads=("text_embeds", "neg_text_embeds"),
                          writes=("denoise_out", )),
            ComponentNode("audio_decode",
                          fn=_audio_decode,
                          reads=("denoise_out", "audio_start_in_s", "audio_end_in_s"),
                          writes=("audio", "audio_sample_rate")),
        ],
        output_artifacts={
            "audio": "audio",
            "latents": "denoise_out",
        },
    ).validate()
