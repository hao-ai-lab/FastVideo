"""Stable Audio Open (text→audio) — a v-prediction (EDM-v / VDenoiser) DPM++ recipe in the v2 substrate.

Self-contained recipe package (the bucket-C pattern): the card declares its torch adapters via
``ComponentSpec.adapter`` (``StableAudioDiT``/``OobleckVAE``/``StableAudioConditioner`` in
``v2/platform/backends/torch_stable_audio.py``) and a new ``StableAudioDenoiseLoop`` (polyexponential
schedule + VDenoiser v->x0 + DPM++ multistep), reusing ``stamp_wan21_checkpoints``. AUDIO modality —
the audio output artifact + a TEXT_TO_AUDIO capability/request field are the request-API extension
(BRINGUP). Registered in ``v2/registry.py`` by the orchestrator.
"""
from __future__ import annotations

from v2.recipes.stable_audio.card import build_stable_audio_card, build_stable_audio_small_card
from v2.recipes.stable_audio.loop import StableAudioDenoiseLoop
from v2.recipes.stable_audio.program import build_stable_audio_program

__all__ = [
    "build_stable_audio_card",
    "build_stable_audio_small_card",
    "build_stable_audio_program",
    "StableAudioDenoiseLoop",
]
