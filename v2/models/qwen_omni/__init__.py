"""Qwen-Omni thinker→talker→vocoder model (vllm-omni ``qwen2_5_omni``): three disjoint experts, three
loops (ar_decode → ar_decode → audio_decode), cascaded conditioning + streaming codec→waveform. The
third weight-sharing topology proving the Card/Loop/Program vocabulary holds (design_v3 §4)."""
from __future__ import annotations

from v2.models.qwen_omni.card import build_qwen_omni_card
from v2.models.qwen_omni.program import build_qwen_omni_program

__all__ = ["build_qwen_omni_card", "build_qwen_omni_program"]
