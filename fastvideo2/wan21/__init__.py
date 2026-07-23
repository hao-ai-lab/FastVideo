"""Wan2.1 — the MVP model family.

One card (`WAN21_T2V_1_3B`), one driven loop (`WanDenoiseLoop`), one pipeline
(text_encode -> denoise -> vae_decode), and one self-contained eager oracle
(`reference.py`). ``reference.py`` is deliberately not imported here: it is a
standalone file you can copy out of the repo, and the production path must
never depend on it (the T2 gate compares the two).
"""
from fastvideo2.wan21.card import FASTWAN_QAD_FP8_1_3B, FASTWAN_T2V_1_3B, SFWAN_T2V_1_3B, WAN21_T2V_1_3B
from fastvideo2.wan21.pipeline import build_wan_t2v_pipeline

__all__ = ["FASTWAN_QAD_FP8_1_3B", "FASTWAN_T2V_1_3B", "SFWAN_T2V_1_3B", "WAN21_T2V_1_3B", "build_wan_t2v_pipeline"]
