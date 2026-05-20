"""Vendored Synchformer (Iashin et al., ICASSP 2024) — audio-visual sync.

Copied verbatim from ``hkchengrex/av-benchmark`` under
``av_bench/synchformer/``. Internal imports rewritten to
``fastvideo.third_party.eval.synchformer.*``; no other changes. MIT
licensed (see ``LICENSE`` alongside).
"""
from fastvideo.third_party.eval.synchformer.synchformer import Synchformer, make_class_grid

__all__ = ["Synchformer", "make_class_grid"]
