"""Vendored Synchformer (Iashin et al., 2024) — audio-visual sync.

Copied verbatim from ``hkchengrex/av-benchmark`` (commit at clone
time) under ``av_bench/synchformer/``. Upstream sources:

* Paper: Iashin et al., ICASSP 2024 — Synchformer.
* Repo (model): https://github.com/v-iashin/Synchformer
* Repo (eval glue): https://github.com/hkchengrex/av-benchmark

MIT licensed (see ``LICENSE`` alongside).

The leading underscore in the package name keeps the eval metric
auto-discovery from importing this as a metric — same pattern as
``_glmasr/``. Internal imports were mechanically rewritten from
``av_bench.synchformer.*`` to
``fastvideo.eval.metrics.audio._synchformer.*``; no other changes.
"""
from fastvideo.eval.metrics.audio._synchformer.synchformer import Synchformer, make_class_grid

__all__ = ["Synchformer", "make_class_grid"]
