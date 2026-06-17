#!/usr/bin/env python3
"""04 — Concurrent inference: step-interleaved batching + the parity gate (Wan2.1-1.3B).

Many requests share ONE resident instance; the engine interleaves their denoise *steps* (loop
inversion). Two guarantees this shows:

  * **The interleave parity gate (designv4 §6):** serial == interleaved, bit-for-bit. This is the test
    the whole loop-inversion bet lives on — per-request state lives in `LoopState`, so concurrency
    cannot smear one request into another.
  * **Shared-prompt feature-cache reuse:** a prompt repeated across requests (e.g. a K-sample group)
    encodes the text encoder ONCE; the rest are cache hits.

Run:  python3 v2_examples/inference/04_concurrent_interleaved.py
"""
from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from v2.models import build_default_engine
from v2.parity import assert_interleave_parity
from v2.request import DiffusionParams, TaskType, make_request

MODEL = "wan2.1-1.3b"


def _t2v(prompt, seed, steps=6):
    return make_request(TaskType.T2V, MODEL, prompt, diffusion=DiffusionParams(num_steps=steps, seed=seed))


def main() -> None:
    eng = build_default_engine()

    # a mixed batch: two distinct prompts + a repeated ("shared") prompt
    reqs = [_t2v("a desert at dawn", 11), _t2v("a city in the rain", 22),
            _t2v("a shared prompt", 7), _t2v("a shared prompt", 7)]

    serial = eng.run_serial(reqs)
    shapes = [tuple(serial[r.request_id].artifacts["video"].frames.shape) for r in reqs[:2]]
    print(f"ran {len(reqs)} requests; sample serial videos: {shapes} ...")

    # the gate runs them serially AND step-interleaved (round-robin one step/request/tick) and
    # bit-compares; any divergence is a list of Divergence; empty ⇒ PASS
    divs = assert_interleave_parity(eng, reqs)
    print(f"  interleave parity  : {'PASS — serial == interleaved, bit-identical ✓' if not divs else divs}")

    # shared-prompt feature-cache reuse (the text encoder runs once for the repeated prompt)
    fc = eng._registry[MODEL][0].caches.stats()["feature"]
    print(f"  text feature cache : {fc['hits']} hits / {fc['misses']} misses "
          f"(the repeated prompt encodes once, then reuses)")


if __name__ == "__main__":
    main()
