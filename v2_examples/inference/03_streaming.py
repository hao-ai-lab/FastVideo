#!/usr/bin/env python3
"""03 — Streaming previews (Wan2.1-1.3B).

Pass `OutputSpec(stream={"video": True})` and the denoise loop emits one preview chunk per step (a
`StreamChunk` carrying the in-progress latent), surfaced as `media.chunk` events and counted in
`metrics["stream_chunks"]`. Streaming is **off by default** — the serve path is byte-for-byte the same
whether or not anyone subscribes, so previews never change the result.

Run:  python3 v2_examples/inference/03_streaming.py
"""
from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import numpy as np

from v2._vendor.models import build_default_engine
from v2.core.request import DiffusionParams, OutputSpec, TaskType, make_request

MODEL = "wan2.1-1.3b"
STEPS = 10


def _run(eng, *, stream: bool):
    outputs = OutputSpec(stream={"video": True}) if stream else OutputSpec()
    req = make_request(TaskType.T2V, MODEL, "a hot air balloon over a valley",
                       diffusion=DiffusionParams(num_steps=STEPS, seed=5), outputs=outputs)
    return eng.run(req)


def main() -> None:
    eng = build_default_engine()

    off = _run(eng, stream=False)
    on = _run(eng, stream=True)

    print(f"streaming OFF : stream_chunks={off.metrics.get('stream_chunks', 0):.0f}")
    print(f"streaming ON  : stream_chunks={on.metrics.get('stream_chunks', 0):.0f}  "
          f"(one preview per denoise step ⇒ == num_steps={STEPS})")

    # streaming must NOT change the result — it's a read-only preview channel
    same = np.array_equal(np.asarray(off.artifacts["video"].frames),
                          np.asarray(on.artifacts["video"].frames))
    print(f"result identical with/without streaming : {same}")
    print("\n(For LIVE async event streaming — media.chunk as it happens — see 05_async_serving.py.)")


if __name__ == "__main__":
    main()
