#!/usr/bin/env python3
"""05 — Async / concurrent serving of Wan2.1-1.3B (AsyncEngine).

`AsyncEngine` wraps the same step-scheduled engine with a request queue, lifecycle state, live event
streaming, and step-boundary cancellation. This shows the three async entry points:

  * `generate_many(reqs)` — await several requests concurrently (steps interleave under the hood);
  * `submit(req)` — an async iterator of `OmniEvent`s (`request.start` → `media.chunk`* →
    `artifact.ready` → `request.complete`);
  * `cancel(request_id)` — trips the CancelScope; the request stops at the next step boundary and ends
    with `request.cancelled`.

Run:  python3 v2_examples/inference/05_async_serving.py
"""
from __future__ import annotations

import asyncio
import pathlib
import sys
from collections import Counter

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from v2.models import build_default_engine
from v2.request import DiffusionParams, OutputSpec, TaskType, make_request
from v2.runtime import AsyncEngine

MODEL = "wan2.1-1.3b"


def _t2v(prompt, seed, *, steps=8, stream=False):
    outputs = OutputSpec(stream={"video": True}) if stream else OutputSpec()
    return make_request(TaskType.T2V, MODEL, prompt,
                        diffusion=DiffusionParams(num_steps=steps, seed=seed), outputs=outputs)


async def main() -> None:
    ae = AsyncEngine(build_default_engine())

    # (1) concurrent generation
    reqs = [_t2v(f"scene {i}", seed=i) for i in range(3)]
    outs = await ae.generate_many(reqs)
    print("concurrent generate_many:")
    for r in reqs:
        print(f"  {r.request_id[:8]}  video={tuple(outs[r.request_id].artifacts['video'].frames.shape)}")

    # (2) live event stream for one request
    print("\nlive event stream (submit):")
    events: Counter[str] = Counter()
    async for ev in ae.submit(_t2v("a streamed clip", seed=99, steps=6, stream=True)):
        events[ev.type] += 1
    print(f"  events: {dict(events)}")
    print("  (request.start → media.chunk per step → artifact.ready → request.complete)")

    # (3) step-boundary cancellation: cancel after the first streamed chunk
    print("\ncancellation (mid-rollout):")
    req = _t2v("a clip we abort", seed=7, steps=20, stream=True)
    seq = []
    async for ev in ae.submit(req):
        seq.append(ev.type)
        if ev.type == "media.chunk" and "cancel-sent" not in seq:
            ae.cancel(req.request_id)            # stop at the next step boundary
            seq.append("cancel-sent")
    print(f"  final state : {ae.state(req.request_id)}")
    print(f"  ended with  : {seq[-1]}  (cancelled before all {20} steps ran)")


if __name__ == "__main__":
    asyncio.run(main())
