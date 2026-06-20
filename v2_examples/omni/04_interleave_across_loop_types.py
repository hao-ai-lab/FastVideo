#!/usr/bin/env python3
"""04 — The interleave parity gate holds across loop *types* (omni).

The loop-inversion safety guarantee isn't limited to homogeneous denoise batches: concurrent omni
requests — each running an `ar_decode` loop THEN a `diffusion_denoise` loop — interleaved at step
granularity are still bit-identical to serial. Per-request state lives in `LoopState`, so mixing AR
tokens and denoise steps from different requests in one schedule cannot smear them (designv4 §6, §9.5).

Run:  python3 v2_examples/omni/04_interleave_across_loop_types.py
"""
from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from v2._vendor.models import build_omni_engine
from v2.core.parity import assert_interleave_parity
from v2.core.request import DiffusionParams, SamplingParams, TaskType, make_request


def _cosmos(prompt, seed):
    return make_request(TaskType.T2V, "cosmos3-vfm", prompt,
                        sampling=SamplingParams(max_tokens=5, seed=seed),
                        diffusion=DiffusionParams(num_steps=4, seed=seed))


def main() -> None:
    eng = build_omni_engine()
    reqs = [_cosmos("a comet", 11), _cosmos("a glacier", 22), _cosmos("a comet", 11)]
    divs = assert_interleave_parity(eng, reqs)
    print("3 omni requests (each ar_decode → diffusion_denoise), serial vs step-interleaved:")
    print(f"  interleave parity : {'PASS — bit-identical across AR + diffusion steps ✓' if not divs else divs}")
    print("  (mixing AR tokens and denoise steps from different requests cannot smear state)")


if __name__ == "__main__":
    main()
