#!/usr/bin/env python3
"""01 — Cosmos3: reason → joint denoise on ONE resident MoT instance.

A single request runs the reasoner (`ar_decode`, the und pathway) to upsample the prompt, packs its
tokens into conditioning, then the joint generation (`diffusion_denoise`) — **both loops bound to the
same `transformer`** (shared weights). The §16 claim no DAG-of-engines can express, made native: both
the AR tokens and the denoise steps are WorkUnits the scheduler prices and interleaves.

Run:  python3 v2_examples/omni/01_cosmos3.py
"""
from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import numpy as np

from v2.recipes import build_omni_engine
from v2.core.request import DiffusionParams, SamplingParams, TaskType, make_request


def main() -> None:
    eng = build_omni_engine()                           # registers cosmos3-vfm, bagel-mot, qwen-omni-tts
    out = eng.run(make_request(
        TaskType.T2V, "cosmos3-vfm", "a phoenix rising over a volcano",
        sampling=SamplingParams(max_tokens=6, seed=1),  # the reasoner (ar_decode)
        diffusion=DiffusionParams(num_steps=4, seed=1)))  # the joint denoise (diffusion_denoise)

    print("Cosmos3 (one MoT instance, two loop types on shared weights):")
    print(f"  reasoner text : {out.artifacts['text'].text}")
    print(f"  video         : {np.asarray(out.artifacts['video'].frames).shape}")
    print(f"  metrics       : reasoner tokens + denoise steps both ran as WorkUnits ({out.metrics})")


if __name__ == "__main__":
    main()
