#!/usr/bin/env python3
"""02 — BAGEL: generate_text → generate_image on ONE resident MoT instance.

The canonical vllm-omni unified AR+diffusion model. `generate_text` (`ar_decode`) and `generate_image`
(`diffusion_denoise`) both bind the one resident `transformer`. Unlike vllm-omni's opaque `DIFFUSION`
stage, BOTH loops are runtime-visible here — the scheduler prices `ar_token` AND `diffusion_step`
WorkUnits. (BAGEL is MoT/shared-weight, the same topology row as Cosmos3 — see designv4 §2.3.)

Run:  python3 v2_examples/omni/02_bagel.py
"""
from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import numpy as np

from v2._vendor.models import build_omni_engine
from v2.core.request import DiffusionParams, SamplingParams, TaskType, make_request


def main() -> None:
    eng = build_omni_engine()
    out = eng.run(make_request(
        TaskType.T2I, "bagel-mot", "a porcelain teapot on a windowsill",
        sampling=SamplingParams(max_tokens=6, seed=2),
        diffusion=DiffusionParams(num_steps=4, seed=2)))

    print("BAGEL (generate_text → generate_image, shared MoT weights):")
    print(f"  text  : {out.artifacts['text'].text}")
    print(f"  image : {np.asarray(out.artifacts['image'].tensor).shape}")
    # the scheduler saw BOTH WorkUnit kinds (not one opaque stage)
    by_kind = dict(eng.admission.metrics.by_kind)
    print(f"  WorkUnit kinds priced by the scheduler: {by_kind}")


if __name__ == "__main__":
    main()
