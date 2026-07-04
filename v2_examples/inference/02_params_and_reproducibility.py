#!/usr/bin/env python3
"""02 — Inference knobs + reproducibility (Wan2.1-1.3B).

`DiffusionParams` controls the denoise: number of steps, CFG `guidance_scale`, `seed`, and resolution
(`height`/`width`/`num_frames`). Two properties this demonstrates:

  * **Reproducibility (a C1 property):** same seed ⇒ bit-identical output; different seed ⇒ different.
  * **Resolution drives geometry + flow-shift bucket:** 480p and 720p produce different latent shapes
    and select different flow-shift (480p→3.0, 720p→5.0, per the Wan card's bucket table).

Run:  python3 v2_examples/inference/02_params_and_reproducibility.py
"""
from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import numpy as np

from v2.recipes import build_default_engine
from v2.core.request import DiffusionParams, TaskType, make_request

MODEL = "wan2.1-1.3b"


def _video(eng, **dp) -> np.ndarray:
    req = make_request(TaskType.T2V, MODEL, "a forest stream over mossy rocks",
                       diffusion=DiffusionParams(**dp))
    return np.asarray(eng.run(req).artifacts["video"].frames)


def main() -> None:
    eng = build_default_engine()

    print("== reproducibility (fixed seed) ==")
    a = _video(eng, num_steps=8, seed=42)
    b = _video(eng, num_steps=8, seed=42)
    c = _video(eng, num_steps=8, seed=43)
    print(f"  same seed (42 vs 42)      bit-identical : {np.array_equal(a, b)}")
    print(f"  different seed (42 vs 43) differs       : {not np.array_equal(a, c)}")

    print("\n== num_steps (latency/quality knob) ==")
    for steps in (2, 8, 25):
        out = eng.run(make_request(TaskType.T2V, MODEL, "a comet", diffusion=DiffusionParams(num_steps=steps, seed=1)))
        print(f"  num_steps={steps:>2}  metrics={out.metrics}")

    print("\n== guidance_scale (CFG strength) ==")
    base = _video(eng, num_steps=6, seed=1, guidance_scale=1.0)        # 1.0 ⇒ effectively no CFG
    for g in (5.0, 9.0):
        v = _video(eng, num_steps=6, seed=1, guidance_scale=g)
        print(f"  guidance={g:>3}  differs from guidance=1.0 : {not np.array_equal(base, v)}")

    print("\n== resolution → latent geometry + flow-shift bucket ==")
    lo = _video(eng, num_steps=4, seed=1, height=480, width=832, num_frames=81)
    hi = _video(eng, num_steps=4, seed=1, height=720, width=1280, num_frames=81)
    print(f"  480x832  → video {lo.shape}")
    print(f"  720x1280 → video {hi.shape}   (larger grid; the card maps the 720p flow-shift bucket)")


if __name__ == "__main__":
    main()
