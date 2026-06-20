#!/usr/bin/env python3
"""01 — Basic text→video with Wan2.1-1.3B.

The minimal inference path on the v2 runtime:

    build_default_engine()              # registers the (recipe, runtime) cards
      → make_request(T2V, "wan2.1-1.3b", prompt, DiffusionParams(...))   # a typed Request
      → engine.run(req)                 # drives the diffusion_denoise loop to completion
      → out.artifacts["video"].frames   # the decoded video tensor [C, T, H, W]

Run:  python3 v2_examples/inference/01_basic_t2v.py
"""
from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))  # repo root → `import v2`

import numpy as np

from v2._vendor.models import build_default_engine
from v2.core.request import DiffusionParams, TaskType, make_request

MODEL = "wan2.1-1.3b"


def main() -> None:
    eng = build_default_engine()                      # one resident instance per card
    print("registered cards:", list(eng._registry))

    req = make_request(
        TaskType.T2V, MODEL, "a cat surfing a wave at sunset",
        diffusion=DiffusionParams(num_steps=12, guidance_scale=5.0, seed=7))
    out = eng.run(req)                                # synchronous, single request

    video = np.asarray(out.artifacts["video"].frames)         # [C, T, H, W]
    latent = np.asarray(out.artifacts["latents"].latent)      # the final denoised latent
    print(f"prompt   : {req.prompt()!r}")
    print(f"video    : shape={video.shape} dtype={video.dtype} "
          f"range=[{video.min():.3f}, {video.max():.3f}]")
    print(f"latent   : shape={latent.shape}")
    print(f"metrics  : denoise_steps={out.metrics['denoise_steps']:.0f} "
          f"gpu_seconds={out.metrics['gpu_seconds']:.2e}")

    print("\nThe shapes and control flow are real; the pixels are a numpy toy. On GPU, swap "
          "ComponentSpec.factory\nfor the torch Wan adapter — this script is unchanged.")


if __name__ == "__main__":
    main()
