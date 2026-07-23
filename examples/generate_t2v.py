#!/usr/bin/env python3
"""Text-to-video with the fastvideo2 SDK — the canonical example.

    python examples/generate_t2v.py --prompt "a cat surfing a wave" --out cat.mp4

Loads the card resident once, generates with card defaults (50 steps, 81
frames, 480x832 — override anything via flags), saves an mp4. Requires a CUDA
box; weights resolve from the HF cache on first use.
"""
from __future__ import annotations

import argparse

import fastvideo2 as fv2


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prompt",
                   default="a golden retriever puppy running through a sprinkler "
                           "on a sunny lawn, water droplets sparkling, slow motion, cinematic")
    p.add_argument("--model", default="wan2.1-t2v-1.3b",
                   help="a model id from the catalog (see `python -m fastvideo2 describe`)")
    p.add_argument("--out", default="out.mp4")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-steps", dest="num_steps", type=int, default=None,
                   help="unset -> the card's default")
    p.add_argument("--num-frames", dest="num_frames", type=int, default=None)
    p.add_argument("--guidance-scale", dest="guidance_scale", type=float, default=None)
    args = p.parse_args()

    model = fv2.load(args.model)
    print(model)

    overrides = {k: getattr(args, k) for k in ("seed", "num_steps", "num_frames", "guidance_scale")
                 if getattr(args, k) is not None}
    result = model.generate(args.prompt, **overrides)

    steps = [t for t in result.trace if "/denoise." in t["label"]]
    print(f"video {result.video.shape} | {len(steps)} denoise steps, "
          f"{sum(t['seconds'] for t in steps) / max(len(steps), 1):.2f}s/step, "
          f"{result.seconds:.1f}s total")
    print(f"saved -> {result.save(args.out)}")


if __name__ == "__main__":
    main()
