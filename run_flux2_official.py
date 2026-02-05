#!/usr/bin/env python3
"""
Run official FLUX.2 Klein 4B via Diffusers for comparison with FastVideo.
Use same prompt, seed, resolution, and steps as your FastVideo run.

Usage:
  pip install -U diffusers transformers accelerate
  python run_flux2_official.py
"""
import torch
from diffusers import Flux2KleinPipeline

def main():
    device = "cuda"
    dtype = torch.bfloat16

    print("Loading Flux2KleinPipeline from black-forest-labs/FLUX.2-klein-4B ...")
    pipe = Flux2KleinPipeline.from_pretrained(
        "black-forest-labs/FLUX.2-klein-4B",
        torch_dtype=dtype,
    )
    pipe = pipe.to(device)

    prompt = "a red apple on a table"
    generator = torch.Generator(device=device).manual_seed(0)

    # Klein is 4-step distilled; guidance_scale 1.0
    print("Generating: prompt=%r, seed=0, 1024x1024, 4 steps" % (prompt,))
    image = pipe(
        prompt=prompt,
        height=1024,
        width=1024,
        guidance_scale=1.0,
        num_inference_steps=4,
        generator=generator,
    ).images[0]

    out_path = "flux-klein-official.png"
    image.save(out_path)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
