#!/usr/bin/env python3
"""
Dump step-0 inputs and official transformer output from Flux2KleinPipeline.
Uses same seed (0), prompt, and settings as run_flux2_official.py so you can
compare FastVideo's DiT to the official one.

Run once (requires diffusers from source with Flux2KleinPipeline):
  python dump_flux2_step0.py

Saves: flux2_step0_dump.pt (latent, timestep, prompt_embeds, official noise_pred, etc.)
"""
import torch

try:
    from diffusers import Flux2KleinPipeline
except ImportError:
    from diffusers.pipelines.flux2 import Flux2KleinPipeline

from diffusers.pipelines.flux2.pipeline_flux2_klein import (
    compute_empirical_mu,
    retrieve_timesteps,
)

DUMP_PATH = "flux2_step0_dump.pt"
PROMPT = "a red apple on a table"
HEIGHT, WIDTH = 1024, 1024
NUM_STEPS = 4
SEED = 0


def main():
    device = "cuda"
    dtype = torch.bfloat16
    generator = torch.Generator(device=device).manual_seed(SEED)

    print("Loading Flux2KleinPipeline ...")
    pipe = Flux2KleinPipeline.from_pretrained(
        "black-forest-labs/FLUX.2-klein-4B",
        torch_dtype=dtype,
    )
    pipe = pipe.to(device)

    # 1. Encode prompt
    print("Encoding prompt ...")
    prompt_embeds, text_ids = pipe.encode_prompt(
        prompt=PROMPT,
        device=device,
        num_images_per_prompt=1,
        max_sequence_length=512,
        text_encoder_out_layers=(9, 18, 27),
    )

    # 2. Prepare latents (packed: B, H*W, C)
    num_channels_latents = pipe.transformer.config.in_channels // 4
    latents, latent_ids = pipe.prepare_latents(
        batch_size=1,
        num_latents_channels=num_channels_latents,
        height=HEIGHT,
        width=WIDTH,
        dtype=prompt_embeds.dtype,
        device=device,
        generator=generator,
        latents=None,
    )

    # 3. Timesteps with same mu as official
    image_seq_len = latents.shape[1]
    mu = compute_empirical_mu(image_seq_len=image_seq_len, num_steps=NUM_STEPS)
    timesteps, _ = retrieve_timesteps(
        pipe.scheduler,
        NUM_STEPS,
        device=device,
        sigmas=None,
        mu=mu,
    )
    pipe.scheduler.set_begin_index(0)

    # 4. Step 0 only
    i = 0
    t = timesteps[i]
    timestep = t.expand(latents.shape[0]).to(latents.dtype)
    latent_model_input = latents.to(pipe.transformer.dtype)
    latent_image_ids = latent_ids

    print("Running official transformer for step 0 ...")
    with pipe.transformer.cache_context("cond"):
        noise_pred = pipe.transformer(
            hidden_states=latent_model_input,
            timestep=timestep / 1000.0,
            guidance=None,
            encoder_hidden_states=prompt_embeds,
            txt_ids=text_ids,
            img_ids=latent_image_ids,
            joint_attention_kwargs=getattr(pipe, "attention_kwargs", None) or {},
            return_dict=False,
        )[0]
    noise_pred = noise_pred[:, : latents.size(1)]

    # 5. Save (cpu, float32 for portability)
    to_save = {
        "latent_model_input": latent_model_input.detach().cpu().float(),
        "timestep_scaled": (timestep / 1000.0).detach().cpu().float(),
        "timestep_raw": timestep.detach().cpu(),
        "prompt_embeds": prompt_embeds.detach().cpu().float(),
        "text_ids": text_ids.detach().cpu(),
        "latent_ids": latent_ids.detach().cpu(),
        "noise_pred_official": noise_pred.detach().cpu().float(),
        "height": HEIGHT,
        "width": WIDTH,
        "num_steps": NUM_STEPS,
        "seed": SEED,
        "prompt": PROMPT,
    }
    torch.save(to_save, DUMP_PATH)
    print(f"Saved {DUMP_PATH}")
    print(f"  latent_model_input: {to_save['latent_model_input'].shape}")
    print(f"  timestep_scaled: {to_save['timestep_scaled'].shape} (value ~{to_save['timestep_scaled'].item():.6f})")
    print(f"  prompt_embeds: {to_save['prompt_embeds'].shape}")
    print(f"  noise_pred_official: {to_save['noise_pred_official'].shape}")


if __name__ == "__main__":
    main()
