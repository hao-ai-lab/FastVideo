#!/usr/bin/env python3
"""
Create step-0 dump for Flux (FLUX.1-dev) using diffusers and save:
  latent_model_input, timestep_scaled, prompt_embeds, pooled_projections,
  noise_pred_official

  python scripts/debug/dump_flux_step0.py [--model-id ID] [--dump PATH]
"""
import argparse
import os
import sys
from typing import Any

import torch

DEFAULT_MODEL_ID = "black-forest-labs/FLUX.1-dev"
DEFAULT_DUMP_PATH = "flux_step0_dump.pt"
DEFAULT_PROMPT = "A cinematic portrait of a fox, 35mm film, soft light, gentle grain."


def _encode_prompt(pipe, prompt: str, device: str, dtype: torch.dtype):
    # Try common diffusers Flux/SDXL-style signatures
    if hasattr(pipe, "encode_prompt"):
        try:
            return pipe.encode_prompt(
                prompt=prompt,
                prompt_2=None,
                device=device,
                num_images_per_prompt=1,
                do_classifier_free_guidance=False,
            )
        except TypeError:
            try:
                return pipe.encode_prompt(
                    prompt,
                    device=device,
                    num_images_per_prompt=1,
                    do_classifier_free_guidance=False,
                )
            except TypeError:
                return pipe.encode_prompt(prompt)
    if hasattr(pipe, "_encode_prompt"):
        return pipe._encode_prompt(
            prompt,
            device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=False,
        )
    raise AttributeError("Pipeline has no encode_prompt/_encode_prompt")


def _extract_prompt_embeds(encoded):
    # Handle common diffusers return patterns
    if isinstance(encoded, tuple):
        if len(encoded) == 2:
            prompt_embeds, pooled = encoded
            return prompt_embeds, pooled
        if len(encoded) == 4:
            prompt_embeds, _negative_prompt_embeds, pooled, _negative_pooled = encoded
            return prompt_embeds, pooled
        if len(encoded) >= 2:
            prompt_embeds = encoded[0]
            pooled = None
            if torch.is_tensor(prompt_embeds):
                for item in encoded[1:]:
                    if torch.is_tensor(item) and item.ndim == 2 and item.shape[0] == prompt_embeds.shape[0]:
                        pooled = item
                        break
            return prompt_embeds, pooled
    if isinstance(encoded, torch.Tensor):
        return encoded, None

    # Handle dict-like or ModelOutput objects
    if isinstance(encoded, dict):
        prompt_embeds = encoded.get("prompt_embeds") or encoded.get("text_embeds")
        pooled = (
            encoded.get("pooled_prompt_embeds")
            or encoded.get("pooled_projections")
            or encoded.get("clip_pooled")
        )
        if prompt_embeds is not None:
            return prompt_embeds, pooled

    prompt_embeds = getattr(encoded, "prompt_embeds", None)
    pooled = (
        getattr(encoded, "pooled_prompt_embeds", None)
        or getattr(encoded, "pooled_projections", None)
        or getattr(encoded, "clip_pooled", None)
    )
    if prompt_embeds is not None:
        return prompt_embeds, pooled

    raise ValueError(
        f"Unrecognized encode_prompt output format: {type(encoded)}"
    )


def _prepare_latents(pipe, batch_size: int, height: int, width: int, dtype: torch.dtype, device: str):
    if not hasattr(pipe, "prepare_latents"):
        raise AttributeError("Pipeline has no prepare_latents")

    # Try to infer latent channels from transformer config
    num_channels = getattr(getattr(pipe, "transformer", None), "config", None)
    num_channels = getattr(num_channels, "in_channels", None) or 16

    return pipe.prepare_latents(
        batch_size,
        num_channels,
        height,
        width,
        dtype,
        device,
        generator=None,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Dump Flux step-0 tensors using diffusers")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID, help="HF model id")
    parser.add_argument("--dump", default=DEFAULT_DUMP_PATH, help="Output dump path")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT, help="Prompt string")
    parser.add_argument("--device", default="cuda", help="Device")
    parser.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"], help="Torch dtype")
    parser.add_argument("--height", type=int, default=None, help="Image height")
    parser.add_argument("--width", type=int, default=None, help="Image width")
    parser.add_argument("--num-steps", type=int, default=1, help="Number of inference steps")
    args = parser.parse_args()

    try:
        from diffusers import FluxPipeline
    except Exception as exc:
        print(f"diffusers FluxPipeline not available: {exc}")
        sys.exit(1)

    dtype_map = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }
    torch_dtype = dtype_map[args.dtype]

    pipe = FluxPipeline.from_pretrained(args.model_id, torch_dtype=torch_dtype)
    pipe = pipe.to(args.device)
    if hasattr(pipe, "set_progress_bar_config"):
        pipe.set_progress_bar_config(disable=True)

    height = args.height or getattr(pipe, "default_sample_size", None) or 1024
    width = args.width or getattr(pipe, "default_sample_size", None) or 1024

    encoded = _encode_prompt(pipe, args.prompt, args.device, torch_dtype)
    prompt_embeds, pooled_projections = _extract_prompt_embeds(encoded)

    if isinstance(prompt_embeds, torch.Tensor):
        prompt_embeds = prompt_embeds.to(args.device, dtype=torch_dtype)
    if pooled_projections is not None:
        pooled_projections = pooled_projections.to(args.device, dtype=torch_dtype)

    pipe.scheduler.set_timesteps(args.num_steps, device=args.device)
    timesteps = pipe.scheduler.timesteps
    timestep_scaled = timesteps[0]

    latent_model_input = _prepare_latents(
        pipe,
        batch_size=1,
        height=height,
        width=width,
        dtype=torch_dtype,
        device=args.device,
    )

    with torch.no_grad():
        noise_pred_official = pipe.transformer(
            hidden_states=latent_model_input,
            encoder_hidden_states=prompt_embeds,
            pooled_projections=pooled_projections,
            timestep=timestep_scaled,
            guidance=None,
        )

    dump = {
        "latent_model_input": latent_model_input.detach().cpu(),
        "timestep_scaled": timestep_scaled.detach().cpu(),
        "prompt_embeds": prompt_embeds.detach().cpu(),
        "pooled_projections": pooled_projections.detach().cpu() if pooled_projections is not None else None,
        "noise_pred_official": noise_pred_official.detach().cpu(),
        "model_id": args.model_id,
        "prompt": args.prompt,
        "height": height,
        "width": width,
    }

    os.makedirs(os.path.dirname(args.dump) or ".", exist_ok=True)
    torch.save(dump, args.dump)
    print(f"Saved dump to {args.dump}")


if __name__ == "__main__":
    main()
