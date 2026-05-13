# SPDX-License-Identifier: Apache-2.0
"""Pipeline latent parity for Flux2 Klein.

Compares a FastVideo ``Flux2KleinPipeline`` run against ``diffusers.FluxPipeline``
for the canonical Klein prompt, seed, 1024x1024 resolution, and four denoising
steps. The comparison uses final denoised latents for determinism and speed.

Activate locally with:

    FLUX2_MODEL_DIR=/path/to/black-forest-labs__FLUX.2-klein-4B \
    pytest tests/local_tests/pipelines/test_flux2_pipeline_parity.py -v -s

The test is skipped in CI unless CUDA and ``FLUX2_MODEL_DIR`` are available.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch
from torch.testing import assert_close


PROMPT = "a photo of a banana on a wooden table, studio lighting"
SEED = 0
HEIGHT = 1024
WIDTH = 1024
NUM_FRAMES = 1
NUM_INFERENCE_STEPS = 4
GUIDANCE_SCALE = 1.0
ATOL = 0.1
RTOL = 0.1
MODEL_DIR = Path(os.getenv("FLUX2_MODEL_DIR", ""))


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="Flux2 Klein pipeline parity requires CUDA",
)


def _log_tensor_stats(label: str, tensor: torch.Tensor) -> None:
    tensor_f32 = tensor.detach().float()
    print(
        f"[FLUX2 PIPELINE] {label}: shape={tuple(tensor.shape)} "
        f"dtype={tensor.dtype} device={tensor.device} "
        f"abs_mean={tensor_f32.abs().mean().item():.6f} "
        f"min={tensor_f32.min().item():.6f} max={tensor_f32.max().item():.6f}"
    )


def _require_model_dir() -> Path:
    if not MODEL_DIR.exists():
        pytest.skip("Set FLUX2_MODEL_DIR to activate Flux2 Klein pipeline parity")
    return MODEL_DIR


def _run_diffusers_flux_pipeline(model_dir: Path) -> torch.Tensor:
    from diffusers import FluxPipeline

    device = torch.device("cuda:0")
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    generator = torch.Generator(device=device).manual_seed(SEED)
    pipe = FluxPipeline.from_pretrained(
        str(model_dir),
        local_files_only=True,
        torch_dtype=dtype,
    ).to(device)
    pipe.set_progress_bar_config(disable=True)
    try:
        with torch.no_grad():
            output = pipe(
                prompt=PROMPT,
                prompt_2=None,
                height=HEIGHT,
                width=WIDTH,
                num_inference_steps=NUM_INFERENCE_STEPS,
                guidance_scale=GUIDANCE_SCALE,
                generator=generator,
                output_type="latent",
                return_dict=True,
            )
        latents = output.images
        if not torch.is_tensor(latents):
            latents = torch.as_tensor(latents)
        return latents.detach().float().cpu()
    finally:
        del pipe
        torch.cuda.empty_cache()


def _run_fastvideo_flux2_pipeline(model_dir: Path) -> torch.Tensor:
    from fastvideo import VideoGenerator

    generator = VideoGenerator.from_pretrained(
        str(model_dir),
        num_gpus=1,
        use_fsdp_inference=False,
        dit_cpu_offload=False,
        vae_cpu_offload=True,
        text_encoder_cpu_offload=True,
        pin_cpu_memory=False,
        output_type="latent",
        override_pipeline_cls_name="Flux2KleinPipeline",
    )
    try:
        result = generator.generate_video(
            prompt=PROMPT,
            output_path="outputs_video/flux2_klein_pipeline_parity",
            save_video=False,
            return_frames=True,
            height=HEIGHT,
            width=WIDTH,
            num_frames=NUM_FRAMES,
            num_inference_steps=NUM_INFERENCE_STEPS,
            guidance_scale=GUIDANCE_SCALE,
            seed=SEED,
        )
        latents = result["samples"]
        assert torch.is_tensor(latents), "FastVideo did not return latent samples"
        return latents.detach().float().cpu()
    finally:
        generator.shutdown()
        torch.cuda.empty_cache()


def test_flux2_klein_pipeline_latent_parity() -> None:
    model_dir = _require_model_dir()
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

    diffusers_latents = _run_diffusers_flux_pipeline(model_dir)
    _log_tensor_stats("diffusers_latents", diffusers_latents)

    fastvideo_latents = _run_fastvideo_flux2_pipeline(model_dir)
    _log_tensor_stats("fastvideo_latents", fastvideo_latents)

    assert diffusers_latents.shape == fastvideo_latents.shape, (
        f"shape mismatch: diffusers={diffusers_latents.shape} "
        f"fastvideo={fastvideo_latents.shape}"
    )

    diff = (diffusers_latents - fastvideo_latents).abs()
    print(
        f"[FLUX2 PIPELINE] diff max={diff.max().item():.6f} "
        f"mean={diff.mean().item():.6f} median={diff.median().item():.6f}"
    )
    print(
        "[FLUX2 PIPELINE] abs-mean drift "
        f"diffusers={diffusers_latents.abs().mean().item():.6f} "
        f"fastvideo={fastvideo_latents.abs().mean().item():.6f}"
    )

    assert_close(diffusers_latents, fastvideo_latents, atol=ATOL, rtol=RTOL)
