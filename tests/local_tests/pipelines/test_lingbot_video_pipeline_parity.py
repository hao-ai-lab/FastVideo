# SPDX-License-Identifier: Apache-2.0
"""End-to-end latent parity for the Dense LingBot-Video T2V pipeline."""

from __future__ import annotations

import gc
import os
from pathlib import Path
from typing import Any, cast

import pytest
import torch
from torch.testing import assert_close

from tests.local_tests.lingbot_video.hf_assets import (
    FASTVIDEO_DENSE,
    OFFICIAL_DENSE,
    download_components,
)

PROMPT = "A red fox runs through fresh snow at sunrise."
NEGATIVE_PROMPT = '{"universal_negative": {"visual_quality": ["low quality", "blurry"]}}'
HEIGHT = int(os.environ.get("LINGBOT_VIDEO_PARITY_HEIGHT", "32"))
WIDTH = int(os.environ.get("LINGBOT_VIDEO_PARITY_WIDTH", "32"))
NUM_FRAMES = int(os.environ.get("LINGBOT_VIDEO_PARITY_NUM_FRAMES", "1"))
NUM_INFERENCE_STEPS = int(os.environ.get("LINGBOT_VIDEO_PARITY_NUM_INFERENCE_STEPS", "1"))
NUM_GPUS = int(os.environ.get("LINGBOT_VIDEO_PARITY_NUM_GPUS", "1"))
SP_SIZE = int(os.environ.get("LINGBOT_VIDEO_PARITY_SP_SIZE", "1"))
USE_FSDP = os.environ.get("LINGBOT_VIDEO_PARITY_USE_FSDP") == "1"
FORCE_MATH_SDPA = os.environ.get("LINGBOT_VIDEO_PARITY_FORCE_MATH_SDPA") == "1"


def _force_math_sdpa(_worker: Any) -> dict[str, bool]:
    """Select the deterministic math SDPA kernel inside one FastVideo worker."""
    torch.backends.cuda.enable_cudnn_sdp(False)
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    return {
        "cudnn": torch.backends.cuda.cudnn_sdp_enabled(),
        "flash": torch.backends.cuda.flash_sdp_enabled(),
        "mem_efficient": torch.backends.cuda.mem_efficient_sdp_enabled(),
        "math": torch.backends.cuda.math_sdp_enabled(),
    }


def _require_gpu_test() -> None:
    """Require an explicitly allocated CUDA worker for heavyweight pipeline parity."""
    if os.environ.get("LINGBOT_VIDEO_RUN_GPU_TESTS") != "1":
        pytest.skip("set LINGBOT_VIDEO_RUN_GPU_TESTS=1 on an allocated GPU")
    if not torch.cuda.is_available():
        pytest.skip("LingBot-Video pipeline parity requires CUDA")
    if torch.cuda.device_count() < NUM_GPUS:
        pytest.skip(f"LingBot-Video pipeline parity requires {NUM_GPUS} CUDA devices")


def _run_official(latents: torch.Tensor, model_dir: Path) -> torch.Tensor:
    """Run the released production loader with the configured parity dimensions."""
    from lingbot_video.runner import _load_diffusers_pipe

    dtype_map = {
        "default": torch.bfloat16,
        "transformer": torch.bfloat16,
        "text_encoder": torch.bfloat16,
        "vae": torch.float32,
    }
    pipe = _load_diffusers_pipe(
        model_dir,
        dtype_map,
        mode="t2v",
        transformer_subfolder="transformer",
    )
    try:
        with torch.inference_mode():
            output = pipe(
                prompt=PROMPT,
                negative_prompt=NEGATIVE_PROMPT,
                height=HEIGHT,
                width=WIDTH,
                num_frames=NUM_FRAMES,
                num_inference_steps=NUM_INFERENCE_STEPS,
                guidance_scale=3.0,
                shift=3.0,
                generator=torch.Generator(device="cuda").manual_seed(42),
                latents=latents.clone(),
                output_type="latent",
                batch_cfg=False,
                return_dict=True,
            )
        return output.frames.detach().float().cpu()
    finally:
        del pipe
        gc.collect()
        torch.cuda.empty_cache()


def _run_fastvideo(latents: torch.Tensor, model_dir: Path, output_dir: Path) -> torch.Tensor:
    """Run the converted native pipeline with the same prompt, seed, shape, and schedule."""
    from fastvideo import VideoGenerator

    generator = VideoGenerator.from_pretrained(
        str(model_dir),
        num_gpus=NUM_GPUS,
        sp_size=SP_SIZE,
        use_fsdp_inference=USE_FSDP,
        dit_cpu_offload=False,
        dit_layerwise_offload=False,
        vae_cpu_offload=True,
        text_encoder_cpu_offload=True,
        pin_cpu_memory=False,
        output_type="latent",
    )
    try:
        if FORCE_MATH_SDPA:
            worker_backends = generator.executor.collective_rpc(_force_math_sdpa)
            assert all(
                backend == {"cudnn": False, "flash": False, "mem_efficient": False, "math": True}
                for backend in worker_backends
            )
        result = generator.generate_video(
            prompt=PROMPT,
            negative_prompt=NEGATIVE_PROMPT,
            output_path=str(output_dir),
            save_video=False,
            return_frames=True,
            height=HEIGHT,
            width=WIDTH,
            num_frames=NUM_FRAMES,
            num_inference_steps=NUM_INFERENCE_STEPS,
            guidance_scale=3.0,
            batch_cfg=False,
            seed=42,
            latents=latents.clone(),
        )
        samples = cast(dict[str, Any], result)["samples"]
        if not torch.is_tensor(samples):
            samples = torch.as_tensor(samples)
        return samples.detach().float().cpu()
    finally:
        generator.shutdown()


def test_lingbot_video_latents_match(tmp_path: Path) -> None:
    """Compare official and FastVideo latent outputs for the configured schedule."""
    _require_gpu_test()
    official_root = download_components(
        OFFICIAL_DENSE,
        "scheduler",
        "text_encoder",
        "processor",
        "transformer",
        "vae",
    )
    fastvideo_root = download_components(
        FASTVIDEO_DENSE,
        "scheduler",
        "text_encoder",
        "tokenizer",
        "transformer",
        "vae",
    )
    latents = torch.randn(
        (1, 16, (NUM_FRAMES - 1) // 4 + 1, HEIGHT // 8, WIDTH // 8),
        generator=torch.Generator(device="cpu").manual_seed(42),
        dtype=torch.float32,
    )
    original_sdp_backends = (
        torch.backends.cuda.cudnn_sdp_enabled(),
        torch.backends.cuda.flash_sdp_enabled(),
        torch.backends.cuda.mem_efficient_sdp_enabled(),
        torch.backends.cuda.math_sdp_enabled(),
    )
    torch.backends.cuda.enable_cudnn_sdp(False)
    if FORCE_MATH_SDPA:
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
    try:
        expected = _run_official(latents, official_root)
        actual = _run_fastvideo(latents, fastvideo_root, tmp_path)
    finally:
        torch.backends.cuda.enable_cudnn_sdp(original_sdp_backends[0])
        torch.backends.cuda.enable_flash_sdp(original_sdp_backends[1])
        torch.backends.cuda.enable_mem_efficient_sdp(original_sdp_backends[2])
        torch.backends.cuda.enable_math_sdp(original_sdp_backends[3])
    assert actual.shape == expected.shape
    drift = (actual - expected).abs()
    print(f"pipeline_max_abs={drift.max().item():.8f} pipeline_mean_abs={drift.mean().item():.8f}")
    assert drift.mean().item() <= 1e-2
    assert_close(actual, expected, atol=5e-2, rtol=5e-2)
