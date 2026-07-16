# SPDX-License-Identifier: Apache-2.0
"""End-to-end latent or decoded-pixel parity for LingBot-Video T2V."""

from __future__ import annotations

import gc
import json
import os
from pathlib import Path
from typing import Any, cast

import pytest
import torch

from tests.local_tests.lingbot_video.hf_assets import (
    FASTVIDEO_DENSE,
    FASTVIDEO_MOE,
    OFFICIAL_DENSE,
    OFFICIAL_MOE,
    download_components,
    materialize_component_view,
)

PROMPT_PATH = Path(__file__).parents[1] / "lingbot_video" / "t2v_example_1_prompt.json"
HEIGHT = int(os.environ.get("LINGBOT_VIDEO_PARITY_HEIGHT", "32"))
WIDTH = int(os.environ.get("LINGBOT_VIDEO_PARITY_WIDTH", "32"))
NUM_FRAMES = int(os.environ.get("LINGBOT_VIDEO_PARITY_NUM_FRAMES", "1"))
NUM_INFERENCE_STEPS = int(os.environ.get("LINGBOT_VIDEO_PARITY_NUM_INFERENCE_STEPS", "1"))
NUM_GPUS = int(os.environ.get("LINGBOT_VIDEO_PARITY_NUM_GPUS", "1"))
SP_SIZE = int(os.environ.get("LINGBOT_VIDEO_PARITY_SP_SIZE", "1"))
USE_FSDP = os.environ.get("LINGBOT_VIDEO_PARITY_USE_FSDP") == "1"
FORCE_MATH_SDPA = os.environ.get("LINGBOT_VIDEO_PARITY_FORCE_MATH_SDPA") == "1"
PARITY_VARIANT = os.environ.get("LINGBOT_VIDEO_PARITY_VARIANT", "dense")
BATCH_CFG = os.environ.get("LINGBOT_VIDEO_PARITY_BATCH_CFG") == "1"
PARITY_OUTPUT_TYPE = os.environ.get("LINGBOT_VIDEO_PARITY_OUTPUT_TYPE", "latent")
DETERMINISTIC = os.environ.get("LINGBOT_VIDEO_PARITY_DETERMINISTIC") == "1"
BATCH_CFG_MIN_SSIM = 0.78
BATCH_CFG_PIXEL_MAX_ABS = 3e-3
BATCH_CFG_PIXEL_MEAN_ABS = 6e-4


def _load_structured_prompt() -> str:
    """Load the released LingBot T2V example in the format expected by Qwen3-VL."""
    sample = json.loads(PROMPT_PATH.read_text(encoding="utf-8"))
    return str(sample["caption"]["comprehensive_description"])


PROMPT = _load_structured_prompt()


def _as_channel_first_tensor(value: Any) -> torch.Tensor:
    """Normalize latent or decoded pipeline output to a CPU tensor."""
    tensor = value if torch.is_tensor(value) else torch.as_tensor(value)
    if tensor.ndim == 5 and tensor.shape[-1] == 3:
        tensor = tensor.permute(0, 4, 1, 2, 3)
    return tensor.detach().float().cpu()


def _configure_worker_backends(_worker: Any) -> dict[str, bool]:
    """Mirror the parent process's deterministic and SDPA settings in a worker."""
    torch.backends.cuda.enable_cudnn_sdp(False)
    if DETERMINISTIC:
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.enabled = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    if FORCE_MATH_SDPA:
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
    return {
        "deterministic": torch.are_deterministic_algorithms_enabled(),
        "cudnn": torch.backends.cudnn.enabled,
        "matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_tf32": torch.backends.cudnn.allow_tf32,
        "cudnn_sdp": torch.backends.cuda.cudnn_sdp_enabled(),
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


def _mean_frame_ssim(actual: torch.Tensor, expected: torch.Tensor) -> float:
    """Measure visible frame similarity without introducing MP4 encoding loss."""
    from pytorch_msssim import ssim

    actual_frames = actual.permute(0, 2, 1, 3, 4).flatten(0, 1).clamp(0, 1)
    expected_frames = expected.permute(0, 2, 1, 3, 4).flatten(0, 1).clamp(0, 1)
    scores = []
    for start in range(0, actual_frames.shape[0], 8):
        frame_slice = slice(start, start + 8)
        scores.append(
            ssim(
                actual_frames[frame_slice].cuda(),
                expected_frames[frame_slice].cuda(),
                data_range=1.0,
                size_average=False,
            ).cpu())
    return torch.cat(scores).mean().item()


def _run_official(latents: torch.Tensor, model_dir: Path) -> tuple[torch.Tensor, bool, str]:
    """Run the released production loader with the configured parity dimensions."""
    from lingbot_video.pipeline_lingbot_video import DEFAULT_NEGATIVE_PROMPT
    from lingbot_video.runner import _load_diffusers_pipe
    from lingbot_video.transformer_lingbot_video import flash_attn_varlen_func_v3

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
    official_batch_cfg = BATCH_CFG and flash_attn_varlen_func_v3 is not None
    if BATCH_CFG and not official_batch_cfg:
        print("official_batch_cfg=sequential reason=flash_attn_interface.flash_attn_varlen_func unavailable")
    try:
        with torch.inference_mode():
            output = pipe(
                prompt=PROMPT,
                negative_prompt=DEFAULT_NEGATIVE_PROMPT,
                height=HEIGHT,
                width=WIDTH,
                num_frames=NUM_FRAMES,
                num_inference_steps=NUM_INFERENCE_STEPS,
                guidance_scale=3.0,
                shift=3.0,
                generator=torch.Generator(device="cuda").manual_seed(42),
                latents=latents.clone(),
                output_type=PARITY_OUTPUT_TYPE,
                batch_cfg=official_batch_cfg,
                return_dict=True,
            )
        return _as_channel_first_tensor(output.frames), official_batch_cfg, DEFAULT_NEGATIVE_PROMPT
    finally:
        del pipe
        gc.collect()
        torch.cuda.empty_cache()


def _run_fastvideo(
    latents: torch.Tensor,
    model_dir: Path,
    output_dir: Path,
    negative_prompt: str,
) -> torch.Tensor:
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
        output_type=PARITY_OUTPUT_TYPE,
        refine_enabled=False,
    )
    try:
        worker_backends = generator.executor.collective_rpc(_configure_worker_backends)
        for backend in worker_backends:
            assert backend["cudnn_sdp"] is False
            if DETERMINISTIC:
                assert backend["deterministic"] is True
                assert backend["cudnn"] is False
                assert backend["matmul_tf32"] is False
                assert backend["cudnn_tf32"] is False
            if FORCE_MATH_SDPA:
                assert backend["flash"] is False
                assert backend["mem_efficient"] is False
                assert backend["math"] is True
        result = generator.generate_video(
            prompt=PROMPT,
            negative_prompt=negative_prompt,
            output_path=str(output_dir),
            save_video=False,
            return_frames=True,
            height=HEIGHT,
            width=WIDTH,
            num_frames=NUM_FRAMES,
            num_inference_steps=NUM_INFERENCE_STEPS,
            guidance_scale=3.0,
            batch_cfg=BATCH_CFG,
            seed=42,
            latents=latents.clone(),
        )
        samples = cast(dict[str, Any], result)["samples"]
        return _as_channel_first_tensor(samples)
    finally:
        generator.shutdown()


def test_lingbot_video_pipeline_matches(tmp_path: Path) -> None:
    """Compare official and FastVideo latent or decoded outputs."""
    _require_gpu_test()
    if PARITY_VARIANT == "dense":
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
    elif PARITY_VARIANT == "moe":
        official_root = materialize_component_view(
            OFFICIAL_MOE,
            tmp_path / "official_moe_base",
            "scheduler",
            "text_encoder",
            "processor",
            "transformer",
            "vae",
        )
        fastvideo_root = materialize_component_view(
            FASTVIDEO_MOE,
            tmp_path / "fastvideo_moe_base",
            "scheduler",
            "text_encoder",
            "tokenizer",
            "transformer",
            "vae",
        )
    else:
        raise ValueError(f"unsupported LingBot-Video parity variant: {PARITY_VARIANT}")
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
    original_deterministic = torch.are_deterministic_algorithms_enabled()
    original_cudnn = torch.backends.cudnn.enabled
    original_matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
    original_cudnn_tf32 = torch.backends.cudnn.allow_tf32
    torch.backends.cuda.enable_cudnn_sdp(False)
    if DETERMINISTIC:
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.enabled = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    if FORCE_MATH_SDPA:
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
    try:
        expected, official_batch_cfg, negative_prompt = _run_official(latents, official_root)
        actual = _run_fastvideo(latents, fastvideo_root, tmp_path, negative_prompt)
    finally:
        torch.backends.cuda.enable_cudnn_sdp(original_sdp_backends[0])
        torch.backends.cuda.enable_flash_sdp(original_sdp_backends[1])
        torch.backends.cuda.enable_mem_efficient_sdp(original_sdp_backends[2])
        torch.backends.cuda.enable_math_sdp(original_sdp_backends[3])
        torch.use_deterministic_algorithms(original_deterministic)
        torch.backends.cudnn.enabled = original_cudnn
        torch.backends.cuda.matmul.allow_tf32 = original_matmul_tf32
        torch.backends.cudnn.allow_tf32 = original_cudnn_tf32
    assert actual.shape == expected.shape
    drift = (actual - expected).abs()
    print(f"pipeline_max_abs={drift.max().item():.8f} pipeline_mean_abs={drift.mean().item():.8f}")
    if not BATCH_CFG:
        assert torch.equal(actual, expected)
        return
    if PARITY_OUTPUT_TYPE != "np":
        raise ValueError("the batched-CFG semantic gate requires decoded pixels")
    if official_batch_cfg:
        mean_ssim = _mean_frame_ssim(actual, expected)
        print(f"pipeline_mean_ssim={mean_ssim:.8f}")
        assert mean_ssim >= BATCH_CFG_MIN_SSIM
        return
    if not FORCE_MATH_SDPA:
        raise ValueError("the dependency-free batched-CFG semantic gate requires math SDPA")
    assert drift.max().item() <= BATCH_CFG_PIXEL_MAX_ABS
    assert drift.mean().item() <= BATCH_CFG_PIXEL_MEAN_ABS
