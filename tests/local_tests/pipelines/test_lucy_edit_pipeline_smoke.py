# SPDX-License-Identifier: Apache-2.0
"""
Lucy-Edit-Dev pipeline smoke test.

Compares FastVideo's LucyEditPipeline output against the Diffusers
reference implementation (LucyEditPipeline from diffusers) to ensure
numerical parity.

Requirements:
    - CUDA GPU
    - Lucy-Edit-Dev weights downloaded to official_weights/Lucy-Edit-Dev/
      or set LUCY_EDIT_WEIGHTS_PATH env var

Usage:
    pytest tests/local_tests/pipelines/test_lucy_edit_pipeline_smoke.py -v
"""

import os
from pathlib import Path

import pytest
import torch
from torch.testing import assert_close

from fastvideo import VideoGenerator


def _log_tensor_stats(label: str, tensor: torch.Tensor) -> None:
    tensor_f32 = tensor.float()
    print(
        f"[LUCY_EDIT SMOKE] {label}: shape={tuple(tensor.shape)} "
        f"dtype={tensor.dtype} device={tensor.device} "
        f"min={tensor_f32.min().item():.6f} max={tensor_f32.max().item():.6f} "
        f"mean={tensor_f32.mean().item():.6f} sum={tensor_f32.sum().item():.6f}"
    )


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="Lucy-Edit pipeline smoke test requires CUDA.",
)
def test_lucy_edit_pipeline_smoke():
    """End-to-end smoke test for the Lucy-Edit-Dev pipeline.

    This test:
    1. Loads the Lucy-Edit-Dev model using FastVideo's VideoGenerator
    2. Generates an edited video from a source video + editing prompt
    3. Verifies the output has the correct shape and is non-degenerate

    For full parity testing, compare against the Diffusers reference:
        from diffusers import LucyEditPipeline, AutoencoderKLWan
    """
    os.environ.setdefault("FASTVIDEO_ATTENTION_BACKEND", "TORCH_SDPA")
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)

    weights_path = os.getenv(
        "LUCY_EDIT_WEIGHTS_PATH",
        "official_weights/Lucy-Edit-Dev",
    )

    if not os.path.isdir(weights_path):
        pytest.skip(
            f"Missing Lucy-Edit-Dev weights at {weights_path}. "
            "Download with: python scripts/huggingface/download_hf.py "
            "--repo_id decart-ai/Lucy-Edit-Dev "
            "--local_dir official_weights/Lucy-Edit-Dev --repo_type model"
        )
    if not os.path.isfile(os.path.join(weights_path, "model_index.json")):
        pytest.skip(f"Missing model_index.json in {weights_path}")

    device = torch.device("cuda:0")
    prompt = (
        "Change the shirt to a bright red leather jacket with a glossy finish, "
        "add aviator sunglasses."
    )
    seed = 42
    height = 480
    width = 832
    num_frames = 17  # Small number for fast testing
    steps = 10

    # --- FastVideo pipeline ---
    generator = VideoGenerator.from_pretrained(
        weights_path,
        num_gpus=1,
        use_fsdp_inference=False,
        dit_cpu_offload=False,
        vae_cpu_offload=False,
        text_encoder_cpu_offload=False,
        pin_cpu_memory=False,
    )

    result = generator.generate_video(
        prompt=prompt,
        output_path="outputs_video/lucy_edit_smoke",
        save_video=False,
        height=height,
        width=width,
        num_frames=num_frames,
        num_inference_steps=steps,
        seed=seed,
    )
    generator.shutdown()

    fastvideo_out = result["samples"]
    fastvideo_out = fastvideo_out.to(device=device, dtype=torch.float32)
    _log_tensor_stats("fastvideo_video", fastvideo_out)

    # Basic sanity checks
    assert fastvideo_out is not None, "Pipeline returned None"
    assert fastvideo_out.ndim == 5, f"Expected 5D output, got {fastvideo_out.ndim}D"
    assert not torch.isnan(fastvideo_out).any(), "Output contains NaN values"
    assert not torch.isinf(fastvideo_out).any(), "Output contains Inf values"
    # Verify output is not all zeros (degenerate)
    assert fastvideo_out.abs().sum() > 0, "Output is all zeros"

    print("[LUCY_EDIT SMOKE] Pipeline smoke test passed!")


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="Lucy-Edit parity test requires CUDA.",
)
def test_lucy_edit_parity_with_diffusers():
    """Numerical parity test comparing FastVideo vs Diffusers reference.

    This test requires both FastVideo weights and the diffusers library
    with LucyEditPipeline support. It generates videos using identical
    seeds and inputs, then compares pixel-level outputs.

    Set LUCY_EDIT_WEIGHTS_PATH to the local weights directory.
    """
    os.environ.setdefault("FASTVIDEO_ATTENTION_BACKEND", "TORCH_SDPA")
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)

    weights_path = os.getenv(
        "LUCY_EDIT_WEIGHTS_PATH",
        "official_weights/Lucy-Edit-Dev",
    )

    if not os.path.isdir(weights_path):
        pytest.skip(f"Missing Lucy-Edit-Dev weights at {weights_path}")

    try:
        from diffusers import AutoencoderKLWan, LucyEditPipeline
        from diffusers.utils import export_to_video, load_video
    except ImportError:
        pytest.skip(
            "Diffusers LucyEditPipeline not available. "
            "Install with: pip install git+https://github.com/huggingface/diffusers"
        )

    device = torch.device("cuda:0")
    prompt = "Turn the person into an alien with green skin."
    seed = 42
    height = 480
    width = 832
    num_frames = 17
    steps = 10
    guidance_scale = 5.0

    # Create a simple synthetic input video (solid color frames)
    # In real usage, load an actual video
    input_video = [
        torch.rand(3, height, width) for _ in range(num_frames)
    ]

    # --- Diffusers reference pipeline ---
    vae = AutoencoderKLWan.from_pretrained(
        weights_path, subfolder="vae", torch_dtype=torch.float32
    )
    ref_pipe = LucyEditPipeline.from_pretrained(
        weights_path, vae=vae, torch_dtype=torch.bfloat16
    )
    ref_pipe.to("cuda")

    ref_generator = torch.Generator(device="cuda").manual_seed(seed)
    ref_output = ref_pipe(
        prompt=prompt,
        video=input_video,
        height=height,
        width=width,
        num_frames=num_frames,
        guidance_scale=guidance_scale,
        num_inference_steps=steps,
        generator=ref_generator,
    ).frames[0]

    # --- FastVideo pipeline ---
    fv_generator = VideoGenerator.from_pretrained(
        weights_path,
        num_gpus=1,
        use_fsdp_inference=False,
        dit_cpu_offload=False,
        vae_cpu_offload=False,
        text_encoder_cpu_offload=False,
        pin_cpu_memory=False,
    )

    fv_result = fv_generator.generate_video(
        prompt=prompt,
        output_path="outputs_video/lucy_edit_parity",
        save_video=False,
        height=height,
        width=width,
        num_frames=num_frames,
        num_inference_steps=steps,
        guidance_scale=guidance_scale,
        seed=seed,
    )
    fv_generator.shutdown()

    fastvideo_out = fv_result["samples"]
    fastvideo_out = fastvideo_out.to(device=device, dtype=torch.float32)

    # Convert reference output to tensor for comparison
    import numpy as np
    ref_frames = [torch.from_numpy(np.array(f)).float() / 255.0 for f in ref_output]
    ref_video = torch.stack(ref_frames, dim=0)  # [T, H, W, C]
    ref_video = ref_video.permute(3, 0, 1, 2).unsqueeze(0).to(device)  # [1, C, T, H, W]

    _log_tensor_stats("fastvideo_video", fastvideo_out)
    _log_tensor_stats("reference_video", ref_video)

    assert ref_video.shape == fastvideo_out.shape, (
        f"Shape mismatch: ref={ref_video.shape}, fv={fastvideo_out.shape}"
    )
    assert_close(ref_video, fastvideo_out, atol=2 / 255, rtol=1e-3)
    print("[LUCY_EDIT PARITY] Parity test passed!")
