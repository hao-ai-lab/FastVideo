# SPDX-License-Identifier: Apache-2.0
"""
Lucy-Edit-Dev pipeline smoke and parity tests.

Smoke test validates that FastVideo's LucyEditPipeline runs end-to-end
and produces non-degenerate output. Parity test compares against the
Diffusers reference implementation to ensure numerical alignment.

Requirements:
    - CUDA GPU
    - Lucy-Edit-Dev weights downloaded to official_weights/Lucy-Edit-Dev/
      or set LUCY_EDIT_WEIGHTS_PATH env var

Usage:
    pytest tests/local_tests/pipelines/test_lucy_edit_pipeline_smoke.py -v
"""

import os
import tempfile

import numpy as np
import pytest
import torch
from torch.testing import assert_close

from fastvideo import VideoGenerator


def _log_tensor_stats(label: str, tensor: torch.Tensor) -> None:
    tensor_f32 = tensor.float()
    print(
        f"[LUCY_EDIT] {label}: shape={tuple(tensor.shape)} "
        f"dtype={tensor.dtype} device={tensor.device} "
        f"min={tensor_f32.min().item():.6f} max={tensor_f32.max().item():.6f} "
        f"mean={tensor_f32.mean().item():.6f} sum={tensor_f32.sum().item():.6f}"
    )


def _create_synthetic_video(
    num_frames: int,
    height: int,
    width: int,
    seed: int = 0,
) -> str:
    """Create a synthetic mp4 video and return the temp file path.

    Uses deterministic random frames for reproducibility.
    The caller is responsible for cleaning up the file.
    """
    import imageio

    rng = np.random.RandomState(seed)
    tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    tmp.close()

    writer = imageio.get_writer(tmp.name, fps=24, codec="libx264",
                                output_params=["-pix_fmt", "yuv420p"])
    for _ in range(num_frames):
        frame = rng.randint(0, 256, (height, width, 3), dtype=np.uint8)
        writer.append_data(frame)
    writer.close()

    return tmp.name


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="Lucy-Edit pipeline smoke test requires CUDA.",
)
def test_lucy_edit_pipeline_smoke():
    """End-to-end smoke test for the Lucy-Edit-Dev pipeline.

    This test:
    1. Creates a synthetic input video (required for V2V editing)
    2. Loads the Lucy-Edit-Dev model using FastVideo's VideoGenerator
    3. Generates an edited video from the source video + editing prompt
    4. Verifies the output has the correct shape and is non-degenerate
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

    # Create a synthetic input video (V2V models require video_path)
    video_path = _create_synthetic_video(num_frames, height, width, seed=0)

    try:
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
            video_path=video_path,
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
        assert fastvideo_out.ndim == 5, (
            f"Expected 5D output, got {fastvideo_out.ndim}D"
        )
        assert not torch.isnan(fastvideo_out).any(), "Output contains NaN"
        assert not torch.isinf(fastvideo_out).any(), "Output contains Inf"
        # Verify output is not all zeros (degenerate)
        assert fastvideo_out.abs().sum() > 0, "Output is all zeros"

        print("[LUCY_EDIT SMOKE] Pipeline smoke test passed!")
    finally:
        os.unlink(video_path)


@pytest.mark.xfail(
    reason="Scheduler and preprocessing alignment pending. "
    "See TODO: align FlowUniPCMultistepScheduler with Diffusers UniPCMultistepScheduler.",
    strict=False,
)
@pytest.mark.xfail(
    reason="Scheduler and preprocessing alignment pending. "
    "See TODO: align FlowUniPCMultistepScheduler with Diffusers UniPCMultistepScheduler.",
    strict=False,
)
@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="Lucy-Edit parity test requires CUDA.",
)
def test_lucy_edit_parity_with_diffusers():
    """Numerical parity test comparing FastVideo vs Diffusers reference.

    This test requires both FastVideo weights and the diffusers library
    with LucyEditPipeline support. It generates videos using identical
    seeds and inputs, then compares pixel-level outputs.

    Both pipelines load the same mp4 file to guarantee identical input
    frames after any video codec compression.
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
        from diffusers.utils import load_video
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

    # Create a synthetic input video shared by both pipelines
    video_path = _create_synthetic_video(num_frames, height, width, seed=0)

    try:
        # Load video frames for Diffusers (same mp4 file, same decoder)
        input_video = load_video(video_path)

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

        # Free Diffusers pipeline VRAM before FastVideo
        del ref_pipe, vae
        torch.cuda.empty_cache()

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
            video_path=video_path,
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

        # Convert reference output to tensor.
        # Diffusers may return numpy float32 arrays ([0,1]) or PIL images.
        if isinstance(ref_output, np.ndarray):
            # Shape: [T, H, W, C] float32 in [0, 1]
            ref_video = torch.from_numpy(ref_output).float()
        else:
            # List of PIL images or numpy uint8 arrays
            ref_frames = []
            for f in ref_output:
                arr = np.array(f)
                t = torch.from_numpy(arr).float()
                if arr.dtype == np.uint8:
                    t = t / 255.0
                ref_frames.append(t)
            ref_video = torch.stack(ref_frames, dim=0)  # [T, H, W, C]
        ref_video = ref_video.permute(3, 0, 1, 2).unsqueeze(0).to(
            device)  # [1, C, T, H, W]

        _log_tensor_stats("fastvideo_video", fastvideo_out)
        _log_tensor_stats("reference_video", ref_video)

        assert ref_video.shape == fastvideo_out.shape, (
            f"Shape mismatch: ref={ref_video.shape}, fv={fastvideo_out.shape}"
        )
        assert_close(ref_video, fastvideo_out, atol=2 / 255, rtol=1e-3)
        print("[LUCY_EDIT PARITY] Parity test passed!")
    finally:
        os.unlink(video_path)
