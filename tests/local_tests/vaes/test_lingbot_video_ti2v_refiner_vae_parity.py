# SPDX-License-Identifier: Apache-2.0
"""Exact LingBot-Video TI2V refiner-state parity from one shared MP4."""

from __future__ import annotations

import gc
import json
import os
from pathlib import Path

import pytest
import torch
from diffusers import AutoencoderKLWan as OfficialAutoencoderKLWan
from PIL import Image
from safetensors.torch import load_file
from torch.nn.attention import SDPBackend, sdpa_kernel

from fastvideo.configs.models.vaes.wanvae import WanVAEConfig
from fastvideo.models.vaes.wanvae import AutoencoderKLWan as FastVideoAutoencoderKLWan
from fastvideo.pipelines.basic.lingbot_video.stages import LingBotVideoRefinerPreparationStage
from tests.local_tests.lingbot_video.hf_assets import FASTVIDEO_MOE, OFFICIAL_MOE, download_components


def _load_native(checkpoint: Path, device: torch.device) -> FastVideoAutoencoderKLWan:
    """Strict-load the released full VAE into FastVideo's native implementation."""
    vae_dir = checkpoint / "vae"
    raw_config = json.loads((vae_dir / "config.json").read_text())
    config = WanVAEConfig()
    config.update_model_arch({key: value for key, value in raw_config.items() if not key.startswith("_")})
    config.load_encoder = True
    config.load_decoder = True
    model = FastVideoAutoencoderKLWan(config)
    model.load_state_dict(load_file(vae_dir / "diffusion_pytorch_model.safetensors"), strict=True)
    return model.to(device=device, dtype=torch.float32).eval()


def _encode(vae: torch.nn.Module, video: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
    """Run the released refiner VAE encode and DiT-space normalization boundary."""
    device = next(vae.parameters()).device
    normalized = video.to(device=device, dtype=torch.float32).mul(2.0).sub(1.0)
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16), sdpa_kernel(SDPBackend.MATH):
        encoded = vae.encode(normalized)
    distribution = encoded.latent_dist if hasattr(encoded, "latent_dist") else encoded
    latents = distribution.sample(generator)
    mean = torch.tensor(vae.config.latents_mean, device=device, dtype=torch.float32).view(1, -1, 1, 1, 1)
    std_inverse = 1.0 / torch.tensor(
        vae.config.latents_std,
        device=device,
        dtype=torch.float32,
    ).view(1, -1, 1, 1, 1)
    return ((latents.float() - mean) * std_inverse).to(latents)


def _prepare_state(
    vae: torch.nn.Module,
    video: torch.Tensor,
    clean_frame: torch.Tensor,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Encode video then clean frame and create the seeded TI2V refiner state."""
    generator = torch.Generator(device=device).manual_seed(42)
    encoded = _encode(vae, video, generator)
    clean_latent = _encode(vae, clean_frame, generator)[:, :, :1].contiguous()
    encoded[:, :, :1] = clean_latent.to(encoded.dtype)
    noise = torch.randn(encoded.shape, generator=generator, device=device, dtype=encoded.dtype)
    initial = (1.0 - 0.85) * encoded + 0.85 * noise
    return {
        "encoded": encoded.detach().cpu(),
        "clean_latent": clean_latent.detach().cpu(),
        "noise": noise.detach().cpu(),
        "initial_latent": initial.detach().cpu(),
    }


def _report_exact(name: str, expected: torch.Tensor, actual: torch.Tensor) -> None:
    """Print exact drift metrics and require zero differing values."""
    difference = (actual.float() - expected.float()).abs()
    print(
        f"{name}: equal={torch.equal(actual, expected)} "
        f"differing={torch.count_nonzero(actual != expected).item()} "
        f"max_abs={difference.max().item():.8f} mean_abs={difference.mean().item():.8f}"
    )
    assert torch.equal(actual, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Wan VAE refiner parity requires CUDA")
def test_lingbot_video_ti2v_refiner_state_exact_parity() -> None:
    """Require exact official/native refiner initialization from one reloaded MP4."""
    if os.environ.get("LINGBOT_VIDEO_RUN_GPU_TESTS") != "1":
        pytest.skip("set LINGBOT_VIDEO_RUN_GPU_TESTS=1 on a scheduled GPU")
    shared_mp4 = os.environ.get("LINGBOT_VIDEO_TI2V_SHARED_BASE_MP4")
    condition_image = os.environ.get("LINGBOT_VIDEO_TI2V_CONDITION_IMAGE")
    if not shared_mp4 or not condition_image:
        pytest.skip("set the shared base MP4 and clean condition image paths")
    from decord import VideoReader, cpu
    from lingbot_video.utils import (
        compute_training_aligned_indices,
        load_first_frame_condition_tensor,
        load_refiner_video_tensor,
    )

    expected_video, metadata = load_refiner_video_tensor(shared_mp4, 1088, 1920, sample_fps=24, vae_tc=4)
    assert metadata["sample_frame"] == 121
    reader = VideoReader(shared_mp4, ctx=cpu(0))
    indices = compute_training_aligned_indices(len(reader), metadata["sample_frame"])
    raw_frames = torch.from_numpy(reader.get_batch(indices).asnumpy()).permute(0, 3, 1, 2).float().div(255.0)
    raw_video = raw_frames.permute(1, 0, 2, 3).unsqueeze(0).contiguous()
    actual_video = LingBotVideoRefinerPreparationStage._resize_video(raw_video, 1088, 1920)
    _report_exact("resized_video_pixels", expected_video, actual_video)
    expected_clean = load_first_frame_condition_tensor(condition_image, 1088, 1920, 480, 832)
    with Image.open(condition_image) as image:
        actual_clean = LingBotVideoRefinerPreparationStage._resize_clean_condition(
            image,
            1088,
            1920,
            480,
            832,
        )
    _report_exact("clean_condition_pixels", expected_clean, actual_clean)
    device = torch.device("cuda")
    official_path = os.environ.get("LINGBOT_VIDEO_OFFICIAL_CHECKPOINT")
    official_checkpoint = Path(official_path) if official_path else download_components(OFFICIAL_MOE, "vae")
    converted_path = os.environ.get("LINGBOT_VIDEO_TI2V_CHECKPOINT")
    converted_checkpoint = Path(converted_path) if converted_path else download_components(FASTVIDEO_MOE, "vae")
    official = OfficialAutoencoderKLWan.from_pretrained(
        official_checkpoint / "vae",
        torch_dtype=torch.float32,
        local_files_only=True,
    ).to(device).eval()
    expected = _prepare_state(official, expected_video, expected_clean, device)
    del official
    gc.collect()
    torch.cuda.empty_cache()
    native = _load_native(converted_checkpoint, device)
    actual = _prepare_state(native, actual_video, actual_clean, device)
    for name in ("encoded", "clean_latent", "noise", "initial_latent"):
        _report_exact(name, expected[name], actual[name])
