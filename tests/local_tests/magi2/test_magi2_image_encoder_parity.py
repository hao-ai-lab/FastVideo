# SPDX-License-Identifier: Apache-2.0
"""Strict production-loader parity for the MAGI-2 Wan 2.2 image encoder.

Coverage scope: both. The test loads the published ``Wan2_2_VAE`` reference
and the FastVideo ``Magi2WanImageEncoder`` through its production loader. A
synthetic image passes through the I2V letterbox, ``VideoProcessor``, and BF16
round trip before FP32 encoding.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from fastvideo.configs.models.vaes.magi2_wanvae import Magi2WanVAEConfig
from fastvideo.models.vaes.magi2_wan_loader import (
    Magi2WanImageEncoder,
    load_magi2_wan_image_encoder,
)
from tests.local_tests.magi2._parity_utils import (
    LOCAL_WEIGHTS_DIR,
    assert_tensor_exact,
    import_official_module,
    require_path,
)


PARITY_COVERAGE = "both"
WAN_VAE_PATH = LOCAL_WEIGHTS_DIR / "vae" / "Wan2.2_VAE.pth"


def _resizepad(image, target_height: int, target_width: int):
    """Fit an RGB image inside a white canvas without cropping its edges."""
    image_module = pytest.importorskip("PIL.Image")
    width, height = image.size
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid image size, width: {width}, height: {height}")
    scale = min(target_width / width, target_height / height)
    resized_width = max(1, int(round(width * scale)))
    resized_height = max(1, int(round(height * scale)))
    resized_image = image.convert("RGB").resize(
        (resized_width, resized_height),
        resample=image_module.Resampling.LANCZOS,
    )
    canvas = image_module.new("RGB", (target_width, target_height), (255, 255, 255))
    canvas.paste(
        resized_image,
        ((target_width - resized_width) // 2, (target_height - resized_height) // 2),
    )
    return canvas


def _prepare_reference_image(device: torch.device) -> torch.Tensor:
    """Apply the I2V resize, letterbox, and BF16 input conversion.

    The source dimensions force the long-edge scaling branch and a non-square
    letterbox. The returned tensor has the ``[B, C, T, H, W]`` layout consumed
    by ``Wan2_2_VAE.encode``.
    """
    pil_image_module = pytest.importorskip("PIL.Image")
    video_processor_module = pytest.importorskip("diffusers.video_processor")
    pixels = (
        np.arange(48 * 80 * 3, dtype=np.uint32).reshape(48, 80, 3) % 251
    ).astype(np.uint8)
    source_image = pil_image_module.fromarray(pixels, mode="RGB")
    generation_height, generation_width = 64, 96
    max_length = max(generation_height, generation_width)
    target_width = max_length
    target_height = int(source_image.height * max_length / source_image.width)
    resized_image = _resizepad(
        source_image,
        target_height,
        target_width,
    )
    video_processor = video_processor_module.VideoProcessor(vae_scale_factor=32)
    image_tensor = video_processor.preprocess(
        resized_image,
        height=target_height,
        width=target_width,
    )
    return image_tensor.to(device=device, dtype=torch.bfloat16).unsqueeze(2)[:, :3].float()


def _assert_magi2_wan_config(
    official_vae,
    fastvideo_encoder: Magi2WanImageEncoder,
) -> None:
    """Validate the production encoder configuration against published values."""
    config = Magi2WanVAEConfig()
    assert config.base_dim == 160
    assert config.decoder_base_dim == 256
    assert config.z_dim == 48
    assert config.in_channels == 12
    assert config.out_channels == 12
    assert config.temperal_downsample == (False, True, True)
    assert config.patch_size == 2
    assert config.is_residual is True
    assert config.clip_output is False
    assert torch.equal(
        torch.tensor(config.latents_mean, dtype=torch.float32),
        official_vae.mean.detach().cpu(),
    )
    assert torch.equal(
        torch.tensor(config.latents_std, dtype=torch.float32),
        official_vae.std.detach().cpu(),
    )
    assert torch.equal(fastvideo_encoder.mean.detach().cpu(), official_vae.mean.detach().cpu())
    assert torch.equal(fastvideo_encoder.std.detach().cpu(), official_vae.std.detach().cpu())
    assert torch.equal(
        fastvideo_encoder.inverse_std.detach().cpu(),
        official_vae.scale[1].detach().cpu(),
    )


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="MAGI-2 I2V image-encoder parity requires CUDA.",
)
def test_load_magi2_wan_image_encoder_i2v_exact_parity() -> None:
    """Require exact post-normalization image latents from both encoders."""
    vae_path = require_path(WAN_VAE_PATH, "Wan 2.2 VAE weights")
    official_vae_module = import_official_module("inference.model.vae2_2")
    device = torch.device("cuda:0")
    image_tensor = _prepare_reference_image(device)

    official_vae = official_vae_module.get_vae2_2(
        str(vae_path),
        device=str(device),
        weight_dtype=torch.float32,
    )
    fastvideo_encoder = load_magi2_wan_image_encoder(vae_path, device)
    _assert_magi2_wan_config(official_vae, fastvideo_encoder)

    with torch.inference_mode():
        official_latent = official_vae.encode(image_tensor).detach().cpu()
        fastvideo_latent = fastvideo_encoder.encode(image_tensor).detach().cpu()

    assert image_tensor.dtype == torch.float32
    assert image_tensor.shape[:3] == (1, 3, 1)
    assert_tensor_exact(fastvideo_latent, official_latent, "I2V Wan latent")
