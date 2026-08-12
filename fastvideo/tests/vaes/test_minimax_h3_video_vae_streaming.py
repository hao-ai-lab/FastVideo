# SPDX-License-Identifier: Apache-2.0
import pytest
import torch
from torch.testing import assert_close

from fastvideo.configs.models.vaes.minimax_h3_video import (
    MiniMaxH3VideoVAEArchConfig,
    MiniMaxH3VideoVAEConfig,
)
from fastvideo.models.vaes.minimax_h3_video import AutoencoderKLMiniMaxH3


def _tiny_vae() -> AutoencoderKLMiniMaxH3:
    arch = MiniMaxH3VideoVAEArchConfig(
        latent_channels=4,
        block_out_channels=(32, 32),
        layers_per_block=1,
        spatial_downsample_factors=(2, 2),
        temporal_downsample_factors=(2, 2),
        decoder_num_layers=1,
        decoder_num_attention_heads=1,
        decoder_attention_head_dim=8,
        decoder_num_register_tokens=2,
        decoder_ffn_mult=1,
        latents_mean=(0.0, ) * 4,
        latents_std=(1.0, ) * 4,
    )
    return AutoencoderKLMiniMaxH3(
        MiniMaxH3VideoVAEConfig(
            arch_config=arch,
            use_tiling=False,
            use_temporal_tiling=False,
            use_parallel_tiling=False,
        )).eval()


@torch.inference_mode()
def test_encode_pixels_matches_encode() -> None:
    torch.manual_seed(20260810)
    vae = _tiny_vae()
    pixels = torch.randint(0, 256, (1, 3, 22, 16, 16), dtype=torch.uint8)
    expected = vae.encode(vae.normalize_pixels(pixels.float().div(255))).latent_dist.parameters
    assert_close(vae.encode_pixels(pixels).latent_dist.parameters, expected, atol=0.0, rtol=0.0)

    float_pixels = torch.rand(1, 3, 22, 16, 16)
    original_pixels = float_pixels.clone()
    expected = vae.encode(vae.normalize_pixels(float_pixels)).latent_dist.parameters
    actual = vae.encode_pixels(float_pixels).latent_dist.parameters
    assert_close(float_pixels, original_pixels, atol=0.0, rtol=0.0)
    assert_close(actual, expected, atol=0.0, rtol=0.0)

    with pytest.raises(ValueError, match="must remain on CPU"):
        vae.encode_pixels(torch.empty(1, 3, 1, 16, 16, device="meta"))


@pytest.mark.parametrize("latent_frames", (2, 12))
@torch.inference_mode()
def test_decode_to_pixels_matches_decode(latent_frames: int) -> None:
    torch.manual_seed(20260810)
    vae = _tiny_vae()
    latents = torch.randn(1, 4, latent_frames, 4, 4)
    expected = vae.denormalize_pixels(vae.decode(latents).sample.float()).clamp_(0, 1)
    actual = torch.empty(vae.decoded_pixel_shape(latents.shape), dtype=torch.float32)

    vae.decode_to_pixels(latents, actual)

    assert_close(actual, expected, atol=0.0, rtol=0.0)


def test_decode_to_pixels_rejects_incomplete_output(monkeypatch) -> None:
    vae = _tiny_vae()
    latents = torch.randn(1, 4, 2, 4, 4)
    output = torch.empty(vae.decoded_pixel_shape(latents.shape), dtype=torch.float32)
    monkeypatch.setattr(vae, "_decode_chunks", lambda _: iter(()))

    with pytest.raises(RuntimeError, match="wrote 0 frames"):
        vae.decode_to_pixels(latents, output)
