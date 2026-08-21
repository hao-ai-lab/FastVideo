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


def _legacy_decode(vae: AutoencoderKLMiniMaxH3, z: torch.Tensor) -> torch.Tensor:
    """Verbatim pre-streaming ``_decode`` (main @ 8208536cd) as a reference oracle."""
    tokens_chunk_size = vae.tokens_chunk_size
    token_drop = vae.config.token_drop
    temporal_ratio = vae.temporal_compression_ratio
    chunk_num_frames = tokens_chunk_size * temporal_ratio
    num_tokens = z.shape[2] + token_drop
    pad_tokens = (-num_tokens) % tokens_chunk_size
    num_chunks = (num_tokens + pad_tokens) // tokens_chunk_size - int(token_drop > 0)
    if pad_tokens > 0:
        z = torch.cat([z, z[:, :, -1:].repeat(1, 1, pad_tokens, 1, 1)], dim=2)

    decoded_chunks = []
    overlap = None
    for index in range(num_chunks):
        start = index * tokens_chunk_size
        clip = vae._decode_clip(z[:, :, start:start + tokens_chunk_size + vae.token_overlap])
        for overlap_index in range(int(token_drop > 0) + 1):
            frame_start = overlap_index * chunk_num_frames
            chunk = clip[:, :, frame_start:frame_start + chunk_num_frames]
            chunk = chunk[:, :, vae.frame_pre_padding:]
            if overlap_index == 0:
                if overlap is not None:
                    chunk = vae._blend(overlap, chunk, vae.frame_overlap, dim=-3)
                decoded_chunks.append(chunk)
            else:
                overlap = chunk
    if overlap is not None:
        decoded_chunks.append(overlap)
    decoded = torch.cat(decoded_chunks, dim=2)

    if pad_tokens > 0:
        intra_tail = vae.config.clip_length % temporal_ratio
        num_tokens_before_pad = z.shape[2] - pad_tokens
        pad_frames = sum(intra_tail if intra_tail and (num_tokens_before_pad + offset) %
                         tokens_chunk_size == 0 else temporal_ratio for offset in range(pad_tokens))
        decoded = decoded[:, :, :-pad_frames]
    return decoded


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


# 3: one chunk with pad tokens; 6: pad hitting the intra-clip tail;
# 12: two blended chunks without padding; 13: three chunks plus pad trim.
@pytest.mark.parametrize("latent_frames", (3, 6, 12, 13))
@torch.inference_mode()
def test_decode_matches_legacy_algorithm(latent_frames: int) -> None:
    """The chunk iterator must stay bit-exact with the pre-streaming decode."""
    torch.manual_seed(20260811 + latent_frames)
    vae = _tiny_vae()
    latents = torch.randn(1, 4, latent_frames, 4, 4)
    expected = _legacy_decode(vae, latents)

    decoded = vae.decode(latents).sample
    assert_close(decoded, expected, atol=0.0, rtol=0.0)

    streamed = torch.empty(vae.decoded_pixel_shape(latents.shape), dtype=torch.float32)
    vae.decode_to_pixels(latents, streamed)
    assert_close(streamed, vae.denormalize_pixels(expected.float()).clamp_(0, 1), atol=0.0, rtol=0.0)


@torch.inference_mode()
def test_streaming_slicing_matches_unbatched() -> None:
    torch.manual_seed(20260812)
    vae = _tiny_vae()
    vae.enable_slicing()

    latents = torch.randn(2, 4, 7, 4, 4)
    expected = vae.denormalize_pixels(vae.decode(latents).sample.float()).clamp_(0, 1)
    actual = torch.empty(vae.decoded_pixel_shape(latents.shape), dtype=torch.float32)
    vae.decode_to_pixels(latents, actual)
    assert_close(actual, expected, atol=0.0, rtol=0.0)

    pixels = torch.randint(0, 256, (2, 3, 22, 16, 16), dtype=torch.uint8)
    expected_moments = vae.encode(vae.normalize_pixels(pixels.float().div(255))).latent_dist.parameters
    assert_close(vae.encode_pixels(pixels).latent_dist.parameters, expected_moments, atol=0.0, rtol=0.0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="pinned-buffer streaming requires CUDA")
@torch.inference_mode()
def test_decode_to_pixels_pinned_buffer_matches_dense_on_cuda() -> None:
    """Async chunk copies into a pinned buffer must equal the dense decode."""
    torch.manual_seed(20260813)
    vae = _tiny_vae().to("cuda")
    latents = torch.randn(1, 4, 12, 4, 4, device="cuda")
    expected = vae.denormalize_pixels(vae.decode(latents).sample.float()).clamp_(0, 1).cpu()

    pinned = torch.empty(vae.decoded_pixel_shape(latents.shape), dtype=torch.float32, pin_memory=True)
    vae.decode_to_pixels(latents, pinned)

    assert_close(pinned, expected, atol=0.0, rtol=0.0)


def test_decode_to_pixels_rejects_incomplete_output(monkeypatch) -> None:
    vae = _tiny_vae()
    latents = torch.randn(1, 4, 2, 4, 4)
    output = torch.empty(vae.decoded_pixel_shape(latents.shape), dtype=torch.float32)
    monkeypatch.setattr(vae, "_decode_chunks", lambda _: iter(()))

    with pytest.raises(RuntimeError, match="wrote 0 frames"):
        vae.decode_to_pixels(latents, output)
