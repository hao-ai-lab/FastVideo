# Copyright 2026 The MiniMax and HuggingFace Teams. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU synthetic parity for the native MiniMax-H3 video VAE."""

import sys

import torch
from torch.testing import assert_close

from tests.local_tests.minimax_h3._reference import REFERENCE_SRC, assert_pinned_reference

assert_pinned_reference(
    "src/diffusers/models/autoencoders/autoencoder_kl_minimax_h3.py",
    "e59dc54caebff9ecb6d88c957b1ff189bf6a88419ed3551f84d3f68c20ce3192",
)
sys.path.insert(0, str(REFERENCE_SRC))

from diffusers import AutoencoderKLMiniMaxH3 as ReferenceAutoencoderKLMiniMaxH3  # noqa: E402

from fastvideo.configs.models.vaes.minimax_h3_video import (  # noqa: E402
    MiniMaxH3VideoVAEArchConfig,
    MiniMaxH3VideoVAEConfig,
)
from fastvideo.models.vaes.minimax_h3_video import AutoencoderKLMiniMaxH3  # noqa: E402


TINY_ARCH = {
    "in_channels": 3,
    "out_channels": 3,
    "latent_channels": 4,
    "block_out_channels": (8, 16),
    "layers_per_block": 1,
    "spatial_downsample_factors": (2, 2),
    "temporal_downsample_factors": (2, 2),
    "norm_num_groups": 8,
    "decoder_num_layers": 2,
    "decoder_num_attention_heads": 2,
    "decoder_attention_head_dim": 8,
    "decoder_num_register_tokens": 2,
    "decoder_ffn_mult": 2,
    "clip_length": 17,
    "token_drop": 3,
    "latents_mean": (0.5, -0.25, 1.0, -1.5),
    "latents_std": (2.0, 0.5, 1.5, 4.0),
}


def _build_models() -> tuple[ReferenceAutoencoderKLMiniMaxH3, AutoencoderKLMiniMaxH3]:
    torch.manual_seed(1234)
    reference = ReferenceAutoencoderKLMiniMaxH3(**TINY_ARCH).float().eval()
    fastvideo = AutoencoderKLMiniMaxH3(
        MiniMaxH3VideoVAEConfig(arch_config=MiniMaxH3VideoVAEArchConfig(**TINY_ARCH))
    ).eval()

    reference_state = reference.state_dict()
    fastvideo_state = fastvideo.state_dict()
    assert reference_state.keys() == fastvideo_state.keys()
    assert {key: value.shape for key, value in reference_state.items()} == {
        key: value.shape for key, value in fastvideo_state.items()
    }
    fastvideo.load_state_dict(reference_state, strict=True)
    return reference, fastvideo


def _capture_activations(model: torch.nn.Module, names: tuple[str, ...]) -> tuple[dict[str, list[torch.Tensor]], list]:
    activations: dict[str, list[torch.Tensor]] = {name: [] for name in names}

    def capture(name: str):
        def hook(_module, _inputs, output) -> None:
            assert isinstance(output, torch.Tensor)
            activations[name].append(output.detach().clone())

        return hook

    modules = dict(model.named_modules())
    handles = [modules[name].register_forward_hook(capture(name)) for name in names]
    return activations, handles


def test_minimax_h3_video_vae_encode_decode_and_geometry_match_reference() -> None:
    reference, fastvideo = _build_models()
    generator = torch.Generator(device="cpu").manual_seed(2026)
    video = torch.randn((1, 3, 22, 8, 8), generator=generator, dtype=torch.float32)

    activation_names = (
        "encoder.down_blocks.0",
        "quant_conv",
        "decoder.transformer_blocks.0",
        "decoder.proj_out",
    )
    reference_activations, reference_handles = _capture_activations(reference, activation_names)
    fastvideo_activations, fastvideo_handles = _capture_activations(fastvideo, activation_names)
    with torch.inference_mode():
        reference_latents = reference.encode(video, return_dict=False)[0].mode()
        fastvideo_latents = fastvideo.encode(video, return_dict=False)[0].mode()
        reference_decoded = reference.decode(reference_latents, return_dict=False)[0]
        fastvideo_decoded = fastvideo.decode(reference_latents, return_dict=False)[0]
    for handle in reference_handles + fastvideo_handles:
        handle.remove()

    assert_close(fastvideo_latents, reference_latents, atol=1e-6, rtol=1e-6)
    assert_close(fastvideo_decoded, reference_decoded, atol=1e-6, rtol=1e-6)
    assert fastvideo_latents.shape == (1, 4, 7, 2, 2)
    assert fastvideo_decoded.shape == video.shape
    assert fastvideo.spatial_compression_ratio == 4
    assert fastvideo.temporal_compression_ratio == 4
    for name in activation_names:
        assert len(fastvideo_activations[name]) == len(reference_activations[name])
        for result, expected in zip(fastvideo_activations[name], reference_activations[name], strict=True):
            assert_close(result, expected, atol=1e-6, rtol=1e-6)


def test_minimax_h3_video_vae_normalization_tiling_and_fp32_contract() -> None:
    _, fastvideo = _build_models()
    generator = torch.Generator(device="cpu").manual_seed(7)
    latents = torch.randn((1, 4, 7, 2, 2), generator=generator, dtype=torch.float32)
    original_state = {key: value.clone() for key, value in fastvideo.state_dict().items()}

    normalized = fastvideo.normalize_latents(latents)
    assert_close(fastvideo.denormalize_latents(normalized), latents, atol=1e-6, rtol=1e-6)
    assert_close(normalized, (latents - fastvideo.latents_mean) / fastvideo.latents_std, atol=0, rtol=0)

    assert fastvideo.use_tiling
    assert fastvideo.tile_sample_min_height == 256
    assert fastvideo.tile_sample_min_width == 256
    assert fastvideo.tile_sample_min_overlap_height == 64
    assert fastvideo.tile_sample_min_overlap_width == 64

    fastvideo.to(dtype=torch.bfloat16)
    assert all(parameter.dtype == torch.float32 for parameter in fastvideo.parameters())
    assert all(buffer.dtype == torch.float32 for buffer in fastvideo.buffers() if buffer.is_floating_point())
    for key, value in fastvideo.state_dict().items():
        assert_close(value, original_state[key], atol=0, rtol=0)


def test_minimax_h3_video_vae_small_multitile_path_matches_reference() -> None:
    reference, fastvideo = _build_models()
    reference.enable_tiling(8, 8, 4, 4)
    fastvideo.enable_tiling(8, 8, 4, 4)
    generator = torch.Generator(device="cpu").manual_seed(99)
    video = torch.randn((1, 3, 22, 12, 12), generator=generator, dtype=torch.float32)

    with torch.inference_mode():
        reference_latents = reference.encode(video, return_dict=False)[0].mode()
        fastvideo_latents = fastvideo.encode(video, return_dict=False)[0].mode()
        reference_decoded = reference.decode(reference_latents, return_dict=False)[0]
        fastvideo_decoded = fastvideo.decode(reference_latents, return_dict=False)[0]

    assert_close(fastvideo_latents, reference_latents, atol=1e-6, rtol=1e-6)
    assert_close(fastvideo_decoded, reference_decoded, atol=1e-6, rtol=1e-6)
    assert fastvideo_latents.shape == (1, 4, 7, 3, 3)
    assert fastvideo_decoded.shape == video.shape
