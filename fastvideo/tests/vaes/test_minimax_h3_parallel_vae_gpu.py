# SPDX-License-Identifier: Apache-2.0
"""GPU regression test for sequence-parallel MiniMax-H3 VAE decode/encode.

Requires a multi-GPU torchrun launch (real NCCL collectives across an SP
group); skipped otherwise:

    torchrun --nproc-per-node=4 -m pytest \
        fastvideo/tests/vaes/test_minimax_h3_parallel_vae_gpu.py -q

Asserts the parallel drivers are bitwise equal to the serial rank-local
decode/encode under the pipeline's fp16 autocast, for both transport
strategies, and that repeated parallel runs are deterministic.
"""

import os

import pytest
import torch
from torch.testing import assert_close

from fastvideo.configs.models.vaes.minimax_h3_video import (
    MiniMaxH3VideoVAEArchConfig,
    MiniMaxH3VideoVAEConfig,
)
from fastvideo.models.vaes.minimax_h3_video import AutoencoderKLMiniMaxH3

_WORLD_SIZE = int(os.environ.get("WORLD_SIZE", "1"))


def _tiny_vae() -> AutoencoderKLMiniMaxH3:
    """Same tiny geometry as test_minimax_h3_parallel_vae (test dirs are not packages)."""
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

pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA"),
    pytest.mark.skipif(_WORLD_SIZE < 2, reason="requires a torchrun launch with WORLD_SIZE > 1"),
]


@pytest.fixture(scope="module")
def sp_group():
    from fastvideo.distributed import get_sp_group, maybe_init_distributed_environment_and_model_parallel
    maybe_init_distributed_environment_and_model_parallel(1, _WORLD_SIZE)
    return get_sp_group()


@pytest.mark.parametrize("strategy", ("gather", "all_gather"))
@pytest.mark.parametrize("latent_frames", (3, 13))
@torch.no_grad()
def test_parallel_decode_bitwise_matches_serial_on_gpu(sp_group, strategy: str, latent_frames: int) -> None:
    from fastvideo.models.vaes.minimax_h3_parallel import decode_to_pixels_parallel

    device = torch.device("cuda", torch.cuda.current_device())
    torch.manual_seed(20260821)  # identical weights on every rank
    vae = _tiny_vae().to(device)
    latents = torch.randn(1, 4, latent_frames, 4, 4, generator=torch.Generator().manual_seed(7)).to(device)

    with torch.autocast(device_type="cuda", dtype=torch.float16):
        expected = torch.empty(vae.decoded_pixel_shape(latents.shape), dtype=torch.float32, pin_memory=True)
        vae.decode_to_pixels(latents, expected)

        outputs = []
        for _ in range(3):  # repeat-determinism
            output = None
            if sp_group.is_first_rank:
                output = torch.empty(vae.decoded_pixel_shape(latents.shape), dtype=torch.float32, pin_memory=True)
            result = decode_to_pixels_parallel(vae, latents, output, sp_group, strategy=strategy)
            outputs.append(result.clone() if result is not None else None)

    if sp_group.is_first_rank:
        for output in outputs:
            assert_close(output, expected, atol=0.0, rtol=0.0)
    else:
        assert all(output is None for output in outputs)


@torch.no_grad()
def test_parallel_encode_bitwise_matches_serial_on_gpu(sp_group) -> None:
    from fastvideo.models.vaes.minimax_h3_parallel import encode_pixels_parallel

    device = torch.device("cuda", torch.cuda.current_device())
    torch.manual_seed(20260821)
    vae = _tiny_vae().to(device)
    pixels = torch.randint(0, 256, (1, 3, 40, 16, 16), dtype=torch.uint8,
                           generator=torch.Generator().manual_seed(9))

    expected = vae.encode_pixels(pixels).latent_dist.parameters
    for _ in range(3):
        moments = encode_pixels_parallel(vae, pixels, sp_group).latent_dist.parameters
        assert_close(moments, expected, atol=0.0, rtol=0.0)
