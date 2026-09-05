# SPDX-License-Identifier: Apache-2.0
"""Deterministic contracts for Helios pyramid sampling geometry."""

import math
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F


def _helpers():
    try:
        from fastvideo.pipelines.basic.helios.pipeline_utils import (
            build_helios_frame_indices,
            calculate_shift,
            downsample_to_pyramid_base,
            get_generated_pixel_frames,
            get_num_latent_chunks,
            sample_block_noise,
        )
    except ImportError as exc:
        raise AssertionError("Helios pyramid helpers have not been implemented") from exc
    return {
        "build_indices": build_helios_frame_indices,
        "calculate_shift": calculate_shift,
        "downsample": downsample_to_pyramid_base,
        "generated_frames": get_generated_pixel_frames,
        "num_chunks": get_num_latent_chunks,
        "block_noise": sample_block_noise,
    }


def test_calculate_shift_matches_official_flux_formula() -> None:
    calculate_shift = _helpers()["calculate_shift"]
    expected_middle = 0.5 + (1024 - 256) * (1.15 - 0.5) / (4096 - 256)

    assert calculate_shift(256) == 0.5
    assert calculate_shift(1024) == expected_middle
    assert calculate_shift(4096) == 1.15


def test_history_frame_indices_match_official_keep_first_frame_layout() -> None:
    current, short, mid, long = _helpers()["build_indices"](
        history_sizes=[16, 2, 1],
        num_latent_frames_per_chunk=9,
        keep_first_frame=True,
        device=torch.device("cpu"),
    )

    assert current.tolist() == [list(range(20, 29))]
    assert short.tolist() == [[0, 19]]
    assert mid.tolist() == [[17, 18]]
    assert long.tolist() == [list(range(1, 17))]


def test_downsample_to_pyramid_base_matches_official_bilinear_loop() -> None:
    latents = torch.arange(1 * 2 * 3 * 8 * 12, dtype=torch.float32).reshape(1, 2, 3, 8, 12)
    expected = latents.permute(0, 2, 1, 3, 4).reshape(3, 2, 8, 12)
    expected = F.interpolate(expected, size=(4, 6), mode="bilinear") * 2
    expected = F.interpolate(expected, size=(2, 3), mode="bilinear") * 2
    expected = expected.reshape(1, 3, 2, 2, 3).permute(0, 2, 1, 3, 4)

    actual = _helpers()["downsample"](latents, num_stages=3)

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_block_noise_matches_official_covariance_sampling() -> None:
    if not torch.cuda.is_available():
        pytest.skip("The official FP32 block-noise Cholesky contract requires CUDA")
    device = torch.device("cuda:0")
    scheduler = SimpleNamespace(config=SimpleNamespace(gamma=1 / 3))
    actual = _helpers()["block_noise"](
        scheduler=scheduler,
        shape=(1, 1, 2, 4, 4),
        patch_size=(1, 2, 2),
        device=device,
        generator=torch.Generator("cpu").manual_seed(123),
    )

    block_size = 4
    covariance = (torch.eye(block_size, device=device) * (1 + 1 / 3) -
                  torch.ones(block_size, block_size, device=device) * (1 / 3) +
                  torch.eye(block_size, device=device) * 1e-8).float()
    cholesky = torch.linalg.cholesky(covariance)
    standard = torch.randn(
        1 * 1 * 2 * 2 * 2,
        block_size,
        generator=torch.Generator("cpu").manual_seed(123),
    ).to(device)
    expected = (standard @ cholesky.T).view(1, 1, 2, 2, 2, 2, 2)
    expected = expected.permute(0, 1, 2, 3, 5, 4, 6).reshape(1, 1, 2, 4, 4)

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_pixel_frame_count_rounds_up_to_complete_helios_chunks() -> None:
    get_num_latent_chunks = _helpers()["num_chunks"]

    assert get_num_latent_chunks(1, 9, 4) == 1
    assert get_num_latent_chunks(33, 9, 4) == 1
    assert get_num_latent_chunks(34, 9, 4) == 2
    assert get_num_latent_chunks(240, 9, 4) == 8


def test_pixel_frame_count_matches_official_chunk_decode_contract() -> None:
    get_generated_pixel_frames = _helpers()["generated_frames"]

    assert get_generated_pixel_frames(33, 4) == 33
    assert get_generated_pixel_frames(66, 4) == 65
    assert get_generated_pixel_frames(264, 4) == 261


def test_pyramid_shift_stays_linear_between_official_endpoints() -> None:
    calculate_shift = _helpers()["calculate_shift"]

    assert math.isclose(calculate_shift(2176), 0.825, rel_tol=0, abs_tol=1e-12)
