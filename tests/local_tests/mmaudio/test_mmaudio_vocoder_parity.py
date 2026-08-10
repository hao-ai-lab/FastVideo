# SPDX-License-Identifier: Apache-2.0
"""MMAudio BigVGAN-v2 parity against the canonical NVIDIA checkpoint."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
import torch


REPO_ROOT = Path(__file__).resolve().parents[3]
BIGVGAN_DIR = Path(
    os.environ.get(
        "MMAUDIO_BIGVGAN_DIR",
        REPO_ROOT / "official_weights/mmaudio/bigvgan_v2_44khz_128band_512x",
    )
)


def _require_assets() -> None:
    required = (BIGVGAN_DIR / "config.json", BIGVGAN_DIR / "bigvgan_generator.pt")
    if not all(path.is_file() for path in required):
        pytest.skip(
            "Canonical BigVGAN-v2 assets are absent. Set MMAUDIO_BIGVGAN_DIR "
            "to a local nvidia/bigvgan_v2_44khz_128band_512x snapshot."
        )


def test_bigvgan_v2_implementation_parity() -> None:
    from mmaudio.ext.bigvgan_v2.bigvgan import BigVGAN
    from mmaudio.ext.bigvgan_v2.env import AttrDict

    from fastvideo.models.audio.bigvgan import BigVGANV2

    config = {
        "num_mels": 4,
        "upsample_initial_channel": 16,
        "resblock": "1",
        "resblock_kernel_sizes": [3],
        "resblock_dilation_sizes": [[1, 3, 5]],
        "upsample_rates": [2],
        "upsample_kernel_sizes": [4],
        "activation": "snakebeta",
        "snake_logscale": True,
        "use_bias_at_final": True,
        "use_tanh_at_final": True,
    }
    official = BigVGAN(AttrDict(config), use_cuda_kernel=False)
    fastvideo = BigVGANV2(config)
    assert {name: tensor.shape for name, tensor in official.state_dict().items()} == {
        name: tensor.shape for name, tensor in fastvideo.state_dict().items()
    }
    fastvideo.load_state_dict(official.state_dict(), strict=True)
    official.remove_weight_norm()
    fastvideo.remove_weight_norm()
    mel = torch.randn((1, 4, 8), generator=torch.Generator().manual_seed(1234))
    with torch.inference_mode():
        expected = official(mel)
        actual = fastvideo(mel)
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)


def test_mmaudio_bigvgan_v2_parity() -> None:
    _require_assets()
    if not torch.cuda.is_available():
        pytest.skip("MMAudio BigVGAN parity requires CUDA")

    from mmaudio.ext.bigvgan_v2.bigvgan import BigVGAN, load_hparams_from_json

    from fastvideo.models.audio.bigvgan import BigVGANV2

    config_path = BIGVGAN_DIR / "config.json"
    with config_path.open(encoding="utf-8") as handle:
        config = json.load(handle)
    official = BigVGAN(load_hparams_from_json(config_path), use_cuda_kernel=False)
    fastvideo = BigVGANV2(config)
    state = torch.load(BIGVGAN_DIR / "bigvgan_generator.pt", map_location="cpu", weights_only=True)["generator"]
    official.load_state_dict(state, strict=True)
    fastvideo.load_state_dict(state, strict=True)
    assert {name: tensor.shape for name, tensor in official.state_dict().items()} == {
        name: tensor.shape for name, tensor in fastvideo.state_dict().items()
    }

    device = torch.device("cuda:0")
    official.remove_weight_norm()
    fastvideo.remove_weight_norm()
    official.to(device).eval()
    fastvideo.to(device).eval()
    mel = torch.randn((1, 128, 8), generator=torch.Generator(device=device).manual_seed(1234), device=device)
    with torch.inference_mode():
        expected = official(mel)
        actual = fastvideo(mel)
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)
