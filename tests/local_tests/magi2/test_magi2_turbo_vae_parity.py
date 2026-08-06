# SPDX-License-Identifier: Apache-2.0
"""Strict temporal sliding-window parity for the MAGI-2 Turbo VAE decoder.

Coverage scope: both. The official side uses
``inference.model.turbo_vaed.get_turbo_vaed`` with the published distilled
decoder. The FastVideo side targets
``fastvideo.models.vaes.magi2_turbo_vae.Magi2TurboVAEModel``. Fifteen latent
frames exercise the published first, middle, and last temporal windows after
padding from 15 to 21 frames.
"""

from __future__ import annotations

import json

import pytest
import torch

from fastvideo.models.vaes.magi2_turbo_vae import Magi2TurboVAEModel
from tests.local_tests.magi2._parity_utils import (
    LOCAL_WEIGHTS_DIR,
    assert_tensor_exact,
    import_official_module,
    require_path,
)


PARITY_COVERAGE = "both"
TURBO_VAE_DIR = LOCAL_WEIGHTS_DIR / "turbo_vae"
TURBO_VAE_CONFIG_PATH = TURBO_VAE_DIR / "TurboV3-Wan22-TinyShallow_7_7.json"
TURBO_VAE_CHECKPOINT_PATH = TURBO_VAE_DIR / "checkpoint.ckpt"


def _load_turbo_config() -> dict:
    """Load the exact configuration paired with the distilled checkpoint."""
    config_path = require_path(TURBO_VAE_CONFIG_PATH, "Turbo VAE configuration")
    with config_path.open(encoding="utf-8") as config_file:
        return json.load(config_file)


def _deterministic_video_latent(device: torch.device) -> torch.Tensor:
    """Create a reproducible BF16 latent that crosses two window boundaries."""
    latent_values = torch.arange(1 * 48 * 15 * 4 * 4, dtype=torch.float32)
    latent_values = ((latent_values % 257) - 128) / 128
    return latent_values.reshape(1, 48, 15, 4, 4).to(
        device=device,
        dtype=torch.bfloat16,
    )


def _temporal_window_roles(
    num_frames: int,
    first_chunk_size: int,
    step_size: int,
) -> tuple[str, ...]:
    """Return the Turbo VAE window roles selected after temporal padding."""
    padding_frames = 0
    if num_frames < first_chunk_size:
        padding_frames = first_chunk_size - num_frames
    elif (num_frames - first_chunk_size) % step_size != 0:
        padding_frames = step_size - (num_frames - first_chunk_size) % step_size
    padded_frames = num_frames + padding_frames
    if padded_frames == first_chunk_size:
        return ("single",)

    window_roles = ["first"]
    for window_start in range(first_chunk_size, padded_frames, step_size):
        is_last_window = window_start + step_size == padded_frames
        window_roles.append("last" if is_last_window else "middle")
    return tuple(window_roles)


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="MAGI-2 Turbo VAE parity requires CUDA.",
)
def test_magi2_turbo_vae_sliding_window_decode_exact_parity() -> None:
    """Require exact decoded video values across first, middle, and last windows."""
    checkpoint_path = require_path(
        TURBO_VAE_CHECKPOINT_PATH,
        "Turbo VAE checkpoint",
    )
    config = _load_turbo_config()
    assert config["first_chunk_size"] == 7
    assert config["step_size"] == 7
    assert config["temporal_compression_ratio"] == 4

    official_module = import_official_module("inference.model.turbo_vaed")
    device = torch.device("cuda:0")
    official_vae = official_module.get_turbo_vaed(
        str(TURBO_VAE_CONFIG_PATH),
        str(checkpoint_path),
        device=str(device),
        weight_dtype=torch.bfloat16,
    )
    from fastvideo.configs.models.vaes.magi2_turbo_vae import Magi2TurboVAEConfig

    fastvideo_config = Magi2TurboVAEConfig(
        config_path=str(TURBO_VAE_CONFIG_PATH),
        checkpoint_path=str(checkpoint_path),
        pretrained_dtype="bfloat16",
    )
    fastvideo_vae = Magi2TurboVAEModel(fastvideo_config).eval()
    latent = _deterministic_video_latent(device)
    assert _temporal_window_roles(
        latent.shape[2],
        config["first_chunk_size"],
        config["step_size"],
    ) == ("first", "middle", "last")

    with torch.inference_mode():
        official_video = official_vae.decode(latent).float().detach().cpu()
        fastvideo_video = fastvideo_vae.decode(latent).float().detach().cpu()

    assert_tensor_exact(fastvideo_video, official_video, "Turbo VAE decoded video")
