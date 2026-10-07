# SPDX-License-Identifier: Apache-2.0
"""TAEHV decode parity tests for the MLX implementation.

TAEHV MLX vs torch is bit-close (atol 1e-5) for z_dim=16 and 48.
"""

from __future__ import annotations

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core", reason="MLX required for Wan VAE/TAEHV tests")

from fastvideo.mlx_runtime.wan_vae import (  # noqa: E402
    decode_latents_taehv_mlx,
    ensure_taehv_checkpoint,
)


@pytest.mark.parametrize("z_dim", [16, 48])
def test_taehv_mlx_matches_torch(z_dim: int) -> None:
    import torch
    from fastvideo.third_party.taehv import TAEHV

    ckpt = ensure_taehv_checkpoint(z_dim=z_dim)
    rng = np.random.default_rng(0)
    h = w = 8
    t = 5
    lat = (rng.standard_normal((1, z_dim, t, h, w)) * 0.5).astype(np.float32)

    model = TAEHV(str(ckpt)).eval()
    with torch.no_grad():
        out_t = model.decode_video(torch.from_numpy(lat).transpose(1, 2), parallel=True, show_progress_bar=False)
    torch_np = out_t[0].permute(0, 2, 3, 1).float().numpy()
    mlx_np = decode_latents_taehv_mlx(lat, z_dim=z_dim)[0]
    # Validate complete temporal output shape before comparing values.
    assert mlx_np.shape[0] == torch_np.shape[0], (
        f"MLX output frame count {mlx_np.shape[0]} != torch {torch_np.shape[0]}"
    )
    np.testing.assert_allclose(mlx_np, torch_np, atol=1e-5, rtol=1e-5)
