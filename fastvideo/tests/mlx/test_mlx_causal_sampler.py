# SPDX-License-Identifier: Apache-2.0
"""Track C rung 5: unit test for the streaming causal DMD sampler.

Runs the block-autoregressive sampler on the tiny random-weight causal model and
checks it streams one finite latent block per frame-block while the KV cache
advances across the whole clip. Numerics quality is out of scope here (that is a
visual check vs the CUDA reference); this guards the control flow and shapes.
"""

from __future__ import annotations

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core", reason="MLX is required for the streaming sampler test")

from fastvideo.mlx_runtime.causal_sampler import build_dmd_schedule, stream_causal_latents  # noqa: E402
from fastvideo.tests.mlx.test_mlx_causal_dit_parity import (  # noqa: E402
    ARCH,
    NUM_FRAMES,
    _build_torch_model,
    _full_rotary,
    _mlx_from_torch,
)


@pytest.mark.usefixtures("distributed_setup")
def test_streaming_sampler_yields_finite_blocks() -> None:
    model = _mlx_from_torch(_build_torch_model())
    height = width = 8
    frame_seqlen = (height // 2) * (width // 2)
    _, _, cos_full, sin_full = _full_rotary()

    schedule, timesteps = build_dmd_schedule([1000, 750, 500, 250], flow_shift=8.0, warp_denoising_step=True)
    assert len(timesteps) == 4

    rng = np.random.default_rng(0)
    noise = mx.array(rng.standard_normal((1, ARCH["in_channels"], NUM_FRAMES, height, width)).astype(np.float32))
    text = mx.array((rng.standard_normal((1, 24, ARCH["text_dim"])) * 0.1).astype(np.float32))

    blocks = list(
        stream_causal_latents(
            model, text, noise, cos_full, sin_full, schedule, timesteps, frame_seqlen=frame_seqlen, seed=0))

    # One block per latent frame (num_frames_per_block == 1), each finite and shaped right.
    assert len(blocks) == NUM_FRAMES
    for index, (block_index, latent) in enumerate(blocks):
        assert block_index == index
        arr = np.array(latent.astype(mx.float32))
        assert arr.shape == (1, ARCH["out_channels"], 1, height, width)
        assert np.isfinite(arr).all(), f"block {index} produced non-finite latents"
