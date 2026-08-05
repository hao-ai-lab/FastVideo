# SPDX-License-Identifier: Apache-2.0
"""Track D Rung 4: I2V input-path parity for Wan2.2-TI2V-5B (DiT-level).

Proves that first-latent-frame replacement + per-token timestep (frame0=0,
rest=video_t) yields matching outputs between torch ``WanTransformer3DModel``
and ``MLXWan22DiT``. No VAE required — the "image" is a fixed latent frame.
Backend-agnostic (Metal or mlx[cpu]).
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

mx = pytest.importorskip("mlx.core", reason="MLX is required for the Wan2.2 I2V parity test")

from fastvideo.forward_context import set_forward_context  # noqa: E402
from fastvideo.mlx_runtime.wan22_i2v import (  # noqa: E402
    build_i2v_inputs,
    build_i2v_per_token_timestep,
    num_patch_tokens,
    replace_first_latent_frame,
    tokens_per_frame,
)
from fastvideo.pipelines.pipeline_batch_info import ForwardBatch  # noqa: E402
from fastvideo.tests.mlx.test_mlx_wan22_parity import _mlx_wan22_from_torch  # noqa: E402
from fastvideo.tests.mlx.tiny_wan import (  # noqa: E402
    TINY_ARCH,
    build_hf_config,
    build_tiny_wan_config,
    build_torch_model,
    mlx_rotary_embeddings,
)


def test_tokens_per_frame_frame_major_layout() -> None:
    """Document frame-major token order used by patch_embed / I2V timesteps."""
    frames, height, width = 4, 8, 8
    patch = TINY_ARCH["patch_size"]
    tpf = tokens_per_frame(height, width, patch)
    assert tpf == (height // patch[1]) * (width // patch[2]) == 16
    assert num_patch_tokens(frames, height, width, patch) == (frames // patch[0]) * tpf == 64

    ts = build_i2v_per_token_timestep(
        batch=1, frames=frames, height=height, width=width, patch_size=patch, video_timestep=500.0)
    assert ts.shape == (1, 64)
    # First frame's tokens are clean (0); remaining frames noised (500).
    assert np.all(ts[0, :tpf] == 0.0)
    assert np.all(ts[0, tpf:] == 500.0)


def test_replace_first_latent_frame_numpy_and_torch() -> None:
    rng = np.random.default_rng(0)
    noise = rng.standard_normal((2, 16, 4, 8, 8)).astype(np.float32)
    image = rng.standard_normal((2, 16, 8, 8)).astype(np.float32)
    out = replace_first_latent_frame(noise, image)
    np.testing.assert_array_equal(out[:, :, 0], image)
    np.testing.assert_array_equal(out[:, :, 1:], noise[:, :, 1:])

    noise_t = torch.from_numpy(noise)
    image_t = torch.from_numpy(image)
    out_t = replace_first_latent_frame(noise_t, image_t)
    assert torch.equal(out_t[:, :, 0], image_t)
    assert torch.equal(out_t[:, :, 1:], noise_t[:, :, 1:])


@pytest.mark.usefixtures("distributed_setup")
def test_wan22_i2v_inputs_match_torch_per_token_path() -> None:
    """I2V latents + timestep: MLXWan22DiT matches torch expand_timesteps path."""
    torch_model = build_torch_model()
    hf_config = build_hf_config(build_tiny_wan_config())
    mlx_model = _mlx_wan22_from_torch(torch_model, hf_config)

    frames, height, width = 4, 8, 8
    patch = TINY_ARCH["patch_size"]
    gen = torch.Generator().manual_seed(42)
    noise = torch.randn(1, TINY_ARCH["in_channels"], frames, height, width, generator=gen, dtype=torch.float32)
    # Distinct "image" content in frame 0 so a missed replacement would fail parity.
    image_frame = torch.randn(1, TINY_ARCH["in_channels"], height, width, generator=gen, dtype=torch.float32)
    text = torch.randn(1, 8, TINY_ARCH["text_dim"], generator=gen, dtype=torch.float32)

    latents_t, timestep_np = build_i2v_inputs(
        noise, image_frame, video_timestep=500.0, image_timestep=0.0, patch_size=patch)
    assert isinstance(latents_t, torch.Tensor)
    assert torch.equal(latents_t[:, :, 0], image_frame)
    timestep = torch.from_numpy(timestep_np).long()

    with torch.no_grad(), set_forward_context(
            current_timestep=0, attn_metadata=None, forward_batch=ForwardBatch(data_type="dummy")):
        ref = torch_model(hidden_states=latents_t, encoder_hidden_states=text, timestep=timestep)
        ref_np = ref.detach().float().numpy()

    # MLX path: same I2V construction via numpy/mx.
    latents_mx, ts_np = build_i2v_inputs(
        mx.array(noise.numpy()),
        mx.array(image_frame.numpy()),
        video_timestep=500.0,
        patch_size=patch,
    )
    freqs_cis = mlx_rotary_embeddings(latents_t)
    out = mlx_model(
        latents_mx if isinstance(latents_mx, mx.array) else mx.array(latents_mx),
        mx.array(text.numpy()),
        mx.array(ts_np),
        freqs_cis,
    )
    mx.eval(out)
    mlx_np = np.array(out.astype(mx.float32))

    np.testing.assert_allclose(mlx_np, ref_np, atol=2e-3, rtol=2e-3)
