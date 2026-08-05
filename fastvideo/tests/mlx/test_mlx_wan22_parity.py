# SPDX-License-Identifier: Apache-2.0
"""Track D Rung 2: per-token-timestep parity for the Wan2.2-TI2V-5B MLX port.

The 5B (FullAttn) differs from the ported 2.1 only in scale (config) and
*per-token* timestep conditioning (``expand_timesteps``: timestep is ``[B, L]``,
one noise level per patch token — how TI2V keeps the image frame at t=0 while
video frames are noised). Passing a 2-D timestep drives the torch model's
per-token path (``ts_seq_len``), so this compares ``MLXWan22DiT`` against the
torch reference on a tiny random-weight config. This is the run-6 prereq gate;
runs on CPU in CI. Real 5B-weight parity is a separate Metal-gated test.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

mx = pytest.importorskip("mlx.core", reason="MLX is required for the Wan2.2 parity test")

from fastvideo.forward_context import set_forward_context  # noqa: E402
from fastvideo.mlx_runtime.fastwan import mlx_block_weights_from_torch  # noqa: E402
from fastvideo.mlx_runtime.wan22 import (  # noqa: E402
    MLXWan22DiT,
    MLXWan22TransformerBlock,
)
from fastvideo.pipelines.pipeline_batch_info import ForwardBatch  # noqa: E402
from fastvideo.tests.mlx.tiny_wan import (  # noqa: E402
    TINY_ARCH,
    TOP_LEVEL_KEY_MAP,
    build_hf_config,
    build_tiny_wan_config,
    build_torch_model,
    mlx_rotary_embeddings,
)


def _mlx_wan22_from_torch(model, hf_config) -> MLXWan22DiT:
    state = {name: value.detach().float() for name, value in model.state_dict().items()}
    inner_dim = int(hf_config["num_attention_heads"]) * int(hf_config["attention_head_dim"])
    weights = {}
    for mlx_name, torch_name in TOP_LEVEL_KEY_MAP.items():
        tensor = state[torch_name]
        if mlx_name == "patch_embedding.weight":
            tensor = tensor.reshape(inner_dim, -1)
        weights[mlx_name] = mx.array(tensor.numpy())
    blocks = [
        MLXWan22TransformerBlock(
            mlx_block_weights_from_torch(tb), dim=inner_dim, ffn_dim=int(hf_config["ffn_dim"]),
            num_heads=int(hf_config["num_attention_heads"]), eps=float(hf_config["eps"])) for tb in model.blocks
    ]
    return MLXWan22DiT(weights, blocks, dict(hf_config))


@pytest.mark.usefixtures("distributed_setup")
def test_wan22_per_token_timestep_matches_torch() -> None:
    torch_model = build_torch_model()
    hf_config = build_hf_config(build_tiny_wan_config())
    mlx_model = _mlx_wan22_from_torch(torch_model, hf_config)

    # Tiny latent: 4 frames, 8x8 latent, patch (1,2,2) -> 4*4*4 = 64 tokens (L).
    frames, height, width = 4, 8, 8
    p_t, p_h, p_w = TINY_ARCH["patch_size"]
    num_tokens = (frames // p_t) * (height // p_h) * (width // p_w)
    tokens_per_frame = num_tokens // (frames // p_t)

    gen = torch.Generator().manual_seed(11)
    hidden = torch.randn(1, TINY_ARCH["in_channels"], frames, height, width, generator=gen, dtype=torch.float32)
    text = torch.randn(1, 8, TINY_ARCH["text_dim"], generator=gen, dtype=torch.float32)

    # Per-token timestep [B, L]: frame 0 clean (t=0, I2V-style), rest noised (t=500).
    per_frame = [0] + [500] * (frames // p_t - 1)
    timestep = torch.tensor([[per_frame[i // tokens_per_frame] for i in range(num_tokens)]], dtype=torch.long)

    with torch.no_grad(), set_forward_context(
            current_timestep=0, attn_metadata=None, forward_batch=ForwardBatch(data_type="dummy")):
        ref = torch_model(hidden_states=hidden, encoder_hidden_states=text, timestep=timestep).detach().float().numpy()

    freqs_cis = mlx_rotary_embeddings(hidden)
    out = mlx_model(mx.array(hidden.numpy()), mx.array(text.numpy()), mx.array(timestep.float().numpy()), freqs_cis)
    mx.eval(out)
    mlx_out = np.array(out.astype(mx.float32))

    np.testing.assert_allclose(mlx_out, ref, atol=2e-3, rtol=2e-3)
