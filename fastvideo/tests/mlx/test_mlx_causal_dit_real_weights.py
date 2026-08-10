# SPDX-License-Identifier: Apache-2.0
"""Track C rung 4: real-weight smoke for MLXCausalWanDiT.

Loads the released Self-Forcing causal checkpoint
(``wlsaidhi/SFWan2.1-T2V-1.3B-Diffusers``) into the MLX causal DiT and runs a
few streaming chunks, asserting finite, correctly-shaped output. This exercises
the Diffusers loader + full 30-layer forward at real scale/dtype — complementary
to the numeric parity test, which runs on a tiny random-weight config.

Gated on the checkpoint being present locally (set ``FASTVIDEO_SFWAN_ROOT`` or
place it at ``~/models/sfwan_t2v_1.3b``); skipped otherwise so CI and other
machines stay green.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core", reason="MLX is required for the real-weight smoke")

_ROOT = Path(os.environ.get("FASTVIDEO_SFWAN_ROOT", str(Path.home() / "models" / "sfwan_t2v_1.3b")))
_CHECKPOINT = _ROOT / "transformer" / "diffusion_pytorch_model.safetensors"
_CONFIG = _ROOT / "transformer" / "config.json"

pytestmark = pytest.mark.skipif(
    not (_CHECKPOINT.exists() and _CONFIG.exists()),
    reason=f"SFWan causal checkpoint not found under {_ROOT} (set FASTVIDEO_SFWAN_ROOT)")


def test_real_causal_dit_streams_finite_output() -> None:
    from fastvideo.mlx_runtime.causal_dit import mlx_causal_dit_from_diffusers_safetensors

    model = mlx_causal_dit_from_diffusers_safetensors(
        _CHECKPOINT, _CONFIG, dtype="fp16", local_attn_size=-1, sink_size=0, num_frames_per_block=1)

    # Small latent shape for a fast smoke: 32x32 latent (patch 1x2x2 -> 16x16 tokens/frame).
    in_channels = int(model.config["in_channels"])
    text_dim = int(model.config["text_dim"])
    height = width = 32
    frame_seqlen = (height // 2) * (width // 2)
    num_chunks = 3

    rng = np.random.default_rng(0)
    text = mx.array((rng.standard_normal((1, 24, text_dim)) * 0.1).astype(np.float16))
    kv_caches, crossattn_caches = model.allocate_caches(batch=1, frame_seqlen=frame_seqlen, dtype=mx.float16)

    # Rotary tables for the full clip (frame-major), sliced per chunk.
    from fastvideo.layers.rotary_embedding import get_rotary_pos_embed
    import torch

    head_dim = int(model.config["attention_head_dim"])
    num_heads = int(model.config["num_attention_heads"])
    rope_dim_list = [head_dim - 4 * (head_dim // 6), 2 * (head_dim // 6), 2 * (head_dim // 6)]
    cos, sin = get_rotary_pos_embed(
        (num_chunks, height // 2, width // 2), num_heads * head_dim, num_heads, rope_dim_list,
        dtype=torch.float32, rope_theta=10000)
    cos = mx.array(cos.float().numpy())
    sin = mx.array(sin.float().numpy())

    for i in range(num_chunks):
        chunk = mx.array((rng.standard_normal((1, in_channels, 1, height, width)) * 0.5).astype(np.float16))
        out = model.forward_chunk(
            chunk, text, mx.array([[900.0]]),
            cos[i * frame_seqlen:(i + 1) * frame_seqlen], sin[i * frame_seqlen:(i + 1) * frame_seqlen],
            kv_caches, crossattn_caches, current_start=i * frame_seqlen)
        mx.eval(out)
        arr = np.array(out.astype(mx.float32))
        assert arr.shape == (1, in_channels, 1, height, width)
        assert np.isfinite(arr).all(), f"chunk {i} produced non-finite output"
        assert np.abs(arr).max() < 1e3, f"chunk {i} output magnitude {np.abs(arr).max()} unreasonable"

    # The KV cache accumulated all chunks; cross-attn cache was populated once.
    assert kv_caches[0].global_end_index == num_chunks * frame_seqlen
    assert crossattn_caches[0]["is_init"] is True
