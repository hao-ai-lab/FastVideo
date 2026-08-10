# SPDX-License-Identifier: Apache-2.0
"""Track C Rung 4: MLX causal DiT full-forward parity vs the torch reference.

Feeds the same latent frame-blocks chunk-by-chunk through the torch
``CausalWanTransformer3DModel._forward_inference`` (dense-SDPA KV-cache path,
CPU) and the MLX ``MLXCausalWanDiT`` and asserts the streaming outputs match.
This is the model-level version of the mask-free-cached == block-causal
equivalence proven at the attention level in ``test_mlx_causal_attention.py``.
Tiny random-weight config; no real checkpoint needed.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

mx = pytest.importorskip("mlx.core", reason="MLX is required for the causal DiT parity test")

from fastvideo.configs.models.dits.wanvideo import WanVideoArchConfig, WanVideoConfig  # noqa: E402
from fastvideo.forward_context import set_forward_context  # noqa: E402
from fastvideo.layers.rotary_embedding import get_rotary_pos_embed  # noqa: E402
from fastvideo.mlx_runtime.causal_dit import MLXCausalWanDiT, MLXCausalWanTransformerBlock  # noqa: E402
from fastvideo.mlx_runtime.fastwan import mlx_block_weights_from_torch  # noqa: E402
from fastvideo.models.dits.causal_wanvideo import CausalWanTransformer3DModel  # noqa: E402
from fastvideo.pipelines.pipeline_batch_info import ForwardBatch  # noqa: E402
from fastvideo.tests.mlx.tiny_wan import TOP_LEVEL_KEY_MAP  # noqa: E402

SEED = 2026
NUM_HEADS, HEAD_DIM = 4, 16
NUM_LAYERS = 2
NUM_FRAMES = 4  # latent frames; num_frames_per_block = 1 -> 4 chunks
HEIGHT, WIDTH = 8, 8  # latent HxW; patch (1,2,2) -> 4x4 = 16 tokens/frame
TEXT_LEN = 8
ARCH = dict(
    num_attention_heads=NUM_HEADS, attention_head_dim=HEAD_DIM, in_channels=16, out_channels=16,
    text_dim=64, freq_dim=64, ffn_dim=128, num_layers=NUM_LAYERS, patch_size=(1, 2, 2),
    rope_max_seq_len=64, local_attn_size=-1, sink_size=0, num_frames_per_block=1)
HF = dict(
    num_attention_heads=NUM_HEADS, attention_head_dim=HEAD_DIM, in_channels=16, out_channels=16,
    text_dim=64, freq_dim=64, ffn_dim=128, num_layers=NUM_LAYERS, patch_size=(1, 2, 2), text_len=TEXT_LEN,
    rope_max_seq_len=64, eps=1e-6)


def _build_torch_model() -> CausalWanTransformer3DModel:
    cfg = WanVideoConfig(arch_config=WanVideoArchConfig(**ARCH))
    model = CausalWanTransformer3DModel(config=cfg, hf_config=HF).to("cpu", torch.float32).eval()
    torch.manual_seed(SEED + 3)
    with torch.no_grad():
        for name, param in model.named_parameters():
            if param.ndim <= 1:
                param.fill_(1.0) if (name.endswith("weight") and "norm" in name) else param.normal_(0.0, 0.02)
            else:
                torch.nn.init.xavier_uniform_(param)
    return model


def _mlx_from_torch(model: CausalWanTransformer3DModel) -> MLXCausalWanDiT:
    state = {name: value.detach().float() for name, value in model.state_dict().items()}
    inner_dim = NUM_HEADS * HEAD_DIM
    weights = {}
    for mlx_name, torch_name in TOP_LEVEL_KEY_MAP.items():
        tensor = state[torch_name]
        if mlx_name == "patch_embedding.weight":
            tensor = tensor.reshape(inner_dim, -1)
        weights[mlx_name] = mx.array(tensor.numpy())
    blocks = [
        MLXCausalWanTransformerBlock(mlx_block_weights_from_torch(tb), dim=inner_dim, ffn_dim=HF["ffn_dim"],
                                     num_heads=NUM_HEADS, eps=HF["eps"]) for tb in model.blocks
    ]
    config = dict(HF)
    config["text_len"] = model.text_len  # match the model's padding target (WanVideoConfig default 512)
    return MLXCausalWanDiT(weights, blocks, config, local_attn_size=-1, sink_size=0, num_frames_per_block=1)


def _full_rotary():
    d = (NUM_HEADS * HEAD_DIM) // NUM_HEADS
    rope_dim_list = [d - 4 * (d // 6), 2 * (d // 6), 2 * (d // 6)]
    cos, sin = get_rotary_pos_embed(
        (NUM_FRAMES, HEIGHT // 2, WIDTH // 2), NUM_HEADS * HEAD_DIM, NUM_HEADS, rope_dim_list,
        dtype=torch.float64, rope_theta=10000)
    return cos, sin, mx.array(cos.float().numpy()), mx.array(sin.float().numpy())


@pytest.mark.usefixtures("distributed_setup")
def test_mlx_causal_dit_matches_torch_streaming() -> None:
    torch_model = _build_torch_model()
    mlx_model = _mlx_from_torch(torch_model)
    frame_seqlen = (HEIGHT // 2) * (WIDTH // 2)

    gen = torch.Generator().manual_seed(SEED + 1)
    latents = torch.randn(1, ARCH["in_channels"], NUM_FRAMES, HEIGHT, WIDTH, generator=gen, dtype=torch.float32)
    text = torch.randn(1, TEXT_LEN, ARCH["text_dim"], generator=gen, dtype=torch.float32)
    _, _, cos_mx, sin_mx = _full_rotary()

    # torch reference: chunked _forward_inference with per-block kv/crossattn caches.
    window = 21 * frame_seqlen
    kv_cache = [{
        "k": torch.zeros(1, window, NUM_HEADS, HEAD_DIM), "v": torch.zeros(1, window, NUM_HEADS, HEAD_DIM),
        "global_end_index": torch.tensor([0]), "local_end_index": torch.tensor([0])
    } for _ in range(NUM_LAYERS)]
    crossattn_cache = [{
        "k": torch.zeros(1, TEXT_LEN, NUM_HEADS, HEAD_DIM), "v": torch.zeros(1, TEXT_LEN, NUM_HEADS, HEAD_DIM),
        "is_init": False
    } for _ in range(NUM_LAYERS)]

    torch_outs, mlx_outs = [], []
    mlx_kv, mlx_cx = mlx_model.allocate_caches(batch=1, frame_seqlen=frame_seqlen, dtype=mx.float32)
    with torch.no_grad(), set_forward_context(
            current_timestep=0, attn_metadata=None, forward_batch=ForwardBatch(data_type="dummy")):
        for i in range(NUM_FRAMES):
            chunk = latents[:, :, i:i + 1]
            timestep = torch.tensor([[10]], dtype=torch.long)
            torch_outs.append(torch_model(
                hidden_states=chunk, encoder_hidden_states=text, timestep=timestep,
                kv_cache=kv_cache, crossattn_cache=crossattn_cache,
                current_start=i * frame_seqlen, cache_start=0, start_frame=i).detach().float().numpy())

            cos_i = cos_mx[i * frame_seqlen:(i + 1) * frame_seqlen]
            sin_i = sin_mx[i * frame_seqlen:(i + 1) * frame_seqlen]
            out = mlx_model.forward_chunk(
                mx.array(chunk.numpy()), mx.array(text.numpy()), mx.array(timestep.float().numpy()),
                cos_i, sin_i, mlx_kv, mlx_cx, current_start=i * frame_seqlen)
            mx.eval(out)
            mlx_outs.append(np.array(out.astype(mx.float32)))

    torch_all = np.concatenate(torch_outs, axis=2)
    mlx_all = np.concatenate(mlx_outs, axis=2)
    np.testing.assert_allclose(mlx_all, torch_all, atol=2e-3, rtol=2e-3)
