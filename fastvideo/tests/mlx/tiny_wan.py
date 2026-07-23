# SPDX-License-Identifier: Apache-2.0
"""Tiny random-weight Wan DiT fixtures shared by the MLX runtime tests.

Builds a miniature ``WanTransformer3DModel`` (all matmul dims are multiples of
the int8 group size 64, so the same model exercises the quantized paths) plus
the conversions into ``fastvideo.mlx_runtime.fastwan.MLXWanDiT``.
"""

from __future__ import annotations

import os

import numpy as np
import torch

os.environ.setdefault("FASTVIDEO_ATTENTION_BACKEND", "TORCH_SDPA")
os.environ.setdefault("MASTER_ADDR", "localhost")
os.environ.setdefault("MASTER_PORT", "29513")

from fastvideo.configs.models.dits.wanvideo import (  # noqa: E402
    WanVideoArchConfig,
    WanVideoConfig,
)
from fastvideo.forward_context import set_forward_context  # noqa: E402
from fastvideo.layers.rotary_embedding import get_rotary_pos_embed  # noqa: E402
from fastvideo.mlx_runtime.fastwan import (  # noqa: E402
    MLXQuantizationSpec,
    MLXWanDiT,
    MLXWanTransformerBlock,
    mlx_block_weights_from_torch,
    quantize_matrix,
)
from fastvideo.models.dits.wanvideo import WanTransformer3DModel  # noqa: E402
from fastvideo.pipelines.pipeline_batch_info import ForwardBatch  # noqa: E402

SEED = 2026

# All matmul dims must be multiples of the int8 group size (64) so the
# quantized variants of the tests can reuse the same tiny model.
TINY_ARCH = dict(
    num_attention_heads=4,
    attention_head_dim=16,
    in_channels=16,
    out_channels=16,
    text_dim=64,
    freq_dim=64,
    ffn_dim=128,
    num_layers=2,
    patch_size=(1, 2, 2),
    rope_max_seq_len=64,
)


def build_tiny_wan_config() -> WanVideoConfig:
    return WanVideoConfig(arch_config=WanVideoArchConfig(**TINY_ARCH))


def build_hf_config(config: WanVideoConfig) -> dict[str, object]:
    return {
        "num_attention_heads": config.num_attention_heads,
        "attention_head_dim": config.attention_head_dim,
        "in_channels": config.in_channels,
        "out_channels": config.out_channels,
        "text_dim": config.text_dim,
        "freq_dim": config.freq_dim,
        "ffn_dim": config.ffn_dim,
        "num_layers": config.num_layers,
        "patch_size": config.patch_size,
        "text_len": config.text_len,
        "rope_max_seq_len": config.rope_max_seq_len,
        "eps": 1e-6,
    }


def initialize_model_parameters(model: torch.nn.Module) -> None:
    # ReplicatedLinear parameters are allocated with torch.empty and need an
    # explicit initialization in tests to avoid undefined values.
    torch.manual_seed(SEED + 3)
    with torch.no_grad():
        for name, param in model.named_parameters():
            if param.ndim <= 1:
                if name.endswith("weight") and "norm" in name:
                    param.fill_(1.0)
                else:
                    param.normal_(mean=0.0, std=0.02)
                continue
            torch.nn.init.xavier_uniform_(param)


def build_torch_model() -> WanTransformer3DModel:
    config = build_tiny_wan_config()
    model = WanTransformer3DModel(config=config, hf_config=build_hf_config(config))
    model = model.to(device="cpu", dtype=torch.float32)
    initialize_model_parameters(model)
    model.eval()
    return model


def build_inputs() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(SEED + 1)
    hidden_states = torch.randn(1, TINY_ARCH["in_channels"], 4, 8, 8, generator=generator, dtype=torch.float32)
    encoder_hidden_states = torch.randn(1, 8, TINY_ARCH["text_dim"], generator=generator, dtype=torch.float32)
    timestep = torch.tensor([10], dtype=torch.long)
    return hidden_states, encoder_hidden_states, timestep


# Top-level weights: the MLX runtime uses the Diffusers key layout; the torch
# model uses FastVideo's module names (see WanVideoConfig.param_names_mapping).
TOP_LEVEL_KEY_MAP = {
    "patch_embedding.weight": "patch_embedding.proj.weight",
    "patch_embedding.bias": "patch_embedding.proj.bias",
    "condition_embedder.time_embedder.linear_1.weight": "condition_embedder.time_embedder.mlp.fc_in.weight",
    "condition_embedder.time_embedder.linear_1.bias": "condition_embedder.time_embedder.mlp.fc_in.bias",
    "condition_embedder.time_embedder.linear_2.weight": "condition_embedder.time_embedder.mlp.fc_out.weight",
    "condition_embedder.time_embedder.linear_2.bias": "condition_embedder.time_embedder.mlp.fc_out.bias",
    "condition_embedder.time_proj.weight": "condition_embedder.time_modulation.linear.weight",
    "condition_embedder.time_proj.bias": "condition_embedder.time_modulation.linear.bias",
    "condition_embedder.text_embedder.linear_1.weight": "condition_embedder.text_embedder.fc_in.weight",
    "condition_embedder.text_embedder.linear_1.bias": "condition_embedder.text_embedder.fc_in.bias",
    "condition_embedder.text_embedder.linear_2.weight": "condition_embedder.text_embedder.fc_out.weight",
    "condition_embedder.text_embedder.linear_2.bias": "condition_embedder.text_embedder.fc_out.bias",
    "scale_shift_table": "scale_shift_table",
    "proj_out.weight": "proj_out.weight",
    "proj_out.bias": "proj_out.bias",
}


def mlx_dit_from_torch_model(
    model: WanTransformer3DModel,
    hf_config: dict[str, object],
    *,
    quantization: MLXQuantizationSpec | None = None,
) -> MLXWanDiT:
    import mlx.core as mx

    state = {name: value.detach().float() for name, value in model.state_dict().items()}
    inner_dim = int(hf_config["num_attention_heads"]) * int(hf_config["attention_head_dim"])  # type: ignore[arg-type]

    weights = {}
    for mlx_name, torch_name in TOP_LEVEL_KEY_MAP.items():
        tensor = state[torch_name]
        if mlx_name == "patch_embedding.weight":
            tensor = tensor.reshape(inner_dim, -1)
        array = mx.array(tensor.numpy())
        if quantization is not None and mlx_name.endswith(".weight") and mlx_name != "scale_shift_table":
            weights[mlx_name] = quantize_matrix(array, quantization)
        else:
            weights[mlx_name] = array

    blocks = []
    for torch_block in model.blocks:
        block_weights = mlx_block_weights_from_torch(torch_block)
        if quantization is not None:
            block_weights = {
                name: (quantize_matrix(value, quantization)
                       if name.endswith(".weight") and "norm" not in name and len(value.shape) >= 2 else value)
                for name, value in block_weights.items()
            }
        blocks.append(
            MLXWanTransformerBlock(
                block_weights,
                dim=inner_dim,
                ffn_dim=int(hf_config["ffn_dim"]),  # type: ignore[arg-type]
                num_heads=int(hf_config["num_attention_heads"]),  # type: ignore[arg-type]
                eps=float(hf_config["eps"]),  # type: ignore[arg-type]
            ))
    return MLXWanDiT(weights, blocks, dict(hf_config))


def mlx_rotary_embeddings(hidden_states: torch.Tensor):
    """The rotary table the torch model builds internally, converted to MLX."""
    import mlx.core as mx

    _, _, frames, height, width = hidden_states.shape
    p_t, p_h, p_w = TINY_ARCH["patch_size"]
    head_dim = TINY_ARCH["attention_head_dim"]
    hidden_size = TINY_ARCH["num_attention_heads"] * head_dim
    rope_dim_list = [head_dim - 4 * (head_dim // 6), 2 * (head_dim // 6), 2 * (head_dim // 6)]
    freqs_cos, freqs_sin = get_rotary_pos_embed(
        (frames // p_t, height // p_h, width // p_w),
        hidden_size,
        TINY_ARCH["num_attention_heads"],
        rope_dim_list,
        dtype=torch.float64,
        rope_theta=10000,
    )
    return (
        mx.array(freqs_cos.float().numpy()).astype(mx.float32),
        mx.array(freqs_sin.float().numpy()).astype(mx.float32),
    )


def torch_reference_output(model, hidden_states, encoder_hidden_states, timestep) -> np.ndarray:
    with torch.no_grad(), set_forward_context(
            current_timestep=0,
            attn_metadata=None,
            forward_batch=ForwardBatch(data_type="dummy"),
    ):
        output = model(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            timestep=timestep,
        )
    return output.detach().float().cpu().numpy()


def mlx_output(dit, hidden_states, encoder_hidden_states, timestep, freqs_cis) -> np.ndarray:
    import mlx.core as mx

    out = dit(
        mx.array(hidden_states.numpy()),
        mx.array(encoder_hidden_states.numpy()),
        mx.array(timestep.float().numpy()),
        freqs_cis,
    )
    mx.eval(out)
    return np.array(out.astype(mx.float32))
