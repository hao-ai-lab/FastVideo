# SPDX-License-Identifier: Apache-2.0
"""Tiny random-weight MiniMax H3 fixtures shared by the MLX runtime tests.

Builds a miniature ``MiniMaxH3Transformer3DModel`` (upstream torch reference,
merged in #1674) plus the conversion into
``fastvideo.mlx_runtime.minimax_h3.MLXMiniMaxH3DiT``. All matmul dims are
multiples of the int8 group size (64) so the quantized variants of the tests
reuse the same model; ``rope_freq_dim`` keeps the rotary width (6 * dim)
inside the head dim, as in the real config (96 <= 128).
"""

from __future__ import annotations

import os

import numpy as np
import torch

os.environ.setdefault("FASTVIDEO_ATTENTION_BACKEND", "TORCH_SDPA")
os.environ.setdefault("MASTER_ADDR", "localhost")
os.environ.setdefault("MASTER_PORT", "29513")

from fastvideo.configs.models.dits.minimax_h3 import (  # noqa: E402
    MiniMaxH3ArchConfig,
    MiniMaxH3Config,
)
from fastvideo.forward_context import set_forward_context  # noqa: E402
from fastvideo.mlx_runtime.minimax_h3 import (  # noqa: E402
    MLXMiniMaxH3DiT,
    MiniMaxH3SchedulerState,
    MINIMAX_H3_AUDIO_SHIFT,
    MINIMAX_H3_VIDEO_SHIFT,
    build_packed_layout,
    build_row_timesteps,
    patchify_video_latents,
)
from fastvideo.models.dits.minimax_h3 import MiniMaxH3Transformer3DModel  # noqa: E402
from fastvideo.pipelines.pipeline_batch_info import ForwardBatch  # noqa: E402

SEED = 2027

TINY_ARCH = dict(
    num_attention_heads=8,
    attention_head_dim=64,  # inner 512 > hidden 256, like the real 7168 > 5376
    hidden_size=256,
    num_layers=2,
    num_refiner_layers=1,
    ffn_dim=384,
    in_channels=8,
    audio_in_channels=4,
    patch_size=(1, 2, 2),
    text_dim=128,
    freq_dim=64,
    time_embed_hidden_dim=256,
    time_embed_dim=192,
    rope_freq_dim=8,  # rotary width 48 <= 64 head dim
    rope_theta=10000.0,
    norm_eps=1e-5,
    qk_norm_eps=1e-5,
    final_norm_eps=1e-5,
)

# Geometry: 2 latent frames of 4x4, patch (1,2,2) -> 4 rows/frame -> 8 video
# rows; 6 text rows; 3 audio latents -> 6 audio rows. Sequence length 20.
NUM_TEXT_TOKENS = 6
NUM_LATENT_FRAMES = 2
LATENT_HEIGHT = 4
LATENT_WIDTH = 4
NUM_AUDIO_LATENTS = 3
VIDEO_TIMESTEP = 0.7
AUDIO_TIMESTEP = 0.4


def build_tiny_h3_config() -> MiniMaxH3Config:
    return MiniMaxH3Config(arch_config=MiniMaxH3ArchConfig(**TINY_ARCH))


def build_hf_config(config: MiniMaxH3Config) -> dict[str, object]:
    arch = config.arch_config
    return {
        "num_attention_heads": arch.num_attention_heads,
        "attention_head_dim": arch.attention_head_dim,
        "hidden_size": arch.hidden_size,
        "num_layers": arch.num_layers,
        "num_refiner_layers": arch.num_refiner_layers,
        "ffn_dim": arch.ffn_dim,
        "in_channels": arch.in_channels,
        "audio_in_channels": arch.audio_in_channels,
        "patch_size": list(arch.patch_size),
        "text_dim": arch.text_dim,
        "freq_dim": arch.freq_dim,
        "time_embed_hidden_dim": arch.time_embed_hidden_dim,
        "time_embed_dim": arch.time_embed_dim,
        "rope_freq_dim": arch.rope_freq_dim,
        "rope_theta": arch.rope_theta,
        "norm_eps": arch.norm_eps,
        "qk_norm_eps": arch.qk_norm_eps,
        "final_norm_eps": arch.final_norm_eps,
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


def build_torch_model() -> MiniMaxH3Transformer3DModel:
    config = build_tiny_h3_config()
    model = MiniMaxH3Transformer3DModel(config=config, hf_config=build_hf_config(config))
    model = model.to(device="cpu", dtype=torch.float32)
    initialize_model_parameters(model)
    model.eval()
    return model


def build_layout():
    return build_packed_layout(
        NUM_TEXT_TOKENS,
        NUM_LATENT_FRAMES,
        LATENT_HEIGHT,
        LATENT_WIDTH,
        NUM_AUDIO_LATENTS,
        patch_size=tuple(TINY_ARCH["patch_size"]),
    )


def build_inputs():
    """Random rows + the packed layout, in both torch and numpy form."""
    layout = build_layout()
    generator = torch.Generator(device="cpu").manual_seed(SEED + 1)
    channels = TINY_ARCH["in_channels"]
    latents = torch.randn(
        1,
        channels,
        NUM_LATENT_FRAMES,
        LATENT_HEIGHT,
        LATENT_WIDTH,
        generator=generator,
        dtype=torch.float32,
    )
    video_rows_np = patchify_video_latents(latents.numpy(), tuple(TINY_ARCH["patch_size"]))
    patch_dim = video_rows_np.shape[-1]
    audio_rows_np = np.random.default_rng(SEED + 2).standard_normal(
        (NUM_AUDIO_LATENTS * 2, TINY_ARCH["audio_in_channels"])).astype(np.float32)
    text_rows_np = np.random.default_rng(SEED + 5).standard_normal(
        (NUM_TEXT_TOKENS, TINY_ARCH["text_dim"])).astype(np.float32)

    unique, inverse = build_row_timesteps(layout, VIDEO_TIMESTEP, AUDIO_TIMESTEP)

    torch_inputs = dict(
        hidden_states=torch.from_numpy(video_rows_np)[None],
        audio_hidden_states=torch.from_numpy(audio_rows_np)[None],
        encoder_hidden_states=torch.from_numpy(text_rows_np)[None],
        timestep=torch.from_numpy(unique),
        timestep_indices=torch.from_numpy(inverse),
        token_tags=torch.from_numpy(layout.token_tags),
        position_ids=torch.from_numpy(layout.position_ids),
        video_indices=torch.from_numpy(layout.video_indices),
        audio_indices=torch.from_numpy(layout.audio_indices),
        text_indices=torch.from_numpy(layout.text_indices),
    )
    mlx_inputs = dict(
        video_rows=video_rows_np,
        audio_rows=audio_rows_np,
        text_rows=text_rows_np,
        timesteps=unique,
        timestep_indices=inverse,
    )
    return layout, torch_inputs, mlx_inputs


# torch (FastVideo module names) -> MLX (released checkpoint / diffusers names,
# per MiniMaxH3ArchConfig.param_names_mapping read in reverse).
def torch_key_to_mlx(name: str) -> str:
    if name.startswith("time_embedder.fc_in."):
        return name.replace("time_embedder.fc_in.", "time_embedder.linear_1.", 1)
    if name.startswith("time_embedder.fc_out."):
        return name.replace("time_embedder.fc_out.", "time_embedder.linear_2.", 1)
    name = name.replace(".attn.to_out.", ".attn.to_out.0.")
    name = name.replace(".ff.fc_in.", ".ff.net.0.proj.")
    name = name.replace(".ff.fc_out.", ".ff.net.2.")
    return name


def mlx_dit_from_torch_model(
    model: MiniMaxH3Transformer3DModel,
    hf_config: dict[str, object],
    *,
    quantization=None,
) -> MLXMiniMaxH3DiT:
    import mlx.core as mx

    from fastvideo.mlx_runtime.minimax_h3 import _is_quantizable, quantize_matrix

    weights: dict[str, object] = {}
    blocks: list[dict[str, object]] = [{} for _ in range(TINY_ARCH["num_layers"])]
    refiner: list[dict[str, object]] = [{} for _ in range(TINY_ARCH["num_refiner_layers"])]

    for torch_name, param in model.state_dict().items():
        name = torch_key_to_mlx(torch_name)
        if name.startswith("rope."):
            continue  # non-persistent analytic buffer, rebuilt on the fly
        array = mx.array(param.detach().float().numpy())
        if quantization is not None and _is_quantizable(name):
            value = quantize_matrix(array, quantization)
        else:
            value = array
        if name.startswith("transformer_blocks."):
            _, index_str, sub = name.split(".", 2)
            blocks[int(index_str)][sub] = value
        elif name.startswith("token_refiner.refiner_blocks."):
            _, _, index_str, sub = name.split(".", 3)
            refiner[int(index_str)][sub] = value
        else:
            weights[name] = value
    return MLXMiniMaxH3DiT(weights, blocks, refiner, dict(hf_config))


def torch_reference_output(model, torch_inputs) -> tuple[np.ndarray, np.ndarray]:
    with torch.no_grad(), set_forward_context(
            current_timestep=0,
            attn_metadata=None,
            forward_batch=ForwardBatch(data_type="dummy"),
    ):
        video_output, audio_output = model(**torch_inputs)
    return video_output[0].float().numpy(), audio_output[0].float().numpy()


def mlx_output(dit: MLXMiniMaxH3DiT, layout, mlx_inputs) -> tuple[np.ndarray, np.ndarray]:
    import mlx.core as mx

    video_output, audio_output = dit.forward(
        mx.array(mlx_inputs["video_rows"]),
        mx.array(mlx_inputs["audio_rows"]),
        mx.array(mlx_inputs["text_rows"]),
        position_ids=mx.array(layout.position_ids.astype(np.float32)),
        token_tags=mx.array(layout.token_tags),
        timestep_indices=mx.array(mlx_inputs["timestep_indices"]),
        timesteps=mx.array(mlx_inputs["timesteps"]),
        video_indices=mx.array(layout.video_indices),
        audio_indices=mx.array(layout.audio_indices),
        text_indices=mx.array(layout.text_indices),
    )
    mx.eval(video_output, audio_output)
    return np.asarray(video_output), np.asarray(audio_output)


def mlx_cache_output(dit: MLXMiniMaxH3DiT, layout, mlx_inputs) -> tuple[np.ndarray, np.ndarray]:
    import mlx.core as mx

    union = np.unique(
        np.concatenate([
            mlx_inputs["timesteps"],
            np.array([1.0], dtype=np.float32),  # condition rows, if any
        ]))
    dit.precompute_adaln(union, drop_weights=True)
    video_output, audio_output = dit.forward_with_cache(
        mx.array(mlx_inputs["video_rows"]),
        mx.array(mlx_inputs["audio_rows"]),
        mx.array(mlx_inputs["text_rows"]),
        layout=layout,
        step_timesteps=mlx_inputs["timesteps"],
        row_timestep_inverse=mlx_inputs["timestep_indices"],
    )
    mx.eval(video_output, audio_output)
    return np.asarray(video_output), np.asarray(audio_output)


def build_schedulers(num_denoise_steps: int = 6) -> tuple[MiniMaxH3SchedulerState, MiniMaxH3SchedulerState]:
    return (
        MiniMaxH3SchedulerState.create(MINIMAX_H3_VIDEO_SHIFT, num_denoise_steps),
        MiniMaxH3SchedulerState.create(MINIMAX_H3_AUDIO_SHIFT, num_denoise_steps),
    )
