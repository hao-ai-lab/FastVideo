"""Smoke-test the real FastWan DiT block stack in MLX FP16.

This loads real block weights from the FastWan 1.3B checkpoint and runs the
Wan transformer blocks sequentially on precomputed hidden/text/timestep
embeddings. It is the first "runtime stack" test before wiring patch embedding,
text/time conditioning, scheduler, and decode for full video generation.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

DEFAULT_MODEL_ROOT = (
    Path.home()
    / ".cache/huggingface/hub/models--FastVideo--FastWan2.1-T2V-1.3B-Diffusers/"
    "snapshots/25e7ed7f41fd8ce2fdd108688c65e8caf0ce3aef"
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run real FastWan DiT blocks in the experimental MLX runtime")
    parser.add_argument("--model-root", type=Path, default=DEFAULT_MODEL_ROOT)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=448)
    parser.add_argument("--num-frames", type=int, default=33)
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--text-len", type=int, default=64)
    parser.add_argument("--num-blocks", type=int, default=None, help="Defaults to all blocks in the model config.")
    parser.add_argument("--dtype", choices=("fp16", "fp32"), default="fp16")
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    import mlx.core as mx
    import torch

    from fastvideo.layers.rotary_embedding import get_rotary_pos_embed
    from fastvideo.mlx_runtime.fastwan import MLXWanTransformerBlock, mlx_block_weights_from_diffusers_safetensors

    mx.random.seed(args.seed)
    torch.manual_seed(args.seed)

    config_path = args.model_root / "transformer/config.json"
    checkpoint_path = args.model_root / "transformer/diffusion_pytorch_model.safetensors"
    config = json.loads(config_path.read_text())
    dim = int(config["num_attention_heads"]) * int(config["attention_head_dim"])
    num_heads = int(config["num_attention_heads"])
    head_dim = int(config["attention_head_dim"])
    ffn_dim = int(config["ffn_dim"])
    num_blocks = int(config["num_layers"]) if args.num_blocks is None else args.num_blocks
    patch_size = tuple(config["patch_size"])

    latent_frames = (args.num_frames - 1) // 4 + 1
    latent_height = args.height // 8
    latent_width = args.width // 8
    post_patch = (
        latent_frames // patch_size[0],
        latent_height // patch_size[1],
        latent_width // patch_size[2],
    )
    full_seq_len = post_patch[0] * post_patch[1] * post_patch[2]
    seq_len = min(args.seq_len, full_seq_len)
    rope_dim_list = [head_dim - 4 * (head_dim // 6), 2 * (head_dim // 6), 2 * (head_dim // 6)]
    freqs_cos, freqs_sin = get_rotary_pos_embed(
        post_patch,
        dim,
        num_heads,
        rope_dim_list,
        dtype=torch.float32,
        rope_theta=10000,
    )
    freqs_mx = (
        mx.array(freqs_cos[:seq_len].numpy()).astype(mx.float32),
        mx.array(freqs_sin[:seq_len].numpy()).astype(mx.float32),
    )

    mx_dtype = mx.float16 if args.dtype == "fp16" else mx.float32
    hidden_states = mx.random.normal((1, seq_len, dim), dtype=mx_dtype)
    encoder_hidden_states = mx.random.normal((1, args.text_len, dim), dtype=mx_dtype)
    temb = mx.random.normal((1, 6, dim), dtype=mx_dtype)

    print(
        f"MLX FastWan block stack: blocks={num_blocks}, dtype={args.dtype}, seq={seq_len}/{full_seq_len}, "
        f"text={args.text_len}, dim={dim}, heads={num_heads}"
    )

    load_start = time.perf_counter()
    blocks = []
    for block_index in range(num_blocks):
        weights = mlx_block_weights_from_diffusers_safetensors(checkpoint_path, block_index=block_index)
        if args.dtype == "fp16":
            weights = {name: value.astype(mx.float16) for name, value in weights.items()}
        blocks.append(MLXWanTransformerBlock(weights, dim=dim, ffn_dim=ffn_dim, num_heads=num_heads))
    mx.eval(hidden_states, encoder_hidden_states, temb)
    load_ms = (time.perf_counter() - load_start) * 1000

    run_start = time.perf_counter()
    for block in blocks:
        hidden_states = block(hidden_states, encoder_hidden_states, temb, freqs_cis=freqs_mx)
    mx.eval(hidden_states)
    run_ms = (time.perf_counter() - run_start) * 1000

    print(f"Loaded blocks in: {load_ms:.2f} ms")
    print(f"Ran block stack in: {run_ms:.2f} ms")
    print(f"Output shape: {hidden_states.shape}, dtype={hidden_states.dtype}")


if __name__ == "__main__":
    main()
