"""Run a full real FastWan DiT forward pass in the experimental MLX FP16 runtime.

This wires the pieces around the block stack:

- patch embedding
- time embedding + time modulation
- text projection
- all Wan DiT blocks
- output norm/projection/unpatchify

Inputs are random latent/text tensors for now; this is a runtime correctness and
shape smoke test before integrating the scheduler, text encoder, and decoder.
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
    parser = argparse.ArgumentParser(description="Full FastWan DiT forward smoke for the experimental MLX runtime")
    parser.add_argument("--model-root", type=Path, default=DEFAULT_MODEL_ROOT)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=448)
    parser.add_argument("--num-frames", type=int, default=33)
    parser.add_argument("--text-len", type=int, default=512)
    parser.add_argument("--dtype", choices=("fp16", "fp32"), default="fp16")
    parser.add_argument(
        "--quantization",
        choices=("none", "int8", "int4", "mxfp8", "mxfp4", "nvfp4"),
        default="none",
    )
    parser.add_argument("--num-blocks", type=int, default=None, help="Defaults to all blocks.")
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    import mlx.core as mx
    import torch

    from fastvideo.layers.rotary_embedding import get_rotary_pos_embed
    from fastvideo.mlx_runtime.fastwan import mlx_dit_from_diffusers_safetensors

    mx.random.seed(args.seed)
    torch.manual_seed(args.seed)

    config_path = args.model_root / "transformer/config.json"
    checkpoint_path = args.model_root / "transformer/diffusion_pytorch_model.safetensors"
    config = json.loads(config_path.read_text())
    num_heads = int(config["num_attention_heads"])
    head_dim = int(config["attention_head_dim"])
    hidden_size = num_heads * head_dim
    text_dim = int(config["text_dim"])
    patch_size = tuple(config["patch_size"])
    in_channels = int(config["in_channels"])
    total_blocks = int(config["num_layers"])
    num_blocks = total_blocks if args.num_blocks is None else args.num_blocks

    latent_frames = (args.num_frames - 1) // 4 + 1
    latent_height = args.height // 8
    latent_width = args.width // 8
    post_patch = (
        latent_frames // patch_size[0],
        latent_height // patch_size[1],
        latent_width // patch_size[2],
    )
    rope_dim_list = [head_dim - 4 * (head_dim // 6), 2 * (head_dim // 6), 2 * (head_dim // 6)]
    freqs_cos, freqs_sin = get_rotary_pos_embed(
        post_patch,
        hidden_size,
        num_heads,
        rope_dim_list,
        dtype=torch.float32,
        rope_theta=10000,
    )
    freqs_mx = (mx.array(freqs_cos.numpy()).astype(mx.float32), mx.array(freqs_sin.numpy()).astype(mx.float32))

    mx_dtype = mx.float16 if args.dtype == "fp16" else mx.float32
    hidden_states = mx.random.normal((1, in_channels, latent_frames, latent_height, latent_width), dtype=mx_dtype)
    encoder_hidden_states = mx.random.normal((1, args.text_len, text_dim), dtype=mx_dtype)
    timestep = mx.array([500.0]).astype(mx.float32)

    print(
        f"MLX FastWan full DiT: blocks={num_blocks}/{total_blocks}, dtype={args.dtype}, "
        f"quant={args.quantization}, latent={latent_frames}x{latent_height}x{latent_width}, text={args.text_len}"
    )

    load_start = time.perf_counter()
    dit = mlx_dit_from_diffusers_safetensors(
        checkpoint_path,
        config_path,
        dtype=args.dtype,
        num_blocks=args.num_blocks,
        quantization=None if args.quantization == "none" else args.quantization,
    )
    load_ms = (time.perf_counter() - load_start) * 1000

    run_start = time.perf_counter()
    output = dit(hidden_states, encoder_hidden_states, timestep, freqs_mx)
    mx.eval(output)
    run_ms = (time.perf_counter() - run_start) * 1000

    print(f"Loaded DiT in: {load_ms:.2f} ms")
    print(f"Ran DiT forward in: {run_ms:.2f} ms")
    print(f"Output shape: {output.shape}, dtype={output.dtype}")


if __name__ == "__main__":
    main()
