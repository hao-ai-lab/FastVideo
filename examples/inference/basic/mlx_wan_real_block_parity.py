"""Run a real FastWan checkpoint block through the experimental MLX runtime.

This is the first meaningful FP16 MLX runtime test:

- loads one real block from the downloaded FastWan 1.3B transformer checkpoint,
- applies Wan rotary embeddings,
- runs the block in MLX,
- compares against the FastVideo/PyTorch block on MPS when available.

It is still a block-level runtime test, not full video generation.
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


def _patch_single_process_attention() -> None:
    import fastvideo.attention.layer as attention_layer

    attention_layer.get_sp_parallel_rank = lambda: 0
    attention_layer.get_sp_world_size = lambda: 1
    attention_layer.sequence_model_parallel_all_to_all_4D = lambda x, scatter_dim, gather_dim: x
    attention_layer.sequence_model_parallel_all_gather = lambda x, dim=0: x


def main() -> None:
    parser = argparse.ArgumentParser(description="Real FastWan block parity for the experimental MLX runtime")
    parser.add_argument("--model-root", type=Path, default=DEFAULT_MODEL_ROOT)
    parser.add_argument("--block-index", type=int, default=0)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=448)
    parser.add_argument("--num-frames", type=int, default=33)
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--text-len", type=int, default=64)
    parser.add_argument("--dtype", choices=("fp16", "fp32"), default="fp16")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--atol", type=float, default=5e-2)
    parser.add_argument("--rtol", type=float, default=5e-2)
    parser.add_argument("--skip-reference", action="store_true", help="Only run the MLX block timing path.")
    args = parser.parse_args()

    import mlx.core as mx
    import numpy as np
    import torch

    from fastvideo.forward_context import set_forward_context
    from fastvideo.layers.rotary_embedding import get_rotary_pos_embed
    from fastvideo.mlx_runtime.fastwan import (
        MLXWanTransformerBlock,
        mlx_block_weights_from_diffusers_safetensors,
        torch_block_state_from_diffusers_safetensors,
    )
    from fastvideo.models.dits.wanvideo import WanTransformerBlock
    from fastvideo.platforms import AttentionBackendEnum

    _patch_single_process_attention()
    torch.manual_seed(args.seed)

    config_path = args.model_root / "transformer/config.json"
    checkpoint_path = args.model_root / "transformer/diffusion_pytorch_model.safetensors"
    config = json.loads(config_path.read_text())
    dim = int(config["num_attention_heads"]) * int(config["attention_head_dim"])
    num_heads = int(config["num_attention_heads"])
    head_dim = int(config["attention_head_dim"])
    ffn_dim = int(config["ffn_dim"])
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
    freqs_cos = freqs_cos[:seq_len]
    freqs_sin = freqs_sin[:seq_len]

    mlx_weights = mlx_block_weights_from_diffusers_safetensors(checkpoint_path, block_index=args.block_index)
    if args.dtype == "fp16":
        mlx_weights = {name: value.astype(mx.float16) for name, value in mlx_weights.items()}
        mx_dtype = mx.float16
        torch_dtype = torch.float16
    else:
        mx_dtype = mx.float32
        torch_dtype = torch.float32

    hidden_np = torch.randn(1, seq_len, dim, dtype=torch.float32).numpy()
    encoder_np = torch.randn(1, args.text_len, dim, dtype=torch.float32).numpy()
    temb_np = torch.randn(1, 6, dim, dtype=torch.float32).numpy()

    mlx_block = MLXWanTransformerBlock(mlx_weights, dim=dim, ffn_dim=ffn_dim, num_heads=num_heads)
    hidden_mx = mx.array(hidden_np).astype(mx_dtype)
    encoder_mx = mx.array(encoder_np).astype(mx_dtype)
    temb_mx = mx.array(temb_np).astype(mx_dtype)
    freqs_mx = (mx.array(freqs_cos.numpy()).astype(mx.float32), mx.array(freqs_sin.numpy()).astype(mx.float32))

    start = time.perf_counter()
    mlx_output = mlx_block(hidden_mx, encoder_mx, temb_mx, freqs_cis=freqs_mx)
    mx.eval(mlx_output)
    mlx_ms = (time.perf_counter() - start) * 1000

    print(
        f"real block: block={args.block_index}, dtype={args.dtype}, seq={seq_len}/{full_seq_len}, "
        f"text={args.text_len}, dim={dim}, heads={num_heads}"
    )
    print(f"MLX block latency: {mlx_ms:.2f} ms")

    if args.skip_reference:
        return

    device = torch.device("mps" if torch.backends.mps.is_available() and args.dtype == "fp16" else "cpu")
    torch_block = WanTransformerBlock(
        dim,
        ffn_dim,
        num_heads,
        cross_attn_norm=True,
        supported_attention_backends=(AttentionBackendEnum.TORCH_SDPA,),
    ).eval()
    torch_block.load_state_dict(torch_block_state_from_diffusers_safetensors(checkpoint_path, block_index=args.block_index))
    torch_block = torch_block.to(device=device, dtype=torch_dtype)

    hidden_torch = torch.from_numpy(hidden_np).to(device=device, dtype=torch_dtype)
    encoder_torch = torch.from_numpy(encoder_np).to(device=device, dtype=torch_dtype)
    temb_torch = torch.from_numpy(temb_np).to(device=device, dtype=torch_dtype)
    freqs_torch = (freqs_cos.to(device=device), freqs_sin.to(device=device))

    with torch.no_grad(), set_forward_context(current_timestep=0, attn_metadata=None):
        start = time.perf_counter()
        torch_output = torch_block(
            hidden_torch,
            encoder_torch,
            temb_torch,
            freqs_cis=freqs_torch,
            original_seq_len=seq_len,
        )
        if device.type == "mps":
            torch.mps.synchronize()
        torch_ms = (time.perf_counter() - start) * 1000

    torch_np = torch_output.detach().cpu().float().numpy()
    mlx_np = np.array(mlx_output.astype(mx.float32))
    abs_diff = np.abs(torch_np - mlx_np)
    max_abs = float(abs_diff.max())
    mean_abs = float(abs_diff.mean())
    ok = bool(np.allclose(torch_np, mlx_np, atol=args.atol, rtol=args.rtol))

    print(f"PyTorch reference device: {device}")
    print(f"PyTorch block latency: {torch_ms:.2f} ms")
    print(f"max_abs_diff: {max_abs:.8f}")
    print(f"mean_abs_diff: {mean_abs:.8f}")
    print(f"allclose(atol={args.atol}, rtol={args.rtol}): {ok}")
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
