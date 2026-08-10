"""Check a single Wan T2V transformer block against the experimental MLX port.

The parity target is intentionally tiny and dense:

- one non-VSA Wan T2V transformer block,
- no rotary embeddings,
- no sequence parallelism,
- CPU PyTorch reference vs MLX GPU implementation,
- float32 inputs/weights for a stable first correctness check.

This is the first real bridge from "MLX primitive benchmark" to "Wan block
runtime." Once this is stable, the next layer is rotary embedding parity and
then loading one real FastWan block's weights.
"""

from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="Tiny Wan block parity check for the experimental MLX runtime")
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--ffn-dim", type=int, default=128)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=8)
    parser.add_argument("--text-len", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--atol", type=float, default=2e-4)
    parser.add_argument("--rtol", type=float, default=2e-4)
    args = parser.parse_args()

    import mlx.core as mx
    import numpy as np
    import torch

    import fastvideo.attention.layer as attention_layer
    from fastvideo.forward_context import set_forward_context
    from fastvideo.mlx_runtime.fastwan import MLXWanTransformerBlock, mlx_block_weights_from_torch
    from fastvideo.models.dits.wanvideo import WanTransformerBlock
    from fastvideo.platforms import AttentionBackendEnum

    # The block's PyTorch attention wrapper expects sequence-parallel runtime
    # globals, even for world_size=1. For this tiny standalone parity probe,
    # patch them to the single-process identities used mathematically.
    attention_layer.get_sp_parallel_rank = lambda: 0
    attention_layer.get_sp_world_size = lambda: 1
    attention_layer.sequence_model_parallel_all_to_all_4D = lambda x, scatter_dim, gather_dim: x
    attention_layer.sequence_model_parallel_all_gather = lambda x, dim=0: x

    torch.manual_seed(args.seed)
    torch.set_default_dtype(torch.float32)

    torch_block = WanTransformerBlock(
        dim=args.dim,
        ffn_dim=args.ffn_dim,
        num_heads=args.num_heads,
        cross_attn_norm=True,
        supported_attention_backends=(AttentionBackendEnum.TORCH_SDPA,),
    ).eval()
    for name, param in torch_block.named_parameters():
        if name.endswith("weight") and param.ndim > 1:
            torch.nn.init.normal_(param, mean=0.0, std=0.02)
        elif name.endswith("weight"):
            torch.nn.init.ones_(param)
        elif name.endswith("bias"):
            torch.nn.init.zeros_(param)

    hidden_states = torch.randn(args.batch_size, args.seq_len, args.dim)
    encoder_hidden_states = torch.randn(args.batch_size, args.text_len, args.dim)
    temb = torch.randn(args.batch_size, 6, args.dim)

    with torch.no_grad(), set_forward_context(current_timestep=0, attn_metadata=None):
        torch_output = torch_block(
            hidden_states,
            encoder_hidden_states,
            temb,
            freqs_cis=None,
            original_seq_len=args.seq_len,
        )

    mlx_weights = mlx_block_weights_from_torch(torch_block)
    mlx_block = MLXWanTransformerBlock(
        mlx_weights,
        dim=args.dim,
        ffn_dim=args.ffn_dim,
        num_heads=args.num_heads,
    )
    mlx_output = mlx_block(
        mx.array(hidden_states.numpy()),
        mx.array(encoder_hidden_states.numpy()),
        mx.array(temb.numpy()),
    )
    mx.eval(mlx_output)

    torch_np = torch_output.detach().cpu().numpy()
    mlx_np = np.array(mlx_output)
    abs_diff = np.abs(torch_np - mlx_np)
    max_abs = float(abs_diff.max())
    mean_abs = float(abs_diff.mean())
    ok = bool(np.allclose(torch_np, mlx_np, atol=args.atol, rtol=args.rtol))

    print(f"shape: batch={args.batch_size}, seq={args.seq_len}, text={args.text_len}, dim={args.dim}")
    print(f"max_abs_diff: {max_abs:.8f}")
    print(f"mean_abs_diff: {mean_abs:.8f}")
    print(f"allclose(atol={args.atol}, rtol={args.rtol}): {ok}")
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
