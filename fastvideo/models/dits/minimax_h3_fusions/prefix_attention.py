# SPDX-License-Identifier: Apache-2.0
"""Opt-in dense NVFP4 attention for the H3 Omni prefix on sm_12x (approximate)."""
import torch


def dense_fp4_prefix_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor | None:
    """BHSD operands -> ``[B, H, Lq, D]`` from the in-tree sm_12x dense NVFP4 kernel, or None.

    ``k``/``v`` are the already-smoothed prefix K8/V8, so per-block Q/K smoothing stays off and
    ``single_level_p_quant=False`` matches the recipe's two-level P. Approximate vs bf16 SDPA.
    """
    if (torch.is_grad_enabled() or torch.compiler.is_compiling() or not q.is_cuda
            or q.dtype != torch.bfloat16 or k.dtype != q.dtype or v.dtype != q.dtype
            or q.shape[-1] != 128 or k.shape[-1] != 128
            or torch.cuda.get_device_capability(q.device)[0] != 12):
        return None
    from fastvideo.attention.backends.attn_qat_infer import _get_attn_qat_infer
    kernel = _get_attn_qat_infer()
    if kernel is None:
        return None
    return kernel(q, k, v, is_causal=False, per_block_mean=False, single_level_p_quant=False)
