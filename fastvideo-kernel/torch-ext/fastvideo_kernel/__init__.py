from typing import Optional

import torch

from ._ops import ops


def sta_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    kernel_t_size: int,
    kernel_h_size: int,
    kernel_w_size: int,
    text_length: int,
    process_text: bool,
    has_text: bool,
    kernel_aspect_ratio_flag: int,
) -> torch.Tensor:
    return ops.sta_fwd(
        q,
        k,
        v,
        out,
        kernel_t_size,
        kernel_h_size,
        kernel_w_size,
        text_length,
        process_text,
        has_text,
        kernel_aspect_ratio_flag,
    )


def block_sparse_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q2k_block_sparse_index: torch.Tensor,
    q2k_block_sparse_num: torch.Tensor,
    kv_block_size: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    out, lse = ops.block_sparse_fwd(
        q,
        k,
        v,
        q2k_block_sparse_index,
        q2k_block_sparse_num,
        kv_block_size,
    )
    return out, lse


def block_sparse_bwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    l_vec: torch.Tensor,
    out_grad: torch.Tensor,
    k2q_block_sparse_index: torch.Tensor,
    k2q_block_sparse_num: torch.Tensor,
    kv_block_size: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    q_grad, k_grad, v_grad = ops.block_sparse_bwd(
        q,
        k,
        v,
        out,
        l_vec,
        out_grad,
        k2q_block_sparse_index,
        k2q_block_sparse_num,
        kv_block_size,
    )
    return q_grad, k_grad, v_grad


def rms_norm(
    input: torch.Tensor,
    eps: float,
    weight: Optional[torch.Tensor] = None,
    output: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    return ops.rms_norm_cuda(input, eps, weight, output)


def layer_norm(
    input: torch.Tensor,
    eps: float,
    weight: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
    output: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    return ops.layer_norm_cuda(input, eps, weight, bias, output)


def int8_quant(input: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    quantized, scale = ops.quant_cuda(input)
    return quantized, scale


def int8_gemm(
    a: torch.Tensor,
    a_scale: torch.Tensor,
    b: torch.Tensor,
    b_scale: torch.Tensor,
    out: torch.Tensor,
) -> torch.Tensor:
    return ops.gemm_cuda(a, a_scale, b, b_scale, out)


__all__ = [
    "sta_fwd",
    "block_sparse_fwd",
    "block_sparse_bwd",
    "rms_norm",
    "layer_norm",
    "int8_quant",
    "int8_gemm",
]
