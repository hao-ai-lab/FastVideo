# SPDX-License-Identifier: Apache-2.0
"""Measured Ascend Wan VSA forward; BF16 inference, logical 64-token tiles.

The distributed layer owns sequence/head collectives. This backend receives
the entire sequence for its head subset. No dense or training fallback exists.
Kernels load only on the first supported NPU forward; importing this module
does not import torch_npu, Triton, or fastvideo_kernel.
"""
import math

import torch

from fastvideo.attention.backends.video_sparse_attn import (
    VideoSparseAttentionBackend,
    VideoSparseAttentionImpl,
    VideoSparseAttentionMetadata,
)


def sparse_forward(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, indices: torch.Tensor, sizes: torch.Tensor,
                   counts: torch.Tensor) -> torch.Tensor:
    from .video_sparse_attn_npu_kernels import _attn_fwd_sparse

    b, h, s, d = q.shape
    out = torch.empty_like(q)
    lse = torch.empty((b, h, s), dtype=torch.float32, device=q.device)
    _attn_fwd_sparse[(s // 64, b * h, 1)](
        q,
        k,
        v,
        d**-.5,
        indices,
        counts,
        indices.shape[-1],
        sizes,
        lse,
        out,
        *q.stride(),
        *k.stride(),
        *v.stride(),
        *out.stride(),
        b,
        h,
        s,
        k.shape[2],
        HEAD_DIM=d,
        BLOCK_M=64,
        BLOCK_N=64,
        STAGE=3,
        num_stages=1,
        num_warps=4,
    )
    return out


def tiled_forward(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, gate: torch.Tensor | None, sizes: torch.Tensor,
                  sparsity: float) -> torch.Tensor:
    """BHSD input with zero padding; fresh output/workspace for each call."""
    if q.device.type != 'npu' or q.dtype != torch.bfloat16:
        raise NotImplementedError('Ascend VSA requires NPU BF16; FP16 P/V mixed dtype is unsupported')
    if q.ndim != 4 or q.shape[0] != 1 or q.shape[-1] not in (64, 128):
        raise NotImplementedError('Ascend VSA supports batch 1, D=64/128')
    if any(x.shape != q.shape or x.dtype != q.dtype or x.device != q.device for x in (k, v)):
        raise ValueError('Q/K/V metadata must match')
    if any(x.requires_grad for x in (q, k, v) + (() if gate is None else (gate, ))):
        raise NotImplementedError('Ascend VSA backward is unsupported')
    if gate is not None and (gate.shape != q.shape or gate.dtype != q.dtype or gate.device != q.device):
        raise ValueError('Gate must match the complete Q shape, dtype and device')
    n = sizes.numel()
    if n < 1 or n > 624 or q.shape[2] != n * 64 or not 0 <= sparsity <= 1:
        raise NotImplementedError('Ascend VSA measured domain requires 1..624 logical blocks and sparsity in [0,1]')
    from .video_sparse_attn_npu_kernels import _fused_block_mean_fwd, fused_topk_mask

    q, k, v = (x.contiguous() for x in (q, k, v))
    sizes = sizes.to(torch.int32)
    qc, kc, vc = (_fused_block_mean_fwd(x, sizes, 64) for x in (q, k, v))
    scores = (qc @ kc.transpose(-1, -2)) / math.sqrt(q.shape[-1])
    coarse = scores.softmax(-1) @ vc
    keep = max(1, min(math.ceil((1 - sparsity) * n), n))
    mask = fused_topk_mask(scores, keep)
    counts = mask.sum(-1, dtype=torch.int32)
    maximum = int(counts.max().item())
    keys = torch.arange(n, dtype=torch.float32, device=q.device)
    indices = torch.where(mask, keys, float(n)).sort(-1).values[..., :maximum].clamp_max(n - 1).to(torch.int32)
    sparse = sparse_forward(q, k, v, indices, sizes, counts)
    coarse = coarse.repeat_interleave(64, -2)
    return coarse + sparse if gate is None else torch.addcmul(sparse, coarse, gate)


class NPUVideoSparseAttentionImpl(VideoSparseAttentionImpl):

    def __init__(self,
                 num_heads: int,
                 head_size: int,
                 causal: bool,
                 softmax_scale: float,
                 num_kv_heads: int | None = None,
                 prefix: str = '',
                 **kwargs) -> None:
        if causal or head_size not in (64, 128) or softmax_scale != head_size**-.5:
            raise NotImplementedError('Ascend Wan VSA supports noncausal default-scale D=64/128 only')
        if num_kv_heads not in (None, num_heads):
            raise NotImplementedError('Ascend VSA does not support grouped-query attention')
        from fastvideo.distributed.parallel_state import get_sp_world_size, model_parallel_is_initialized
        sp_size = get_sp_world_size() if model_parallel_is_initialized() else 1
        if sp_size not in (1, 4) or num_heads < 1 or num_heads % sp_size:
            raise NotImplementedError('Ascend VSA supports SP1/SP4 with positive heads divisible by the SP size')
        super().__init__(num_heads, head_size, causal, softmax_scale, num_kv_heads, prefix, **kwargs)

    def preprocess_qkv(self, qkv: torch.Tensor, attn_metadata: VideoSparseAttentionMetadata) -> torch.Tensor:
        # Reuse only shape metadata: mutable tile workspace is owned by this call.
        from .video_sparse_attn import scatter_into_tile_buf
        n = attn_metadata.variable_block_sizes.numel()
        return scatter_into_tile_buf(qkv, (qkv.shape[0], n * 64, qkv.shape[2], qkv.shape[3]),
                                     attn_metadata.non_pad_index, None, attn_metadata.tile_partition_indices)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, gate_compress: torch.Tensor | None,
                attn_metadata: VideoSparseAttentionMetadata) -> torch.Tensor:
        out = tiled_forward(query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2),
                            None if gate_compress is None else gate_compress.transpose(1, 2),
                            attn_metadata.variable_block_sizes, attn_metadata.VSA_sparsity)
        return out.transpose(1, 2)


class NPUVideoSparseAttentionBackend(VideoSparseAttentionBackend):

    @staticmethod
    def get_impl_cls() -> type[NPUVideoSparseAttentionImpl]:
        return NPUVideoSparseAttentionImpl
