"""FA4 CuTe-DSL block-sparse attention adapter.

This module adapts VSA's ``(block_map, variable_block_sizes)`` inputs into
FA4's forward and backward ``BlockSparseTensorsTorch`` representations.
FA4's public ``flash_attn_func`` owns the forward/backward autograd bridge.

Both [B, H, S, D] (BHSD) and [B, S, H, D] (BSHD) entrypoints are provided.
The BSHD variant is preferred from VSA-128/256 callers to avoid layout
round-trips on the hot path.

The FA4 CuTe block-sparse kernel (``flash_attn.cute`` with
``block_sparsity``) is an *optional* dependency: it is imported lazily and
only exercised when the VSA-128/256 CuTe fastpath is explicitly selected
(``FASTVIDEO_VSA_CUTEDSL=1``). The default path is Triton and does not require
it. Also needs ``nvidia-cutlass-dsl`` and ``quack-kernels``.
"""

from __future__ import annotations

import functools
from typing import Tuple

import torch

_FA4_IMPORT_HINT = ("VSA-128/256 CuTe fastpath requires a FlashAttention-4 CuTe build that "
                    "provides `flash_attn.cute` with block-sparsity support (plus "
                    "`nvidia-cutlass-dsl` and `quack-kernels`). This is an optional "
                    "dependency; the default path is Triton. Install the FA4 CuTe "
                    "build and set FASTVIDEO_VSA_CUTEDSL=1 to enable the CuTe fastpath.")


@functools.lru_cache(maxsize=1)
def _load_fa4_cute():
    """Lazily import the optional FA4 CuTe block-sparse symbols.

    Raising a clear, actionable error here keeps the optional FA4 CuTe
    build from being a hard import-time dependency of this module (and of
    the default Triton VSA-256 path).
    """
    try:
        from flash_attn.cute.block_sparsity import BlockSparseTensorsTorch
        from flash_attn.cute.interface import (
            _flash_attn_bwd,
            _flash_attn_fwd,
            flash_attn_func,
        )
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(_FA4_IMPORT_HINT) from exc
    return BlockSparseTensorsTorch, flash_attn_func, _flash_attn_fwd, _flash_attn_bwd


# FA4's physical Q tile size; KV block size comes from the VSA caller.
_FA4_Q_BLOCK_SIZE = 128


class _SingleQStageLength(int):
    """Keep the real length while selecting FA4's one-stage Q128 path.

    On sm_100 FA4 derives ``q_stage`` from ``max_seqlen_q > tile_m``. Its
    kernel supports one 128-token Q stage, but the fixed-length public wrapper
    does not expose that choice. VSA-128 must select it explicitly; otherwise
    adjacent logical Q blocks are merged into a 256-token sparse block.
    """

    def __mul__(self, other):
        return type(self)(int(self) * int(other))

    def __rmul__(self, other):
        return type(self)(int(other) * int(self))

    def __gt__(self, other):
        if int(other) == _FA4_Q_BLOCK_SIZE:
            return False
        return int(self) > int(other)


def _map_to_index(block_map: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    if block_map.dim() == 3:
        block_map = block_map.unsqueeze(0)
    if block_map.dim() != 4:
        raise ValueError(f"block_map must be [B,H,Q,KV] (or [H,Q,KV]), "
                         f"got shape={tuple(block_map.shape)}")
    if block_map.dtype != torch.bool:
        block_map = block_map.to(torch.bool)
    if not block_map.is_cuda:
        raise RuntimeError("block_map must be a CUDA tensor.")
    from fastvideo_kernel.triton_kernels.index import (
        map_to_index as triton_map_to_index, )

    return triton_map_to_index(block_map)


def _choose_q_sparse_block_size(q_len: int, q_tile_size: int = _FA4_Q_BLOCK_SIZE) -> int:
    # FA4 supports a doubled Q sparsity granularity on sm_100+ when q_len > q_tile_size.
    major, _ = torch.cuda.get_device_capability()
    if major >= 10 and q_len > q_tile_size:
        return 2 * q_tile_size
    return q_tile_size


def _aggregate_q_block_map(
    block_map: torch.Tensor,
    q_sparse_block_size: int,
    q_block_size: int,
) -> torch.Tensor:
    factor = q_sparse_block_size // q_block_size
    if factor <= 0 or q_sparse_block_size % q_block_size != 0:
        raise ValueError(f"q_sparse_block_size must be a positive multiple of "
                         f"q_block_size ({q_block_size}), got {q_sparse_block_size}")
    bsz, nhead, q_blocks, kv_blocks = block_map.shape
    q_blocks_sparse = (q_blocks + factor - 1) // factor
    pad_q = q_blocks_sparse * factor - q_blocks
    if pad_q > 0:
        pad = torch.zeros(
            bsz,
            nhead,
            pad_q,
            kv_blocks,
            dtype=torch.bool,
            device=block_map.device,
        )
        block_map = torch.cat([block_map, pad], dim=2)
    block_map = block_map.view(bsz, nhead, q_blocks_sparse, factor, kv_blocks)
    return block_map.any(dim=3)


@functools.lru_cache(maxsize=4)
def _build_vbs_mask_mod(kv_block_size: int):
    """Build a CuTe mask_mod that trims per-KV-block valid tokens.

    aux_tensors[0] must be an int32 tensor of shape [kv_blocks] giving the
    valid token count in [0, kv_block_size] for each KV block.
    """
    import cutlass
    import cutlass.cute as cute
    from flash_attn.cute import utils
    from flash_attn.cute.block_sparsity import fast_sampling

    kv_block_size_const = int(kv_block_size)

    @fast_sampling
    @cute.jit
    def _vbs_mask_mod(
        batch: cute.TensorSSA,
        head: cute.TensorSSA,
        m_idx: cute.TensorSSA,
        n_idx: cute.TensorSSA,
        seqlen_info,
        aux_tensors,
    ) -> cute.TensorSSA:
        del batch, head, m_idx, seqlen_info
        block_size_ssa = utils.scalar_to_ssa(kv_block_size_const, cutlass.Int32)
        zero_ssa = utils.scalar_to_ssa(0, cutlass.Int32)
        kv_blk = n_idx // block_size_ssa
        kv_off = n_idx % block_size_ssa
        kv_sizes = aux_tensors[0]
        valid = utils.scalar_to_ssa(kv_sizes[kv_blk[0]], cutlass.Int32)
        return (valid > zero_ssa) & (kv_off < valid)

    return _vbs_mask_mod


def _build_sparse_tensors(
    block_map: torch.Tensor,
    variable_block_sizes: torch.Tensor,
    *,
    q_len: int,
    q_block_size: int,
    kv_block_size: int,
    need_backward: bool,
    force_q_sparse_block_size: int | None = None,
) -> Tuple[object, object | None]:
    """Build the Q-owned forward and KV-owned backward sparse metadata.

    ``need_backward`` is False on inference-only calls: the backward metadata
    is a pair of dense ``[B, H, kv_blocks, q_blocks]`` int32 index tensors that
    FA4 keeps alive on its autograd ctx until backward runs, so building it
    when nothing requires grad is pure overhead (~80 MiB per call at Wan-14B
    720p shape).
    """
    BlockSparseTensorsTorch, _, _, _ = _load_fa4_cute()
    if force_q_sparse_block_size is None:
        q_sparse_candidate = _choose_q_sparse_block_size(q_len)
        q_sparse_block_size = max(
            q_block_size,
            ((q_sparse_candidate + q_block_size - 1) // q_block_size) * q_block_size,
        )
    else:
        q_sparse_block_size = force_q_sparse_block_size
        if q_sparse_block_size < q_block_size or q_sparse_block_size % q_block_size != 0:
            raise ValueError("force_q_sparse_block_size must be a positive multiple of q_block_size")
    sparse_map = _aggregate_q_block_map(
        block_map,
        q_sparse_block_size=q_sparse_block_size,
        q_block_size=q_block_size,
    )
    kv_full = (variable_block_sizes == kv_block_size).view(1, 1, 1, -1)
    kv_partial = ((variable_block_sizes > 0) & (variable_block_sizes < kv_block_size)).view(1, 1, 1, -1)

    def from_maps(full_map: torch.Tensor, mask_map: torch.Tensor) -> object:
        full_block_idx, full_block_cnt = _map_to_index(full_map.contiguous())
        mask_block_idx, mask_block_cnt = _map_to_index(mask_map.contiguous())
        return BlockSparseTensorsTorch(
            full_block_cnt=full_block_cnt.to(torch.int32).contiguous(),
            full_block_idx=full_block_idx.to(torch.int32).contiguous(),
            mask_block_cnt=mask_block_cnt.to(torch.int32).contiguous(),
            mask_block_idx=mask_block_idx.to(torch.int32).contiguous(),
            block_size=(q_sparse_block_size, kv_block_size),
        )

    forward_sparse_tensors = from_maps(
        sparse_map & kv_full,
        sparse_map & kv_partial,
    )

    if not need_backward:
        return forward_sparse_tensors, None

    # FA4 backward is KV-owned: for each physical KV tile, list the sparse
    # query tiles that selected it. Full and partial KV tiles stay separate
    # so the token-level validity mask only runs for padded tiles.
    backward_sparse_tensors = from_maps(
        (sparse_map & kv_full).transpose(2, 3),
        (sparse_map & kv_partial).transpose(2, 3),
    )
    return forward_sparse_tensors, backward_sparse_tensors


def _cute_attention_q128_forward(
    q_bshd: torch.Tensor,
    k_bshd: torch.Tensor,
    v_bshd: torch.Tensor,
    block_map: torch.Tensor,
    variable_block_sizes: torch.Tensor,
    *,
    need_backward: bool,
) -> Tuple[torch.Tensor, torch.Tensor, object | None]:
    """Run FA4 with one physical Q stage per logical VSA-128 block."""
    _, _, flash_attn_fwd, _ = _load_fa4_cute()
    forward_sparse_tensors, backward_sparse_tensors = _build_sparse_tensors(
        block_map,
        variable_block_sizes,
        q_len=q_bshd.shape[1],
        q_block_size=_FA4_Q_BLOCK_SIZE,
        kv_block_size=_FA4_Q_BLOCK_SIZE,
        need_backward=need_backward,
        force_q_sparse_block_size=_FA4_Q_BLOCK_SIZE,
    )
    out, lse = flash_attn_fwd(
        q_bshd,
        k_bshd,
        v_bshd,
        tile_mn=(_FA4_Q_BLOCK_SIZE, _FA4_Q_BLOCK_SIZE),
        max_seqlen_q=_SingleQStageLength(q_bshd.shape[1]),
        mask_mod=_build_vbs_mask_mod(_FA4_Q_BLOCK_SIZE),
        block_sparse_tensors=forward_sparse_tensors,
        aux_tensors=[variable_block_sizes],
        causal=False,
        return_lse=True,
    )[:2]
    return out, lse, backward_sparse_tensors


class _CuteAttentionQ128(torch.autograd.Function):

    @staticmethod
    def forward(ctx, q_bshd, k_bshd, v_bshd, block_map, variable_block_sizes):
        out, lse, backward_sparse_tensors = _cute_attention_q128_forward(
            q_bshd,
            k_bshd,
            v_bshd,
            block_map,
            variable_block_sizes,
            need_backward=True,
        )
        ctx.save_for_backward(q_bshd, k_bshd, v_bshd, out, lse, variable_block_sizes)
        ctx.backward_sparse_tensors = backward_sparse_tensors
        ctx.mark_non_differentiable(lse)
        ctx.set_materialize_grads(False)
        return out, lse

    @staticmethod
    def backward(ctx, grad_out, grad_lse):
        del grad_lse
        q_bshd, k_bshd, v_bshd, out, lse, variable_block_sizes = ctx.saved_tensors
        if grad_out is None:
            grad_out = torch.zeros_like(out)
        _, _, _, flash_attn_bwd = _load_fa4_cute()
        dq, dk, dv = flash_attn_bwd(
            q_bshd,
            k_bshd,
            v_bshd,
            out,
            grad_out.contiguous(),
            lse,
            softmax_scale=q_bshd.shape[-1]**-0.5,
            mask_mod=_build_vbs_mask_mod(_FA4_Q_BLOCK_SIZE),
            aux_tensors=[variable_block_sizes],
            block_sparse_tensors=ctx.backward_sparse_tensors,
        )
        return dq, dk, dv, None, None


def _cute_attention_q128(
    q_bshd: torch.Tensor,
    k_bshd: torch.Tensor,
    v_bshd: torch.Tensor,
    block_map: torch.Tensor,
    variable_block_sizes: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    need_backward = torch.is_grad_enabled() and any(t.requires_grad for t in (q_bshd, k_bshd, v_bshd))
    if need_backward:
        return _CuteAttentionQ128.apply(q_bshd, k_bshd, v_bshd, block_map, variable_block_sizes)
    out, lse, _ = _cute_attention_q128_forward(
        q_bshd,
        k_bshd,
        v_bshd,
        block_map,
        variable_block_sizes,
        need_backward=False,
    )
    return out, lse


def _cute_attention(
    q_bshd: torch.Tensor,
    k_bshd: torch.Tensor,
    v_bshd: torch.Tensor,
    block_map: torch.Tensor,
    variable_block_sizes: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Run FA4's autograd-enabled block-sparse attention with BSHD inputs."""
    _, flash_attn_func, _, _ = _load_fa4_cute()
    q_block_size = q_bshd.shape[1] // block_map.shape[2]
    kv_block_size = k_bshd.shape[1] // block_map.shape[3]
    if q_block_size == kv_block_size == _FA4_Q_BLOCK_SIZE:
        return _cute_attention_q128(q_bshd, k_bshd, v_bshd, block_map, variable_block_sizes)
    need_backward = torch.is_grad_enabled() and any(t.requires_grad for t in (q_bshd, k_bshd, v_bshd))
    forward_sparse_tensors, backward_sparse_tensors = _build_sparse_tensors(
        block_map,
        variable_block_sizes,
        q_len=q_bshd.shape[1],
        q_block_size=q_block_size,
        kv_block_size=kv_block_size,
        need_backward=need_backward,
    )
    return flash_attn_func(
        q_bshd,
        k_bshd,
        v_bshd,
        mask_mod=_build_vbs_mask_mod(kv_block_size),
        aux_tensors=[variable_block_sizes],
        block_sparse_tensors=forward_sparse_tensors,
        block_sparse_tensors_bwd=backward_sparse_tensors,
        return_lse=True,
    )


def block_sparse_attn_cute_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    block_map: torch.Tensor,
    variable_block_sizes: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Autograd-enabled CuTe block-sparse attention for [B, H, S, D]."""
    if block_map.dim() == 3:
        block_map = block_map.unsqueeze(0)

    q_bshd = q.transpose(1, 2).contiguous()
    k_bshd = k.transpose(1, 2).contiguous()
    v_bshd = v.transpose(1, 2).contiguous()
    out_bshd, lse = _cute_attention(
        q_bshd,
        k_bshd,
        v_bshd,
        block_map,
        variable_block_sizes,
    )
    out = out_bshd.transpose(1, 2).contiguous()
    # FA4 already returns lse as [B, H, S], matching the Triton path's aux
    # contract, so it needs no transpose. Detach before any further op: the
    # value is informational and callers never backprop through it.
    return out, lse.detach()


def block_sparse_attn_cute_fwd_bshd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    block_map: torch.Tensor,
    variable_block_sizes: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Autograd-enabled CuTe block-sparse attention for [B, S, H, D]."""
    if block_map.dim() == 3:
        block_map = block_map.unsqueeze(0)

    out, lse = _cute_attention(
        q,
        k,
        v,
        block_map,
        variable_block_sizes,
    )
    # lse is [B, H, S] regardless of the q/k/v layout; see above.
    return out, lse.detach()
