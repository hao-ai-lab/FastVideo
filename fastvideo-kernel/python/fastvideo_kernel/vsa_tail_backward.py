"""Opt-in VSA256 backward packing for small residual KV tails.

The bounded plan stays on device. Overflow keeps the original partial blocks;
packing never changes the selected tokens or forward softmax normalization.
"""
import functools
import inspect

import torch
import triton
import triton.language as tl

from fastvideo_kernel import block_sparse_attn_cute_fwd as adapter


@functools.lru_cache(maxsize=1)
def _check_workspace_support(backward):
    if "_workspace" not in inspect.signature(backward).parameters:
        raise RuntimeError("FASTVIDEO_VSA_PACK_TAILS requires FA4 backward workspace support")


@functools.lru_cache(maxsize=1)
def _tail_mask_mod():
    import cutlass
    import cutlass.cute as cute
    from flash_attn.cute import utils

    @cute.jit
    def mask(batch, head, q, kv, seqlen, aux):
        routes, parents, valid = aux
        return utils.scalar_to_ssa(
            valid[kv[0]] & routes[batch[0], head[0], q[0] // 256, parents[kv[0]]], cutlass.Boolean,
        )

    mask.__q_block_invariant__ = 256
    return mask


def _prepare(routes, sizes):
    lengths = sizes.clamp(0, 256)
    counts = lengths.remainder(128)
    ends = counts.cumsum(0)
    # Bounded scratch: eight residual slots per parent; overflow uses original blocks.
    capacity = triton.cdiv(max(128, sizes.numel() * 8), 128) * 128
    positions = torch.arange(capacity, device=sizes.device)
    fits = ends[-1] <= capacity
    parents = torch.searchsorted(ends, positions, right=True).clamp_max(sizes.numel() - 1)
    starts = torch.where(parents > 0, ends[(parents - 1).clamp_min(0)], 0)
    valid = (positions < ends[-1]) & fits
    offsets = lengths[parents].div(128, rounding_mode='floor') * 128 + positions - starts
    index = torch.where(valid, parents * 256 + offsets, 0)
    parents = parents.to(torch.int32)

    child_sizes = torch.stack((lengths.clamp_max(128), (lengths - 128).clamp(0, 128)), -1).flatten()
    child_routes = routes.repeat_interleave(2, -1)
    _, full = adapter._build_sparse_tensors(
        child_routes, child_sizes, q_len=routes.shape[2] * 256,
        q_block_size=256, kv_block_size=128, need_backward=True,
        need_forward=False, force_q_sparse_block_size=256,
    )
    full = full._replace(mask_block_cnt=torch.where(fits, 0, full.mask_block_cnt))
    sparse_type = adapter._load_fa4_cute()[0]

    selected = routes.index_select(-1, parents.long()) & valid
    union = selected.reshape(*routes.shape[:-1], capacity // 128, 128).any(-1).transpose(2, 3).contiguous()
    idx, cnt = adapter._map_to_index(union)
    tails = sparse_type(full_block_idx=torch.full_like(idx, -1), full_block_cnt=torch.zeros_like(cnt),
                        mask_block_idx=idx, mask_block_cnt=cnt, block_size=(256, 128))
    return full, child_sizes, (index, parents, valid, tails)


@triton.jit
def _scatter(DK, DV, TK, TV, Index, Valid, N: tl.constexpr, C: tl.constexpr,
             H: tl.constexpr, D: tl.constexpr, S: tl.constexpr):
    # BSHD contiguous inputs; each valid packed token owns one distinct destination.
    x = tl.program_id(0) * S + tl.arange(0, S)
    batch = tl.program_id(1)
    token = x // (H * D)
    channel = x % (H * D)
    present = tl.load(Valid + token, token < C, other=False)
    original = tl.load(Index + token, token < C, other=0)
    src = batch * C * H * D + x
    dst = batch * N * H * D + original * H * D + channel
    kg = tl.load(TK + src, (token < C) & present, other=0)
    vg = tl.load(TV + src, (token < C) & present, other=0)
    tl.store(DK + dst, kg, (token < C) & present)
    tl.store(DV + dst, vg, (token < C) & present)


def scatter(dk, dv, tk, tv, index, valid):
    assert all(x.is_contiguous() for x in (dk, dv, tk, tv))
    assert dk.shape == dv.shape and tk.shape == tv.shape
    assert tk.shape[0] == dk.shape[0] and tk.shape[2:] == dk.shape[2:]
    assert tk.shape[1] == index.numel() == valid.numel()
    batch, n, heads, dim = dk.shape
    capacity = index.numel()
    _scatter[(triton.cdiv(capacity * heads * dim, 512), batch)](
        dk, dv, tk, tv, index, valid, n, capacity, heads, dim, 512)


class TailTraining(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, routes, sizes):
        _, _, forward, backward = adapter._load_fa4_cute()
        _check_workspace_support(backward)
        sparse, _ = adapter._build_sparse_tensors(
            routes, sizes, q_len=q.shape[1], q_block_size=256,
            kv_block_size=256, need_backward=False,
        )
        out, lse = forward(
            q, k, v, mask_mod=adapter._build_vbs_vector_mask_mod(256),
            aux_tensors=[sizes], block_sparse_tensors=sparse, return_lse=True,
        )[:2]
        ctx.save_for_backward(q, k, v, out, lse, routes, sizes)
        ctx.set_materialize_grads(False)
        return out, lse

    @staticmethod
    def backward(ctx, dout, dlse):
        q, k, v, out, lse, routes, sizes = ctx.saved_tensors
        dout = torch.zeros_like(out) if dout is None else dout.contiguous()
        dlse = dlse.contiguous() if dlse is not None else None
        full, child_sizes, tail = _prepare(routes, sizes)
        index, parents, valid, sparse = tail
        _, _, _, backward = adapter._load_fa4_cute()
        dq, dk, dv, workspace = backward(
            q, k, v, out, dout, lse, mask_mod=adapter._build_vbs_mask_mod(128),
            aux_tensors=[child_sizes], block_sparse_tensors=full, dlse=dlse,
            _return_workspace=True,
        )
        _, packed_dk, packed_dv = backward(
            q, k.index_select(1, index), v.index_select(1, index), out, dout, lse,
            mask_mod=_tail_mask_mod(), aux_tensors=[routes, parents, valid],
            block_sparse_tensors=sparse, dlse=dlse, dq=dq, _workspace=workspace,
        )
        scatter(dk, dv, packed_dk, packed_dv, index, valid)
        return dq, dk, dv, None, None
