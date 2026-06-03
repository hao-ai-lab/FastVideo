"""VSA-256 block-sparse attention wrapper.

The default 256-block path is Triton: it expands the logical 256-block map
to the existing 64-block Triton kernel via a dense 4x4 expansion per logical
edge ("route A"), and requires no optional dependencies.

The FA4 CuTe block-sparse fastpath (intended for Blackwell sm_100+) is
*opt-in* via ``FASTVIDEO_VSA_CUTEDSL=1``. It routes to
:mod:`fastvideo_kernel.block_sparse_attn_cute_fwd`, which natively operates
on 128-token KV blocks (this wrapper expands the logical 256-block map /
sizes into that physical 128-block representation). The CuTe kernel
(``flash_attn.cute`` with block-sparsity) is an optional dependency,
imported lazily only when this fastpath is selected.

``FASTVIDEO_VSA_TRITON=1`` (or the legacy
``FASTVIDEO_KERNEL_VSA_FORCE_TRITON=1``) forces Triton explicitly.
"""

from __future__ import annotations

import os
from typing import Tuple

import torch

from .block_sparse_attn import block_sparse_attn_triton, _force_triton

# NOTE: ``block_sparse_attn_cute_fwd`` is imported lazily inside the CuTe
# branches below. Importing it at module load would pull in the optional
# FA4 CuTe build (``flash_attn.cute``) and make it a hard dependency of the
# default Triton path.


_KV_BLOCK_PHYS = 128  # FA4 CuTe BSA forward uses 128-token KV blocks.
_KV_BLOCK_TRITON = 64  # Existing Triton path uses 64-token KV blocks.


def _resolve_backend() -> str:
    """Pick the backend for the 256-block VSA path.

    Default is Triton (no optional deps). The FA4 CuTe fastpath is opt-in
    via ``FASTVIDEO_VSA_CUTEDSL=1`` and requires the optional FA4 CuTe
    build. ``FASTVIDEO_VSA_TRITON=1`` / the legacy force-triton flag force
    Triton explicitly and take precedence over the CuTe opt-in.
    """
    if _force_triton():
        return "triton"
    if os.environ.get("FASTVIDEO_VSA_CUTEDSL", "0") == "1":
        return "cutedsl"
    return "triton"


def _expand_mask_and_sizes_256_to_128(
    logical_mask_256: torch.Tensor,
    logical_kv_sizes_256: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Expand a [B, H, Qb256, KVb256] map to [B, H, Qb256, KVb128].

    Each logical 256-token KV block splits into two physical 128-token
    children. Each child inherits the logical mask edge; its valid-token
    count is the logical count clamped into the child's window.
    """
    expanded_mask = logical_mask_256.repeat_interleave(2, dim=3)

    sizes_i32 = logical_kv_sizes_256.to(torch.int32)
    child0 = torch.clamp(sizes_i32, min=0, max=_KV_BLOCK_PHYS)
    child1 = torch.clamp(sizes_i32 - _KV_BLOCK_PHYS, min=0, max=_KV_BLOCK_PHYS)
    expanded_sizes = torch.empty(
        (sizes_i32.numel() * 2,),
        dtype=torch.int32,
        device=sizes_i32.device,
    )
    expanded_sizes[0::2] = child0
    expanded_sizes[1::2] = child1
    return expanded_mask, expanded_sizes


def _expand_mask_and_sizes_256_to_64(
    logical_mask_256: torch.Tensor,
    logical_kv_sizes_256: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Expand a [B, H, Qb256, KVb256] map to [B, H, Qb64, KVb64] (route A).

    Each logical 256-token tile splits 4-ways along both Q and KV. Sizes
    are computed by chopping the logical count into 64-token strides.
    """
    expanded_mask = logical_mask_256.repeat_interleave(4, dim=2).repeat_interleave(4, dim=3)
    sizes_i32 = logical_kv_sizes_256.to(torch.int32)
    offsets = torch.tensor(
        [0, _KV_BLOCK_TRITON, 2 * _KV_BLOCK_TRITON, 3 * _KV_BLOCK_TRITON],
        dtype=torch.int32,
        device=sizes_i32.device,
    )
    expanded_sizes = torch.clamp(
        sizes_i32[:, None] - offsets[None, :],
        min=0,
        max=_KV_BLOCK_TRITON,
    ).reshape(-1)
    return expanded_mask, expanded_sizes


def _triton_via_route_a(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    logical_mask_256: torch.Tensor,
    logical_kv_sizes_256: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    from .triton_kernels.index import map_to_index as triton_map_to_index

    mask_64, sizes_64 = _expand_mask_and_sizes_256_to_64(
        logical_mask_256, logical_kv_sizes_256
    )
    q2k_idx, q2k_num = triton_map_to_index(mask_64.to(torch.bool))
    return block_sparse_attn_triton(q, k, v, q2k_idx, q2k_num, sizes_64)


def block_sparse_attn_256(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    logical_block_map_256: torch.Tensor,
    logical_variable_block_sizes_256: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """VSA-256 sparse-branch entrypoint for [B, H, S, D] inputs."""
    if logical_block_map_256.dim() == 3:
        logical_block_map_256 = logical_block_map_256.unsqueeze(0)

    if _resolve_backend() == "triton":
        return _triton_via_route_a(
            q, k, v, logical_block_map_256, logical_variable_block_sizes_256
        )

    mask_128, sizes_128 = _expand_mask_and_sizes_256_to_128(
        logical_block_map_256, logical_variable_block_sizes_256
    )
    from .block_sparse_attn_cute_fwd import block_sparse_attn_cute_fwd
    return block_sparse_attn_cute_fwd(q, k, v, mask_128, sizes_128)


def block_sparse_attn_256_bshd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    logical_block_map_256: torch.Tensor,
    logical_variable_block_sizes_256: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """VSA-256 sparse-branch entrypoint for [B, S, H, D] inputs.

    Default CuTe path consumes BSHD directly; Triton fallback transposes
    to BHSD as the legacy path expects.
    """
    if logical_block_map_256.dim() == 3:
        logical_block_map_256 = logical_block_map_256.unsqueeze(0)

    if _resolve_backend() == "triton":
        out_bhsd, aux = _triton_via_route_a(
            q.transpose(1, 2).contiguous(),
            k.transpose(1, 2).contiguous(),
            v.transpose(1, 2).contiguous(),
            logical_block_map_256,
            logical_variable_block_sizes_256,
        )
        return out_bhsd.transpose(1, 2).contiguous(), aux

    mask_128, sizes_128 = _expand_mask_and_sizes_256_to_128(
        logical_block_map_256, logical_variable_block_sizes_256
    )
    from .block_sparse_attn_cute_fwd import block_sparse_attn_cute_fwd_bshd
    return block_sparse_attn_cute_fwd_bshd(q, k, v, mask_128, sizes_128)
