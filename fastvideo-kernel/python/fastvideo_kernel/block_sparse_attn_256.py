"""VSA-256 block-sparse attention wrapper.

The default 256-block path is Triton: it expands the logical 256-block map
to the existing 64-block Triton kernel via a dense 4x4 expansion per logical
edge ("route A"), and requires no optional dependencies.

The FA4 CuTe block-sparse fastpath (intended for Blackwell sm_100+) is
*opt-in* via ``FASTVIDEO_VSA_CUTEDSL=1``. It routes to
:mod:`fastvideo_kernel.block_sparse_attn_cute_fwd`. By default this wrapper
expands logical KV256 blocks to the historical physical KV128 tiles. Set
``FASTVIDEO_VSA_FA4_BLOCK_SHAPE=128x64`` for the fine-grained FA4 schedule.
The production ``64x64`` selection pairs adjacent Q64 children into physical
Q128 tiles: their KV lists are identical because this wrapper derives them
from one original Q256 metadata row. Direct Q64 callers with arbitrary masks
continue to use the native Q64/KV64 path. The CuTe kernel
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

_LOGICAL_BLOCK_SIZE = 256
_FA4_BLOCK_SHAPES = {
    # The public name describes the original VSA mask. FA4 internally splits
    # its logical KV256 edge into two physical KV128 tiles.
    "256x256": (256, 128),
    "128x64": (128, 64),
    # This pairing is safe only at this original-Q256 wrapper boundary: all
    # four Q64 children inherit exactly the same KV list from their parent.
    # Keep arbitrary/distinct Q64 maps on block_sparse_attn_cute_fwd's native
    # Q64/KV64 path, where they are not coalesced.
    "64x64": (128, 64),
}
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


def _resolve_fa4_block_shape() -> Tuple[int, int]:
    requested = os.environ.get("FASTVIDEO_VSA_FA4_BLOCK_SHAPE", "256x256").lower()
    try:
        return _FA4_BLOCK_SHAPES[requested]
    except KeyError as exc:
        choices = ", ".join(_FA4_BLOCK_SHAPES)
        raise ValueError(
            f"unsupported FASTVIDEO_VSA_FA4_BLOCK_SHAPE={requested!r}; choose one of {choices}"
        ) from exc


def _expand_mask_and_sizes_256_for_fa4(
    logical_mask_256: torch.Tensor,
    logical_kv_sizes_256: torch.Tensor,
    q_block_size: int,
    kv_block_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Expand a logical Q256/KV256 map to one supported physical FA4 shape.

    Every child inherits its parent edge. KV valid-token counts are clamped
    into each child's window, preserving partial and empty tail blocks.
    """
    if _LOGICAL_BLOCK_SIZE % q_block_size or _LOGICAL_BLOCK_SIZE % kv_block_size:
        raise ValueError(
            f"FA4 QxKV blocks must divide {_LOGICAL_BLOCK_SIZE}, got "
            f"{q_block_size}x{kv_block_size}"
        )
    q_factor = _LOGICAL_BLOCK_SIZE // q_block_size
    kv_factor = _LOGICAL_BLOCK_SIZE // kv_block_size
    expanded_mask = logical_mask_256.repeat_interleave(q_factor, dim=2)
    expanded_mask = expanded_mask.repeat_interleave(kv_factor, dim=3)

    sizes_i32 = logical_kv_sizes_256.to(torch.int32)
    offsets = torch.arange(
        kv_factor,
        dtype=torch.int32,
        device=sizes_i32.device,
    ) * kv_block_size
    expanded_sizes = torch.clamp(
        sizes_i32[:, None] - offsets[None, :],
        min=0,
        max=kv_block_size,
    ).reshape(-1)
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

    mask_64, sizes_64 = _expand_mask_and_sizes_256_to_64(logical_mask_256, logical_kv_sizes_256)
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
        return _triton_via_route_a(q, k, v, logical_block_map_256, logical_variable_block_sizes_256)

    q_block_size, kv_block_size = _resolve_fa4_block_shape()
    physical_mask, physical_sizes = _expand_mask_and_sizes_256_for_fa4(
        logical_block_map_256,
        logical_variable_block_sizes_256,
        q_block_size,
        kv_block_size,
    )
    from .block_sparse_attn_cute_fwd import block_sparse_attn_cute_fwd
    return block_sparse_attn_cute_fwd(q, k, v, physical_mask, physical_sizes)


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

    q_block_size, kv_block_size = _resolve_fa4_block_shape()
    physical_mask, physical_sizes = _expand_mask_and_sizes_256_for_fa4(
        logical_block_map_256,
        logical_variable_block_sizes_256,
        q_block_size,
        kv_block_size,
    )
    from .block_sparse_attn_cute_fwd import block_sparse_attn_cute_fwd_bshd
    return block_sparse_attn_cute_fwd_bshd(q, k, v, physical_mask, physical_sizes)
