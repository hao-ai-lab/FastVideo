# SPDX-License-Identifier: Apache-2.0
"""Dense FP4 attention emulator for QAD of models deployed with FlashAttention-4 NVFP4 Q/K (dense attention).

The deployed dense FP4 path (``FLASH_ATTN`` with ``nvfp4_fa4``) quantizes Q and K to NVFP4 per 16 channels and keeps
P and V in BF16. The FP4 block-sparse training kernel (``fp4_vsa_attn_qat`` with ``quantize_pv=False``) computes the
same numerics with straight-through gradients; given every key block for every query block it is dense attention.

``install_fp4_dense_attention(transformer, numerics)`` routes every dense attention impl of ``transformer``
(``FlashAttentionImpl``, or ``SDPAImpl`` where the ``FLASH_ATTN`` selection fell back to SDPA because flash-attn is
not installed) through that kernel (``None`` restores the original forward). Inputs arrive as ``[B, L, H, D]`` after
the sequence-parallel all-to-all (full sequence, local heads); they are padded to 128-token tiles, attended over the
full block lists and cut back.
"""
from __future__ import annotations

import re
import types
from typing import Any

import torch

from fastvideo.attention.backends.fp4_vsa_qat import FP4AttentionNumerics, load_qat_vsa_modules
from fastvideo.attention.utils.nvfp4_qk_precondition import precondition_qk

TILE = 128
_ORIGINAL = "_fp4_dense_original_forward"
_BLOCK = re.compile(r"(?:^|\.)transformer_blocks\.(\d+)\.")


def dense_block_lists(seq_len: int, batch: int, heads: int, reference: Any,
                      device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Kernel block lists for full (dense) attention over ``seq_len`` tokens in 128-token tiles."""
    n_tiles = -(-seq_len // TILE)
    tile_valid = torch.full((n_tiles, ), TILE, dtype=torch.int32, device=device)
    tile_valid[-1] = seq_len - (n_tiles - 1) * TILE
    tile_mask = torch.ones((batch, heads, n_tiles, n_tiles), dtype=torch.bool, device=device)
    return reference.tile128_mask_to_fp4_blocks(tile_mask, tile_valid)


class FP4DenseAttention:
    """Dense attention ``forward`` replacement: ``(q, k, v, metadata) -> [B, L, H, D]`` through the FP4 kernel."""

    def __init__(self,
                 numerics: FP4AttentionNumerics,
                 modules: tuple[Any, Any] | None = None,
                 qk_transform: tuple[bool, bool] = (False, False)) -> None:
        self.numerics = numerics
        # (smooth K, Hadamard-rotate Q/K) applied before quantization, exactly as the deployed FA4 NVFP4 path does.
        self.smooth_k_mean, self.hadamard = qk_transform
        self._reference, self._train = modules or load_qat_vsa_modules()
        self._lists: dict[tuple[int, int, int, str], tuple[torch.Tensor, ...]] = {}

    def _block_lists(self, seq_len: int, batch: int, heads: int, device: torch.device) -> tuple[torch.Tensor, ...]:
        key = (seq_len, batch, heads, str(device))
        if key not in self._lists:
            self._lists = {key: dense_block_lists(seq_len, batch, heads, self._reference, device)}
        return self._lists[key]

    def __call__(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, metadata: Any) -> torch.Tensor:
        if getattr(metadata, "packed_sequence", None) is not None:
            raise NotImplementedError("dense FP4 attention emulator does not support packed sequences")
        batch, seq_len, heads, _ = query.shape
        query, key = precondition_qk(query, key, smooth_k=self.smooth_k_mean, hadamard=self.hadamard)
        pad = -seq_len % TILE
        q, k, v = (torch.nn.functional.pad(t.to(torch.bfloat16).transpose(1, 2), (0, 0, 0, pad))
                   for t in (query, key, value))
        q2k_idx, q2k_num, kv_valid = self._block_lists(seq_len, batch, heads, query.device)
        out = self._train.fp4_vsa_attn_qat(q,
                                           k,
                                           v,
                                           q2k_idx,
                                           q2k_num,
                                           kv_valid,
                                           quantize=self.numerics.quantize,
                                           quantize_pv=self.numerics.quantize_pv,
                                           two_level_p=self.numerics.two_level_p,
                                           smooth_k=self.numerics.smooth_k,
                                           first_block_max_floor=self.numerics.first_block_max_floor)
        return out[:, :, :seq_len].transpose(1, 2).to(query.dtype)


def _dense_impl_types() -> tuple[type, ...]:
    from fastvideo.attention.backends.sdpa import SDPAImpl

    try:
        from fastvideo.attention.backends.flash_attn import FlashAttentionImpl
    except (ImportError, RuntimeError):  # no flash-attn install: FLASH_ATTN selection resolves to SDPA
        return (SDPAImpl, )
    return (FlashAttentionImpl, SDPAImpl)


def named_dense_attention_impls(transformer: torch.nn.Module) -> list[tuple[str, Any]]:
    """``(module name, impl)`` for every dense (FlashAttention or SDPA) impl behind a distributed attention layer."""
    types_ = _dense_impl_types()
    impls = []
    for name, module in transformer.named_modules():
        impl = getattr(getattr(module, "distributed_attention", None), "attn_impl", None)
        if isinstance(impl, types_):
            impls.append((name, impl))
    return impls


def dense_attention_impls(transformer: torch.nn.Module) -> list[Any]:
    """Every dense (FlashAttention or SDPA) impl reached through the transformer's distributed attention layers."""
    return [impl for _, impl in named_dense_attention_impls(transformer)]


def block_index(module_name: str) -> int | None:
    """``transformer_blocks.<i>`` index of a module name (``None`` outside the block stack)."""
    match = _BLOCK.search(module_name + ".")
    return int(match.group(1)) if match else None


def install_fp4_dense_attention(
    transformer: torch.nn.Module,
    numerics: FP4AttentionNumerics | None,
    modules: tuple[Any, Any] | None = None,
    *,
    qk_transform: tuple[bool, bool] = (False, False),
    bf16_blocks: tuple[int, ...] = ()) -> int:
    """Route dense attention layers through the FP4 emulator; returns how many now run FP4.

    ``numerics=None`` restores every original forward. ``bf16_blocks`` keeps the attention of those
    ``transformer_blocks`` indices on its original (BF16) forward, matching a deployment that opts them out of NVFP4.
    """
    override = FP4DenseAttention(numerics, modules, qk_transform) if numerics is not None else None
    named = named_dense_attention_impls(transformer)
    skip = frozenset(int(b) for b in bf16_blocks)
    present = {index for name, _ in named if (index := block_index(name)) is not None}
    missing = skip - present
    if missing:
        raise ValueError(f"bf16_blocks {sorted(missing)} match no transformer_blocks attention layer")
    installed = 0
    for name, impl in named:
        if _ORIGINAL not in impl.__dict__:
            impl.__dict__[_ORIGINAL] = impl.forward
        if override is None or block_index(name) in skip:
            impl.forward = impl.__dict__[_ORIGINAL]
        else:
            impl.forward = types.MethodType(lambda self, q, k, v, meta, _o=override: _o(q, k, v, meta), impl)
            installed += 1
    return installed if override is not None else len(named)


__all__ = [
    "FP4DenseAttention", "block_index", "dense_block_lists", "dense_attention_impls", "install_fp4_dense_attention",
    "named_dense_attention_impls"
]
