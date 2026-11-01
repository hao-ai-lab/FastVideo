# SPDX-License-Identifier: Apache-2.0
"""Dense FP4 attention emulator for QAD of models deployed with FlashAttention-4 NVFP4 Q/K (dense attention).

The deployed dense FP4 path (``FLASH_ATTN`` with ``nvfp4_fa4``) quantizes Q and K to NVFP4 per 16 channels and keeps
P and V in BF16. The FP4 block-sparse training kernel (``fp4_vsa_attn_qat`` with ``quantize_pv=False``) computes the
same numerics with straight-through gradients; given every key block for every query block it is dense attention.

``install_fp4_dense_attention(transformer, numerics)`` routes every ``FlashAttentionImpl`` of ``transformer`` through
that kernel (``None`` restores FlashAttention). Inputs arrive as ``[B, L, H, D]`` after the sequence-parallel
all-to-all (full sequence, local heads); they are padded to 128-token tiles, attended over the full block lists and
cut back.
"""
from __future__ import annotations

import types
from typing import Any

import torch

from fastvideo.attention.backends.fp4_vsa_qat import FP4AttentionNumerics, load_qat_vsa_modules

TILE = 128
_ORIGINAL = "_fp4_dense_original_forward"


def dense_block_lists(seq_len: int, batch: int, heads: int, reference: Any,
                      device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Kernel block lists for full (dense) attention over ``seq_len`` tokens in 128-token tiles."""
    n_tiles = -(-seq_len // TILE)
    tile_valid = torch.full((n_tiles, ), TILE, dtype=torch.int32, device=device)
    tile_valid[-1] = seq_len - (n_tiles - 1) * TILE
    tile_mask = torch.ones((batch, heads, n_tiles, n_tiles), dtype=torch.bool, device=device)
    return reference.tile128_mask_to_fp4_blocks(tile_mask, tile_valid)


class FP4DenseAttention:
    """``FlashAttentionImpl.forward`` replacement: ``(q, k, v, metadata) -> [B, L, H, D]`` through the FP4 kernel."""

    def __init__(self, numerics: FP4AttentionNumerics, modules: tuple[Any, Any] | None = None) -> None:
        self.numerics = numerics
        self._reference, self._train = modules or load_qat_vsa_modules()
        self._lists: dict[tuple[int, int, int, str], tuple[torch.Tensor, ...]] = {}

    def _block_lists(self, seq_len: int, batch: int, heads: int, device: torch.device) -> tuple[torch.Tensor, ...]:
        key = (seq_len, batch, heads, str(device))
        if key not in self._lists:
            self._lists = {key: dense_block_lists(seq_len, batch, heads, self._reference, device)}
        return self._lists[key]

    def __call__(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, metadata: Any) -> torch.Tensor:
        batch, seq_len, heads, _ = query.shape
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


def flash_attention_impls(transformer: torch.nn.Module) -> list[Any]:
    """Every ``FlashAttentionImpl`` reached through the transformer's distributed attention layers."""
    from fastvideo.attention.backends.flash_attn import FlashAttentionImpl

    impls = []
    for module in transformer.modules():
        impl = getattr(getattr(module, "distributed_attention", None), "attn_impl", None)
        if isinstance(impl, FlashAttentionImpl):
            impls.append(impl)
    return impls


def install_fp4_dense_attention(transformer: torch.nn.Module,
                                numerics: FP4AttentionNumerics | None,
                                modules: tuple[Any, Any] | None = None) -> int:
    """Route every FlashAttention layer through the dense FP4 emulator (``None`` restores FlashAttention)."""
    override = FP4DenseAttention(numerics, modules) if numerics is not None else None
    impls = flash_attention_impls(transformer)
    for impl in impls:
        if _ORIGINAL not in impl.__dict__:
            impl.__dict__[_ORIGINAL] = impl.forward
        if override is None:
            impl.forward = impl.__dict__[_ORIGINAL]
        else:
            impl.forward = types.MethodType(lambda self, q, k, v, meta, _o=override: _o(q, k, v, meta), impl)
    return len(impls)


__all__ = ["FP4DenseAttention", "dense_block_lists", "flash_attention_impls", "install_fp4_dense_attention"]
