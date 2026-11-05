# SPDX-License-Identifier: Apache-2.0
"""VSA-H3 fine attention (64- or 128-token tiles) with the deployed sparse-FP4 numerics, for quantization-aware training.

``FP4VSAFineAttention`` plugs into ``MiniMaxH3VSAImpl.fine_attention_override``:
VSA-H3's own selection (tile pooling, top-k mask, gated compression branch)
runs unchanged and in BF16, and only the block-sparse attention is replaced by
``fastvideo_kernel.triton_kernels.attn_qat_vsa_train.fp4_vsa_attn_qat``, which
emulates ``attn_qat_infer``'s sparse FP4 kernel on any CUDA GPU (forward) and
provides straight-through gradients (backward).

Training images may carry a prebuilt ``fastvideo_kernel`` that predates these
modules or their current API; ``load_qat_vsa_modules`` then loads both from this
checkout's ``fastvideo-kernel/python`` tree under a private package name.
"""
from __future__ import annotations

import importlib
import importlib.util
import inspect
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

import torch

_PRIVATE_PACKAGE = "_fastvideo_qat_vsa_kernels"
_KERNEL_DIR = Path(
    __file__).resolve().parents[3] / "fastvideo-kernel" / "python" / "fastvideo_kernel" / "triton_kernels"


def _load_installed(name: str) -> ModuleType | None:
    try:
        return importlib.import_module(f"fastvideo_kernel.triton_kernels.{name}")
    except ImportError:
        return None
    except KeyError:
        # An earlier failed `fastvideo_kernel` import leaves a None entry in
        # sys.modules, and the machinery then raises KeyError instead of
        # ImportError. Same meaning for us: fall back to this checkout's tree.
        return None


def _load_checkout(name: str) -> ModuleType:
    if _PRIVATE_PACKAGE not in sys.modules:
        if not (_KERNEL_DIR / f"{name}.py").is_file():
            raise ImportError(f"{name}.py is neither in the installed fastvideo_kernel nor in {_KERNEL_DIR}")
        spec = importlib.util.spec_from_loader(_PRIVATE_PACKAGE, loader=None, is_package=True)
        assert spec is not None and spec.submodule_search_locations is not None
        spec.submodule_search_locations.append(str(_KERNEL_DIR))
        sys.modules[_PRIVATE_PACKAGE] = importlib.util.module_from_spec(spec)
    return importlib.import_module(f"{_PRIVATE_PACKAGE}.{name}")


def _has_current_api(reference: ModuleType, train: ModuleType) -> bool:
    attn = getattr(train, "fp4_vsa_attn_qat", None)
    return (hasattr(reference, "tile_mask_to_fp4_blocks") and attn is not None
            and "q2k_quad" in inspect.signature(attn).parameters)


def load_qat_vsa_modules() -> tuple[ModuleType, ModuleType]:
    """``(attn_qat_vsa_reference, attn_qat_vsa_train)``; the second needs Triton.

    The installed pair is used only when both modules have the current API; otherwise both come from the checkout,
    so the reference and the kernel always match.
    """
    reference, train = _load_installed("attn_qat_vsa_reference"), _load_installed("attn_qat_vsa_train")
    if reference is not None and train is not None and _has_current_api(reference, train):
        return reference, train
    return _load_checkout("attn_qat_vsa_reference"), _load_checkout("attn_qat_vsa_train")


@dataclass(frozen=True)
class FP4AttentionNumerics:
    """Numeric mode of the deployed kernel; mirror whatever inference ships.

    ``quantize=False`` keeps the emulator's sparse structure with exact
    softmax (a BF16, grad-capable route). ``quantize_pv=False``
    quantizes Q/K only, with P and V in BF16 (FA4 NVFP4-QK + BF16 PV).
    """
    quantize: bool = True
    quantize_pv: bool = True
    two_level_p: bool = False
    smooth_k: bool = False
    first_block_max_floor: float | None = None
    # Q/K-only error-reduction studies (forward-only except ``hadamard``); see ``fp4_vsa_attn_qat``.
    hadamard: bool = False
    smooth_q: str | None = None
    prefix_precision: str | None = None


class FP4VSAFineAttention:
    """``fine_attention_override`` callable: ``(q, k, v, mask, metadata) -> [B, S, H, D]``."""

    def __init__(self, numerics: FP4AttentionNumerics) -> None:
        self.numerics = numerics
        self._reference, self._train = load_qat_vsa_modules()

    def __call__(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask: torch.Tensor,
                 metadata: Any) -> torch.Tensor:
        tile_elems = int(metadata.tile_elems)
        if tile_elems not in (64, 128):
            raise ValueError(f"the FP4 VSA emulator models 64- or 128-token tiles, got {tile_elems}")
        if tile_elems != 128 and (self.numerics.prefix_precision or self.numerics.smooth_q == "segment"):
            raise ValueError("prefix_precision and smooth_q='segment' count 128-token prefix tiles")
        q2k_idx, q2k_num, kv_valid, q2k_quad = self._reference.tile_mask_to_fp4_blocks(
            mask, tile_elems, metadata.variable_block_sizes)
        q, k, v = (t.transpose(1, 2) for t in (query, key, value))
        seq_len = q.shape[2]
        padded = (seq_len + 127) // 128 * 128
        if padded != seq_len:
            q, k, v = (torch.nn.functional.pad(t, (0, 0, 0, padded - seq_len)) for t in (q, k, v))
        num_prefix_tiles = int(getattr(metadata, "num_prefix_tiles", 0) or 0)
        out = self._train.fp4_vsa_attn_qat(q,
                                           k,
                                           v,
                                           q2k_idx,
                                           q2k_num,
                                           kv_valid,
                                           q2k_quad=q2k_quad,
                                           quantize=self.numerics.quantize,
                                           quantize_pv=self.numerics.quantize_pv,
                                           two_level_p=self.numerics.two_level_p,
                                           smooth_k=self.numerics.smooth_k,
                                           first_block_max_floor=self.numerics.first_block_max_floor,
                                           hadamard=self.numerics.hadamard,
                                           smooth_q=self.numerics.smooth_q,
                                           prefix_precision=self.numerics.prefix_precision,
                                           num_prefix_tiles=num_prefix_tiles)
        if padded != seq_len:
            out = out[:, :, :seq_len]
        return out.transpose(1, 2)


def vsa_impls(transformer: torch.nn.Module) -> list[Any]:
    """Every VSA-H3 attention implementation in a MiniMax-H3 transformer."""
    impls = []
    for module in transformer.modules():
        impl = getattr(getattr(module, "distributed_attention", None), "attn_impl", None)
        if impl is not None and hasattr(impl, "fine_attention_override"):
            impls.append(impl)
    return impls


def install_fp4_vsa_attention(transformer: torch.nn.Module, numerics: FP4AttentionNumerics | None) -> int:
    """Route every VSA-H3 layer's fine attention through the emulator (``None`` restores the kernels)."""
    override = FP4VSAFineAttention(numerics) if numerics is not None else None
    impls = vsa_impls(transformer)
    for impl in impls:
        impl.fine_attention_override = override
    return len(impls)


__all__ = ["FP4AttentionNumerics", "FP4VSAFineAttention", "install_fp4_vsa_attention", "load_qat_vsa_modules"]
