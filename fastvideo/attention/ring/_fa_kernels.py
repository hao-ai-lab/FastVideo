# SPDX-License-Identifier: Apache-2.0
#
# Minimal FlashAttention dispatch shim for the vendored Ring Attention
# kernels in this package (see ring_flash_attn.py).
#
# Upstream yunchang (long-context-attention: https://github.com/feifeibear/
# long-context-attention) provides this dispatch via
# ``yunchang.kernels.select_flash_attn_impl`` / ``yunchang.kernels.AttnType``,
# which would pull in ``yunchang`` as a runtime dependency just to reach the
# plain FlashAttention kernel FastVideo already depends on. FastVideo does
# not want that dependency, so this module reimplements only the
# ``AttnType.FA`` path required by the initial pure-Ring integration,
# calling straight into the ``flash_attn`` package used elsewhere by
# ``fastvideo.attention.backends.flash_attn``.
#
# The other ``AttnType`` members exist only so the vendored modules that
# reference them (default kwargs, equality checks) continue to import
# cleanly; selecting one of them raises ``NotImplementedError``.

from __future__ import annotations

import enum
from typing import Any
from collections.abc import Callable

import torch
from flash_attn import flash_attn_func


class AttnType(enum.Enum):
    FA = "fa"
    FA3 = "fa3"
    TORCH = "torch"
    SPARSE_SAGE = "sparse_sage"
    NPU = "npu"
    FLASHINFER = "flashinfer"


def _fa_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    dropout_p: float = 0.0,
    softmax_scale: float | None = None,
    causal: bool = False,
    window_size: tuple[int, int] = (-1, -1),
    softcap: float = 0.0,
    alibi_slopes: torch.Tensor | None = None,
    return_softmax: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    del return_softmax  # only affects the (unused) S_dmask output below
    out, softmax_lse, _ = flash_attn_func(
        q,
        k,
        v,
        dropout_p=dropout_p,
        softmax_scale=softmax_scale,
        causal=causal,
        window_size=window_size,
        softcap=softcap,
        alibi_slopes=alibi_slopes,
        deterministic=False,
        return_attn_probs=True,
    )
    return out, softmax_lse


def _fa_backward_unsupported(*args: Any, **kwargs: Any) -> Any:
    raise NotImplementedError(
        "Ring Attention backward is not supported by FastVideo's minimal AttnType.FA dispatch shim. "
        "Training/backward is out of scope for the initial Ring Attention integration.")


def select_flash_attn_impl(
    attn_type: AttnType,
    stage: str = "fwd-only",
    attn_processor: Any | None = None,
) -> Callable[..., Any]:
    if attn_type != AttnType.FA:
        raise NotImplementedError(
            f"FastVideo's Ring Attention integration only implements AttnType.FA, got {attn_type!r}. "
            "The other variants vendored here (FA3, TORCH, SPARSE_SAGE, NPU, FLASHINFER) come from "
            "yunchang but are not wired up without the yunchang runtime dependency.")
    if attn_processor is not None:
        raise NotImplementedError("attn_processor overrides are not supported by the FastVideo AttnType.FA shim.")
    if stage == "fwd-only":
        return _fa_forward
    if stage == "bwd-only":
        return _fa_backward_unsupported
    raise ValueError(f"Unknown stage {stage!r}, expected 'fwd-only' or 'bwd-only'.")
