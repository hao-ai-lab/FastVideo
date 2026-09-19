# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any

from fastvideo.pipelines.lazy_module import is_lazy_module


def refuse_zero_initialized_h3_vsa(transformer: Any) -> None:
    if is_lazy_module(transformer) and not transformer.is_materialized:
        return
    blocks = getattr(transformer, "transformer_blocks", None)
    if blocks is None:
        return
    saw_weight = False
    any_trained = False
    for block in blocks:
        attention = getattr(block, "attn", None)
        gate = getattr(attention, "to_gate_compress", None) if attention is not None else None
        if gate is None:
            continue
        weight = getattr(gate, "weight", None)
        if weight is None:
            continue
        saw_weight = True
        if bool((weight != 0).any()):
            any_trained = True
            break
    if saw_weight and not any_trained:
        raise RuntimeError("VIDEO_SPARSE_ATTN_H3 is loaded but every to_gate_compress weight is zero. "
                           "Load a VSA-trained student or a LoRA that carries gates; dense CompactH3 cannot run VSA.")
