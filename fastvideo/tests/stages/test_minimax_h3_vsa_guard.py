# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from fastvideo.pipelines.basic.minimax_h3.vsa_guard import refuse_zero_initialized_h3_vsa
from fastvideo.pipelines.lazy_module import LazyModule


class _GateAttention(nn.Module):

    def __init__(self, trained: bool, *, has_weight: bool = True) -> None:
        super().__init__()
        if has_weight:
            self.to_gate_compress = nn.Linear(4, 4, bias=False)
            with torch.no_grad():
                if trained:
                    self.to_gate_compress.weight.fill_(0.1)
                else:
                    self.to_gate_compress.weight.zero_()
        else:
            self.to_gate_compress = SimpleNamespace(weight=None)


class _Block(nn.Module):

    def __init__(self, trained: bool, *, has_weight: bool = True) -> None:
        super().__init__()
        self.attn = _GateAttention(trained, has_weight=has_weight)


class _Transformer(nn.Module):

    def __init__(self, trained: bool, *, empty: bool = False, has_weight: bool = True) -> None:
        super().__init__()
        blocks = [] if empty else [_Block(trained, has_weight=has_weight)]
        self.transformer_blocks = nn.ModuleList(blocks)


def test_refuse_zero_initialized_h3_vsa_raises() -> None:
    with pytest.raises(RuntimeError, match="to_gate_compress"):
        refuse_zero_initialized_h3_vsa(_Transformer(trained=False))


def test_refuse_zero_initialized_h3_vsa_allows_trained_gates() -> None:
    refuse_zero_initialized_h3_vsa(_Transformer(trained=True))


def test_refuse_zero_initialized_h3_vsa_skips_dense_transformer() -> None:
    refuse_zero_initialized_h3_vsa(SimpleNamespace())


def test_refuse_zero_initialized_h3_vsa_skips_empty_blocks() -> None:
    refuse_zero_initialized_h3_vsa(_Transformer(trained=False, empty=True))


def test_refuse_zero_initialized_h3_vsa_skips_missing_gate_weight() -> None:
    refuse_zero_initialized_h3_vsa(_Transformer(trained=False, has_weight=False))


def test_refuse_zero_initialized_h3_vsa_skips_unmaterialized_lazy_module() -> None:
    loads: list[int] = []

    def loader() -> nn.Module:
        loads.append(1)
        return _Transformer(trained=False)

    refuse_zero_initialized_h3_vsa(LazyModule("transformer", loader))
    assert loads == []
