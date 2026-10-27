# SPDX-License-Identifier: Apache-2.0
"""Routing guards for MiniMax-H3's packed-varlen FA4 inference path."""

from __future__ import annotations

import sys
from types import ModuleType

import pytest
import torch


flash_attn_module = pytest.importorskip(
    "fastvideo.attention.backends.flash_attn",
    reason="no usable flash-attention package installed",
    exc_type=ImportError,
)
FlashAttentionImpl = flash_attn_module.FlashAttentionImpl


def _build_impl(*, fa4_packed_varlen: bool) -> FlashAttentionImpl:
    return FlashAttentionImpl(
        num_heads=2,
        head_size=8,
        causal=False,
        softmax_scale=0.125,
        num_kv_heads=2,
        fa4_packed_varlen=fa4_packed_varlen,
    )


def _qkv(batch: int = 1) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    query = torch.randn(batch, 7, 2, 8)
    return query, torch.randn_like(query), torch.randn_like(query)


def test_h3_fa4_no_grad_uses_single_sequence_varlen(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []

    def varlen(query, key, value, cu_q, cu_k, max_q, max_k, **kwargs):
        calls.append((query.shape, key.shape, value.shape, cu_q.tolist(), cu_k.tolist(), max_q, max_k, kwargs))
        return value + 1

    def unexpected_fixed(*args, **kwargs):
        raise AssertionError("H3 FA4 inference took the fixed-length path")

    monkeypatch.setattr(flash_attn_module, "fa_version", "4")
    monkeypatch.setattr(flash_attn_module, "flash_attn_varlen_func_compilable", varlen)
    monkeypatch.setattr(flash_attn_module, "flash_attn_func_compilable", unexpected_fixed)
    query, key, value = _qkv()

    with torch.no_grad():
        output = _build_impl(fa4_packed_varlen=True)._forward_impl(query, key, value, None)

    torch.testing.assert_close(output, value + 1)
    assert calls == [
        (
            torch.Size([7, 2, 8]),
            torch.Size([7, 2, 8]),
            torch.Size([7, 2, 8]),
            [0, 7],
            [0, 7],
            7,
            7,
            {"dropout_p": 0.0, "softmax_scale": 0.125, "causal": False},
        )
    ]


@pytest.mark.parametrize(
    ("fa4_packed_varlen", "version", "batch", "requires_grad"),
    [
        (False, "4", 1, False),
        (True, "3", 1, False),
        (True, "4", 2, False),
        (True, "4", 1, True),
    ],
)
def test_h3_varlen_guard_preserves_fixed_path(
    monkeypatch: pytest.MonkeyPatch,
    fa4_packed_varlen: bool,
    version: str,
    batch: int,
    requires_grad: bool,
) -> None:
    def unexpected_varlen(*args, **kwargs):
        raise AssertionError("packed-varlen route escaped its H3 FA4 inference guard")

    def fixed(query, key, value, **kwargs):
        del key, value, kwargs
        return query + 2

    monkeypatch.setattr(flash_attn_module, "fa_version", version)
    monkeypatch.setattr(flash_attn_module, "flash_attn_varlen_func_compilable", unexpected_varlen)
    monkeypatch.setattr(flash_attn_module, "flash_attn_func_compilable", fixed)
    query, key, value = _qkv(batch)
    query.requires_grad_(requires_grad)

    context = torch.enable_grad() if requires_grad else torch.no_grad()
    with context:
        output = _build_impl(fa4_packed_varlen=fa4_packed_varlen)._forward_impl(query, key, value, None)

    torch.testing.assert_close(output, query + 2)
    if requires_grad:
        output.sum().backward()
        torch.testing.assert_close(query.grad, torch.ones_like(query))


def test_h3_varlen_guard_prioritizes_masked_path(monkeypatch: pytest.MonkeyPatch) -> None:
    def unexpected_varlen(*args, **kwargs):
        raise AssertionError("packed-varlen route bypassed attention metadata")

    def masked(qkv, key_padding_mask, **kwargs):
        del key_padding_mask, kwargs
        return qkv[:, :, 2]

    monkeypatch.setattr(flash_attn_module, "fa_version", "4")
    monkeypatch.setattr(flash_attn_module, "flash_attn_varlen_func_compilable", unexpected_varlen)
    # Keep this routing test independent of whether an FA4-only environment
    # also installed FA2's top-level packed-QKV helper. The production branch
    # imports these names lazily only after observing attention metadata.
    no_pad_module = ModuleType("fastvideo.attention.utils.flash_attn_no_pad")
    no_pad_module.flash_attn_no_pad_compilable = masked
    no_pad_module.flash_attn_varlen_qk_no_pad_compilable = unexpected_varlen
    monkeypatch.setitem(sys.modules, no_pad_module.__name__, no_pad_module)
    query, key, value = _qkv()
    metadata = flash_attn_module.FlashAttnMetadata(
        current_timestep=0,
        attn_mask=torch.ones((1, query.shape[1]), dtype=torch.bool),
    )

    with torch.no_grad():
        output = _build_impl(fa4_packed_varlen=True)._forward_impl(query, key, value, metadata)

    torch.testing.assert_close(output, value)


def test_h3_varlen_guard_prioritizes_nvfp4_path(monkeypatch: pytest.MonkeyPatch) -> None:
    def unexpected_varlen(*args, **kwargs):
        raise AssertionError("packed-varlen route bypassed NVFP4")

    monkeypatch.setattr(flash_attn_module, "fa_version", "4")
    monkeypatch.setattr(flash_attn_module, "flash_attn_varlen_func_compilable", unexpected_varlen)
    impl = _build_impl(fa4_packed_varlen=True)
    impl.nvfp4_fa4 = True
    monkeypatch.setattr(impl, "_forward_nvfp4", lambda query, key, value: value + 3)
    query, key, value = _qkv()

    with torch.no_grad():
        output = impl._forward_impl(query, key, value, None)

    torch.testing.assert_close(output, value + 3)


def test_h3_varlen_guard_preserves_unequal_qk_fixed_path(monkeypatch: pytest.MonkeyPatch) -> None:
    def unexpected_varlen(*args, **kwargs):
        raise AssertionError("packed-varlen route accepted unequal Q/K/V lengths")

    def fixed(query, key, value, **kwargs):
        del key, value, kwargs
        return query + 4

    monkeypatch.setattr(flash_attn_module, "fa_version", "4")
    monkeypatch.setattr(flash_attn_module, "flash_attn_varlen_func_compilable", unexpected_varlen)
    monkeypatch.setattr(flash_attn_module, "flash_attn_func_compilable", fixed)
    query, _, _ = _qkv()
    key = torch.randn(1, 5, 2, 8)
    value = torch.randn_like(key)

    with torch.no_grad():
        output = _build_impl(fa4_packed_varlen=True)._forward_impl(query, key, value, None)

    torch.testing.assert_close(output, query + 4)
