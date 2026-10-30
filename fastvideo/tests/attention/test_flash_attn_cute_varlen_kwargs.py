# SPDX-License-Identifier: Apache-2.0
"""FA4 varlen must not forward ``max_seqlen_q/k`` to ``_flash_attn_fwd``.

``FASTVIDEO_MINIMAX_H3_FA4_PACKED_VARLEN=1`` routes H3 inference through the
varlen custom op. Current ``flash_attn.cute`` builds derive the maximum
sequence lengths from ``cu_seqlens`` and reject ``max_seqlen_q``/``max_seqlen_k``
as keyword arguments, so every packed-varlen forward crashed with
``TypeError: _flash_attn_fwd() got an unexpected keyword argument
'max_seqlen_q'``.

The test installs a fake ``flash_attn.cute.interface`` whose ``_flash_attn_fwd``
has the current (max_seqlen-free) signature, then calls the varlen forward
directly. The old code fails with that TypeError; the fixed code forwards only
the arguments the kernel accepts.
"""

from __future__ import annotations

import importlib
import sys
import types

import pytest
import torch

_MODULE_NAME = "fastvideo.attention.utils.flash_attn_cute"


def _fake_flash_attn(monkeypatch: pytest.MonkeyPatch):
    """Install a minimal flash_attn.cute.interface with the current signature."""
    calls: list[dict] = []

    def fake_flash_attn_fwd(
        q,
        k,
        v,
        *,
        cu_seqlens_q=None,
        cu_seqlens_k=None,
        softmax_scale=None,
        causal=False,
        window_size_left=None,
        window_size_right=None,
        softcap=0.0,
        num_splits=1,
        pack_gqa=None,
        return_lse=True,
        deterministic=False,
        sm_margin=0,
    ):
        calls.append({
            "cu_seqlens_q": cu_seqlens_q,
            "cu_seqlens_k": cu_seqlens_k,
            "softmax_scale": softmax_scale,
            "causal": causal,
            "return_lse": return_lse,
        })
        return torch.empty_like(q), torch.empty((q.shape[1], q.shape[0]), dtype=torch.float32)

    def fake_flash_attn_bwd(*args, **kwargs):
        raise NotImplementedError("the test fake only covers the forward")

    flash_attn = types.ModuleType("flash_attn")
    cute = types.ModuleType("flash_attn.cute")
    interface = types.ModuleType("flash_attn.cute.interface")
    interface._flash_attn_fwd = fake_flash_attn_fwd
    interface._flash_attn_bwd = fake_flash_attn_bwd
    cute.interface = interface
    flash_attn.cute = cute
    monkeypatch.setitem(sys.modules, "flash_attn", flash_attn)
    monkeypatch.setitem(sys.modules, "flash_attn.cute", cute)
    monkeypatch.setitem(sys.modules, "flash_attn.cute.interface", interface)
    return fake_flash_attn_fwd, calls


def _load_module(monkeypatch: pytest.MonkeyPatch):
    fake_flash_attn_fwd, calls = _fake_flash_attn(monkeypatch)
    module = importlib.import_module(_MODULE_NAME)
    # A real FA4 install imports the real kernel; pin the fake so the contract
    # under test is the forwarding call, not the kernel.
    monkeypatch.setattr(module, "_flash_attn_fwd", fake_flash_attn_fwd)
    return module, calls


def test_varlen_forward_omits_max_seqlen_kwargs(monkeypatch: pytest.MonkeyPatch) -> None:
    if not torch.cuda.is_available():
        pytest.skip("flash_attn.cute is only imported on CUDA builds")
    try:
        module, calls = _load_module(monkeypatch)
    except ImportError as exc:
        pytest.skip(f"flash_attn.cute is not importable: {exc}")

    q = torch.randn(7, 2, 8, device="cuda")
    cu_seqlens = torch.tensor([0, 7], dtype=torch.int32, device="cuda")
    with torch.no_grad():
        out, lse = torch.ops.fastvideo._flash_attn_cute_varlen_forward(
            q,
            q,
            q,
            cu_seqlens,
            cu_seqlens,
            7,
            7,
            0.125,
            False,
            False,
        )

    assert out.shape == q.shape
    assert lse.shape == (2, 7)
    assert len(calls) == 1
    forwarded = calls[0]
    assert "max_seqlen_q" not in forwarded
    assert "max_seqlen_k" not in forwarded
    assert forwarded["cu_seqlens_q"] is cu_seqlens
    assert forwarded["softmax_scale"] == 0.125
