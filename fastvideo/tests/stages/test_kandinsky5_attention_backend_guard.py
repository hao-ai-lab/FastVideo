# SPDX-License-Identifier: Apache-2.0
"""Regression tests for ``Kandinsky5DenoisingStage._assert_local_attention_backend_engaged``.

The stage guard must be strict for ``ATTN_QAT_TRAIN`` only:
``fastvideo.platforms.cuda`` raises ``ImportError`` at backend selection if
the training kernel isn't built (training must never silently
de-quantize), so an exact match is guaranteed whenever module construction
succeeded -- a mismatch there really does mean a silent SDPA fallback.

``ATTN_QAT_INFER`` is different: ``fastvideo.platforms.cuda`` deliberately
resolves it to FlashAttention when the sm_120 kernel isn't built, and the
QAD recipe README (examples/train/configs/fine_tuning/kandinsky5/README.md)
documents that fallback as the supported path on non-sm_120 GPUs ("the
attention falls back to a supported dense backend while the FP4 linear
layers still run"). Treating it as exact-match-only would let module
construction succeed with FlashAttention and then abort denoising on every
machine relying on the documented fallback.

Pure logic tests on stand-in modules -- no GPU, no optional kernels, no
model load, no distributed init needed.
"""
from __future__ import annotations

import pytest
import torch

from fastvideo.attention import LocalAttention
from fastvideo.pipelines.stages.kandinsky5 import Kandinsky5DenoisingStage
from fastvideo.platforms import AttentionBackendEnum


class _ResolvedLocalAttention(LocalAttention):
    """``LocalAttention`` stand-in with a pre-resolved backend.

    Bypasses ``LocalAttention.__init__`` (which runs real backend selection
    -- unavailable without the optional kernels these tests are about)
    while still passing the stage guard's
    ``isinstance(module, LocalAttention)`` check.
    """

    def __init__(self, backend: AttentionBackendEnum) -> None:
        torch.nn.Module.__init__(self)
        self.backend = backend


def _make_stage(resolved_backend: AttentionBackendEnum) -> Kandinsky5DenoisingStage:
    """Bypass __init__: only set the field the guard reads."""
    stage = Kandinsky5DenoisingStage.__new__(Kandinsky5DenoisingStage)
    transformer = torch.nn.Module()
    transformer.attn = _ResolvedLocalAttention(resolved_backend)
    stage.transformer = transformer
    return stage


def test_attn_qat_infer_flash_fallback_is_allowed(monkeypatch):
    monkeypatch.setenv("FASTVIDEO_ATTENTION_BACKEND", "ATTN_QAT_INFER")
    stage = _make_stage(AttentionBackendEnum.FLASH_ATTN)

    # Must not raise: FlashAttention is the documented ATTN_QAT_INFER
    # fallback on GPUs without the sm_120 kernel.
    stage._assert_local_attention_backend_engaged()


def test_attn_qat_infer_exact_match_is_allowed(monkeypatch):
    monkeypatch.setenv("FASTVIDEO_ATTENTION_BACKEND", "ATTN_QAT_INFER")
    stage = _make_stage(AttentionBackendEnum.ATTN_QAT_INFER)

    stage._assert_local_attention_backend_engaged()


def test_attn_qat_train_mismatch_raises(monkeypatch):
    monkeypatch.setenv("FASTVIDEO_ATTENTION_BACKEND", "ATTN_QAT_TRAIN")
    stage = _make_stage(AttentionBackendEnum.TORCH_SDPA)

    with pytest.raises(AssertionError, match="ATTN_QAT_TRAIN"):
        stage._assert_local_attention_backend_engaged()


def test_attn_qat_train_exact_match_is_allowed(monkeypatch):
    monkeypatch.setenv("FASTVIDEO_ATTENTION_BACKEND", "ATTN_QAT_TRAIN")
    stage = _make_stage(AttentionBackendEnum.ATTN_QAT_TRAIN)

    stage._assert_local_attention_backend_engaged()
