# SPDX-License-Identifier: Apache-2.0
"""CPU-only test for the NVFP4 QAT attachment receipt.

``_maybe_quantize_model`` is the single post-load walk over the model,
so it is where the definitive "how many linears carry the QAT-train
method" count exists. This test builds a tiny module tree — one linear
whose prefix matches the QAT target set, one that doesn't — and asserts
the receipt reports the counts derived from that walk.
"""
from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import contextmanager

import torch.nn as nn

from fastvideo.layers.linear import ReplicatedLinear
from fastvideo.layers.quantization.nvfp4_qat_train_config import (
    NVFP4QATTrainConfig, )
from fastvideo.models.loader.fsdp_load import _maybe_quantize_model


# ``fastvideo.logger.init_logger`` sets ``propagate=False`` on its
# loggers, so the standard ``caplog`` fixture cannot observe them.
# Attach a temporary handler directly to the target logger instead
# (same pattern as fastvideo/tests/train/callbacks/test_callback.py).
@contextmanager
def _capture_logger(name: str, level: int = logging.INFO) -> Iterator[list[logging.LogRecord]]:
    logger = logging.getLogger(name)
    records: list[logging.LogRecord] = []

    class _Handler(logging.Handler):

        def emit(self, record: logging.LogRecord) -> None:
            records.append(record)

    handler = _Handler(level=level)
    prev_level = logger.level
    logger.addHandler(handler)
    logger.setLevel(level)
    try:
        yield records
    finally:
        logger.removeHandler(handler)
        logger.setLevel(prev_level)


def _model(quant_config: NVFP4QATTrainConfig | None) -> nn.Module:
    model = nn.Module()
    # Prefix in DEFAULT_FP4_LAYERS -> gets the QAT-train method attached.
    model.matched = ReplicatedLinear(16, 16, quant_config=quant_config, prefix="blocks.0.attn1.to_k")
    # Prefix outside the target set -> UnquantizedLinearMethod fallback.
    model.unmatched = ReplicatedLinear(16, 16, quant_config=quant_config, prefix="blocks.0.proj_mlp")
    return model


def test_qat_train_receipt_reports_attached_and_skipped_counts() -> None:
    model = _model(NVFP4QATTrainConfig())
    with _capture_logger("fastvideo.models.loader.fsdp_load") as records:
        _maybe_quantize_model(model)
    messages = [r.getMessage() for r in records]
    assert "NVFP4 QAT: attached 1 linears (1 skipped by prefix filter)" in messages


def test_no_receipt_without_qat_train_layers() -> None:
    model = _model(quant_config=None)
    with _capture_logger("fastvideo.models.loader.fsdp_load") as records:
        _maybe_quantize_model(model)
    assert not any("NVFP4 QAT" in r.getMessage() for r in records)
