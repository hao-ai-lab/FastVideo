#!/usr/bin/env python3
"""Scratch capacity gate: block_skip activation checkpointing with a stride.

The LTX-2 wrapper calls ``apply_activation_checkpointing`` without plumbing
``n_layer``, so ``block_skip`` degenerates to full checkpointing. This driver
forces ``block_skip`` with ``FV_BLOCK_SKIP_N`` (checkpoint every Nth block)
around the frozen packed benchmark harness. Launch with
``--models.student.enable_gradient_checkpointing_type block_skip`` so the
wrapper takes the checkpointing path at all; the patch supplies the stride.
"""

from __future__ import annotations

import os

import benchmark_fastvideo_train_pack_d016 as benchmark
import fastvideo.train.models.ltx2.ltx2 as ltx2_module
from fastvideo.training.activation_checkpoint import apply_activation_checkpointing

BLOCK_SKIP_N = int(os.environ["FV_BLOCK_SKIP_N"])


def _forced_block_skip(module, checkpointing_type="full", n_layer=1):
    del checkpointing_type, n_layer
    wrapped = apply_activation_checkpointing(module, checkpointing_type="block_skip", n_layer=BLOCK_SKIP_N)
    wrapped_blocks = sum(
        1 for _, child in module.transformer_blocks.named_children()
        if type(child).__name__ == "CheckpointWrapper")
    print(f"BLOCK_SKIP_APPLIED n={BLOCK_SKIP_N} wrapped_blocks={wrapped_blocks}", flush=True)
    return wrapped


ltx2_module.apply_activation_checkpointing = _forced_block_skip

if __name__ == "__main__":
    benchmark.main()
