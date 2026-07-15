# SPDX-License-Identifier: Apache-2.0
"""Regression test: Kandinsky5 denoising stages must honor
``pipeline_config.dit_precision`` for a plain (non-FSDP) transformer.

``Kandinsky5DenoisingStage._resolve_target_dtype`` scans parameters and
defaults to bf16 when every one is fp32 -- a heuristic needed only for the
FSDP2-wrapped transformer ``ValidationCallback`` reuses from training
(``maybe_load_fsdp_model`` hardcodes its ``MixedPrecisionPolicy`` to
``param_dtype=torch.bfloat16`` independent of ``dit_precision``, so the
live training module's true compute dtype can disagree with what
``dit_precision`` claims). Applying that same scan unconditionally broke
plain inference: ``TransformerLoader.load()`` asserts every parameter
matches ``dit_precision`` exactly for a non-FSDP load, so an explicit
fp32 pipeline (all params fp32, loop never finds a non-fp32 param) fell
through to the "safe default" of bf16 -- silently running fp32-requested
inference in bf16 instead.

This is a pure logic test on a plain ``torch.nn.Module`` stand-in (never an
``FSDPModule``) -- no GPU, no model load, no distributed init needed.
"""
from __future__ import annotations

import types

import torch

from fastvideo.pipelines.stages.kandinsky5 import Kandinsky5DenoisingStage


def _make_stage(transformer: torch.nn.Module) -> Kandinsky5DenoisingStage:
    """Bypass __init__: only set the field _resolve_target_dtype reads."""
    stage = Kandinsky5DenoisingStage.__new__(Kandinsky5DenoisingStage)
    stage.transformer = transformer
    return stage


def _fastvideo_args(dit_precision: str) -> types.SimpleNamespace:
    return types.SimpleNamespace(pipeline_config=types.SimpleNamespace(dit_precision=dit_precision))


def test_resolve_target_dtype_honors_explicit_fp32_for_plain_transformer():
    transformer = torch.nn.Linear(4, 4).to(torch.float32)
    stage = _make_stage(transformer)

    resolved = stage._resolve_target_dtype(_fastvideo_args("fp32"))

    assert resolved == torch.float32, (
        "an explicit fp32 pipeline_config must not be silently cast to bf16 for a "
        "plain (non-FSDP) transformer")


def test_resolve_target_dtype_honors_explicit_fp16_for_plain_transformer():
    transformer = torch.nn.Linear(4, 4).to(torch.float16)
    stage = _make_stage(transformer)

    resolved = stage._resolve_target_dtype(_fastvideo_args("fp16"))

    assert resolved == torch.float16


def test_resolve_target_dtype_honors_explicit_bf16_for_plain_transformer():
    transformer = torch.nn.Linear(4, 4).to(torch.bfloat16)
    stage = _make_stage(transformer)

    resolved = stage._resolve_target_dtype(_fastvideo_args("bf16"))

    assert resolved == torch.bfloat16
