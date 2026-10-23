# SPDX-License-Identifier: Apache-2.0
"""Regression tests for ``Kandinsky6DenoisingStage._resolve_target_dtype``.

Same logic and same rationale as
``fastvideo/tests/stages/test_kandinsky5_target_dtype.py`` (Kandinsky6DenoisingStage
copies Kandinsky5DenoisingStage's dtype-resolution helper verbatim): a plain
(non-FSDP) transformer must honor ``pipeline_config.dit_precision`` exactly,
while an FSDP2-wrapped transformer must read the active
``MixedPrecisionPolicy`` (always recorded by ``maybe_load_fsdp_model`` before
sharding) rather than the parameter storage dtype, since FSDP2 computes in
the policy's ``param_dtype`` regardless of what dtype parameters are stored
in.

Pure logic tests -- the FSDP "wrap" only swaps ``__class__`` the same way
``fully_shard`` does, without any process group, GPU, or model load.
"""
from __future__ import annotations

import types

import pytest
import torch

from fastvideo import utils as fastvideo_utils
from fastvideo.pipelines.stages.kandinsky6 import Kandinsky6DenoisingStage
from fastvideo.utils import set_mixed_precision_policy


def _make_stage(transformer: torch.nn.Module) -> Kandinsky6DenoisingStage:
    """Bypass __init__: only set the field _resolve_target_dtype reads."""
    stage = Kandinsky6DenoisingStage.__new__(Kandinsky6DenoisingStage)
    stage.transformer = transformer
    return stage


def _fastvideo_args(dit_precision: str) -> types.SimpleNamespace:
    return types.SimpleNamespace(pipeline_config=types.SimpleNamespace(dit_precision=dit_precision))


def _fsdp_wrap(module: torch.nn.Module) -> torch.nn.Module:
    from torch.distributed.fsdp import FSDPModule

    orig_cls = module.__class__
    module.__class__ = type(f"FSDP{orig_cls.__name__}", (FSDPModule, orig_cls), {})
    return module


@pytest.fixture()
def mixed_precision_state_reset():
    state_holder = fastvideo_utils._mixed_precision_state
    had_state = hasattr(state_holder, "state")
    prev_state = getattr(state_holder, "state", None)
    if had_state:
        del state_holder.state
    yield
    if had_state:
        state_holder.state = prev_state
    elif hasattr(state_holder, "state"):
        del state_holder.state


def test_resolve_target_dtype_honors_explicit_fp32_for_plain_transformer():
    transformer = torch.nn.Linear(4, 4).to(torch.float32)
    stage = _make_stage(transformer)

    resolved = stage._resolve_target_dtype(_fastvideo_args("fp32"))

    assert resolved == torch.float32


def test_resolve_target_dtype_honors_explicit_bf16_for_plain_transformer():
    transformer = torch.nn.Linear(4, 4).to(torch.bfloat16)
    stage = _make_stage(transformer)

    resolved = stage._resolve_target_dtype(_fastvideo_args("bf16"))

    assert resolved == torch.bfloat16


def test_resolve_target_dtype_fsdp_reads_policy_not_parameter_storage(mixed_precision_state_reset):
    set_mixed_precision_policy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
    transformer = _fsdp_wrap(torch.nn.Linear(4, 4).to(torch.float16))
    stage = _make_stage(transformer)

    resolved = stage._resolve_target_dtype(_fastvideo_args("fp16"))

    assert resolved == torch.bfloat16


def test_resolve_target_dtype_fsdp_defaults_to_bf16_without_policy_state(mixed_precision_state_reset):
    transformer = _fsdp_wrap(torch.nn.Linear(4, 4).to(torch.float16))
    stage = _make_stage(transformer)

    resolved = stage._resolve_target_dtype(_fastvideo_args("fp16"))

    assert resolved == torch.bfloat16
