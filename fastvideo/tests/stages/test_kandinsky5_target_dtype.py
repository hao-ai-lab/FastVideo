# SPDX-License-Identifier: Apache-2.0
"""Regression tests for ``Kandinsky5DenoisingStage._resolve_target_dtype``.

Plain (non-FSDP) transformer: the stage must honor
``pipeline_config.dit_precision`` exactly -- ``TransformerLoader.load()``
asserts every parameter matches it for a non-FSDP load, so an explicit
fp32 pipeline must not be silently cast to bf16 (an earlier
parameter-scanning version of this helper did exactly that: all-fp32
params fell through to a bf16 "safe default").

FSDP2-wrapped transformer: the stage must read the active
``MixedPrecisionPolicy`` recorded by ``set_mixed_precision_policy`` (which
``maybe_load_fsdp_model`` always calls before sharding -- hardcoded to
``param_dtype=torch.bfloat16``, ``cast_forward_inputs=False`` today), NOT
the parameter storage dtypes. FSDP2 computes in the policy's
``param_dtype`` regardless of the dtype parameters are stored in, so an
fp16-loaded FSDP model still runs its forward in bf16; a parameter scan
resolved fp16 there and, with autocast keyed off the wrong dtype and
input casting disabled, handed bf16-computing modules fp16 inputs.

These are pure logic tests -- the FSDP "wrap" only swaps ``__class__`` the
same way ``fully_shard`` does (so the ``isinstance(..., FSDPModule)``
check fires) without any process group, GPU, or model load.
"""
from __future__ import annotations

import types

import pytest
import torch

from fastvideo import utils as fastvideo_utils
from fastvideo.pipelines.stages.kandinsky5 import Kandinsky5DenoisingStage
from fastvideo.utils import set_mixed_precision_policy


def _make_stage(transformer: torch.nn.Module) -> Kandinsky5DenoisingStage:
    """Bypass __init__: only set the field _resolve_target_dtype reads."""
    stage = Kandinsky5DenoisingStage.__new__(Kandinsky5DenoisingStage)
    stage.transformer = transformer
    return stage


def _fastvideo_args(dit_precision: str) -> types.SimpleNamespace:
    return types.SimpleNamespace(pipeline_config=types.SimpleNamespace(dit_precision=dit_precision))


def _fsdp_wrap(module: torch.nn.Module) -> torch.nn.Module:
    """Make ``isinstance(module, FSDPModule)`` true the way ``fully_shard``
    does -- by swapping ``__class__`` to a dynamic ``(FSDPModule, orig)``
    subclass -- without any actual sharding/distributed setup."""
    from torch.distributed.fsdp import FSDPModule

    orig_cls = module.__class__
    module.__class__ = type(f"FSDP{orig_cls.__name__}", (FSDPModule, orig_cls), {})
    return module


@pytest.fixture()
def mixed_precision_state_reset():
    """Snapshot/restore the process's thread-local mixed-precision state so
    these tests neither depend on nor leak state across the pytest run."""
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


def test_resolve_target_dtype_fsdp_reads_policy_not_parameter_storage(mixed_precision_state_reset):
    """An fp16-*stored* FSDP model still computes in the policy's bf16 --
    the exact mismatch a parameter scan gets wrong."""
    set_mixed_precision_policy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
    transformer = _fsdp_wrap(torch.nn.Linear(4, 4).to(torch.float16))
    stage = _make_stage(transformer)

    resolved = stage._resolve_target_dtype(_fastvideo_args("fp16"))

    assert resolved == torch.bfloat16, (
        "FSDP compute dtype comes from the MixedPrecisionPolicy param_dtype, not from "
        "the dtype the parameters happen to be stored in")


def test_resolve_target_dtype_fsdp_follows_a_non_default_policy(mixed_precision_state_reset):
    """Proves the policy is actually read (not a hardcoded bf16)."""
    set_mixed_precision_policy(param_dtype=torch.float16, reduce_dtype=torch.float32)
    transformer = _fsdp_wrap(torch.nn.Linear(4, 4).to(torch.float32))
    stage = _make_stage(transformer)

    resolved = stage._resolve_target_dtype(_fastvideo_args("fp32"))

    assert resolved == torch.float16


def test_resolve_target_dtype_fsdp_defaults_to_bf16_without_policy_state(mixed_precision_state_reset):
    """maybe_load_fsdp_model always records the policy before sharding, but
    if the thread-local state is somehow unset the stage must still match
    the loader's hardcoded bf16 param_dtype rather than crash or resolve
    the storage dtype."""
    transformer = _fsdp_wrap(torch.nn.Linear(4, 4).to(torch.float16))
    stage = _make_stage(transformer)

    resolved = stage._resolve_target_dtype(_fastvideo_args("fp16"))

    assert resolved == torch.bfloat16
