# SPDX-License-Identifier: Apache-2.0
"""The loader must not hold the whole checkpoint alive while it copies it.

The loader consumes ``iter_hf_to_custom_state_dict`` one tensor at a time:
each source tensor is cast and placed before the next one is pulled from the
weight iterator. Production safetensors values may retain memory-mapped shard
storage, and since the DiT path reads straight onto the GPU, draining the
iterator into a dict first would keep a full-precision checkpoint resident in
full before the bf16 cast.

These tests observe the source iterator itself: at the moment the loader pulls
a tensor, the previously placed one must already be collectable.
"""
from __future__ import annotations

import gc
import weakref

import pytest
import torch
from torch import nn

from fastvideo.models.loader import fsdp_load
from fastvideo.models.loader.fsdp_load import load_model_from_full_model_state_dict

PARAM_NAMES = ("a.weight", "b.weight", "c.weight")


class _TinyModel(nn.Module):
    """Plain module, no device mesh, so the loader takes its unsharded path."""

    def __init__(self) -> None:
        super().__init__()
        self.a = nn.Linear(8, 8, bias=False)
        self.b = nn.Linear(8, 8, bias=False)
        self.c = nn.Linear(8, 8, bias=False)


def _identity_mapping(name: str) -> tuple[str, None, None]:
    return name, None, None


def _source_tensors(scale: bool = False) -> dict[str, torch.Tensor]:
    # FP32 on purpose: the loader casts to param_dtype, and a cast is what makes
    # the copy a real copy. Handing it tensors that already match would let
    # `.to()` return the same object, and the model would then legitimately keep
    # the source alive for reasons that have nothing to do with this fix.
    return {
        name: torch.ones(8, 8, dtype=torch.float32) * (index + 1 if scale else 1)
        for index, name in enumerate(PARAM_NAMES)
    }


def _releasing_iterator(sources: dict[str, torch.Tensor], alive_at_pull: list[list[str]]):
    """Mirror safetensors_weights_iterator (drops each tensor as it yields) and record
    which earlier source tensors are still reachable each time the loader pulls."""
    refs: dict[str, weakref.ref] = {}
    for name in list(sources):
        gc.collect()
        alive_at_pull.append(sorted(n for n, ref in refs.items() if ref() is not None))
        tensor = sources.pop(name)
        refs[name] = weakref.ref(tensor)
        yield name, tensor
        del tensor


def test_each_source_tensor_is_released_before_the_next_is_pulled() -> None:
    alive_at_pull: list[list[str]] = []

    load_model_from_full_model_state_dict(
        _TinyModel(),
        _releasing_iterator(_source_tensors(), alive_at_pull),
        torch.device("cpu"),
        torch.bfloat16,
        strict=False,
        param_names_mapping=_identity_mapping,
        training_mode=False,
    )

    assert alive_at_pull == [[], [], []], ("the loader pulled a tensor while an earlier one was still in hand; on a "
                                           "real model that is the checkpoint accumulating on the GPU")


@pytest.mark.parametrize(
    ("skipped_name", "strict"),
    (("metadata._extra_state", True), ("unexpected.weight", False)),
)
def test_skipped_source_entry_is_not_retained(monkeypatch, skipped_name: str, strict: bool) -> None:
    """Both skip branches must drop their source once the loader moves on."""
    warning_names = []
    monkeypatch.setattr(fsdp_load.logger, "warning", lambda _message, name: warning_names.append(name))
    sources = {skipped_name: torch.ones(8, 8), **_source_tensors()}
    skipped_ref = weakref.ref(sources[skipped_name])
    alive_at_pull: list[list[str]] = []

    load_model_from_full_model_state_dict(
        _TinyModel(),
        _releasing_iterator(sources, alive_at_pull),
        torch.device("cpu"),
        torch.bfloat16,
        strict=strict,
        param_names_mapping=_identity_mapping,
        training_mode=False,
    )

    assert warning_names == [skipped_name]
    # The loop variable keeps a skipped entry alive for one pull; it must be gone by the one after.
    assert skipped_name not in alive_at_pull[2]
    gc.collect()
    assert skipped_ref() is None


def test_weights_still_land_in_the_model() -> None:
    """Releasing early must not cost correctness."""
    sources = _source_tensors(scale=True)
    expected = {name: tensor[0, 0].item() for name, tensor in sources.items()}

    model = _TinyModel()
    load_model_from_full_model_state_dict(
        model,
        iter(list(sources.items())),
        torch.device("cpu"),
        torch.bfloat16,
        strict=False,
        param_names_mapping=_identity_mapping,
        training_mode=False,
    )

    loaded = dict(model.named_parameters())
    for name, value in expected.items():
        assert loaded[name].dtype == torch.bfloat16
        assert loaded[name][0, 0].item() == value


@pytest.mark.parametrize(
    ("cpu_offload", "fsdp_inference", "has_unified_memory", "expected"),
    [
        (True, False, False, True),
        (True, True, True, True),
        (False, True, False, True),
        (False, True, True, False),
        (False, False, False, False),
        (False, False, True, False),
    ],
)
def test_transformer_checkpoint_staging_policy(
    cpu_offload: bool,
    fsdp_inference: bool,
    has_unified_memory: bool,
    expected: bool,
) -> None:
    """FSDP inference stages on CPU for discrete GPUs; training keeps ``cpu_offload``."""
    assert fsdp_load._should_stage_transformer_weights_on_cpu(
        cpu_offload=cpu_offload,
        fsdp_inference=fsdp_inference,
        has_unified_memory=has_unified_memory,
    ) is expected
