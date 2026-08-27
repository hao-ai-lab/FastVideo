# SPDX-License-Identifier: Apache-2.0
"""The loader must not hold the whole checkpoint alive while it copies it.

``hf_to_custom_state_dict`` drains the weight iterator into one dict before a
single parameter is placed. Production safetensors values may retain
memory-mapped shard storage, and mapped or merged parameters can own ordinary
allocations. Keeping every source reference until loading finishes raises the
host or unified-memory working set enough to kill a large-model load.

Checking after the call proves nothing, because the dict is a local and dies
with the frame. So these tests keep a reference to that dict from the outside
and assert it comes back empty: if the loader iterated instead of draining,
every source tensor would still be reachable through the reference we hold.
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


def _capture_state_dict(monkeypatch) -> dict:
    """Hold the loader's internal dict from outside so we can inspect it after."""
    captured: dict = {}
    real = fsdp_load.hf_to_custom_state_dict

    def spy(*args, **kwargs):
        custom_param_sd, reverse = real(*args, **kwargs)
        captured["sd"] = custom_param_sd
        return custom_param_sd, reverse

    monkeypatch.setattr(fsdp_load, "hf_to_custom_state_dict", spy)
    return captured


def test_the_state_dict_is_drained_not_iterated(monkeypatch) -> None:
    captured = _capture_state_dict(monkeypatch)

    load_model_from_full_model_state_dict(
        _TinyModel(),
        iter(list(_source_tensors().items())),
        torch.device("cpu"),
        torch.bfloat16,
        strict=False,
        param_names_mapping=_identity_mapping,
        training_mode=False,
    )

    assert captured["sd"] == {}, ("the loader finished with the checkpoint still in hand; on a real model that is the "
                                  "whole file held resident for the length of the copy")


@pytest.mark.parametrize(
    ("skipped_name", "strict"),
    (("metadata._extra_state", True), ("unexpected.weight", False)),
)
def test_skipped_source_entry_is_popped_before_continue(monkeypatch, skipped_name: str, strict: bool) -> None:
    """Both skip branches must drop their source before advancing the loop."""
    captured = _capture_state_dict(monkeypatch)
    warning_names = []

    def assert_popped_before_warning(_message, warned_name) -> None:
        assert warned_name not in captured["sd"]
        warning_names.append(warned_name)

    monkeypatch.setattr(fsdp_load.logger, "warning", assert_popped_before_warning)
    sources = {**_source_tensors(), skipped_name: torch.ones(8, 8)}

    load_model_from_full_model_state_dict(
        _TinyModel(),
        iter(sources.items()),
        torch.device("cpu"),
        torch.bfloat16,
        strict=strict,
        param_names_mapping=_identity_mapping,
        training_mode=False,
    )

    assert warning_names == [skipped_name]
    assert captured["sd"] == {}


def test_source_tensors_become_collectable(monkeypatch) -> None:
    """The reason the drain matters: the tensors have to actually go."""
    captured = _capture_state_dict(monkeypatch)
    sources = _source_tensors()
    refs = {name: weakref.ref(tensor) for name, tensor in sources.items()}

    # Mirror safetensors_weights_iterator, which drops each tensor as it yields.
    def iterator():
        while sources:
            yield sources.popitem()

    load_model_from_full_model_state_dict(
        _TinyModel(),
        iterator(),
        torch.device("cpu"),
        torch.bfloat16,
        strict=False,
        param_names_mapping=_identity_mapping,
        training_mode=False,
    )

    # captured["sd"] is still in scope here on purpose: it is the reference that
    # would keep them alive if the loader had not popped.
    gc.collect()
    alive = sorted(name for name, ref in refs.items() if ref() is not None)
    assert not alive, f"still reachable through the loader's state dict: {alive}"


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
