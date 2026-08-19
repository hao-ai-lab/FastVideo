# SPDX-License-Identifier: Apache-2.0
"""Regional training compile policy regression tests."""

from __future__ import annotations

import pytest
import torch

from fastvideo.models.loader.fsdp_load import _compile_model_regions
from fastvideo.train.utils.activation_checkpoint import apply_activation_checkpointing


class _RepeatedModel(torch.nn.Module):
    _compile_conditions = [
        lambda name, module: name.startswith("blocks.") and name.count(".") == 1
    ]

    def __init__(self) -> None:
        super().__init__()
        self.blocks = torch.nn.ModuleList([
            torch.nn.Linear(4, 4),
            torch.nn.Linear(4, 4),
        ])

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            value = block(value)
        return value


def test_regional_compile_preserves_checkpoint_state_dict(monkeypatch) -> None:
    model = apply_activation_checkpointing(_RepeatedModel())
    state_dict_keys = list(model.state_dict())
    calls: list[tuple[torch.nn.Module, dict]] = []

    def _fake_compile(forward, **kwargs):
        calls.append((forward.__self__, kwargs))
        return forward

    monkeypatch.setattr(torch, "compile", _fake_compile)

    assert _compile_model_regions(model, {}) == 2
    assert [target for target, _ in calls] == [
        block._checkpoint_wrapped_module for block in model.blocks
    ]
    assert [kwargs for _, kwargs in calls] == [
        {"fullgraph": True, "options": {"emulate_precision_casts": True}},
        {"fullgraph": True, "options": {"emulate_precision_casts": True}},
    ]
    assert list(model.state_dict()) == state_dict_keys


def test_regional_compile_dispatches_grad_and_no_grad_calls(monkeypatch) -> None:
    model = _RepeatedModel()
    compiled_calls = 0

    def _fake_compile(eager_forward, **kwargs):
        del kwargs

        def _compiled(*args, **forward_kwargs):
            nonlocal compiled_calls
            compiled_calls += 1
            return eager_forward(*args, **forward_kwargs)

        return _compiled

    monkeypatch.setattr(torch, "compile", _fake_compile)
    _compile_model_regions(model, {})

    value = torch.randn(2, 4)
    model(value)
    assert compiled_calls == 2

    with torch.no_grad():
        model(value)
    assert compiled_calls == 4


def test_regional_compile_forwards_supported_kwargs(monkeypatch) -> None:
    model = _RepeatedModel()
    calls: list[dict] = []

    def _fake_compile(forward, **kwargs):
        calls.append(kwargs)
        return forward

    monkeypatch.setattr(torch, "compile", _fake_compile)

    _compile_model_regions(model, {
        "dynamic": False,
        "options": {
            "emulate_precision_casts": False,
            "max_autotune": True,
        },
    })

    assert calls == [
        {
            "fullgraph": True,
            "dynamic": False,
            "options": {
                "emulate_precision_casts": False,
                "max_autotune": True,
            },
        },
        {
            "fullgraph": True,
            "dynamic": False,
            "options": {
                "emulate_precision_casts": False,
                "max_autotune": True,
            },
        },
    ]


def test_regional_compile_rejects_partial_graph_mode() -> None:
    with pytest.raises(ValueError, match="fullgraph=True"):
        _compile_model_regions(_RepeatedModel(), {"fullgraph": False})
