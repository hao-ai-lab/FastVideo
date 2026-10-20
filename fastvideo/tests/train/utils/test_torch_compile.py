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


def test_regional_compile_rejects_mode_kwarg() -> None:
    """`mode` conflicts with the always-injected inductor options.

    torch.compile forbids mode+options together; the loader must fail with an
    actionable message rather than letting torch blame an `options` key the
    user never wrote (the CLI help's own example uses `mode`).
    """
    model = _RepeatedModel()
    with pytest.raises(ValueError, match="mode"):
        _compile_model_regions(model, {"mode": "reduce-overhead"})


def test_checkpoint_wrapper_prefix_normalization() -> None:
    """AC-wrapped blocks must not break name-keyed weight-loader lookups.

    checkpoint_wrapper strips its prefix from state_dict() keys via hooks but
    NOT from named_parameters()/named_buffers(); checkpoint keys are clean.
    Pre-fix, a loaded buffer inside a wrapped block missed the named_buffers
    membership test and was silently converted into a trainable nn.Parameter
    by load_state_dict(assign=True).
    """
    from fastvideo.models.loader.fsdp_load import _strip_checkpoint_wrapper_prefix

    class _BufferBlock(torch.nn.Module):

        def __init__(self) -> None:
            super().__init__()
            self.lin = torch.nn.Linear(4, 4)
            self.register_buffer("freq", torch.arange(4.0))

        def forward(self, value: torch.Tensor) -> torch.Tensor:
            return self.lin(value) + self.freq

    class _BufferModel(torch.nn.Module):

        def __init__(self) -> None:
            super().__init__()
            self.blocks = torch.nn.ModuleList([_BufferBlock()])

    from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
        checkpoint_wrapper, )

    model = _BufferModel()
    model.blocks[0] = checkpoint_wrapper(model.blocks[0])

    # state_dict is clean; raw named_buffers is prefixed.
    assert "blocks.0.freq" in model.state_dict()
    raw_buffer_names = {name for name, _ in model.named_buffers()}
    assert "blocks.0.freq" not in raw_buffer_names
    assert "blocks.0._checkpoint_wrapped_module.freq" in raw_buffer_names

    # The canonicalized views match checkpoint keys exactly.
    clean_buffers = {_strip_checkpoint_wrapper_prefix(name) for name, _ in model.named_buffers()}
    clean_params = {_strip_checkpoint_wrapper_prefix(name) for name, _ in model.named_parameters()}
    assert clean_buffers == {"blocks.0.freq"}
    assert clean_params == {"blocks.0.lin.weight", "blocks.0.lin.bias"}

    # End-to-end: membership keyed on canonical names keeps a loaded buffer a
    # buffer under load_state_dict(assign=True) instead of promoting it to a
    # trainable parameter.
    loaded = {
        "blocks.0.freq": torch.ones(4),
        "blocks.0.lin.weight": torch.ones(4, 4),
        "blocks.0.lin.bias": torch.ones(4),
    }
    sharded_sd = {
        key: (value if key in clean_buffers else torch.nn.Parameter(value))
        for key, value in loaded.items()
    }
    model.load_state_dict(sharded_sd, assign=True)
    assert any("freq" in name for name, _ in model.named_buffers())
    assert not any("freq" in name for name, _ in model.named_parameters())
