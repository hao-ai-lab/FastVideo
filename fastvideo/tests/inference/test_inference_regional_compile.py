# SPDX-License-Identifier: Apache-2.0
"""Contract tests for the inference-side regional torch.compile port.

The loader applies a per-transformer-block fullgraph compile after the
transformer loads (``FastVideoArgs.inference_torch_compile``, env
``FASTVIDEO_INFERENCE_TORCH_COMPILE=1``). These tests pin the two pieces that
must not drift from the #1718 training-port semantics:

- ``_regional_compile_unsupported_reason``: VSA backends (and the attention
  eager escape hatch) degrade to eager with a reason instead of hard-failing
  fullgraph capture at the first denoising forward.
- attention forward dispatch: ordinary instances retain the historical
  compiler-disabled boundary; regional compile opts in only the selected
  model's instances.
- ``_compile_model_regions``: exactly the ``_compile_conditions`` blocks are
  compiled, fullgraph=True plus inductor ``emulate_precision_casts`` are
  injected, and ``mode`` kwargs are rejected (torch.compile forbids
  mode+options).

CPU-safe: the real capture check uses torch's eager recording backend; no CUDA
or inductor kernel build is needed.
"""

import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch
from torch import nn

from fastvideo.attention import layer as attention_layer
from fastvideo.attention.layer import DistributedAttention
from fastvideo.models.loader import fsdp_load
from fastvideo.models.loader.fsdp_load import (
    _compile_model_regions,
    _enable_regional_attention_compile,
    _regional_compile_unsupported_reason,
)


def _init_params_for(backend_name: str | None) -> dict:
    resolved = None if backend_name is None else SimpleNamespace(name=backend_name)
    return {"config": SimpleNamespace(_resolved_attention_backend=resolved)}


@pytest.mark.parametrize("backend_name", ["VIDEO_SPARSE_ATTN", "VIDEO_SPARSE_ATTN_H3"])
def test_vsa_backends_degrade_to_eager(backend_name, monkeypatch) -> None:
    monkeypatch.delenv("FASTVIDEO_DISABLE_ATTENTION_COMPILE", raising=False)
    reason = _regional_compile_unsupported_reason(_init_params_for(backend_name))
    assert reason is not None
    assert backend_name in reason
    assert "eager" in reason


@pytest.mark.parametrize("backend_name", [None, "TORCH_SDPA"])
def test_dense_backends_allow_compile(backend_name, monkeypatch) -> None:
    monkeypatch.delenv("FASTVIDEO_DISABLE_ATTENTION_COMPILE", raising=False)
    assert _regional_compile_unsupported_reason(_init_params_for(backend_name)) is None


def test_attention_compile_escape_hatch_degrades_to_eager(monkeypatch) -> None:
    monkeypatch.setenv("FASTVIDEO_DISABLE_ATTENTION_COMPILE", "1")
    reason = _regional_compile_unsupported_reason(_init_params_for("TORCH_SDPA"))
    assert reason is not None
    assert "FASTVIDEO_DISABLE_ATTENTION_COMPILE" in reason


@pytest.mark.parametrize("fa_version", ["2", "3", "4"])
def test_dense_flash_attention_inference_allows_compile(fa_version, monkeypatch) -> None:
    monkeypatch.delenv("FASTVIDEO_DISABLE_ATTENTION_COMPILE", raising=False)
    fake_module = ModuleType("fastvideo.attention.utils.flash_attn_default")
    fake_module.fa_version = fa_version
    monkeypatch.setitem(sys.modules, fake_module.__name__, fake_module)

    assert _regional_compile_unsupported_reason(_init_params_for("FLASH_ATTN")) is None


def test_default_attention_dispatch_stays_compiler_disabled(monkeypatch) -> None:
    monkeypatch.delenv("FASTVIDEO_DISABLE_ATTENTION_COMPILE", raising=False)
    disabled_calls: list[str] = []

    def _fake_disable(fn):

        def _disabled(self):
            disabled_calls.append("disabled")
            return fn(self)

        return _disabled

    monkeypatch.setattr(attention_layer.torch.compiler, "disable", _fake_disable)

    class _ToyAttention:

        def __init__(self) -> None:
            self._compile_forward_enabled = not attention_layer._attention_compile_disabled()

        @attention_layer._maybe_compiler_disable
        def forward(self) -> str:
            return "forward"

    ordinary = _ToyAttention()
    assert ordinary.forward() == "forward"
    assert disabled_calls == ["disabled"]

    # Preserve the existing process-wide escape hatch for callers that opt in
    # before constructing their attention modules.
    monkeypatch.setenv("FASTVIDEO_DISABLE_ATTENTION_COMPILE", "0")
    explicit = _ToyAttention()
    assert explicit.forward() == "forward"
    assert disabled_calls == ["disabled"]


class _BareDistributedAttention(DistributedAttention):

    def __init__(self) -> None:
        nn.Module.__init__(self)
        self._compile_forward_enabled = False


class _CompileProbe(nn.Module):

    def __init__(self, enabled: bool) -> None:
        super().__init__()
        self._compile_forward_enabled = enabled

    @attention_layer._maybe_compiler_disable
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + 1


def test_attention_instance_flag_controls_real_fullgraph_capture() -> None:
    try:
        ordinary = torch.compile(_CompileProbe(False), backend="eager", fullgraph=True)
        with pytest.raises(torch._dynamo.exc.Unsupported, match="Skip calling"):
            ordinary(torch.ones(2))

        regional = torch.compile(_CompileProbe(True), backend="eager", fullgraph=True)
        torch.testing.assert_close(regional(torch.ones(2)), torch.full((2,), 2.0))
    finally:
        torch._dynamo.reset()


def test_regional_compile_enables_only_loaded_model_attention() -> None:
    selected_model = nn.Sequential(_BareDistributedAttention(), _BareDistributedAttention())
    unrelated_model = nn.Sequential(_BareDistributedAttention())

    assert _enable_regional_attention_compile(selected_model) == 2
    assert all(module._compile_forward_enabled for module in selected_model)
    assert not unrelated_model[0]._compile_forward_enabled


class _Block(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class _Toy(nn.Module):
    _compile_conditions = [lambda name, module: name.startswith("blocks.") and name.count(".") == 1]

    def __init__(self) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([_Block() for _ in range(3)])
        self.proj_out = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return self.proj_out(x)


def test_compile_model_regions_injects_fullgraph_and_precision_casts(monkeypatch) -> None:
    captured: list[dict] = []

    def _fake_compile(fn, **kwargs):
        captured.append(kwargs)
        return fn

    monkeypatch.setattr(fsdp_load.torch, "compile", _fake_compile)
    model = _Toy()
    count = _compile_model_regions(model, {})
    # The three repeated blocks compile; proj_out and the root stay eager.
    assert count == 3
    assert len(captured) == 3
    for kwargs in captured:
        assert kwargs["fullgraph"] is True
        assert kwargs["options"] == {"emulate_precision_casts": True}


def test_compile_model_regions_rejects_mode_kwargs() -> None:
    with pytest.raises(ValueError, match="mode"):
        _compile_model_regions(_Toy(), {"mode": "reduce-overhead"})


def test_compile_model_regions_requires_conditions_and_matches(monkeypatch) -> None:
    monkeypatch.setattr(fsdp_load.torch, "compile", lambda fn, **kwargs: fn)
    plain = nn.Linear(4, 4)
    with pytest.raises(ValueError, match="_compile_conditions"):
        _compile_model_regions(plain, {})

    class _NoMatch(_Toy):
        _compile_conditions = [lambda name, module: False]

    with pytest.raises(ValueError, match="matched"):
        _compile_model_regions(_NoMatch(), {})
