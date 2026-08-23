# SPDX-License-Identifier: Apache-2.0
"""Contract tests for the inference-side regional torch.compile port.

The loader applies a per-transformer-block fullgraph compile after the
transformer loads (``FastVideoArgs.inference_torch_compile``, env
``FASTVIDEO_INFERENCE_TORCH_COMPILE=1``). These tests pin the two pieces that
must not drift from the #1718 training-port semantics:

- ``_regional_compile_unsupported_reason``: legacy VSA (and the attention
  eager escape hatch) degrades to eager, while prepared MiniMax-H3 VSA is
  admitted to regional fullgraph capture.
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
from fastvideo.models.dits.minimax_h3 import MiniMaxH3Attention, MiniMaxH3Transformer3DModel
from fastvideo.models.loader import fsdp_load
from fastvideo.models.loader.fsdp_load import (
    _compile_model_regions,
    _enable_regional_attention_compile,
    _prepare_model_for_compile,
    _regional_compile_unsupported_reason,
)


def _init_params_for(backend_name: str | None) -> dict:
    resolved = None if backend_name is None else SimpleNamespace(name=backend_name)
    return {"config": SimpleNamespace(_resolved_attention_backend=resolved)}


def test_legacy_vsa_backend_degrades_to_eager(monkeypatch) -> None:
    monkeypatch.delenv("FASTVIDEO_DISABLE_ATTENTION_COMPILE", raising=False)
    backend_name = "VIDEO_SPARSE_ATTN"
    reason = _regional_compile_unsupported_reason(_init_params_for(backend_name))
    assert reason is not None
    assert backend_name in reason
    assert "eager" in reason


@pytest.mark.parametrize("backend_name", [None, "TORCH_SDPA"])
def test_supported_backends_allow_compile(backend_name, monkeypatch) -> None:
    monkeypatch.delenv("FASTVIDEO_DISABLE_ATTENTION_COMPILE", raising=False)
    monkeypatch.delenv("FASTVIDEO_H3_VSA_PROBE", raising=False)
    assert _regional_compile_unsupported_reason(_init_params_for(backend_name)) is None


def test_h3_vsa_sm100a_tile64_allows_compile(monkeypatch) -> None:
    monkeypatch.delenv("FASTVIDEO_DISABLE_ATTENTION_COMPILE", raising=False)
    monkeypatch.delenv("FASTVIDEO_H3_VSA_PROBE", raising=False)
    monkeypatch.setenv("FASTVIDEO_VSA_SM100A", "1")

    reason = _regional_compile_unsupported_reason(
        _init_params_for("VIDEO_SPARSE_ATTN_H3"),
        vsa_tile_size=64,
    )

    assert reason is None


@pytest.mark.parametrize(("sm100a", "tile_size"), [(False, 64), (True, 256), (True, None)])
def test_h3_vsa_unsupported_compile_route_degrades_to_eager(sm100a, tile_size, monkeypatch) -> None:
    monkeypatch.delenv("FASTVIDEO_DISABLE_ATTENTION_COMPILE", raising=False)
    monkeypatch.delenv("FASTVIDEO_H3_VSA_PROBE", raising=False)
    if sm100a:
        monkeypatch.setenv("FASTVIDEO_VSA_SM100A", "1")
    else:
        monkeypatch.delenv("FASTVIDEO_VSA_SM100A", raising=False)

    reason = _regional_compile_unsupported_reason(
        _init_params_for("VIDEO_SPARSE_ATTN_H3"),
        vsa_tile_size=tile_size,
    )

    assert reason is not None
    assert "eager" in reason


def test_h3_vsa_probe_degrades_regional_compile_to_eager(monkeypatch) -> None:
    monkeypatch.delenv("FASTVIDEO_DISABLE_ATTENTION_COMPILE", raising=False)
    monkeypatch.setenv("FASTVIDEO_H3_VSA_PROBE", "/tmp/h3-vsa-probe")

    reason = _regional_compile_unsupported_reason(_init_params_for("VIDEO_SPARSE_ATTN_H3"))

    assert reason is not None
    assert "FASTVIDEO_H3_VSA_PROBE" in reason
    assert "eager" in reason


class _RegionalPrepareProbe:

    def __init__(self, unsupported: str | None = None) -> None:
        self.devices: list[torch.device] = []
        self.unsupported = unsupported

    def prepare_for_regional_compile(self, device: torch.device) -> str | None:
        self.devices.append(device)
        return self.unsupported


class _GateBlock(nn.Module):

    def __init__(self, gate_weight: float) -> None:
        super().__init__()
        attention = MiniMaxH3Attention.__new__(MiniMaxH3Attention)
        nn.Module.__init__(attention)
        attention.to_q = nn.Linear(1, 1, bias=False)
        attention.to_gate_compress = nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            attention.to_gate_compress.weight.fill_(gate_weight)
        attention._gate_compress_active = None
        attention.distributed_attention = SimpleNamespace(attn_impl=_RegionalPrepareProbe())
        self.attn = attention


class _GateProbe(nn.Module):

    def __init__(self, attention: MiniMaxH3Attention) -> None:
        super().__init__()
        self.attention = attention

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.attention._gate_active():
            return x + 1
        return x - 1


def _gate_model(*weights: float) -> MiniMaxH3Transformer3DModel:
    model = MiniMaxH3Transformer3DModel.__new__(MiniMaxH3Transformer3DModel)
    nn.Module.__init__(model)
    model.transformer_blocks = nn.ModuleList([_GateBlock(weight) for weight in weights])
    model.enabled_fusions = frozenset()
    return model


def test_minimax_h3_prepare_for_compile_resolves_loaded_vsa_gates() -> None:
    model = _gate_model(0.0, 2.0)

    model.prepare_for_compile()

    assert [block.attn._gate_compress_active for block in model.transformer_blocks] == [False, True]
    for block in model.transformer_blocks:
        impl = block.attn.distributed_attention.attn_impl
        assert impl.devices == []


def test_training_compile_prepare_does_not_probe_inference_kernel() -> None:
    model = _gate_model(2.0)

    reason = _prepare_model_for_compile(model, regional=False)

    assert reason is None
    attention = model.transformer_blocks[0].attn
    assert attention._gate_compress_active is True
    assert attention.distributed_attention.attn_impl.devices == []


def test_regional_compile_prepare_prefers_specialized_hook() -> None:
    model = _gate_model(2.0)
    expected_device = next(model.parameters()).device

    reason = _prepare_model_for_compile(model, regional=True)

    assert reason is None
    impl = model.transformer_blocks[0].attn.distributed_attention.attn_impl
    assert impl.devices == [expected_device]


def test_minimax_h3_prepare_for_regional_compile_does_not_require_quantized_q_weight() -> None:
    model = _gate_model(2.0)
    attention = model.transformer_blocks[0].attn
    removed_weight = attention.to_q._parameters.pop("weight")
    attention.to_q.register_buffer("_fp8_weight", removed_weight.detach(), persistent=False)
    expected_device = next(model.parameters()).device

    reason = model.prepare_for_regional_compile()

    assert reason is None
    impl = attention.distributed_attention.attn_impl
    assert impl.devices == [expected_device]


def test_minimax_h3_prepare_for_regional_compile_propagates_backend_rejection() -> None:
    model = _gate_model(2.0)
    impl = model.transformer_blocks[0].attn.distributed_attention.attn_impl
    impl.unsupported = "sm100a probe failed"

    reason = model.prepare_for_regional_compile()

    assert reason == "sm100a probe failed"


def test_unprepared_vsa_gate_cannot_mutate_cache_during_compile(monkeypatch) -> None:
    attention = _gate_model(2.0).transformer_blocks[0].attn
    monkeypatch.setattr(torch.compiler, "is_compiling", lambda: True)

    with torch.no_grad(), pytest.raises(RuntimeError, match="not resolved before torch.compile"):
        attention._gate_active()
    assert attention._gate_compress_active is None


@pytest.mark.parametrize(("gate_weight", "expected"), [(0.0, -1.0), (2.0, 1.0)])
def test_prepared_vsa_gate_is_static_under_fullgraph_compile(gate_weight, expected) -> None:
    model = _gate_model(gate_weight)
    model.prepare_for_compile()
    attention = model.transformer_blocks[0].attn
    resolved = attention._gate_compress_active

    try:
        compiled = torch.compile(_GateProbe(attention), backend="eager", fullgraph=True)
        with torch.no_grad():
            actual = compiled(torch.zeros(1))
        torch.testing.assert_close(actual, torch.full((1, ), expected))
        assert attention._gate_compress_active is resolved
    finally:
        torch._dynamo.reset()


def test_prepared_zero_vsa_gate_still_runs_in_grad_enabled_training() -> None:
    model = _gate_model(0.0)
    model.prepare_for_compile()
    attention = model.transformer_blocks[0].attn

    with torch.enable_grad():
        assert attention._gate_active() is True
    assert attention._gate_compress_active is False


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
