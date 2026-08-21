# SPDX-License-Identifier: Apache-2.0
"""Contract tests for the inference-side regional torch.compile port.

The loader applies a per-transformer-block fullgraph compile after the
transformer loads (``FastVideoArgs.inference_torch_compile``, env
``FASTVIDEO_INFERENCE_TORCH_COMPILE=1``). These tests pin the two pieces that
must not drift from the #1718 training-port semantics:

- ``_regional_compile_unsupported_reason``: VSA backends (and the attention
  eager escape hatch) degrade to eager with a reason instead of hard-failing
  fullgraph capture at the first denoising forward.
- ``_compile_model_regions``: exactly the ``_compile_conditions`` blocks are
  compiled, fullgraph=True plus inductor ``emulate_precision_casts`` are
  injected, and ``mode`` kwargs are rejected (torch.compile forbids
  mode+options).

CPU-safe: torch.compile is monkeypatched, no CUDA needed.
"""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from fastvideo.models.loader import fsdp_load
from fastvideo.models.loader.fsdp_load import (
    _compile_model_regions,
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
