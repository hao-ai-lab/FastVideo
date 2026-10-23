# SPDX-License-Identifier: Apache-2.0
"""``_supports_fp8_compute`` decides between the FP8 ``_scaled_mm`` path and the
bf16 dequant fallback. On ROCm the CUDA capability tuple is the GFX generation,
so the gate has to look at the architecture name instead."""
from types import SimpleNamespace

import pytest
import torch

from fastvideo.layers import fp8linear
from fastvideo.layers.quantization import fp8_config


def _fake_device(monkeypatch: pytest.MonkeyPatch,
                 *,
                 hip: str | None,
                 arch: str = "",
                 cap: tuple[int, int] = (9, 5),
                 available: bool = True) -> None:
    monkeypatch.setattr(torch.version, "hip", hip, raising=False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: available)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda *_: SimpleNamespace(gcnArchName=arch))
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *_: cap)


@pytest.mark.parametrize(
    ("arch", "expected"),
    [
        ("gfx950:sramecc+:xnack-", True),  # MI350X / MI355X
        ("gfx942:sramecc+:xnack-", False),  # MI300X: fnuz-only FP8
        ("gfx90a:sramecc+:xnack-", False),  # MI250: no FP8
        ("gfx1201", False),
        ("", False),
    ],
)
def test_rocm_gate_admits_only_cdna4(monkeypatch: pytest.MonkeyPatch, arch: str, expected: bool) -> None:
    # The capability tuple is (9, 5) for every case: it must not decide the outcome on ROCm.
    _fake_device(monkeypatch, hip="7.14.60850", arch=arch, cap=(9, 5))
    assert fp8_config._supports_fp8_compute() is expected


@pytest.mark.parametrize(("cap", "expected"), [((8, 0), False), ((8, 6), False), ((8, 9), True), ((9, 0), True),
                                               ((10, 0), True), ((12, 1), True)])
def test_cuda_gate_is_sm89(monkeypatch: pytest.MonkeyPatch, cap: tuple[int, int], expected: bool) -> None:
    _fake_device(monkeypatch, hip=None, cap=cap)
    assert fp8_config._supports_fp8_compute() is expected


def test_gate_is_false_without_an_accelerator(monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_device(monkeypatch, hip="7.14.60850", arch="gfx950", available=False)
    assert fp8_config._supports_fp8_compute() is False


def test_fp8linear_reuses_the_same_gate_and_constants() -> None:
    """The QAT linear path must not re-implement the gate or the FP8 constants."""
    assert fp8linear._supports_fp8_compute is fp8_config._supports_fp8_compute
    assert fp8linear.FP8_MAX == fp8_config.FP8_MAX
    assert fp8linear.FP8_MIN_SCALE == fp8_config.FP8_MIN_SCALE
