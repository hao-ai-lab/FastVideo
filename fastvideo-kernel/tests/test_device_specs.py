# SPDX-License-Identifier: Apache-2.0
"""CPU-only tests for benchmarks/device_specs.py (no torch, no GPU)."""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "benchmarks"))

from device_specs import DENSE_BF16_TFLOPS, resolve_peak_tflops  # noqa: E402


def test_known_devices_resolve_from_table():
    tflops, source = resolve_peak_tflops("NVIDIA GeForce RTX 5090")
    assert tflops == DENSE_BF16_TFLOPS["RTX 5090"]
    assert "RTX 5090" in source

    tflops, source = resolve_peak_tflops("NVIDIA L40S")
    assert tflops == DENSE_BF16_TFLOPS["L40S"]
    assert "L40S" in source


def test_override_wins_even_for_known_device():
    tflops, source = resolve_peak_tflops("NVIDIA H100 80GB HBM3", override=123.4)
    assert tflops == 123.4
    assert "override" in source


def test_unknown_device_without_override_raises_with_guidance():
    with pytest.raises(ValueError, match="peak-tflops"):
        resolve_peak_tflops("NVIDIA GB10")


def test_unknown_device_with_override_resolves():
    tflops, source = resolve_peak_tflops("NVIDIA GB10", override=99.0)
    assert tflops == 99.0
    assert "override" in source
