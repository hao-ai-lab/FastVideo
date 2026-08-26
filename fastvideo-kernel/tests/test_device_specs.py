# SPDX-License-Identifier: Apache-2.0
"""CPU-only tests for benchmarks/device_specs.py (no torch or GPU required)."""

import math
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "benchmarks"))

from device_specs import DENSE_BF16_TFLOPS, format_tflops_with_mfu, resolve_peak_tflops  # noqa: E402


@pytest.mark.parametrize(
    "device_name",
    [
        "NVIDIA GeForce RTX 5090",
        "NVIDIA H100 80GB HBM3",
        "NVIDIA H100 NVL",
        "NVIDIA H100 PCIe",
        "NVIDIA A100-SXM4-80GB",
        "NVIDIA A100-PCIE-40GB",
        "NVIDIA L40S",
    ],
)
def test_known_full_gpus_resolve_from_table(device_name):
    tflops, source = resolve_peak_tflops(device_name)
    assert tflops == DENSE_BF16_TFLOPS[device_name]
    assert device_name in source


def test_override_wins_even_for_known_device():
    tflops, source = resolve_peak_tflops("NVIDIA H100 80GB HBM3", override=123.4)
    assert tflops == 123.4
    assert "override" in source


@pytest.mark.parametrize(
    "device_name",
    [
        "NVIDIA GB10",
        "NVIDIA H100",
        "NVIDIA H100 80GB HBM3 MIG 1g.10gb",
        "NVIDIA A100-SXM4-80GB MIG 1g.10gb",
    ],
)
def test_unknown_or_partitioned_device_keeps_benchmark_available(device_name):
    tflops, source = resolve_peak_tflops(device_name)
    assert tflops is None
    assert "peak-tflops" in source


def test_unknown_device_with_override_resolves():
    tflops, source = resolve_peak_tflops("NVIDIA GB10", override=99.0)
    assert tflops == 99.0
    assert "override" in source


@pytest.mark.parametrize("override", [0.0, -1.0, math.nan, math.inf, -math.inf])
def test_override_must_be_positive_and_finite(override):
    with pytest.raises(ValueError, match="finite and greater than zero"):
        resolve_peak_tflops("NVIDIA GB10", override=override)


def test_format_tflops_reports_mfu_only_with_a_valid_denominator():
    assert format_tflops_with_mfu(123.456, None) == "123.46 TFLOPS/MFU N/A"
    assert format_tflops_with_mfu(123.456, 200.0) == "123.46 TFLOPS/61.73% MFU"
