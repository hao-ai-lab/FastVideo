# SPDX-License-Identifier: Apache-2.0
"""Per-device peak throughput used to turn benchmark timings into %MFU.

Values are dense (non-sparse) BF16 Tensor Core TFLOPS with FP32 accumulation.
Only unambiguous full-GPU names with published peaks are listed. Timings and
achieved TFLOPS remain useful on other devices, so unknown devices report MFU
as unavailable unless the user supplies an explicit peak.
"""

from __future__ import annotations

import math

# Exact torch.cuda.get_device_name() values for supported full GPUs. H100 form
# factors have different published peaks, and exact matching also prevents MIG,
# vGPU, or future variants from silently inheriting a full-GPU denominator.
DENSE_BF16_TFLOPS: dict[str, float] = {
    "NVIDIA H100 80GB HBM3": 989.0,
    "NVIDIA H100 NVL": 835.5,
    "NVIDIA H100 PCIe": 756.5,
    "NVIDIA GeForce RTX 5090": 209.5,  # Matches the value this benchmark historically used.
    "NVIDIA A100-SXM4-80GB": 312.0,
    "NVIDIA A100-SXM4-40GB": 312.0,
    "NVIDIA A100-PCIE-80GB": 312.0,
    "NVIDIA A100-PCIE-40GB": 312.0,
    "NVIDIA L40S": 362.05,
}


def resolve_peak_tflops(device_name: str, override: float | None = None) -> tuple[float | None, str]:
    """Return (peak_tflops, source_description) for *device_name*.

    An explicit positive, finite *override* always wins. Unknown and partitioned
    devices return ``None`` so the benchmark can still report timing and
    achieved TFLOPS without presenting a misleading MFU value.
    """
    if override is not None:
        if not math.isfinite(override) or override <= 0:
            raise ValueError("--peak-tflops must be finite and greater than zero")
        return float(override), "explicit --peak-tflops override"

    normalized_name = device_name.strip()
    if "MIG" in normalized_name.upper():
        return None, "MIG/partitioned device; pass --peak-tflops for this partition to report MFU"

    if normalized_name in DENSE_BF16_TFLOPS:
        return DENSE_BF16_TFLOPS[normalized_name], f"exact device table entry {normalized_name!r}"

    return None, f"no dense BF16 peak for {device_name!r}; pass --peak-tflops to report MFU"


def format_tflops_with_mfu(achieved_tflops: float, peak_tflops: float | None) -> str:
    """Format achieved throughput, including MFU only when its denominator is known."""
    if peak_tflops is None:
        return f"{achieved_tflops:.2f} TFLOPS/MFU N/A"
    return f"{achieved_tflops:.2f} TFLOPS/{100 * achieved_tflops / peak_tflops:.2f}% MFU"
