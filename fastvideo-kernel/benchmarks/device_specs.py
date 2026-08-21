# SPDX-License-Identifier: Apache-2.0
"""Per-device peak throughput used to turn benchmark timings into %MFU.

Values are dense (non-sparse) BF16 Tensor Core TFLOPS with FP32 accumulation,
from vendor datasheets. Only devices with well-established numbers are listed;
for anything else pass an explicit override (--peak-tflops in the benchmarks)
rather than silently assuming another GPU's peak.
"""

from __future__ import annotations

# Keyed by substring of torch.cuda.get_device_name(). First match wins;
# more specific names must come before less specific ones.
DENSE_BF16_TFLOPS: dict[str, float] = {
    "RTX 5090": 209.5,  # boost clock; matches the value this benchmark historically used
    "H100": 989.0,
    "A100": 312.0,
    "L40S": 362.0,
}


def resolve_peak_tflops(device_name: str, override: float | None = None) -> tuple[float, str]:
    """Return (peak_tflops, source_description) for *device_name*.

    An explicit *override* always wins. An unknown device without an override
    raises instead of guessing, since %MFU against the wrong peak is worse
    than no %MFU at all.
    """
    if override is not None:
        return override, "explicit --peak-tflops override"
    for key, tflops in DENSE_BF16_TFLOPS.items():
        if key in device_name:
            return tflops, f"device table entry {key!r}"
    known = ", ".join(DENSE_BF16_TFLOPS)
    raise ValueError(f"No dense BF16 peak-TFLOPS entry for device {device_name!r} "
                     f"(known: {known}). Pass --peak-tflops explicitly, or add the "
                     "device to DENSE_BF16_TFLOPS in benchmarks/device_specs.py.")
