# SPDX-License-Identifier: Apache-2.0
"""Device-name -> reference-folder resolution.

The lookup is a substring scan over an ordered table, which makes it
order-sensitive in a way that is easy to get wrong: ``"B200"`` is a substring of
``"NVIDIA GB200"``, so a table listing B200 first silently routes every
Grace-Blackwell host to the B200 references. That is not a cosmetic mismatch --
whichever host seeds a reference defines the baseline every later run on the
*other* host is scored against.
"""
from __future__ import annotations

import pytest

from fastvideo.tests.ssim.inference_similarity_utils import DEVICE_MAPPINGS
from fastvideo.tests.ssim.reference_utils import resolve_device_reference_folder


@pytest.mark.parametrize(
    ("device_name", "expected"),
    [
        ("NVIDIA GB200", "GB200_reference_videos"),
        ("NVIDIA B200", "B200_reference_videos"),
        ("NVIDIA H200", "H200_reference_videos"),
        ("NVIDIA H100 80GB HBM3", "H100_reference_videos"),
        ("NVIDIA L40S", "L40S_reference_videos"),
        ("NVIDIA A40", "A40_reference_videos"),
    ],
)
def test_device_resolves_to_its_own_folder(device_name: str, expected: str) -> None:
    assert resolve_device_reference_folder(DEVICE_MAPPINGS, device_name=device_name) == expected


def test_gb200_is_not_captured_by_the_b200_pattern() -> None:
    """The regression this file exists for.

    Asserting the mapping order directly, so reordering the table fails here
    rather than silently scoring GB200 runs against B200 references.
    """
    patterns = [pattern for pattern, _ in DEVICE_MAPPINGS]
    assert "GB200" in patterns, "GB200 missing from DEVICE_MAPPINGS"
    assert patterns.index("GB200") < patterns.index("B200"), (
        "GB200 must precede B200: the lookup is a substring scan, so B200 "
        "would otherwise match 'NVIDIA GB200' first")


def test_unknown_device_returns_none() -> None:
    assert resolve_device_reference_folder(DEVICE_MAPPINGS, device_name="NVIDIA RTX 5090") is None
