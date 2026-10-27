# SPDX-License-Identifier: Apache-2.0

import pytest

from .grad_norm_regression import resolve_device_key


@pytest.mark.parametrize(
    ("device_name", "expected_key"),
    [
        ("NVIDIA GB10", "GB10"),
        ("nvidia gb10", "GB10"),
        ("NVIDIA GB200", "GB200"),
        ("NVIDIA B200", "GB200"),
        ("NVIDIA RTX 5090", None),
    ],
)
def test_resolve_device_key(device_name: str, expected_key: str | None) -> None:
    assert resolve_device_key(device_name) == expected_key
