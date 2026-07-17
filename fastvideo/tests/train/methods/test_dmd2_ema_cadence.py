# SPDX-License-Identifier: Apache-2.0
"""EMA cadence regression tests for alternating DMD2 optimization."""

from fastvideo.train.methods.distribution_matching.dmd2 import DMD2Method


def test_dmd2_ema_follows_generator_update_interval() -> None:
    method = object.__new__(DMD2Method)
    object.__setattr__(
        method,
        "method_config",
        {"generator_update_interval": 5},
    )

    observed = [
        DMD2Method.should_update_ema(method, iteration)
        for iteration in range(1, 11)
    ]

    assert observed == [
        False,
        False,
        False,
        False,
        True,
        False,
        False,
        False,
        False,
        True,
    ]
