# SPDX-License-Identifier: Apache-2.0
"""Configuration tests for pure Ring Attention integration."""

from __future__ import annotations

import pytest

from fastvideo.fastvideo_args import FastVideoArgs


MODEL_PATH = "FastVideo/LTX2-Distilled-Diffusers"


def test_sequence_parallel_defaults() -> None:
    args = FastVideoArgs(model_path=MODEL_PATH)

    assert args.sp_size == args.num_gpus
    assert args.ring_size == 1


@pytest.mark.parametrize(
    ("num_gpus", "sp_size", "ring_size"),
    [
        (2, 2, 1),  # Ulysses
        (4, 4, 1),  # Ulysses
        (2, 2, 2),  # pure Ring
        (4, 4, 4),  # pure Ring
    ],
)
def test_supported_sequence_parallel_configurations(
    num_gpus: int,
    sp_size: int,
    ring_size: int,
) -> None:
    args = FastVideoArgs(
        model_path=MODEL_PATH,
        num_gpus=num_gpus,
        sp_size=sp_size,
        ring_size=ring_size,
    )

    assert args.sp_size == sp_size
    assert args.ring_size == ring_size


def test_sp_size_minus_one_resolves_to_num_gpus() -> None:
    args = FastVideoArgs(
        model_path=MODEL_PATH,
        num_gpus=4,
        sp_size=-1,
    )

    assert args.sp_size == 4


@pytest.mark.parametrize("sp_size", [0, -2])
def test_invalid_sp_size_rejected(sp_size: int) -> None:
    with pytest.raises(ValueError, match="sp_size must be >= 1"):
        FastVideoArgs(
            model_path=MODEL_PATH,
            num_gpus=4,
            sp_size=sp_size,
        )


@pytest.mark.parametrize("ring_size", [0, -1])
def test_invalid_ring_size_rejected(ring_size: int) -> None:
    with pytest.raises(ValueError, match="ring_size must be >= 1"):
        FastVideoArgs(
            model_path=MODEL_PATH,
            num_gpus=4,
            sp_size=4,
            ring_size=ring_size,
        )


def test_sp_size_cannot_exceed_num_gpus() -> None:
    with pytest.raises(ValueError, match="cannot exceed"):
        FastVideoArgs(
            model_path=MODEL_PATH,
            num_gpus=2,
            sp_size=4,
        )


def test_num_gpus_must_be_divisible_by_sp_size() -> None:
    with pytest.raises(ValueError, match="must be divisible"):
        FastVideoArgs(
            model_path=MODEL_PATH,
            num_gpus=4,
            sp_size=3,
        )


def test_ring_requires_sequence_parallelism() -> None:
    with pytest.raises(
        ValueError,
        match="requires sequence parallelism",
    ):
        FastVideoArgs(
            model_path=MODEL_PATH,
            num_gpus=2,
            sp_size=1,
            ring_size=2,
        )


def test_partial_ring_group_is_not_supported_yet() -> None:
    with pytest.raises(
        NotImplementedError,
        match="pure Ring only",
    ):
        FastVideoArgs(
            model_path=MODEL_PATH,
            num_gpus=4,
            sp_size=4,
            ring_size=2,
        )