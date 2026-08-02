# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest

from fastvideo.fastvideo_args import FastVideoArgs


MODEL_PATH = "FastVideo/LTX2-Distilled-Diffusers"


def test_sequence_parallel_disabled_by_default() -> None:
    args = FastVideoArgs(model_path=MODEL_PATH)

    assert args.sp_size == 1
    assert args.ring_size == 1


@pytest.mark.parametrize(
    ("num_gpus", "sp_size", "ring_size"),
    [
        (2, 2, 1),  # pure Ulysses
        (2, 2, 2),  # pure Ring
        (4, 4, 1),  # larger Ulysses group
        (4, 4, 2),  # hybrid candidate
        (8, 8, 4),  # hybrid candidate
    ],
)
def test_valid_sequence_parallel_configurations(
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


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("sp_size", 0),
        ("ring_size", 0),
        ("sp_size", -1),
        ("ring_size", -1),
    ],
)
def test_parallel_sizes_must_be_positive(
    field: str,
    value: int,
) -> None:
    kwargs = {
        "model_path": MODEL_PATH,
        "num_gpus": 4,
        "sp_size": 1,
        "ring_size": 1,
    }
    kwargs[field] = value

    with pytest.raises(ValueError):
        FastVideoArgs(**kwargs)


def test_sp_size_cannot_exceed_num_gpus() -> None:
    with pytest.raises(ValueError):
        FastVideoArgs(
            model_path=MODEL_PATH,
            num_gpus=2,
            sp_size=4,
            ring_size=1,
        )


def test_ring_size_cannot_exceed_sp_size() -> None:
    with pytest.raises(ValueError):
        FastVideoArgs(
            model_path=MODEL_PATH,
            num_gpus=4,
            sp_size=2,
            ring_size=4,
        )


@pytest.mark.parametrize(
    ("sp_size", "ring_size"),
    [
        (4, 3),
        (8, 3),
        (8, 6),
    ],
)
def test_sp_size_must_be_divisible_by_ring_size(
    sp_size: int,
    ring_size: int,
) -> None:
    with pytest.raises(ValueError, match="divisible"):
        FastVideoArgs(
            model_path=MODEL_PATH,
            num_gpus=sp_size,
            sp_size=sp_size,
            ring_size=ring_size,
        )


def test_ring_size_one_represents_no_ring_parallelism() -> None:
    args = FastVideoArgs(
        model_path=MODEL_PATH,
        num_gpus=4,
        sp_size=4,
        ring_size=1,
    )

    assert args.sp_size == 4
    assert args.ring_size == 1


def test_sp_size_one_requires_ring_size_one() -> None:
    with pytest.raises(ValueError):
        FastVideoArgs(
            model_path=MODEL_PATH,
            num_gpus=2,
            sp_size=1,
            ring_size=2,
        )