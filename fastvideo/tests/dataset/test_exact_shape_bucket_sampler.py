# SPDX-License-Identifier: Apache-2.0
"""Exact-shape global microbatch scheduling contracts."""

from pathlib import Path

import pytest

from fastvideo.dataset.parquet_dataset_map_style import (
    DP_SP_BatchSampler,
    shape_bucket_ids_from_parquet_files,
)
from fastvideo.dataset.shape_bucket import parse_video_shape_bucket_id


@pytest.mark.parametrize(
    "bucket_id",
    [
        "bucket=0x768-124f",
        "bucket=1344x0-124f",
        "bucket=1344x768-0f",
        "bucket=1344X768-124f",
        "bucket=1344x768-124F",
        "bucket=01344x768-124f",
        "1344x768-124f",
        "bucket=1344x768",
        "bucket=1344x768-124f-extra",
    ],
)
def test_portable_bucket_id_rejects_noncanonical_spelling(bucket_id: str) -> None:
    with pytest.raises(ValueError, match="bucket=<width>x<height>"):
        parse_video_shape_bucket_id(bucket_id)


def test_bucket_ids_expand_from_canonical_parquet_ancestors() -> None:
    files = [
        "/shared/source-a/data/bucket=1344x768-124f/part-0.parquet",
        "/shared/source-b/data/bucket=768x1344-362f/part-1.parquet",
    ]
    assert shape_bucket_ids_from_parquet_files(files, [2, 1]) == [
        "bucket=1344x768-124f",
        "bucket=1344x768-124f",
        "bucket=768x1344-362f",
    ]

    with pytest.raises(ValueError, match="exactly one ancestor"):
        shape_bucket_ids_from_parquet_files(["/shared/data/part.parquet"], [1])
    malformed = str(Path("/shared/data/bucket=1344-768-124f/part.parquet"))
    with pytest.raises(ValueError, match="positive decimal"):
        shape_bucket_ids_from_parquet_files([malformed], [1])


def _rank_samplers(bucket_ids: list[str], *, seed: int = 17) -> list[DP_SP_BatchSampler]:
    return [
        DP_SP_BatchSampler(
            batch_size=1,
            dataset_size=len(bucket_ids),
            num_sp_groups=32,
            sp_world_size=1,
            global_rank=rank,
            drop_last=True,
            seed=seed,
            sample_bucket_ids=bucket_ids,
        ) for rank in range(32)
    ]


def test_rare_bucket_is_padded_not_dropped_and_all_ranks_share_schedule() -> None:
    rare = "bucket=480x832-124f"
    common = "bucket=1344x768-362f"
    bucket_ids = [rare] * 7 + [common] * 64
    samplers = _rank_samplers(bucket_ids)

    assert all(sampler.bucket_schedule == samplers[0].bucket_schedule for sampler in samplers)
    assert samplers[0].bucket_schedule is not None
    assert samplers[0].bucket_schedule.count(rare) == 1
    assert samplers[0].bucket_schedule.count(common) == 2
    assert samplers[0].bucket_padding == {rare: 25}
    assert samplers[0].num_padded_samples == 25

    batches_by_rank = [list(sampler) for sampler in samplers]
    observed_originals: set[int] = set()
    for step, scheduled_bucket in enumerate(samplers[0].bucket_schedule):
        step_indices = [batches_by_rank[rank][step][0] for rank in range(32)]
        assert {bucket_ids[index] for index in step_indices} == {scheduled_bucket}
        observed_originals.update(step_indices)
        if scheduled_bucket == rare:
            assert set(step_indices) == set(range(7))
            assert len(step_indices) == 32

    # Padding may repeat rows, but every frozen row participates at least once.
    assert observed_originals == set(range(len(bucket_ids)))


def test_bucket_schedule_is_seeded_and_sp_ranks_share_indices() -> None:
    bucket_ids = (["bucket=1344x768-124f"] * 16 + ["bucket=768x1344-362f"] * 16)

    def sampler(rank: int, seed: int) -> DP_SP_BatchSampler:
        return DP_SP_BatchSampler(
            batch_size=1,
            dataset_size=len(bucket_ids),
            num_sp_groups=4,
            sp_world_size=2,
            global_rank=rank,
            drop_last=True,
            seed=seed,
            sample_bucket_ids=bucket_ids,
        )

    first = [list(sampler(rank, 9)) for rank in range(8)]
    replay = [list(sampler(rank, 9)) for rank in range(8)]
    changed = [list(sampler(rank, 10)) for rank in range(8)]
    assert first == replay
    assert first != changed
    for sp_leader in range(0, 8, 2):
        assert first[sp_leader] == first[sp_leader + 1]
