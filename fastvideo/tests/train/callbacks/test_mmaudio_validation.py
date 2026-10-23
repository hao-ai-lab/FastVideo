# SPDX-License-Identifier: Apache-2.0

from fastvideo.train.callbacks.mmaudio_validation import (
    _global_inference_indices,
    _inference_sample_index, )


def test_mmaudio_inference_uses_fixed_global_budget() -> None:
    assignments = [_global_inference_indices(16, 4, rank) for rank in range(4)]

    assert all(len(rank_indices) == 4 for rank_indices in assignments)
    assert sorted(index for rank_indices in assignments
                  for index in rank_indices) == list(range(16))


def test_mmaudio_inference_rows_cover_validation_batch() -> None:
    available = 8
    assignments = [_global_inference_indices(16, 4, rank) for rank in range(4)]

    rows = {
        _inference_sample_index(global_index, local_index, available)
        for rank_indices in assignments
        for local_index, global_index in enumerate(rank_indices)
    }

    assert rows == set(range(min(16, available)))


def test_mmaudio_inference_pads_collective_calls() -> None:
    assignments = [_global_inference_indices(16, 6, rank) for rank in range(6)]

    assert all(len(rank_indices) == 3 for rank_indices in assignments)
    valid = [
        index for rank_indices in assignments for index in rank_indices
        if index is not None
    ]
    assert sorted(valid) == list(range(16))
    assert sum(index is None for rank_indices in assignments
               for index in rank_indices) == 2
