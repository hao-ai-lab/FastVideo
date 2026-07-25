# SPDX-License-Identifier: Apache-2.0

from fastvideo.train.callbacks.mmaudio_validation import (
    _global_inference_indices, )


def test_mmaudio_inference_uses_fixed_global_budget() -> None:
    assignments = [_global_inference_indices(16, 4, rank) for rank in range(4)]

    assert all(len(rank_indices) == 4 for rank_indices in assignments)
    assert sorted(index for rank_indices in assignments
                  for index in rank_indices) == list(range(16))


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
