# SPDX-License-Identifier: Apache-2.0
"""Rank-to-caption mapping for the TDM multiprompt report."""

from __future__ import annotations

import pytest

from tests.local_tests.tdm.tools.tdm_multiprompt_report import _row_index, _summarize_schedules


def test_row_index_uses_sp_group_not_global_rank() -> None:
    assert _row_index(rank=0, video_index=1, sp_size=2, rows_per_group=3) == 1
    assert _row_index(rank=2, video_index=1, sp_size=2, rows_per_group=3) == 4
    assert _row_index(rank=1, video_index=0, sp_size=1, rows_per_group=3) == 3


def test_row_index_rejects_non_leader_rank() -> None:
    with pytest.raises(ValueError, match="group leader"):
        _row_index(rank=1, video_index=0, sp_size=2, rows_per_group=3)


def test_summary_keeps_sampling_schedules_separate() -> None:
    records = {
        4: [{
            "train": {
                "within_spread": 0.1,
                "sharpness": 10.0,
            },
            "heldout": {
                "within_spread": 0.3,
                "sharpness": 20.0,
            },
        }],
        8: [{
            "train": {
                "within_spread": 0.5,
                "sharpness": 30.0,
            },
            "heldout": {
                "within_spread": 0.7,
                "sharpness": 40.0,
            },
        }],
    }

    summary = _summarize_schedules(records, {"train"})

    assert summary == {
        "inference_steps_4": {
            "train_mean_within_spread": 0.1,
            "heldout_mean_within_spread": 0.3,
            "train_mean_sharpness": 10.0,
            "heldout_mean_sharpness": 20.0,
        },
        "inference_steps_8": {
            "train_mean_within_spread": 0.5,
            "heldout_mean_within_spread": 0.7,
            "train_mean_sharpness": 30.0,
            "heldout_mean_sharpness": 40.0,
        },
    }
