# SPDX-License-Identifier: Apache-2.0
"""Rank-to-caption mapping for the TDM multiprompt report."""

from __future__ import annotations

import pytest

from tests.local_tests.tdm.tools.tdm_multiprompt_report import _row_index


def test_row_index_uses_sp_group_not_global_rank() -> None:
    assert _row_index(rank=0, video_index=1, sp_size=2, rows_per_group=3) == 1
    assert _row_index(rank=2, video_index=1, sp_size=2, rows_per_group=3) == 4
    assert _row_index(rank=1, video_index=0, sp_size=1, rows_per_group=3) == 3


def test_row_index_rejects_non_leader_rank() -> None:
    with pytest.raises(ValueError, match="group leader"):
        _row_index(rank=1, video_index=0, sp_size=2, rows_per_group=3)
