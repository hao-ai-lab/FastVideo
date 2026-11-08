# SPDX-License-Identifier: Apache-2.0
"""Collect per-rank training measurements and their step-wise maxima."""
from __future__ import annotations

from typing import Any

import torch.distributed as dist


def gather_step_records(records: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[list[dict[str, Any]]]]:
    rank_records: list[list[dict[str, Any]]] = [[] for _ in range(dist.get_world_size())]
    dist.all_gather_object(rank_records, records)
    steps = [
        {
            **rank_records[0][index],
            **{
                key: max(rows[index][key] for rows in rank_records)
                for key in ('step_ms', 'peak_allocated_bytes', 'peak_reserved_bytes')
            },
        }
        for index in range(len(records))
    ]
    return steps, rank_records
