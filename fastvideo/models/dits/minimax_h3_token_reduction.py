# SPDX-License-Identifier: Apache-2.0
"""Training-free horizontal token pairing for MiniMax H3, after antirez/h3.c ``--token-reduction``.

Inside an interval of transformer blocks, every pair of horizontally adjacent
generated-video tokens is averaged into one row, so the middle blocks see the
target video at half width. Text, audio, references and conditioning rows are
never paired and pass through exactly. Before the first full-width block each
original row is restored as its own saved value plus the update its pair
received, so within-pair detail survives::

    row 1, 3, 5, 9 -> pairs 2, 7 -> middle blocks 2.5, 6 -> restored 1.5, 3.5, 4, 8

Pair positions are the mean of the two source positions; a pair takes the
modulation row of its first source (every target-video row shares one).
Unpaired rows, such as the last column of an odd-width video, take the
reduced output directly, as in h3.c, so ``update_scale`` applies to pairs only.

The option parsing and the default schedule live in
``fastvideo/configs/pipelines/minimax_h3_token_reduction.py``.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from fastvideo.attention.backends.abstract import AttentionMetadata
from fastvideo.configs.pipelines.minimax_h3_token_reduction import MiniMaxH3TokenReductionSchedule


@dataclass(frozen=True)
class MiniMaxH3TokenReduction:
    """Row maps for one packed sequence whose generated video is its trailing ``(t, h, w)`` token grid.

    ``first``/``second`` name the full rows averaged into each reduced row
    (equal for unpaired rows); ``parents`` names the reduced row every full
    row restores from. ``pair_reduced_rows`` and ``pair_rows`` list the
    reduced and full rows that belong to a true pair.
    """

    begin: int
    end: int
    first: torch.Tensor
    second: torch.Tensor
    parents: torch.Tensor
    pair_reduced_rows: torch.Tensor
    pair_rows: torch.Tensor
    position_ids: torch.Tensor
    update_scale: float = 1.0
    # Attention metadata of the reduced sequence (VSA-H3 tiles of the half-width grid).
    attn_metadata: AttentionMetadata | None = None

    @property
    def sequence_length(self) -> int:
        return int(self.parents.numel())

    @property
    def reduced_length(self) -> int:
        return int(self.first.numel())

    def to(self, device: torch.device) -> MiniMaxH3TokenReduction:
        return MiniMaxH3TokenReduction(
            begin=self.begin,
            end=self.end,
            first=self.first.to(device),
            second=self.second.to(device),
            parents=self.parents.to(device),
            pair_reduced_rows=self.pair_reduced_rows.to(device),
            pair_rows=self.pair_rows.to(device),
            position_ids=self.position_ids.to(device),
            update_scale=self.update_scale,
            attn_metadata=self.attn_metadata,
        )


def reduced_token_width(width: int) -> int:
    return (width + 1) // 2


def build_token_reduction(
    position_ids: torch.Tensor,
    video_start: int,
    grid: tuple[int, int, int],
    schedule: MiniMaxH3TokenReductionSchedule,
) -> MiniMaxH3TokenReduction:
    """Pair horizontally adjacent rows of the row-major ``(t, h, w)`` video grid ending the sequence."""
    t, h, w = (int(v) for v in grid)
    sequence_length = int(position_ids.shape[0])
    if min(t, h, w) <= 0 or video_start < 0 or video_start + t * h * w != sequence_length:
        raise ValueError(f"Token reduction needs the generated video grid {grid} to end the packed sequence "
                         f"({video_start} + {t * h * w} != {sequence_length}).")
    reduced_w = reduced_token_width(w)
    prefix = torch.arange(video_start, dtype=torch.long)

    reduced_local = torch.arange(t * h * reduced_w, dtype=torch.long)
    pair_column = 2 * (reduced_local % reduced_w)
    first_video = video_start + (reduced_local // reduced_w) * w + pair_column
    has_pair = pair_column + 1 < w
    second_video = first_video + has_pair.long()

    full_local = torch.arange(t * h * w, dtype=torch.long)
    column = full_local % w
    parent_video = video_start + (full_local // w) * reduced_w + column // 2
    in_pair = 2 * (column // 2) + 1 < w

    first = torch.cat([prefix, first_video])
    second = torch.cat([prefix, second_video])
    # (x + x) / 2 == x, so unpaired rows keep their exact positions.
    reduced_positions = (position_ids.index_select(0, first) + position_ids.index_select(0, second)) * 0.5
    return MiniMaxH3TokenReduction(
        begin=schedule.begin,
        end=schedule.end,
        first=first,
        second=second,
        parents=torch.cat([prefix, parent_video]),
        pair_reduced_rows=video_start + torch.nonzero(has_pair).flatten(),
        pair_rows=video_start + torch.nonzero(in_pair).flatten(),
        position_ids=reduced_positions,
        update_scale=schedule.update_scale,
    )


def pool_token_pairs(hidden_states: torch.Tensor, plan: MiniMaxH3TokenReduction) -> torch.Tensor:
    """``[B, S, C]`` -> ``[B, R, C]``: unpaired rows copied, pairs averaged in float32."""
    pooled = hidden_states.index_select(1, plan.first)
    rows = plan.pair_reduced_rows
    if rows.numel():
        pair_mean = (hidden_states.index_select(1, plan.first.index_select(0, rows)).float() +
                     hidden_states.index_select(1, plan.second.index_select(0, rows)).float()) * 0.5
        pooled.index_copy_(1, rows, pair_mean.to(hidden_states.dtype))
    return pooled


def expand_token_pairs(original: torch.Tensor, reduced: torch.Tensor, pooled: torch.Tensor,
                       plan: MiniMaxH3TokenReduction) -> torch.Tensor:
    """Restore ``[B, S, C]`` from the reduced stack's output ``[B, R, C]``.

    Unpaired rows take the reduced output directly. A paired row becomes its
    value before pooling plus ``update_scale`` times the change of its pair,
    ``reduced - pooled``.
    """
    restored = reduced.index_select(1, plan.parents)
    rows = plan.pair_rows
    if rows.numel():
        parents = plan.parents.index_select(0, rows)
        update = reduced.index_select(1, parents).float() - pooled.index_select(1, parents).float()
        paired = original.index_select(1, rows).float() + plan.update_scale * update
        restored.index_copy_(1, rows, paired.to(restored.dtype))
    return restored


__all__ = [
    "MiniMaxH3TokenReduction",
    "build_token_reduction",
    "expand_token_pairs",
    "pool_token_pairs",
    "reduced_token_width",
]
