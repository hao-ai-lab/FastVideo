# SPDX-License-Identifier: Apache-2.0
"""``Embeddings1DConnector`` must left-align each row's valid tokens.

The connector replaces padded positions with learnable register tokens. Rows
in a batch generally have different prompt lengths, so the selection of valid
tokens is not rectangular and cannot be expressed as a boolean index -- it has
to be a per-row gather.

CPU-only, no model download: the method is exercised directly against a stub
carrying just the two attributes it reads.
"""
from __future__ import annotations

import torch

from fastvideo.models.encoders.gemma import Embeddings1DConnector


class _Stub:
    """Only what ``_replace_padded_with_learnable_registers`` touches."""

    def __init__(self, num_registers: int, hidden: int) -> None:
        self.num_learnable_registers = num_registers
        # Register i holds -(i+1) in every channel: distinct from any hidden
        # state below, and distinct from each other, so an assertion can pin
        # not just "a register landed here" but *which* register landed here.
        self.learnable_registers = -(torch.arange(num_registers, dtype=torch.float32) +
                                     1.0).unsqueeze(-1).expand(num_registers, hidden).contiguous()


def _run(hidden_states: torch.Tensor, valid_lengths: list[int], num_registers: int | None = None):
    """Drive the real method with a left-padded mask (Gemma pads on the left)."""
    batch, seq_len, hidden = hidden_states.shape
    mask = torch.full((batch, 1, 1, seq_len), -10000.0, device=hidden_states.device)
    for row, n in enumerate(valid_lengths):
        mask[row, 0, 0, seq_len - n:] = 0.0  # left padding: valid tokens at the end
    stub = _Stub(num_registers=num_registers or seq_len, hidden=hidden)
    stub.learnable_registers = stub.learnable_registers.to(hidden_states.device)
    return Embeddings1DConnector._replace_padded_with_learnable_registers(stub, hidden_states, mask)


def test_ragged_batch_left_aligns_each_row_independently() -> None:
    """The regression: two rows with different prompt lengths.

    Row 0 keeps 3 tokens, row 1 keeps 1. Each row's valid tokens must land at
    the front of that row, with registers filling only that row's tail.
    """
    seq_len, hidden = 4, 2
    # Token values encode (row, position) so misplacement is visible.
    hidden_states = torch.tensor(
        [[[10.0, 10.0], [11.0, 11.0], [12.0, 12.0], [13.0, 13.0]],
         [[20.0, 20.0], [21.0, 21.0], [22.0, 22.0], [23.0, 23.0]]],
    )
    out, _ = _run(hidden_states, valid_lengths=[3, 1])

    # Row 0's valid tokens are the last 3 (11, 12, 13), left-aligned in order.
    assert out[0, 0].tolist() == [11.0, 11.0]
    assert out[0, 1].tolist() == [12.0, 12.0]
    assert out[0, 2].tolist() == [13.0, 13.0]
    assert out[0, 3].tolist() == [-4.0, -4.0]  # register 3, not just "a register"

    # Row 1 keeps only the last token; the other three slots take registers
    # 1, 2 and 3 -- the register bank is indexed by slot, not by fill order.
    assert out[1, 0].tolist() == [23.0, 23.0]
    assert out[1, 1:].tolist() == [[-2.0, -2.0], [-3.0, -3.0], [-4.0, -4.0]]


def test_row_with_no_padding_is_unchanged() -> None:
    seq_len, hidden = 4, 2
    hidden_states = torch.arange(seq_len * hidden, dtype=torch.float32).reshape(1, seq_len, hidden)
    out, _ = _run(hidden_states, valid_lengths=[seq_len])
    assert torch.equal(out, hidden_states)


def test_registers_tile_across_the_sequence() -> None:
    """The shipped ratio is 8 duplications (1024 slots / 128 registers), not 1.

    Every other test here uses ``num_registers == seq_len``, where the tile is
    a no-op -- so a wrong tile, or no tile at all, would pass them. With fewer
    registers than slots, padded slot ``i`` must hold register ``i % n``.
    """
    hidden_states = torch.tensor([[[10.0, 10.0], [11.0, 11.0], [12.0, 12.0], [13.0, 13.0]]])
    out, _ = _run(hidden_states, valid_lengths=[1], num_registers=2)

    assert out[0, 0].tolist() == [13.0, 13.0]  # the single valid token, left-aligned
    # Tiled bank is [r0, r1, r0, r1], so the padded tail reads r1, r0, r1.
    assert out[0, 1].tolist() == [-2.0, -2.0]
    assert out[0, 2].tolist() == [-1.0, -1.0]
    assert out[0, 3].tolist() == [-2.0, -2.0]


def test_single_row_matches_ragged_batch_row() -> None:
    """A row's result must not depend on what else is in the batch."""
    hidden_states = torch.tensor(
        [[[10.0, 10.0], [11.0, 11.0], [12.0, 12.0], [13.0, 13.0]],
         [[20.0, 20.0], [21.0, 21.0], [22.0, 22.0], [23.0, 23.0]]],
    )
    batched, _ = _run(hidden_states, valid_lengths=[3, 1])
    alone, _ = _run(hidden_states[1:2], valid_lengths=[1])
    assert torch.equal(batched[1], alone[0])
