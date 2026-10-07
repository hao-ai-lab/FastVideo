# SPDX-License-Identifier: Apache-2.0

import torch

from fastvideo.models.dits.mmaudio import TimestepEmbedder


def test_timestep_embedder_accepts_integer_timesteps() -> None:
    embedder = TimestepEmbedder(8, 8, 10_000)

    actual = embedder(torch.tensor([1], dtype=torch.long))
    expected = embedder(torch.tensor([1.0]))

    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    assert actual.dtype == embedder.mlp[0].weight.dtype
