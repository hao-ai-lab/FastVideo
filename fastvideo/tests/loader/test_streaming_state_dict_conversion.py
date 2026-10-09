# SPDX-License-Identifier: Apache-2.0
"""``iter_hf_to_custom_state_dict`` converts a checkpoint one tensor at a time (CPU only)."""
from __future__ import annotations

import torch

from fastvideo.models.loader.utils import (get_param_names_mapping, hf_to_custom_state_dict,
                                           iter_hf_to_custom_state_dict)

# q/k/v fuse into one ``qkv`` parameter; ``out`` is a plain rename.
MAPPING = get_param_names_mapping({
    r"^blocks\.(\d+)\.q\.weight$": (r"blocks.\1.qkv.weight", 0, 3),
    r"^blocks\.(\d+)\.k\.weight$": (r"blocks.\1.qkv.weight", 1, 3),
    r"^blocks\.(\d+)\.v\.weight$": (r"blocks.\1.qkv.weight", 2, 3),
    r"^blocks\.(\d+)\.o\.weight$": r"blocks.\1.out.weight",
})


def _source() -> dict[str, torch.Tensor]:
    return {
        "blocks.0.o.weight": torch.full((4, 4), 1.0),
        "blocks.0.v.weight": torch.full((2, 4), 3.0),
        "blocks.0.q.weight": torch.full((2, 4), 1.0),
        "blocks.0.k.weight": torch.full((2, 4), 2.0),
        "blocks.1.o.weight": torch.full((4, 4), 2.0),
    }


def test_first_tensor_is_yielded_before_the_source_is_consumed() -> None:
    pulled = 0

    def counting_source():
        nonlocal pulled
        for item in _source().items():
            pulled += 1
            yield item

    stream = iter_hf_to_custom_state_dict(counting_source(), MAPPING)
    name, _ = next(stream)
    assert name == "blocks.0.out.weight"
    assert pulled == 1


def test_fused_param_is_yielded_once_complete_and_matches_the_dict_path() -> None:
    reverse: dict = {}
    streamed = list(iter_hf_to_custom_state_dict(_source(), MAPPING, reverse))
    expected_sd, expected_reverse = hf_to_custom_state_dict(_source(), MAPPING)

    assert [name for name, _ in streamed] == ["blocks.0.out.weight", "blocks.0.qkv.weight", "blocks.1.out.weight"]
    qkv = dict(streamed)["blocks.0.qkv.weight"]
    assert torch.equal(qkv[:, 0], torch.tensor([1.0, 1.0, 2.0, 2.0, 3.0, 3.0]))
    assert reverse == expected_reverse
    assert list(dict(streamed)) == list(expected_sd)
    assert all(torch.equal(dict(streamed)[name], expected_sd[name]) for name in expected_sd)
