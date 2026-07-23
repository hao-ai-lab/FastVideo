# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
import torch

from fastvideo.models.loader.utils import get_param_names_mapping, hf_to_custom_state_dict
from fastvideo.training import training_utils


def _packed_projection_mapper():
    return get_param_names_mapping({
        r"^q\.weight$": ("attention.to_qkv.weight", 0, 3),
        r"^k\.weight$": ("attention.to_qkv.weight", 1, 3),
        r"^v\.weight$": ("attention.to_qkv.weight", 2, 3),
        r"^norm\.weight$": "model.norm.weight",
    })


def test_merged_parameter_round_trip_preserves_unequal_split_sizes() -> None:
    hf_state = {
        "q.weight": torch.arange(6).reshape(2, 3),
        "k.weight": torch.arange(3).reshape(1, 3) + 10,
        "v.weight": torch.arange(9).reshape(3, 3) + 20,
        "norm.weight": torch.arange(3),
    }

    custom_state, reverse_mapping = hf_to_custom_state_dict(
        iter([
            ("v.weight", hf_state["v.weight"]),
            ("norm.weight", hf_state["norm.weight"]),
            ("q.weight", hf_state["q.weight"]),
            ("k.weight", hf_state["k.weight"]),
        ]),
        _packed_projection_mapper(),
    )

    torch.testing.assert_close(
        custom_state["attention.to_qkv.weight"],
        torch.cat([
            hf_state["q.weight"],
            hf_state["k.weight"],
            hf_state["v.weight"],
        ]),
    )
    assert reverse_mapping["attention.to_qkv.weight"] == [
        ("q.weight", 0, 3, 2),
        ("k.weight", 1, 3, 1),
        ("v.weight", 2, 3, 3),
    ]

    round_trip = training_utils.custom_to_hf_state_dict(
        custom_state.items(),
        reverse_mapping,
    )
    assert round_trip.keys() == hf_state.keys()
    for name, tensor in hf_state.items():
        torch.testing.assert_close(round_trip[name], tensor)


def test_hf_to_custom_rejects_incomplete_merge_group() -> None:
    with pytest.raises(ValueError, match="Incomplete merged parameters"):
        hf_to_custom_state_dict(
            {
                "q.weight": torch.ones(2, 3),
                "k.weight": torch.ones(1, 3),
            },
            _packed_projection_mapper(),
        )


def test_hf_to_custom_rejects_duplicate_merge_index() -> None:
    mapper = get_param_names_mapping({
        r"^q\.weight$": ("attention.to_qk.weight", 0, 2),
        r"^k\.weight$": ("attention.to_qk.weight", 0, 2),
    })

    with pytest.raises(ValueError, match="Duplicate merge index 0"):
        hf_to_custom_state_dict(
            {
                "q.weight": torch.ones(2, 3),
                "k.weight": torch.ones(1, 3),
            },
            mapper,
        )


def test_custom_to_hf_rejects_incomplete_reverse_merge_group() -> None:
    with pytest.raises(ValueError, match="Incomplete reverse merge mapping"):
        training_utils.custom_to_hf_state_dict(
            {"attention.to_qkv.weight": torch.ones(5, 3)},
            {
                "attention.to_qkv.weight": [
                    ("q.weight", 0, 3, 2),
                    ("v.weight", 2, 3, 3),
                ]
            },
        )


def test_clip_grad_norm_uses_local_dtensor_shards_for_foreach(monkeypatch) -> None:

    class FakeDTensor:

        def __init__(self, local: torch.Tensor) -> None:
            self.local = local

        def to_local(self) -> torch.Tensor:
            return self.local

    monkeypatch.setattr(torch.distributed.tensor, "DTensor", FakeDTensor)

    local_grad = torch.tensor([3.0, 4.0])
    parameter = SimpleNamespace(grad=FakeDTensor(local_grad))

    training_utils._clip_grads_with_norm_(
        [parameter],
        max_norm=1.0,
        total_norm=torch.tensor(5.0),
    )

    torch.testing.assert_close(local_grad, torch.tensor([0.6, 0.8]))
