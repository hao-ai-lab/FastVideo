# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from fastvideo.models.loader.fsdp_load import (
    load_model_from_full_model_state_dict, )


class _VideoOnlyModel(torch.nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.video_weight = torch.nn.Parameter(torch.zeros(1))

    @staticmethod
    def _is_ignored_checkpoint_key(key: str) -> bool:
        return key == "audio_weight"


def _load(model: torch.nn.Module, state: dict[str, torch.Tensor]):
    return load_model_from_full_model_state_dict(
        model,
        iter(state.items()),
        device=torch.device("cpu"),
        param_dtype=torch.float32,
        strict=True,
        param_names_mapping=lambda name: (name, None, None),
    )


def test_strict_load_ignores_only_model_declared_checkpoint_extras() -> None:
    model = _VideoOnlyModel()
    incompatible = _load(model, {
        "video_weight": torch.tensor([2.0]),
        "audio_weight": torch.tensor([3.0]),
    })

    assert incompatible.missing_keys == []
    assert incompatible.unexpected_keys == []
    torch.testing.assert_close(model.video_weight, torch.tensor([2.0]))


def test_strict_load_rejects_unrelated_checkpoint_extras() -> None:
    with pytest.raises(ValueError, match="unrelated_weight"):
        _load(
            _VideoOnlyModel(),
            {
                "video_weight": torch.tensor([2.0]),
                "unrelated_weight": torch.tensor([3.0]),
            },
        )


def test_strict_load_still_rejects_missing_model_parameters() -> None:
    with pytest.raises(ValueError, match="video_weight"):
        _load(
            _VideoOnlyModel(),
            {"audio_weight": torch.tensor([3.0])},
        )
