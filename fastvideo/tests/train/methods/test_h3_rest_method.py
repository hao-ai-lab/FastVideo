# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest
import torch

from fastvideo.train.methods.knowledge_distillation.h3_rest import compute_h3_rest_losses
from fastvideo.train.methods.knowledge_distillation.h3_rest_ema import TrainableShardEMA


def test_rest_loss_signs_video_but_not_audio() -> None:
    prediction = torch.tensor([[1.0, 2.0, 3.0]], requires_grad=True)
    losses = compute_h3_rest_losses(
        prediction,
        video_target=torch.zeros(1, 2),
        audio_target=torch.zeros(1, 1),
        ema_prediction=prediction.detach().clone(),
        video_slice=slice(0, 2),
        audio_slice=slice(2, 3),
        coefficient=torch.tensor([-0.5]),
        audio_loss_weight=1.0,
        ema_regularization_weight=0.0,
    )
    assert losses["total_loss"].item() >= 0.0
    losses["total_loss"].backward()
    assert torch.all(prediction.grad[0, :2] < 0)
    assert prediction.grad[0, 2] > 0


def test_rest_ema_regularizer_is_zero_at_initialization() -> None:
    prediction = torch.tensor([[1.0, 2.0, 3.0]], requires_grad=True)
    losses = compute_h3_rest_losses(
        prediction,
        video_target=torch.zeros(1, 2),
        audio_target=torch.zeros(1, 1),
        ema_prediction=prediction.detach().clone(),
        video_slice=slice(0, 2),
        audio_slice=slice(2, 3),
        coefficient=torch.tensor([0.5]),
        audio_loss_weight=1.0,
        ema_regularization_weight=0.2,
    )
    assert losses["video_ema_loss"].item() == 0.0
    assert losses["audio_ema_loss"].item() == 0.0


def test_trainable_shard_ema_updates_swaps_and_restores() -> None:
    module = torch.nn.Linear(2, 1, bias=False)
    with torch.no_grad():
        module.weight.fill_(1.0)
    ema = TrainableShardEMA(module, decay=0.5)
    with torch.no_grad():
        module.weight.fill_(3.0)
    ema.update(module)
    assert torch.allclose(
        ema.shadow["weight"], torch.full_like(ema.shadow["weight"], 2.0)
    )
    with ema.apply_to_model(module):
        assert torch.allclose(module.weight, torch.full_like(module.weight, 2.0))
    assert torch.allclose(module.weight, torch.full_like(module.weight, 3.0))


def test_trainable_shard_ema_state_round_trip() -> None:
    module = torch.nn.Linear(2, 1, bias=False)
    ema = TrainableShardEMA(module, decay=0.9)
    with torch.no_grad():
        module.weight.add_(1.0)
    ema.update(module)
    state = ema.state_dict()
    other = TrainableShardEMA(module, decay=0.9)
    other.load_state_dict(state)
    assert other.num_updates == 1
    assert torch.allclose(other.shadow["weight"], ema.shadow["weight"])
    wrong = TrainableShardEMA(module, decay=0.8)
    with pytest.raises(ValueError, match="does not match"):
        wrong.load_state_dict(state)
