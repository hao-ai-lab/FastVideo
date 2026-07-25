# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from types import SimpleNamespace

import torch

from fastvideo.train.callbacks.posthoc_ema import (
    PostHocEMACallback,
    sigma_rel_to_gamma,
    solve_posthoc_weights,
)


class _FakeMethod:

    def __init__(self) -> None:
        self.student = SimpleNamespace(
            transformer=torch.nn.Linear(2, 2, bias=False),
        )


def test_posthoc_math_is_finite() -> None:
    gamma_005 = sigma_rel_to_gamma(0.05)
    gamma_01 = sigma_rel_to_gamma(0.1)
    weights = solve_posthoc_weights(
        torch.tensor([5, 5, 10, 10]),
        torch.tensor([gamma_005, gamma_01, gamma_005, gamma_01]),
        10,
        gamma_005,
    )
    assert weights.shape == (4, )
    assert torch.isfinite(weights).all()


def test_posthoc_update_matches_nitrous_ema_step_order(tmp_path: Path) -> None:
    callback = PostHocEMACallback(
        checkpoint_every=100,
        checkpoint_folder=str(tmp_path),
    )
    callback.training_config = SimpleNamespace(
        checkpoint=SimpleNamespace(output_dir=str(tmp_path)),
    )
    method = _FakeMethod()
    callback.on_train_start(method)

    first_weight = method.student.transformer.weight.detach().float().clone()
    callback.on_training_step_end(method, {}, iteration=1)
    torch.testing.assert_close(callback._ema_models[0].shadow["weight"], first_weight)

    with torch.no_grad():
        method.student.transformer.weight.add_(1)
    second_weight = method.student.transformer.weight.detach().float().clone()
    callback.on_training_step_end(method, {}, iteration=2)

    gamma = sigma_rel_to_gamma(0.05)
    # nitrous-ema increments KarrasEMA.step before reading beta.
    official_beta = ((1 - 1 / 3)**(1 + gamma) *
                     (1 - 1 / 4)**(1 + gamma))**0.5
    expected = first_weight * official_beta + second_weight * (1 - official_beta)
    torch.testing.assert_close(callback._ema_models[0].shadow["weight"], expected)


def test_posthoc_callback_snapshots_and_synthesizes(tmp_path: Path) -> None:
    callback = PostHocEMACallback(
        checkpoint_every=2,
        checkpoint_folder=str(tmp_path),
    )
    callback.training_config = SimpleNamespace(
        checkpoint=SimpleNamespace(output_dir=str(tmp_path)),
    )
    method = _FakeMethod()
    callback.on_train_start(method)

    callback.on_training_step_end(method, {}, iteration=1)
    with torch.no_grad():
        method.student.transformer.weight.add_(1)
    callback.on_training_step_end(method, {}, iteration=2)

    assert len(list((tmp_path / "rank_00000").glob("*.pt"))) == 2
    synthesized = callback.synthesize_local_shard()
    assert synthesized is not None
    assert set(synthesized) == {"weight"}
    assert torch.isfinite(synthesized["weight"]).all()

    state = callback.state_dict()
    restored = PostHocEMACallback(
        checkpoint_every=2,
        checkpoint_folder=str(tmp_path),
    )
    restored.training_config = callback.training_config
    restored.on_train_start(method)
    restored.load_state_dict(state)
    assert restored.state_dict()["calls"] == 2

    callback.on_train_end(method, iteration=2)
    final_paths = list((tmp_path / "rank_00000").glob(
        "synthesized_sigma_0p05_step_*.pt",
    ))
    assert len(final_paths) == 1
    final_state = torch.load(final_paths[0], weights_only=True)
    assert set(final_state) == {"weight"}
