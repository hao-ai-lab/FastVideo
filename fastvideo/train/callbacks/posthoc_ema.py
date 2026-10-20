# SPDX-License-Identifier: Apache-2.0
"""FSDP2-compatible post-hoc EMA used by the official MMAudio recipe."""

from __future__ import annotations

import contextlib
import os
from collections.abc import Generator
from pathlib import Path
from typing import Any, TYPE_CHECKING

import numpy as np
import torch

from fastvideo.logger import init_logger
from fastvideo.train.callbacks.callback import Callback
from fastvideo.training.training_utils import EMA_FSDP

if TYPE_CHECKING:
    from fastvideo.train.methods.base import TrainingMethod

logger = init_logger(__name__)


def sigma_rel_to_gamma(sigma_rel: float) -> float:
    """Algorithm 2 from Karras et al., matching ``nitrous-ema``."""
    if sigma_rel <= 0:
        raise ValueError("Post-hoc EMA sigma_rel must be positive")
    t = sigma_rel**-2
    return float(np.roots([1, 7, 16 - t, 12 - t]).real.max())


def _p_dot_p(
    t_a: torch.Tensor,
    gamma_a: torch.Tensor,
    t_b: torch.Tensor,
    gamma_b: torch.Tensor,
) -> torch.Tensor:
    t_ratio = t_a / t_b
    t_exp = torch.where(t_a < t_b, gamma_b, -gamma_a)
    t_max = torch.maximum(t_a, t_b)
    numerator = (gamma_a + 1) * (gamma_b + 1) * t_ratio**t_exp
    denominator = (gamma_a + gamma_b + 1) * t_max
    return numerator / denominator


def solve_posthoc_weights(
    timesteps: torch.Tensor,
    gammas: torch.Tensor,
    target_timestep: int,
    target_gamma: float,
) -> torch.Tensor:
    """Algorithm 3 from Karras et al., matching ``nitrous-ema``."""
    t_i = timesteps.double().reshape(-1, 1)
    gamma_i = gammas.double().reshape(-1, 1)
    t_j = timesteps.double().reshape(1, -1)
    gamma_j = gammas.double().reshape(1, -1)
    matrix = _p_dot_p(t_i, gamma_i, t_j, gamma_j)
    target_t = torch.tensor([[target_timestep]], dtype=torch.float64)
    target_g = torch.tensor([[target_gamma]], dtype=torch.float64)
    rhs = _p_dot_p(t_i, gamma_i, target_t, target_g)
    return torch.linalg.solve(matrix, rhs).squeeze(-1)


class PostHocEMACallback(Callback):
    """Maintain and checkpoint multiple Karras EMA profiles on local shards.

    The upstream ``nitrous-ema`` implementation deep-copies a complete model
    on rank zero. That is appropriate for MMAudio's DDP trainer but not for a
    FastVideo FSDP2/HSDP transformer. This callback applies the same update and
    synthesis equations independently to every local FSDP shard. Together the
    rank-local snapshots represent the same full EMA model without gathering
    it on every optimizer step.
    """

    def __init__(
        self,
        *,
        sigma_rels: list[float] | tuple[float, ...] = (0.05, 0.1),
        update_every: int = 1,
        checkpoint_every: int = 5000,
        checkpoint_folder: str | None = None,
        start_iter: int = 0,
        default_output_sigma: float = 0.05,
        step_size_correction: bool = True,
    ) -> None:
        if len(sigma_rels) < 2:
            raise ValueError("Post-hoc EMA requires at least two sigma profiles")
        self.sigma_rels = tuple(float(value) for value in sigma_rels)
        self.gammas = tuple(sigma_rel_to_gamma(value) for value in self.sigma_rels)
        if len(set(self.gammas)) != len(self.gammas):
            raise ValueError("Post-hoc EMA sigma profiles must be unique")
        self.update_every = max(1, int(update_every))
        self.checkpoint_every = max(1, int(checkpoint_every))
        self.checkpoint_folder = checkpoint_folder
        self.start_iter = int(start_iter)
        self.default_output_sigma = float(default_output_sigma)
        self.step_size_correction = bool(step_size_correction)

        self._ema_models: list[EMA_FSDP] = []
        self._calls = 0
        self._initted = False
        self._rank = 0
        self._snapshot_root: Path | None = None

    def on_train_start(
        self,
        method: TrainingMethod,
        iteration: int = 0,
    ) -> None:
        del iteration
        transformer = method.student.transformer
        self._ema_models = [EMA_FSDP(transformer, decay=0.0, mode="local_shard") for _ in self.gammas]
        if torch.distributed.is_initialized():
            self._rank = torch.distributed.get_rank()
        root = self.checkpoint_folder
        if not root:
            root = str(Path(self.training_config.checkpoint.output_dir) / "posthoc_ema")
        self._snapshot_root = Path(root).expanduser().resolve()
        (self._snapshot_root / f"rank_{self._rank:05d}").mkdir(
            parents=True,
            exist_ok=True,
        )
        logger.info(
            "PostHocEMA enabled: sigma_rels=%s, checkpoint_every=%d, folder=%s",
            self.sigma_rels,
            self.checkpoint_every,
            self._snapshot_root,
        )

    def _decay(self, gamma: float) -> float:
        step = self._calls
        if not self.step_size_correction:
            return (1.0 - 1.0 / (step + 1))**(1.0 + gamma)
        first = (1.0 - 1.0 / (step + 1))**(1.0 + gamma)
        second = (1.0 - 1.0 / (step + 1 + self.update_every))**(1.0 + gamma)
        return (first * second)**(self.update_every / 2)

    def on_training_step_end(
        self,
        method: TrainingMethod,
        loss_dict: dict[str, Any],
        iteration: int = 0,
    ) -> None:
        del loss_dict
        if iteration < self.start_iter or not self._ema_models:
            return

        previous_calls = self._calls
        self._calls += 1
        if previous_calls % self.update_every != 0:
            return

        transformer = method.student.transformer
        if not self._initted:
            for ema in self._ema_models:
                ema._init_shadow(transformer)
            self._initted = True
        for gamma, ema in zip(self.gammas, self._ema_models, strict=True):
            ema.decay = self._decay(gamma)
            ema.update(transformer)

        if self._calls % self.checkpoint_every == 0:
            self._save_snapshots()

    def _save_snapshots(self) -> None:
        if self._snapshot_root is None:
            raise RuntimeError("PostHocEMA callback has not been initialized")
        rank_dir = self._snapshot_root / f"rank_{self._rank:05d}"
        for index, ema in enumerate(self._ema_models):
            path = rank_dir / f"{index}.{self._calls}.pt"
            temporary = path.with_suffix(".pt.tmp")
            torch.save(ema.state_dict(), temporary)
            os.replace(temporary, path)
        logger.info("Saved PostHocEMA shard snapshots at step %d", self._calls)

    def _snapshot_records(
        self,
        *,
        max_step: int | None,
    ) -> list[tuple[Path, int, float]]:
        if self._snapshot_root is None:
            return []
        rank_dir = self._snapshot_root / f"rank_{self._rank:05d}"
        records: list[tuple[Path, int, float]] = []
        for path in rank_dir.glob("*.pt"):
            try:
                profile_index, timestep = map(int, path.stem.split("."))
                gamma = self.gammas[profile_index]
            except (ValueError, IndexError):
                continue
            if max_step is None or timestep <= max_step:
                records.append((path, timestep, gamma))
        return sorted(records, key=lambda item: (item[1], item[0].name))

    def synthesize_local_shard(
        self,
        *,
        sigma_rel: float | None = None,
        step: int | None = None,
    ) -> dict[str, torch.Tensor] | None:
        target_sigma = self.default_output_sigma if sigma_rel is None else float(sigma_rel)
        records = self._snapshot_records(max_step=step)
        if not records:
            return None
        target_step = max(record[1] for record in records) if step is None else int(step)
        if target_step > max(record[1] for record in records):
            raise ValueError("Cannot synthesize PostHocEMA beyond the newest snapshot")

        weights = solve_posthoc_weights(
            torch.tensor([record[1] for record in records]),
            torch.tensor([record[2] for record in records]),
            target_step,
            sigma_rel_to_gamma(target_sigma),
        )
        synthesized: dict[str, torch.Tensor] = {}
        for (path, _, _), weight in zip(records, weights.tolist(), strict=True):
            state = torch.load(path, map_location="cpu", weights_only=True)
            if not isinstance(state, dict):
                raise ValueError(f"Invalid PostHocEMA snapshot: {path}")
            for name, tensor in state.items():
                if not isinstance(tensor, torch.Tensor):
                    continue
                if name not in synthesized:
                    synthesized[name] = tensor.float().mul(float(weight))
                else:
                    synthesized[name].add_(tensor.float(), alpha=float(weight))
        return synthesized

    @contextlib.contextmanager
    def ema_context(
        self,
        transformer: torch.nn.Module,
    ) -> Generator[torch.nn.Module, None, None]:
        synthesized = self.synthesize_local_shard()
        if synthesized is None:
            yield transformer
            return
        temporary_ema = EMA_FSDP(transformer, decay=0.0, mode="local_shard")
        temporary_ema.load_state_dict(synthesized)
        with temporary_ema.apply_to_model(transformer):
            yield transformer

    def on_train_end(
        self,
        method: TrainingMethod,
        iteration: int = 0,
    ) -> None:
        del method, iteration
        if not self._initted or self._snapshot_root is None:
            return

        # Upstream MMAudio synthesizes the sigma=0.05 model after training.
        # Save the latest profiles even when a short/early-stopped run did not
        # end exactly on checkpoint_every, then synthesize each FSDP shard.
        self._save_snapshots()
        synthesized = self.synthesize_local_shard(
            sigma_rel=self.default_output_sigma,
            step=self._calls,
        )
        if synthesized is None:
            return
        rank_dir = self._snapshot_root / f"rank_{self._rank:05d}"
        sigma_name = str(self.default_output_sigma).replace(".", "p")
        path = rank_dir / (f"synthesized_sigma_{sigma_name}_step_{self._calls:09d}.pt")
        temporary = path.with_suffix(".pt.tmp")
        torch.save(synthesized, temporary)
        os.replace(temporary, path)
        logger.info("Saved synthesized PostHocEMA shard to %s", path)

    def state_dict(self) -> dict[str, Any]:
        return {
            "calls": self._calls,
            "initted": self._initted,
            "profiles": {
                str(index): ema.state_dict()
                for index, ema in enumerate(self._ema_models)
            },
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self._calls = int(state_dict.get("calls", 0))
        self._initted = bool(state_dict.get("initted", False))
        profiles = state_dict.get("profiles", {})
        if isinstance(profiles, dict):
            for index, ema in enumerate(self._ema_models):
                profile = profiles.get(str(index))
                if isinstance(profile, dict):
                    ema.load_state_dict(profile)


__all__ = [
    "PostHocEMACallback",
    "sigma_rel_to_gamma",
    "solve_posthoc_weights",
]
