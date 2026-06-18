"""FineTuneMethod — plain flow-matching finetune (design_v3 §10; repo: train/methods/fine_tuning).

Loss (repo finetune.py): flow-matching target ``v = ε − x0``; ``L = MSE(student(x_σ), v)`` where
``x_σ = (1−σ)·x0 + σ·ε``. Data-centric (no rollout) — the simplest (recipe, runtime) consumer.
"""
from __future__ import annotations

import numpy as np

from v2._enums import ConsistencyLevel
from v2.models.backend import LATENT_CHANNELS
from v2.models.common import cached_text_encode
from v2.training.methods.base import TrainingMethod, new_instance


class FineTuneMethod(TrainingMethod):
    name = "finetune"
    consistency = ConsistencyLevel.C1

    def _x0(self, seed: int) -> np.ndarray:
        rng = np.random.default_rng(seed ^ 0x5151)
        return (rng.standard_normal((LATENT_CHANNELS, 2, 4, 6)) * 0.5).astype("float32")

    def train_step(self, batch: dict, iteration: int) -> tuple[dict, dict]:
        prompts = batch["prompts"]
        seeds = batch.get("seeds", list(range(len(prompts))))
        latents = batch.get("latents")
        losses, gnorms = [], []
        for i, (p, s) in enumerate(zip(prompts, seeds)):
            x0 = np.asarray(latents[i], dtype="float32") if latents is not None else self._x0(s)
            emb = cached_text_encode(self.student, p)
            rng = np.random.default_rng(s)
            sigma = float(rng.uniform(0.05, 0.95))
            noise = rng.standard_normal(x0.shape).astype("float32")
            noised = ((1.0 - sigma) * x0 + sigma * noise).astype("float32")
            target = (noise - x0).astype("float32")                 # flow-matching velocity target
            loss, gnorm = self.student_dit.mse_grad_step(noised, emb, sigma, target, self.lr)
            losses.append(loss)
            gnorms.append(gnorm)
        self.iteration = iteration
        return ({"loss": float(np.mean(losses))},
                {"loss": float(np.mean(losses)), "grad_norm/student": float(np.mean(gnorms))})


def build_finetune(card, **kw) -> FineTuneMethod:
    return FineTuneMethod(new_instance(card), **kw)
