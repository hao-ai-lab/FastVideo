"""TrainingMethod base (design_v3 §10; mirrors the repo's train/methods contract).

The load-bearing principle: every method's *rollout* drives the SAME loop the engine serves
(via ``_rollout`` → ``rollout_loop``), so there is one numerics surface. Methods differ only in
loss math, roles, and capture — not in a second sampler. ``manages_optimization`` matches the
landed ``TrainingMethod.manages_optimization()`` seam (RL owns its sample→score→inner-train cadence).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from v2._enums import ConsistencyLevel, ExecutionProfile
from v2.cache import CacheManager
from v2.card import load_card
from v2.recipes.common import text_encode_node_fn
from v2.training.rollout import rollout_loop


def new_instance(card) -> Any:
    """A fresh resident instance of a card (a training role: student/teacher/critic/old/reference)."""
    return load_card(card, cache_manager=CacheManager.from_card(card), validate=False)


def predict_x0(velocity: np.ndarray, noised: np.ndarray, sigma: float) -> np.ndarray:
    """Flow prediction → clean sample: x0 = x_σ − σ·v (Wan ``x0 = x_t − σ·model_output``)."""
    return noised - sigma * np.asarray(velocity, dtype="float32")


class TrainingMethod(ABC):
    name: str = "base"
    consistency: ConsistencyLevel = ConsistencyLevel.C1
    student_loop_id: str = "diffusion_denoise"

    def __init__(self, student_instance: Any, *, lr: float = 0.05):
        self.student = student_instance
        self.lr = lr
        self.iteration = 0

    @property
    def student_dit(self) -> Any:
        return self.student.component("transformer")

    def manages_optimization(self) -> bool:
        return False

    def get_grad_clip_targets(self, iteration: int = 0) -> dict[str, Any]:
        return {"student": self.student_dit}

    def consistency_level(self) -> ConsistencyLevel:
        return self.consistency

    # --- rollout helpers: the SAME loop the engine serves ---------------------- #
    def _encode_slots(self, request, instance: Any = None) -> dict:
        slots: dict = {}
        text_encode_node_fn(instance or self.student, slots, request, None)
        return slots

    def _rollout(self, request, *, instance: Any = None, loop_id: str | None = None):
        inst = instance or self.student
        slots = self._encode_slots(request, inst)
        return rollout_loop(inst, loop_id or self.student_loop_id, request, slots=slots,
                            profile=ExecutionProfile.ROLLOUT)

    @abstractmethod
    def train_step(self, batch: dict, iteration: int) -> tuple[dict, dict]:
        """Return (loss_map, metrics). loss_map values are floats; metrics includes grad_norm/*."""
