"""WeightSyncPlan — versioned weight transfer with a declared role.

A sync ships a *role*, not just "the weights": student / EMA / decay-blended old policy is declared
(e.g. an NFT behavior policy is the *old* copy, not the student — the plan must carry that).

Lifecycle: freeze admission -> drain/boundary-stop -> transfer -> bump ``weights_version`` ->
invalidate incompatible caches/graphs -> publish -> resume. Computed from three inputs (mesh specs
+ per-model layout adapters + transport), validated pre-flight, CPU-testable on fake pools. Here the
"transfer" copies/blends the toy DiT params.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class WeightRole(str, Enum):
    STUDENT = "student"  # the current learnable weights
    EMA = "ema"  # exponential moving average of student
    OLD_POLICY = "old_policy"  # decay-blended behavior policy (NFT samples from THIS, not student)
    REFERENCE = "reference"  # frozen KL anchor
    TEACHER = "teacher"  # frozen distillation teacher
    CRITIC = "critic"  # trainable fake-score / value head


_ver = itertools.count(1)


@dataclass
class WeightSyncPlan:
    role: WeightRole
    # three inputs: mesh specs (trainer<->engine), layout adapters, transport
    mesh_specs: dict[str, Any] = field(default_factory=dict)
    layout_adapters: dict[str, Any] = field(default_factory=dict)
    transport: str = "in_proc"  # in_proc | shm | nccl | rdma
    decay: float = 0.0  # for EMA / OLD_POLICY blend
    # which component(s) this plan syncs — versioned + cache-invalidated in isolation.
    # Joint RL (UniRL) ships ONE plan per trainable expert: ``("transformer",)`` and ``("llm",)`` —
    # so the LM weight sync does NOT flush the frozen text-encoder's feature cache, and vice-versa.
    components: tuple[str, ...] = ("transformer", )

    def apply(self, src_dit: Any, dst_dit: Any, dst_instance: Any = None) -> str:
        """Execute the lifecycle. Returns the new weights_version published on dst_instance."""
        # 1) freeze admission + 2) drain/boundary-stop are scheduler ops (no-op in single-thread mini)
        # 3) transfer per role
        if self.role in (WeightRole.EMA, WeightRole.OLD_POLICY):
            dst_dit.blend_from(src_dit, self.decay)  # decay·dst + (1-decay)·src
        else:
            dst_dit.copy_from(src_dit)  # hard copy (student push / reference init)
        # 4) bump version + 5) invalidate ONLY this plan's component caches/graphs + 6) publish.
        # Scoped to ``self.components`` so frozen text/vision encoders keep their feature caches,
        # and the two experts of a joint-RL update version independently.
        version = f"w{next(_ver)}"
        if dst_instance is not None:
            dst_instance.set_weights_version(version, components=list(self.components))
        return version


class WeightSyncController:
    """Hot weight-sync lifecycle over a *live, serving* resident instance — the RL flywheel's hardest
    correctness surface (rollout while serving, swap weights, stay correct):

        freeze() admission -> (drain in-flight loops to a boundary) -> sync() transfer+publish -> resume.

    The load-bearing invariant: an in-flight request finishes on the weights it **started** with — the
    sync waits for it to drain, so its trajectory is never a half-and-half of two policies, and its
    captured behavior is stamped with that version (for correct off-policy correction). ``freeze`` gates
    admission so no NEW request starts the synced loop mid-swap; the caller drains the in-flight set,
    then ``sync`` transfers per-component (bump version + invalidate that component's caches only) and
    resumes. This is the engine-side counterpart to a training ``WeightSyncPlan`` — same lifecycle.
    """

    def __init__(self, instance: Any):
        self.instance = instance
        self.frozen = False
        self.synced = 0

    def freeze(self) -> None:
        self.frozen = True

    def can_admit(self) -> bool:
        return not self.frozen  # admission gate: no new starts of the synced loop while frozen

    def sync(self, src_dit: Any, *, component_id: str = "transformer", role: WeightRole = WeightRole.STUDENT) -> str:
        """Transfer new weights into the resident component, bump its version, invalidate ONLY its
        caches, and resume admission. Call AFTER in-flight loops on this instance have drained."""
        version = WeightSyncPlan(role=role, components=(component_id, )).apply(src_dit,
                                                                               self.instance.component(component_id),
                                                                               self.instance)
        self.frozen = False
        self.synced += 1
        return version
