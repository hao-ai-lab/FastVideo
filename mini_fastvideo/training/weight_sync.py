"""WeightSyncPlan — versioned weight transfer with a declared role (design_v3 §10).

> RL ships a *role*, not "the weights": student / EMA / decay-blended old policy is declared
> (the landed NFT behavior policy is the *old* copy, not the student — the plan must carry that).

The lifecycle (design_v3 §10): freeze admission → drain/boundary-stop → transfer → bump
``weights_version`` → invalidate incompatible caches/graphs → publish → resume. Computed from
three inputs (mesh specs + per-model layout adapters + transport), validated pre-flight,
CPU-testable on fake pools. Here the "transfer" copies/blends the toy DiT params.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class WeightRole(str, Enum):
    STUDENT = "student"          # the current learnable weights
    EMA = "ema"                  # exponential moving average of student
    OLD_POLICY = "old_policy"    # decay-blended behavior policy (NFT samples from THIS, not student)
    REFERENCE = "reference"      # frozen KL anchor
    TEACHER = "teacher"          # frozen distillation teacher
    CRITIC = "critic"            # trainable fake-score / value head


_ver = itertools.count(1)


@dataclass
class WeightSyncPlan:
    role: WeightRole
    # three inputs (design_v3 §10): mesh specs (trainer↔engine), layout adapters, transport
    mesh_specs: dict[str, Any] = field(default_factory=dict)
    layout_adapters: dict[str, Any] = field(default_factory=dict)
    transport: str = "in_proc"                 # in_proc | shm | nccl | rdma
    decay: float = 0.0                         # for EMA / OLD_POLICY blend

    def apply(self, src_dit: Any, dst_dit: Any, dst_instance: Any = None) -> str:
        """Execute the lifecycle. Returns the new weights_version published on dst_instance."""
        # 1) freeze admission + 2) drain/boundary-stop are scheduler ops (no-op in single-thread mini)
        # 3) transfer per role
        if self.role in (WeightRole.EMA, WeightRole.OLD_POLICY):
            dst_dit.blend_from(src_dit, self.decay)     # decay·dst + (1-decay)·src
        else:
            dst_dit.copy_from(src_dit)                  # hard copy (student push / reference init)
        # 4) bump version + 5) invalidate caches/graphs + 6) publish (set_weights_version does 4-6)
        version = f"w{next(_ver)}"
        if dst_instance is not None:
            dst_instance.set_weights_version(version)
        return version
