"""DeploymentCard — what an engine exports to a fleet (design_v3 §14; designv2 Deployment plane).

> The engine exports a ``DeploymentCard`` and lets a fleet orchestrator route. ... The cost model is
> the SAME §6 object — one object, two consumers (the scheduler's budget and the fleet's routing input).

This is the contract our OWN fleet (``deploy/fleet.py``) consumes AND the Dynamo adapter
(``deploy/dynamo.py``) exports — so we are frontable by Dynamo without depending on it.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .._enums import Capability
from ..card import CostModel


@dataclass
class HealthSchema:
    status: str = "healthy"               # "healthy" | "draining" | "unhealthy"
    in_flight: int = 0
    queue_depth: int = 0


@dataclass
class SLOSchema:
    slo_class: str = "standard"           # "latency" | "throughput" | "cost"
    max_concurrent: int = 8


@dataclass
class DeploymentCard:
    engine_id: str
    model_cards: list[str] = field(default_factory=list)
    capabilities: frozenset[Capability] = frozenset()
    role_pools: list = field(default_factory=list)            # list[RolePoolSpec]
    supported_programs: list[str] = field(default_factory=list)
    cost_model: CostModel | None = None                       # the SAME §6 cost model
    health: HealthSchema = field(default_factory=HealthSchema)
    slo: SLOSchema = field(default_factory=SLOSchema)

    def serves(self, model_id: str) -> bool:
        return model_id in self.model_cards


def build_deployment_card(engine_id: str, model_cards: list, *, max_concurrent: int = 8,
                          slo_class: str = "standard", role_pools: list | None = None,
                          supported_programs: list[str] | None = None) -> DeploymentCard:
    """Export a DeploymentCard from the model cards an engine serves (design_v3 §14).

    Picks a representative ``step_cost_model`` so the fleet/Dynamo route on the SAME cost object the
    scheduler budgets with."""
    caps: set = set()
    cost_model = None
    ids: list[str] = []
    for c in model_cards:
        ids.append(c.model_id)
        caps |= set(c.capabilities.capabilities)
        if cost_model is None:
            for lp in c.loops.values():
                if lp.step_cost_model is not None:
                    cost_model = lp.step_cost_model
                    break
    return DeploymentCard(
        engine_id=engine_id, model_cards=ids, capabilities=frozenset(caps),
        role_pools=role_pools or [], supported_programs=supported_programs or [],
        cost_model=cost_model, slo=SLOSchema(slo_class=slo_class, max_concurrent=max_concurrent))
