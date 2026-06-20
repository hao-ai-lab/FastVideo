"""DeploymentCard — what an engine exports to a fleet.

The engine exports a ``DeploymentCard`` and lets a fleet orchestrator route (by capability + load;
pooled run-to-completion prices nothing, so there is no cost model).

This is the contract our OWN fleet (``deploy/fleet.py``) consumes AND the Dynamo adapter
(``deploy/dynamo.py``) exports — so we are frontable by Dynamo without depending on it.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from v2.core.enums import Capability


@dataclass
class HealthSchema:
    status: str = "healthy"  # "healthy" | "draining" | "unhealthy"
    in_flight: int = 0
    queue_depth: int = 0


@dataclass
class SLOSchema:
    slo_class: str = "standard"  # "latency" | "throughput" | "cost"
    max_concurrent: int = 8


@dataclass
class DeploymentCard:
    engine_id: str
    model_cards: list[str] = field(default_factory=list)
    capabilities: frozenset[Capability] = frozenset()
    role_pools: list = field(default_factory=list)  # list[RolePoolSpec]
    supported_programs: list[str] = field(default_factory=list)
    health: HealthSchema = field(default_factory=HealthSchema)
    slo: SLOSchema = field(default_factory=SLOSchema)

    def serves(self, model_id: str) -> bool:
        return model_id in self.model_cards


def build_deployment_card(engine_id: str,
                          model_cards: list,
                          *,
                          max_concurrent: int = 8,
                          slo_class: str = "standard",
                          role_pools: list | None = None,
                          supported_programs: list[str] | None = None) -> DeploymentCard:
    """Export a DeploymentCard from the model cards an engine serves."""
    caps: set = set()
    ids: list[str] = []
    for c in model_cards:
        ids.append(c.model_id)
        caps |= set(c.capabilities.capabilities)
    return DeploymentCard(engine_id=engine_id,
                          model_cards=ids,
                          capabilities=frozenset(caps),
                          role_pools=role_pools or [],
                          supported_programs=supported_programs or [],
                          slo=SLOSchema(slo_class=slo_class, max_concurrent=max_concurrent))
