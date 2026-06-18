"""Role / stage pools (design_v3 §13, design.md §6.3.4; port of sglang multimodal_gen disagg).

A ``RolePool`` is a single-node worker pool for one role (encoder / denoiser / decoder / …) holding
its OWN resident ModelInstance (a component subset), its OWN cache manager, an outbound connector,
and a capacity (max concurrent requests → capacity-aware dispatch). ``DeployConfig`` maps program
nodes onto pools. Pools are single-node; multi-node scale is *multiple pools* fronted by the fleet
(§14) — never a cross-node mesh inside one pool.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from v2.cache import CacheManager
from v2.card import load_card
from v2.transport import Connector, make_connector


@dataclass
class RolePoolSpec:
    role: str                                   # "encoder" | "denoiser" | "decoder" | ...
    pool_id: str
    components: tuple[str, ...] = ()             # components this pool is responsible for
    capacity: int = 2                            # max concurrent requests occupying this pool
    replicas: int = 1
    connector: str = "in_proc"                   # outbound transport backend
    slo_class: str = "throughput"                # "latency" | "throughput" | "cost"


@dataclass
class DeployConfig:
    """Maps a Program's nodes onto role pools (the vllm-omni deploy-YAML analog, §6.3.4)."""
    pools: list[RolePoolSpec] = field(default_factory=list)
    placement: dict[str, str] = field(default_factory=dict)   # node_id -> pool_id

    def pool_id_for(self, node_id: str, default: str | None = None) -> str:
        return self.placement.get(node_id, default or (self.pools[0].pool_id if self.pools else ""))


class RolePool:
    def __init__(self, spec: RolePoolSpec, card: Any):
        self.spec = spec
        self.role = spec.role
        self.pool_id = spec.pool_id
        self.components = set(spec.components)
        self.capacity = spec.capacity
        self.instance = load_card(card, cache_manager=CacheManager.from_card(card), validate=False)
        self.connector: Connector = make_connector(spec.connector, name=spec.pool_id)
        self.in_flight = 0
        self.served = 0
        self.transfers_out = 0

    def can_admit(self) -> bool:
        return self.in_flight < self.capacity

    def enter(self) -> None:
        self.in_flight += 1

    def leave(self) -> None:
        if self.in_flight > 0:
            self.in_flight -= 1
        self.served += 1

    @property
    def utilization(self) -> float:
        return self.in_flight / max(1, self.capacity)

    def stats(self) -> dict[str, Any]:
        return {"role": self.role, "in_flight": self.in_flight, "capacity": self.capacity,
                "served": self.served, "transfers_out": self.connector.transfers,
                "credits_available": self.connector.credits.available}


class PoolSet:
    """The collection of role pools for one deployment + the node→pool placement."""

    def __init__(self, deploy: DeployConfig, card: Any):
        self.deploy = deploy
        self.card = card
        self.by_id: dict[str, RolePool] = {s.pool_id: RolePool(s, card) for s in deploy.pools}
        if not self.by_id:
            raise ValueError("DeployConfig declares no pools")

    def pool_for(self, node_id: str) -> RolePool:
        return self.by_id[self.deploy.pool_id_for(node_id)]

    def warmup(self) -> dict[str, int]:
        """Instantiate each pool's components ahead of first request (design.md §6.3.5 warmup)."""
        built: dict[str, int] = {}
        for pool in self.by_id.values():
            for cid in pool.components:
                if pool.instance.has_component(cid):
                    pool.instance.component(cid)
            built[pool.pool_id] = len(pool.components)
        return built

    def stats(self) -> dict[str, Any]:
        return {pid: p.stats() for pid, p in self.by_id.items()}


def wan_t2v_disaggregated() -> DeployConfig:
    """The canonical N:M:K split for Wan T2V (encoder → denoiser → decoder)."""
    return DeployConfig(
        pools=[
            RolePoolSpec(role="encoder", pool_id="enc", components=("text_encoder",), capacity=4,
                         slo_class="latency"),
            RolePoolSpec(role="denoiser", pool_id="den", components=("transformer",), capacity=1,
                         slo_class="throughput"),       # jumbo video step stays batch-of-1
            RolePoolSpec(role="decoder", pool_id="dec", components=("vae",), capacity=2),
        ],
        placement={"text_encode": "enc", "denoise": "den", "vae_decode": "dec"},
    )
