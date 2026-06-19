"""Dynamo worker adapter — Dynamo as an *option*, not a dependency.

NVIDIA Dynamo is the named first-class partner for the fleet layer; the engine's job is to be a good
Dynamo citizen: registration, health/drain, cost metrics, affinity events.

This adapter exposes exactly that contract over an AsyncEngine + DeploymentCard, so Dynamo CAN front
this engine. It consumes the same DeploymentCard + cost model as our own LocalFleet — one object, two
consumers — so choosing Dynamo vs. our fleet is a deployment decision, not a rewrite.
``FakeDynamoRuntime`` proves the contract is satisfiable end-to-end without importing Dynamo.
"""
from __future__ import annotations

from typing import Any

from v2.request.artifacts import Output
from v2.deploy.card import DeploymentCard


class DynamoWorkerAdapter:
    """Implements the Dynamo worker surface: registration, metrics, affinity, handle."""

    def __init__(self, engine: Any, card: DeploymentCard, *, worker_type: str = "Aggregated"):
        self.engine = engine
        self.card = card
        self.worker_type = worker_type
        self.registered = False
        self.draining = False

    # 1) worker surface — registration payload (ModelType.Videos|Images, roles, endpoints)
    def registration(self) -> dict[str, Any]:
        self.registered = True
        return {
            "engine_id":
            self.card.engine_id,
            "model_type": ["Videos", "Images"] + (["Chat"] if any("omni" in m or "cosmos" in m or "bagel" in m
                                                                  for m in self.card.model_cards) else []),
            "worker_type":
            self.worker_type,
            "models":
            list(self.card.model_cards),
            "capabilities":
            sorted(c.value for c in self.card.capabilities),
            "supported_programs":
            list(self.card.supported_programs),
        }

    # 2) metrics for routing + the SLA Planner (the SAME cost model the scheduler uses)
    def metrics(self) -> dict[str, Any]:
        return {"in_flight": self.engine.in_flight, "queue_depth": self.engine.queue_depth, "draining": self.draining}

    def cost_estimate(self, request: Any) -> float:
        cm = self.card.cost_model
        steps = max(1, int(getattr(request.diffusion, "num_steps", 1) or 1))
        work = max(1, int(getattr(request.diffusion, "height", 1)) * int(getattr(request.diffusion, "width", 1)))
        return steps * (cm.predict(work) if cm is not None else 1e-3)

    # health / graceful drain wired to the engine
    def health(self) -> dict[str, Any]:
        return {"status": "draining" if self.draining else "healthy", **self.metrics()}

    def drain(self) -> None:
        self.draining = True

    # 3) affinity / cache events (KvCacheEventData shape) — checkpoint/session residency
    def cache_event(self, kind: str, key: str) -> dict[str, Any]:
        return {"event": kind, "engine_id": self.card.engine_id, "key": key}

    # the worker entrypoint Dynamo's router calls
    async def handle(self, request: Any) -> Output:
        if self.draining:
            raise RuntimeError(f"worker {self.card.engine_id} is draining")
        return await self.engine.generate(request)


class FakeDynamoRuntime:
    """A minimal stand-in for Dynamo: registers worker adapters and routes by the published cost model
    + health. Demonstrates the engine satisfies the Dynamo contract WITHOUT a Dynamo dependency."""

    def __init__(self) -> None:
        self.workers: list[DynamoWorkerAdapter] = []
        self.registry: list[dict] = []

    def register_worker(self, adapter: DynamoWorkerAdapter) -> None:
        self.registry.append(adapter.registration())
        self.workers.append(adapter)

    def _route(self, request: Any) -> DynamoWorkerAdapter:
        cands = [w for w in self.workers if not w.draining and request.model_id in w.card.model_cards]
        if not cands:
            raise RuntimeError(f"no Dynamo worker serves {request.model_id!r}")
        # the SLA-planner-style choice: cheapest predicted cost, tie-broken by least in-flight
        return min(cands, key=lambda w: (w.cost_estimate(request), w.engine.in_flight))

    async def generate(self, request: Any) -> Output:
        return await self._route(request).handle(request)
