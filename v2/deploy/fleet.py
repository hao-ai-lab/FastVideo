"""LocalFleet — OUR OWN fleet router.

Dynamo is the first-class fleet partner, but every Dynamo ask has a first-class fallback — and this
is it: a self-contained fleet that does discovery, health/drain, and routing (least-loaded /
cost-model / affinity) over multiple engine workers, so we are never *reliant* on Dynamo. The
router's cost input is the SAME cost model the scheduler uses (one object, two consumers). Affinity
routing is sticky-by-key for checkpoint/session residency (least-loaded + engine redirects).
"""
from __future__ import annotations

from typing import Any
from collections.abc import AsyncIterator

from v2.request.artifacts import Output
from v2.deploy.card import DeploymentCard, HealthSchema


class NoWorkerAvailable(RuntimeError):
    pass


class Worker:
    """A registered engine worker (an AsyncEngine + its exported DeploymentCard)."""

    def __init__(self, worker_id: str, engine: Any, card: DeploymentCard):
        self.worker_id = worker_id
        self.engine = engine
        self.card = card
        self.draining = False

    @property
    def load(self) -> float:
        return self.engine.in_flight / max(1, self.card.slo.max_concurrent)

    @property
    def healthy(self) -> bool:
        return not self.draining and self.engine.in_flight < self.card.slo.max_concurrent * 4

    def serves(self, model_id: str) -> bool:
        return self.engine.serves(model_id) or self.card.serves(model_id)

    def cost_estimate(self, request: Any) -> float:
        """Predicted GPU-time for this request — the cost model as the fleet's routing input."""
        cm = self.card.cost_model
        steps = max(1, int(getattr(request.diffusion, "num_steps", 1) or 1))
        work = max(1, int(getattr(request.diffusion, "height", 1)) * int(getattr(request.diffusion, "width", 1)))
        per = cm.predict(work) if cm is not None else 1e-3
        return steps * per

    def health(self) -> HealthSchema:
        return HealthSchema(status=("draining" if self.draining else "healthy"),
                            in_flight=self.engine.in_flight,
                            queue_depth=self.engine.queue_depth)


class LocalFleet:

    def __init__(self, policy: str = "least_loaded", *, max_affinity: int = 100_000):
        assert policy in ("least_loaded", "cost", "affinity")
        self.policy = policy
        self.max_affinity = max_affinity
        self.workers: dict[str, Worker] = {}
        self._affinity: dict[str, str] = {}  # affinity key -> worker_id (sticky), FIFO-bounded

    # --- discovery / health (what Dynamo's registry + planner would do) ------ #
    def register(self, worker_id: str, engine: Any, card: DeploymentCard) -> Worker:
        w = Worker(worker_id, engine, card)
        self.workers[worker_id] = w
        return w

    def deregister(self, worker_id: str) -> None:
        self.workers.pop(worker_id, None)

    def drain(self, worker_id: str) -> None:
        if worker_id in self.workers:
            self.workers[worker_id].draining = True

    def health(self) -> dict[str, HealthSchema]:
        return {wid: w.health() for wid, w in self.workers.items()}

    # --- routing (least-loaded / cost / affinity) ---------------------------- #
    def _candidates(self, request: Any) -> list[Worker]:
        return [w for w in self.workers.values() if w.serves(request.model_id) and w.healthy]

    def route(self, request: Any, *, affinity_key: str | None = None) -> Worker:
        cands = self._candidates(request)
        if not cands:
            raise NoWorkerAvailable(f"no healthy worker serves model {request.model_id!r}")
        if self.policy == "affinity":
            key = affinity_key or request.model_id
            wid = self._affinity.get(key)
            if wid in self.workers and self.workers[wid] in cands:
                return self.workers[wid]
            chosen = min(cands, key=lambda w: w.load)  # cold key → least-loaded, then pin
            if len(self._affinity) >= self.max_affinity:  # FIFO-bound the sticky map
                self._affinity.pop(next(iter(self._affinity)), None)
            self._affinity[key] = chosen.worker_id
            return chosen
        if self.policy == "cost":
            return min(cands, key=lambda w: w.cost_estimate(request) * (1.0 + w.load))
        return min(cands, key=lambda w: w.load)  # least_loaded (default)

    # --- serving (delegates to the chosen worker's engine) ------------------- #
    async def generate(self, request: Any, *, affinity_key: str | None = None) -> Output:
        return await self.route(request, affinity_key=affinity_key).engine.generate(request)

    async def submit(self, request: Any, *, affinity_key: str | None = None) -> AsyncIterator:
        worker = self.route(request, affinity_key=affinity_key)
        async for ev in worker.engine.submit(request):
            yield ev
