"""Disaggregated program execution (design_v3 §13; design.md §6.3.4; sglang multimodal_gen).

Runs a Program across role pools: each node executes on its assigned pool's resident instance, and
cross-pool edges move slots through the producing pool's connector (with credit flow-control +
``chunk_ready`` readiness + a TransferManifest). A request occupies one pool at a time (capacity-aware
dispatch) and walks a state machine waiting → running:<role> → done. Output is bit-identical to inline
execution (same weights, same data) — verified by a test.

Exposes the same ``tick()``/``done``/``output()`` surface as the inline ``ProgramRunner`` so the
AsyncEngine drives inline and disaggregated requests uniformly.
"""
from __future__ import annotations

from typing import Any

from .._enums import ExecutionProfile
from ..loop.driver import LoopRunner
from ..program.specs import ComponentNode, ModelLoopNode, Program
from ..request.artifacts import Output
from ..request.cancel import CancelKind, CancelScope
from ..request.streams import Stream
from ..transport import TransferManifest, payload_nbytes
from .context import RuntimeLoopContext
from .engine import _to_artifact


class DisaggregatedRunner:
    def __init__(self, pools, program: Program, request: Any, *, observers, interceptors):
        self.pools = pools
        self.program = program
        self.request = request
        self.observers = observers
        self.interceptors = interceptors
        self.slots: dict[str, Any] = {}
        self.slot_pool: dict[str, str] = {}          # slot name -> pool_id currently holding it
        self.metrics: dict[str, float] = {}
        self.transfers = 0
        self.node_idx = 0
        self.done = False
        self.state = "waiting"
        self.error: str | None = None
        self.cancel_scope = CancelScope(kind=CancelKind.REQUEST, target_id=request.request_id)
        self.stream = Stream(request.request_id)
        self._cur_pool = None
        self._ctxs: dict[str, RuntimeLoopContext] = {}
        self.loop_runner: LoopRunner | None = None

    # --- the same progress/lifecycle surface the AsyncEngine drives ----------- #
    @property
    def _progress(self) -> tuple:
        sidx = self.loop_runner.state.step_idx if self.loop_runner is not None else -1
        return (self.node_idx, sidx, self._cur_pool.pool_id if self._cur_pool else "")

    def _ctx_for(self, pool) -> RuntimeLoopContext:
        if pool.pool_id not in self._ctxs:
            self._ctxs[pool.pool_id] = RuntimeLoopContext(
                pool.instance, observers=self.observers, interceptors=self.interceptors,
                slots=self.slots, stream=self.stream, cancel_scope=self.cancel_scope,
                profile=ExecutionProfile.SERVE, metrics=self.metrics, request_id=self.request.request_id)
        return self._ctxs[pool.pool_id]

    def _leave_current(self) -> None:
        if self._cur_pool is not None:
            self._cur_pool.leave()
            self._cur_pool = None

    def _ensure_reads(self, node, pool) -> None:
        """Move any read-slots that live on another pool to this pool, via the producer's connector."""
        for r in node.reads:
            holder_id = self.slot_pool.get(r)
            if holder_id and holder_id != pool.pool_id and r in self.slots:
                src = self.pools.by_id[holder_id]
                got = src.connector.acquire_credit()                  # credit-based flow control
                manifest = TransferManifest(keys=(r,), producer_id=holder_id, consumer_id=pool.pool_id,
                                            src_location=src.connector.name, dst_location=pool.connector.name,
                                            nbytes=payload_nbytes(self.slots.get(r)))
                src.connector.put(r, self.slots.get(r), manifest)     # producer publishes (chunk_ready)
                if src.connector.chunk_ready(r):                       # readiness signal
                    self.slots[r] = src.connector.take(r)              # consumer fetches
                if got:
                    src.connector.release_credit()
                self.slot_pool[r] = pool.pool_id
                self.transfers += 1
                self.metrics["transfers"] = self.metrics.get("transfers", 0) + 1

    def _commit_node_writes(self, node, pool) -> None:
        for w in node.writes:
            self.slot_pool[w] = pool.pool_id

    def tick(self) -> bool:
        if self.done:
            return True
        nodes = self.program.active_nodes(self.request)
        while True:
            if self.node_idx >= len(nodes):
                self._leave_current()
                self.done = True
                self.state = "done"
                return True
            node = nodes[self.node_idx]
            pool = self.pools.pool_for(node.node_id)

            if pool is not self._cur_pool:                 # switching pools: capacity-aware dispatch
                if not pool.can_admit():
                    self.state = f"waiting:{pool.role}"
                    return False                            # backpressure — pool at capacity
                self._leave_current()
                pool.enter()
                self._cur_pool = pool
                self.state = f"running:{pool.role}"

            self._ensure_reads(node, pool)

            if isinstance(node, ComponentNode):
                if node.fn is not None:
                    node.fn(pool.instance, self.slots, self.request, self._ctx_for(pool))
                self._commit_node_writes(node, pool)
                self.node_idx += 1
                continue

            if isinstance(node, ModelLoopNode):
                if self.loop_runner is None:
                    self.loop_runner = LoopRunner(pool.instance.loop(node.loop_id),
                                                  self._ctx_for(pool), self.request, pool.instance)
                plan = self.loop_runner.peek()
                if plan is None:
                    self._finish_loop(node, pool)
                    return False
                self.loop_runner.step()
                self.metrics["stepped_units"] = self.metrics.get("stepped_units", 0) + 1
                if self.loop_runner.done:
                    self._finish_loop(node, pool)
                return False                                # yield one loop step (for interleaving)

            self.node_idx += 1

    def _finish_loop(self, node, pool) -> None:
        if self.loop_runner is not None and self.loop_runner.result is not None:
            self.slots[node.output_slot] = self.loop_runner.result.outputs
            self._commit_node_writes(node, pool)
            for k, v in self.loop_runner.result.metrics.items():
                self.metrics[k] = self.metrics.get(k, 0.0) + v
        self.loop_runner = None
        self.node_idx += 1

    def run_to_completion(self) -> None:
        stuck = 0
        while not self.done:
            p0 = self._progress
            self.tick()
            if not self.done and self._progress == p0:
                stuck += 1
                if stuck > 3:                               # single-runner: a stall means real OOM/capacity-0
                    raise RuntimeError(f"disaggregated request {self.request.request_id} stalled "
                                       f"(state={self.state}); a pool cannot admit it")
            else:
                stuck = 0

    def output(self) -> Output:
        artifacts = {name: _to_artifact(name, self.slots.get(slot), producer=slot)
                     for name, slot in self.program.output_artifacts.items()}
        return Output(request_id=self.request.request_id, artifacts=artifacts,
                      metrics=dict(self.metrics, transfers=float(self.transfers)), error=self.error)
