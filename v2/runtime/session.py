"""WorldModelSession — a long-lived interactive world-model rollout (design_v3 §16, §12, §15e).

The realtime/interactive plane, made concrete. A one-shot ``engine.run`` generates a clip and forgets;
a *session* continues the world: each ``.act(action)`` resumes the causal ``chunk_rollout`` loop from
the **persistent world state** carried on the ``Session`` (its ``kv_handle``), streams frames as chunks
complete, and honors **step-boundary cancellation**. This exercises the seams a one-shot path never
touches — cross-request persistent context, the ``CHUNK_ROLLOUT`` loop driven incrementally, the
``Stream`` frame channel, and ``CancelScope`` — using the exact runtime primitives the engine uses
(``RuntimeLoopContext`` + ``LoopRunner``), so interleaving two sessions is still parity-safe.

Cancellation is transactional: the persistent world state advances only on a *completed* act, so a
cancelled rollout leaves the session resumable from the last good frame.
"""
from __future__ import annotations

from typing import Any, Callable

from v2._enums import ExecutionProfile
from v2.loop.driver import LoopRunner
from v2.recipes.common import text_encode_node_fn
from v2.request import DiffusionParams, TaskType, make_request
from v2.request.requests import Session
from v2.request.streams import Stream
from v2.runtime.context import RuntimeLoopContext


class WorldModelSession:
    def __init__(self, engine: Any, model_id: str, *, session_id: str = "world",
                 window: int = 2, loop_id: str = "chunk_rollout"):
        self.engine = engine
        self.model_id = model_id
        self.window = window                 # how many prior chunks of world state to carry forward
        self.loop_id = loop_id
        self.session = Session(session_id=session_id)
        self.stream = Stream(session_id)
        self.session.streams["video"] = self.stream
        self.world_context: list = []        # the persistent world chunks (the §16 cross-request KV)
        self.session.kv_handle = self.world_context
        self.step = 0

    @property
    def instance(self) -> Any:
        if self.model_id not in self.engine._registry:
            raise KeyError(f"no model {self.model_id!r} registered on engine")
        return self.engine._registry[self.model_id][0]

    def act(self, action: str, *, seed: int | None = None,
            step_hook: Callable[["WorldModelSession", int], None] | None = None) -> Any:
        """Advance the world by one action, continuing from persistent context; stream frames.

        Raises ``request.Cancelled`` if cancelled at a step boundary — and because the world state is
        only committed after the loop completes, a cancelled act leaves the session resumable."""
        inst = self.instance
        req = make_request(TaskType.V2W, self.model_id, action,
                           diffusion=DiffusionParams(seed=self.step if seed is None else seed))
        slots: dict[str, Any] = {}
        text_encode_node_fn(inst, slots, req, None)
        slots["world_context"] = list(self.world_context)        # thread the persistent state in
        ctx = RuntimeLoopContext(
            inst, observers=self.engine.observers, interceptors=self.engine.interceptors,
            slots=slots, stream=self.stream, cancel_scope=self.session.cancel_scope,
            profile=ExecutionProfile.SERVE, metrics={},
            request_id=f"{self.session.session_id}.{self.step}")
        runner = LoopRunner(inst.loop(self.loop_id), ctx, req, inst)
        n = 0
        while not runner.done:
            if step_hook is not None:
                step_hook(self, n)                 # test seam: may call self.cancel() at a boundary
            runner.step()                          # peek() raises request.Cancelled if cancelled
            n += 1
        result = runner.result
        new_chunks = list(result.outputs.get("chunks") or [])
        self.world_context = (self.world_context + new_chunks)[-self.window:]   # slide window; commit
        self.session.kv_handle = self.world_context
        self.step += 1
        return result

    def frames(self) -> list:
        """All streamed media chunks so far across the session (the realtime frame channel)."""
        return [e for e in self.stream.events if e.type == "media.chunk"]

    def cancel(self) -> None:
        self.session.cancel_scope.cancel()
