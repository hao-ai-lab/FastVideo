"""Programs compose a card's loops into a task.

> The card says what loops *exist*, the program says how to *run* them for this request.

Kinds: InlineProgram (many loops, one resident instance - the omni default),
DisaggregatedProgram, RealtimeProgram, and WorkflowProgram. A Program is an
ordered list of typed nodes that communicate through named slots. ``when=``
predicates select the per-request sub-sequence; the runtime intentionally does
not interpret a separate graph IR.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any
from collections.abc import Callable

from v2.core.request.tasks import TaskType


class ProgramKind(str, Enum):
    INLINE = "inline"
    DISAGGREGATED = "disaggregated"
    REALTIME = "realtime"
    WORKFLOW = "workflow"


# --- node predicates (declarative, replacing geometry heuristics) ------------- #
def always(_request: Any) -> bool:
    return True


def when_task(*tasks: TaskType) -> Callable[[Any], bool]:
    s = set(tasks)
    return lambda req: req.task in s


def when_opt(node_id: str, key: str) -> Callable[[Any], bool]:
    return lambda req: bool(req.node_override(node_id).get(key, False))


# --- nodes -------------------------------------------------------------------- #
@dataclass
class ProgramNode:
    node_id: str
    when: Callable[[Any], bool] = always
    reads: tuple[str, ...] = ()  # input slot names (typed edges in)
    writes: tuple[str, ...] = ()  # output slot names (typed edges out)


@dataclass
class ComponentNode(ProgramNode):
    """A one-shot transform: text/vision encode, VAE decode, latent upsample, vocoder.

    ``fn(instance, slots, request, ctx)`` reads its ``reads`` slots and writes its ``writes``
    slots. Runs to completion in a single engine tick (not step-interleaved)."""
    fn: Callable[..., None] | None = None


@dataclass
class ModelLoopNode(ProgramNode):
    """Binds one of the card's loops. The engine drives it via a LoopRunner, one step per tick,
    so its steps interleave with other requests' steps."""
    loop_id: str = ""
    output_slot: str = "latents"


# --- program ------------------------------------------------------------------ #
@dataclass
class Program:
    program_id: str
    kind: ProgramKind
    nodes: list[ProgramNode] = field(default_factory=list)
    # artifact name -> slot that holds its value at the end
    output_artifacts: dict[str, str] = field(default_factory=dict)

    def active_nodes(self, request: Any) -> list[ProgramNode]:
        return [n for n in self.nodes if n.when(request)]

    def validate(self) -> Program:
        produced: set[str] = set()
        for n in self.nodes:
            for r in n.reads:
                if r not in produced:
                    # not fatal in mini (some reads come from the request itself), but flag dangling
                    pass
            produced.update(n.writes)
        for name, slot in self.output_artifacts.items():
            if slot not in produced:
                raise ValueError(f"program {self.program_id!r} output {name!r} reads slot "
                                 f"{slot!r} that no node writes")
        return self
