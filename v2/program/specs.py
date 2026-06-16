"""Programs compose a card's loops into a task (design_v3 §13).

> The card says what loops *exist*, the program says how to *run* them for this request.

Kinds: InlineProgram (many loops, one resident instance — the omni default), TrainingProgram,
etc. Nodes are typed (ModelLoopNode / ComponentNode / ...); edges are typed and carry *named*
artifacts (not a god-batch). Linear pipelines are the degenerate case; ``when=`` predicates and
multiple producers give branches/fan-out (LTX-2 base→upsample→refine→decode; A/V fan-out).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

from ..request.tasks import TaskType


class ProgramKind(str, Enum):
    INLINE = "inline"
    DISAGGREGATED = "disaggregated"
    TRAINING = "training"
    REALTIME = "realtime"
    WORKFLOW = "workflow"


# --- node predicates (replace geometry heuristics; design.md P7) -------------- #
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
    reads: tuple[str, ...] = ()     # input slot names (typed edges in)
    writes: tuple[str, ...] = ()    # output slot names (typed edges out)


@dataclass
class ComponentNode(ProgramNode):
    """A one-shot transform: text/vision encode, VAE decode, latent upsample, vocoder.

    ``fn(instance, slots, request, ctx)`` reads its ``reads`` slots and writes its ``writes``
    slots. Runs to completion in a single engine tick (not step-interleaved)."""
    fn: Callable[..., None] | None = None


@dataclass
class ModelLoopNode(ProgramNode):
    """Binds one of the card's loops. The engine drives it via a LoopRunner, one step per tick,
    so its steps interleave with other requests' steps (design_v3 §5, §6.3)."""
    loop_id: str = ""
    output_slot: str = "latents"


# --- typed edges (graph IR; validated, data flows via named slots) ------------ #
class EdgeKind(str, Enum):
    TENSOR = "tensor"
    ARTIFACT = "artifact"
    STREAM = "stream"
    CONTROL = "control"
    CACHE = "cache"
    BEHAVIOR = "behavior"


@dataclass
class Edge:
    src: str             # slot or node id
    dst: str
    kind: EdgeKind = EdgeKind.TENSOR


# --- program ------------------------------------------------------------------ #
@dataclass
class Program:
    program_id: str
    kind: ProgramKind
    nodes: list[ProgramNode] = field(default_factory=list)
    edges: list[Edge] = field(default_factory=list)
    # artifact name -> slot that holds its value at the end (design_v3 §12 named artifacts)
    output_artifacts: dict[str, str] = field(default_factory=dict)

    def active_nodes(self, request: Any) -> list[ProgramNode]:
        return [n for n in self.nodes if n.when(request)]

    def validate(self) -> "Program":
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
