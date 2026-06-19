"""Request + Session — the only currency crossing the product boundary.

A typed ``Request`` carries ``task`` (declared, never inferred), ``inputs``
(ModalParts), AR ``sampling`` vs ``diffusion`` params, an ``OutputSpec``, and
per-node overrides. Products construct Requests; they never reach into the model.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field

from v2.request.cancel import CancelKind, CancelScope
from v2.request.modalpart import ImagePart, ModalPart, TextPart
from v2.request.params import DiffusionParams, OutputSpec, SamplingParams
from v2.request.streams import Stream
from v2.request.tasks import TaskType

_counter = itertools.count(1)


def new_request_id() -> str:
    return f"req-{next(_counter)}"


@dataclass(frozen=True)
class Request:
    request_id: str
    task: TaskType
    model_id: str  # which ModelCard to run
    inputs: tuple[ModalPart, ...] = ()
    sampling: SamplingParams = field(default_factory=SamplingParams)
    diffusion: DiffusionParams = field(default_factory=DiffusionParams)
    outputs: OutputSpec = field(default_factory=OutputSpec)
    node_params: dict[str, dict] = field(default_factory=dict)  # per graph-node overrides
    priority: int = 0

    # --- convenience accessors (do not infer task from these) ---
    def prompt(self) -> str:
        for p in self.inputs:
            if isinstance(p, TextPart):
                return p.text
        return ""

    def image(self):
        for p in self.inputs:
            if isinstance(p, ImagePart):
                return p
        return None

    def node_override(self, node_id: str) -> dict:
        return self.node_params.get(node_id, {})


def make_request(task: TaskType,
                 model_id: str,
                 prompt: str = "",
                 *,
                 inputs: tuple[ModalPart, ...] | None = None,
                 sampling: SamplingParams | None = None,
                 diffusion: DiffusionParams | None = None,
                 outputs: OutputSpec | None = None,
                 node_params: dict[str, dict] | None = None,
                 priority: int = 0,
                 request_id: str | None = None) -> Request:
    """Ergonomic constructor used by the offline API and tests."""
    parts: tuple[ModalPart, ...] = inputs if inputs is not None else ((TextPart(prompt), ) if prompt else ())
    return Request(
        request_id=request_id or new_request_id(),
        task=task,
        model_id=model_id,
        inputs=parts,
        sampling=sampling or SamplingParams(),
        diffusion=diffusion or DiffusionParams(),
        outputs=outputs or OutputSpec(),
        node_params=node_params or {},
        priority=priority,
    )


@dataclass
class Session:
    """A long-lived interactive context.

    Holds prompt memory, media streams, a cancel scope, and a handle to
    cross-request chunk-KV that persists for a game/scene session (self-forcing /
    world-model rollout). Mutable by design.
    """
    session_id: str
    prompt_memory: list[str] = field(default_factory=list)
    streams: dict[str, Stream] = field(default_factory=dict)
    cancel_scope: CancelScope | None = None
    kv_handle: object | None = None  # persistent chunk-KV CacheHandle (see cache/)

    def __post_init__(self):
        if self.cancel_scope is None:
            self.cancel_scope = CancelScope(kind=CancelKind.SESSION, target_id=self.session_id)

    def push_text(self, text: str) -> None:
        self.prompt_memory.append(text)

    def stream(self, name: str) -> Stream:
        return self.streams.setdefault(name, Stream(f"{self.session_id}:{name}"))
