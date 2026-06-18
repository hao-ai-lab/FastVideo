"""ModelInstance — a resident, loaded card (design_v3 §4.3).

> A ``ModelInstance`` is a resident, loaded card: component instances, model state,
> caches, compiled graphs, and a parallel plan. **A request may run several of the
> card's loops against one ``ModelInstance``.**

That single sentence is the difference between this design and a stage-only design,
and it is what makes omni native: when two loops bind the same component
(``shared_weight_components``), ``component()`` returns the *exact same* live object
— no duplicated weights, shared live state.
"""
from __future__ import annotations

from typing import Any

from v2.platform import Platform
from v2.card.specs import ModelCard


class ModelInstance:
    """The live, resident form of a ModelCard. Also serves as the ``ModelState``
    passed to ``Loop.init`` (the loop reads components through it)."""

    def __init__(self, card: ModelCard, parallel_plan: Any = None,
                 cache_manager: Any = None, weights_version: str = "v0",
                 platform: Any = None):
        self.card = card
        self.parallel_plan = parallel_plan
        self.caches = cache_manager
        self.weights_version = weights_version
        # The detected (device, arch). Resolves component/kernel implementations through the two
        # backend registries; defaults to CPU/numpy. Swapping this to a GPU platform is the whole
        # "change only the backend, not the loops/policies/training" story (design_v3 §17).
        self.platform = platform if platform is not None else Platform.cpu()
        # Piecewise CUDA-graph cache for loops declaring graph_capture="breakable_cudagraph". Lazily
        # created by the runtime (kept as a plain attr so card/ stays free of any runtime import —
        # design_v3 §18 boundary); a captured graph is tied to this instance's resident weights.
        self.graphs: Any = None
        self.adapter_versions: dict[str, str] = {}
        # per-component weight versions (§7.1): a component's version changes only when IT is synced,
        # so a transformer-only RL sync never invalidates the frozen text-encoder's feature cache.
        self.component_versions: dict[str, str] = {cid: weights_version for cid in card.components}
        self._components: dict[str, Any] = {}
        self._loops: dict[str, Any] = {}
        self._asleep: set[str] = set()

    # --- components: shared by reference (the MoT requirement) ---------------- #
    def component(self, component_id: str) -> Any:
        if component_id in self._asleep:
            raise RuntimeError(f"component {component_id!r} is asleep; wake it first")
        if component_id not in self._components:
            spec = self.card.components.get(component_id)
            if spec is None:
                raise KeyError(f"component {component_id!r} not declared on card {self.card.model_id!r}")
            # The single materialization seam: the platform resolves (kind, device, variant) through
            # the COMPONENTS registry, falling back to spec.factory as the cpu/numpy terminal rung.
            self._components[component_id] = self.platform.build_component(spec, self)
        return self._components[component_id]

    def has_component(self, component_id: str) -> bool:
        return component_id in self.card.components

    # --- loops: one stateless Loop object per (instance, loop_id) ------------- #
    def loop(self, loop_id: str) -> Any:
        if loop_id not in self._loops:
            spec = self.card.loops.get(loop_id)
            if spec is None:
                raise KeyError(f"loop {loop_id!r} not declared on card {self.card.model_id!r}")
            if spec.loop_factory is None:
                raise RuntimeError(f"loop {loop_id!r} has no loop_factory")
            self._loops[loop_id] = spec.loop_factory()
        return self._loops[loop_id]

    # --- sleep/wake by component (design_v3 §7.3; CuMem tags = component names) #
    def sleep(self, component_ids: list[str]) -> None:
        for cid in component_ids:
            self._components.pop(cid, None)
            self._asleep.add(cid)

    def wake(self, component_ids: list[str]) -> None:
        for cid in component_ids:
            self._asleep.discard(cid)

    # --- weight sync (design_v3 §10): bump version + invalidate caches -------- #
    def version_of(self, component_id: str) -> str:
        """The component's own weights version (defaults to the instance version)."""
        return self.component_versions.get(component_id, self.weights_version)

    def set_weights_version(self, version: str, components: list[str] | None = None) -> None:
        """Publish a new weights version. If ``components`` is given, only those components' versions
        bump and only their caches are invalidated (design_v3 §7.1 partition-not-flush) — so a
        transformer-only RL weight sync leaves the frozen text-encoder's feature cache intact."""
        self.weights_version = version
        changed = components if components is not None else list(self.card.components.keys())
        for c in changed:
            self.component_versions[c] = version
        if self.caches is not None and hasattr(self.caches, "invalidate_components"):
            self.caches.invalidate_components(set(changed))
        # Evict captured CUDA graphs for the synced components too (else they leak on a real box;
        # version-in-key already makes them unreachable). Duck-typed → card/ imports no runtime.
        if self.graphs is not None and hasattr(self.graphs, "invalidate"):
            self.graphs.invalidate(set(changed))

    def __repr__(self) -> str:
        return (f"ModelInstance(card={self.card.model_id!r}, weights={self.weights_version!r}, "
                f"resident={sorted(self._components)})")


def load_card(card: ModelCard, parallel_plan: Any = None, cache_manager: Any = None,
              *, validate: bool = True, platform: Any = None) -> ModelInstance:
    """The card-as-factory entrypoint (design_v3 §4.1: card is a runtime factory).

    ``platform`` selects the backend (device, arch); when omitted it is detected (CPU/numpy here,
    CUDA on a torch+GPU box). The same card loads on any backend — only the resolved component/kernel
    implementations differ.
    """
    if validate:
        card.validate()
    return ModelInstance(card, parallel_plan=parallel_plan, cache_manager=cache_manager,
                         platform=platform if platform is not None else Platform.detect())
