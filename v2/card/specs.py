"""ModelCard and its sub-specs — the Model Plane.

The atomic unit is the (recipe, runtime) pair, owned by a typed ``ModelCard``. The card
is both a declarative contract (strict enough to validate before any GPU touches it — see
``ModelCard.validate``) and a runtime factory (it instantiates components, binds loops, and
resolves caches — see ``card/instance.py``).

Boundary: ``card/`` imports no product/runtime. It depends only on the shared leaf modules
(``_enums``, ``_types``) and references ``ParallelPlan`` under ``TYPE_CHECKING`` so there is
no runtime coupling to ``parallel/``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any
from collections.abc import Callable

from v2._enums import Capability, ConsistencyLevel, LoopKind, WorkUnitKind

if TYPE_CHECKING:  # avoid runtime card -> parallel coupling
    pass




# --------------------------------------------------------------------------- #
# Precision, parallelism, parity, cache, checkpoint contracts                 #
# --------------------------------------------------------------------------- #
@dataclass
class PrecisionContract:
    default_dtype: str = "float32"
    component_overrides: dict[str, str] = field(default_factory=dict)
    quantization_scheme: str | None = None  # "nvfp4" | "int8" | None
    training_precision: str = "float32"  # distinct from serving precision

    def dtype_for(self, component_id: str) -> str:
        return self.component_overrides.get(component_id, self.default_dtype)


@dataclass
class ParallelismContract:
    valid_plans: list = field(default_factory=list)  # list[ParallelPlan]
    default_plan: Any = None  # ParallelPlan | None


@dataclass
class CacheContract:
    """A cache class the card declares it needs."""
    cache_class: str  # "feature" | "residual" | "slab_kv" | "paged_kv"
    max_bytes: int = 1 << 30
    block_bytes: int = 1 << 16  # page bytes (paged) / slab granule (slab)
    eviction: str = "lru"  # "lru" | "fifo" | "none"
    reuse_across_requests: bool = True  # paged/feature=True; slab depends on mode
    per_component: dict[str, int] = field(default_factory=dict)
    training_mode_disables_recycle: bool = False  # chunk-KV training mode


@dataclass
class ParityTestSpec:
    """A named tap + tolerance + ladder level."""
    name: str
    level: ConsistencyLevel
    tap: str  # named activation tap, e.g. "block.0.out"
    rtol: float = 0.0
    atol: float = 0.0


@dataclass
class ParitySpec:
    """The (recipe, runtime) honesty contract. Measured, never assumed."""
    consistency_levels: list[ConsistencyLevel] = field(default_factory=lambda: [ConsistencyLevel.C1])
    tests: list[ParityTestSpec] = field(default_factory=list)
    tap_tolerances: dict[str, float] = field(default_factory=dict)
    interleave_required: bool = True  # the batch-of-N gate is non-negotiable

    @property
    def max_level(self) -> ConsistencyLevel:
        return max(self.consistency_levels, key=lambda c: c.rank) if self.consistency_levels else ConsistencyLevel.C0


@dataclass
class DataRef:
    """What a recipe trained on, for governance/reproduction."""
    dataset_id: str = ""
    revision: str = ""
    description: str = ""


@dataclass
class RecipeSpec:
    """The provenance half of the (recipe, runtime) pair.

    ``assumes_loop`` and ``assumes_precision`` are the teeth: a 4-step distilled
    model whose ``assumes_loop = "ddim_4step"`` cannot be served under a 50-step
    sampler without a typed mismatch error (enforced in ModelCard.validate).
    """
    method: str = "base"  # base | dmd2 | self_forcing | diffusion_nft | attn_qat_nvfp4
    parents: list[str] = field(default_factory=list)  # teacher / base model_ids
    data_contract: DataRef = field(default_factory=DataRef)
    assumes_loop: str = ""  # loop_id this recipe's weights require
    assumes_precision: str = "float32"
    consistency_required: ConsistencyLevel = ConsistencyLevel.C1


@dataclass
class ComponentSpec:
    """A weight-bearing (or processing) component.

    Omni-ready fields ``resident_for`` / ``optional_for`` / ``required_for`` turn the
    Cosmos3 lazy-sound-VAE problem into a declaration, not an ``if env_var`` inside
    ``forward``.
    """
    component_id: str
    kind: str  # dit | vae | text_encoder | audio_vae | reasoner_tower | ...
    load_id: str = ""  # "module:Class" for the real (torch) adapter
    config_schema: type | None = None
    io_schema: tuple[type | None, type | None] = (None, None)
    precision_policy: str | None = None
    placement_policy: str = "colocated"
    parallel_constraints: dict[str, Any] = field(default_factory=dict)
    parity_tests: list[ParityTestSpec] = field(default_factory=list)
    # omni-ready:
    resident_for: list[str] = field(default_factory=list)  # loop_ids that keep this resident mid-request
    optional_for: set[str] = field(default_factory=set)  # tasks that don't need it
    required_for: set[str] = field(default_factory=set)  # tasks that require it
    # v2 wiring: a factory producing the live component (toy numpy or torch adapter)
    factory: Callable[..., Any] | None = None
    # GPU backend: weights source (HF id or local path) for the real torch adapter resolved from
    # ``load_id``. Empty for the CPU toy (its factory needs no weights); a GPU deployment fills it in.
    checkpoint: str = ""
    # GPU backend: optional explicit torch-adapter class "module:Class" (a TorchComponent subclass
    # constructed as cls(module, device=, dtype=)). Lets a NEW architecture declare its own adapter on
    # the card instead of editing the shared backend dispatch — so a port is a self-contained recipe
    # package. Empty -> the backend's built-in per-kind dispatch (Wan/LTX2) by module class name.
    adapter: str = ""


@dataclass
class LoopSpec:
    """Describes one iterative computation the card can run.

    The model owns loop *semantics*; the runtime owns loop *lifecycle*. A single
    ``ModelInstance`` may run several of these against shared components — the MoT
    requirement (``shared_weight_components``).
    """
    loop_id: str
    kind: LoopKind
    work_unit_kind: WorkUnitKind
    state_schema: type | None = None  # the typed LoopState
    step_schema: type | None = None  # the typed WorkPlan a step emits
    result_schema: type | None = None  # the typed StepResult
    behavior_schema: type | None = None  # what to capture for RL (None if not training-relevant)
    extension_schema: type | None = None  # per-model LoopState extension (Cosmos3PackedSeq, etc.)
    cache_policy: list[str] = field(default_factory=list)  # cache class names this loop draws from
    valid_parallel_plans: list = field(default_factory=list)
    graph_capture: str = "eager"  # eager | breakable_cudagraph
    # omni-ready:
    shared_weight_components: list[str] = field(default_factory=list)
    allows_interleaving: bool = True
    # v2: the Loop implementation factory (built at bind time)
    loop_factory: Callable[..., Any] | None = None


@dataclass
class CheckpointManifest:
    """Explicit declared components + key maps — no name-detector guessing."""
    upstream_source: str = ""
    revision: str = ""
    component_ownership: dict[str, list[str]] = field(default_factory=dict)  # component_id -> file globs
    key_mappings: dict[str, str] = field(default_factory=dict)  # ckpt_key -> component.param
    required_for: dict[str, set[str]] = field(default_factory=dict)  # component_id -> tasks requiring it
    optional_for: dict[str, set[str]] = field(default_factory=dict)
    conversion_version: str = "v0"


@dataclass
class CapabilityMatrix:
    capabilities: frozenset[Capability] = frozenset()

    def has(self, cap: Capability) -> bool:
        return cap in self.capabilities

    @classmethod
    def of(cls, *caps: Capability) -> CapabilityMatrix:
        return cls(frozenset(caps))


@dataclass
class SamplingDefaults:
    """Per-model default generation params (the v2 mirror of fastvideo's per-model ``InferencePreset``
    defaults). Applied by the entrypoint when the caller didn't specify a value; ``None`` => fall back to
    the generic default. These are the user-facing knobs surfaced at request time."""
    num_steps: int | None = None
    guidance_scale: float | None = None
    guidance_per_modality: dict[str, float] = field(default_factory=dict)  # joint A/V, e.g. {"video":3,"audio":7}
    height: int | None = None
    width: int | None = None
    num_frames: int | None = None
    fps: int | None = None
    negative_prompt: str | None = None
    shift: float | None = None
    sigmas: tuple[float, ...] | None = None


# --------------------------------------------------------------------------- #
# ModelCard — the (recipe, runtime) pair as one versioned, validatable object  #
# --------------------------------------------------------------------------- #
class CardValidationError(ValueError):
    pass


@dataclass
class ModelCard:
    model_id: str
    family: str
    components: dict[str, ComponentSpec] = field(default_factory=dict)
    loops: dict[str, LoopSpec] = field(default_factory=dict)
    capabilities: CapabilityMatrix = field(default_factory=CapabilityMatrix)
    recipe: RecipeSpec = field(default_factory=RecipeSpec)
    parity: ParitySpec = field(default_factory=ParitySpec)
    caches: dict[str, CacheContract] = field(default_factory=dict)
    parallelism: ParallelismContract = field(default_factory=ParallelismContract)
    precision: PrecisionContract = field(default_factory=PrecisionContract)
    checkpoint: CheckpointManifest = field(default_factory=CheckpointManifest)
    sampling_defaults: SamplingDefaults = field(default_factory=SamplingDefaults)
    # On a GPU box, keep this model's components' I/O on-device (torch tensors) instead of marshalling
    # numpy<->torch at every loop step — the latent stays resident for the whole denoise loop. Opt-in
    # per recipe: set True only when the model's loop+program are array-agnostic (see v2/platform/array_ns).
    device_io: bool = False

    def validate(self) -> ModelCard:
        """Strict enough to validate before any GPU touches it.

        Returns self so it can be chained. Raises CardValidationError on any
        contract violation — the (recipe, runtime) binding is enforced here.
        """
        errs: list[str] = []

        # 1) recipe.assumes_loop must exist (the (recipe, runtime) binding)
        if self.recipe.assumes_loop and self.recipe.assumes_loop not in self.loops:
            errs.append(f"recipe.assumes_loop={self.recipe.assumes_loop!r} is not a declared loop "
                        f"(have {sorted(self.loops)}) — a recipe cannot assume a loop the card does not run")

        # 2) recipe.assumes_precision must be consistent with the precision contract
        if (self.recipe.assumes_precision and self.recipe.assumes_precision
                not in (self.precision.default_dtype, *self.precision.component_overrides.values())):
            errs.append(f"recipe.assumes_precision={self.recipe.assumes_precision!r} not present in precision contract")

        # 3) every loop's shared_weight_components must be declared components
        for lid, loop in self.loops.items():
            for comp in loop.shared_weight_components:
                if comp not in self.components:
                    errs.append(f"loop {lid!r} shares weight component {comp!r} which is not declared")
            for cc in loop.cache_policy:
                if cc not in self.caches:
                    errs.append(f"loop {lid!r} references cache class {cc!r} not in card.caches")

        # 4) required_for / optional_for tasks must be disjoint per component
        for cid, comp_spec in self.components.items():
            overlap = comp_spec.required_for & comp_spec.optional_for
            if overlap:
                errs.append(f"component {cid!r} lists tasks {overlap} as both required and optional")

        if errs:
            raise CardValidationError(f"ModelCard {self.model_id!r} failed validation:\n  - " + "\n  - ".join(errs))
        return self

    def loops_sharing(self, component_id: str) -> list[str]:
        """The loops that bind a given component — the MoT 'many loops, one instance' set."""
        return [lid for lid, lp in self.loops.items() if component_id in lp.shared_weight_components]
