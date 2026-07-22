"""Model cards — the contract surface.

A card is a frozen, pure-data description of one servable artifact: which
components it is made of, which loops its weights assume, and the sampling
defaults that are part of the trained artifact. Cards contain **no callables**:
components and loops are declared as ``"module:attr"`` reference strings, so a
card round-trips through JSON and has a stable content digest. The digest is
the card's identity everywhere (evidence ledger, environment manifests,
deploy configs).

Variants are expressed as diffs against a base card via :func:`derive` — never
as builder functions with keyword arguments. Derivation is additive: a variant
that needs to *remove* something picked the wrong base and should be declared
fresh.

Import discipline: this module is stdlib-only. ``validate()`` imports the
declared loop modules to check their ``semantics`` ids, so loop modules must be
importable without torch.
"""
from __future__ import annotations

import hashlib
import importlib
import json
from dataclasses import asdict, dataclass, field, fields, is_dataclass, replace
from typing import Any


class CardError(ValueError):
    pass


def resolve_ref(ref: str) -> Any:
    """Resolve a ``"module:attr"`` reference string to the live object."""
    mod, _, attr = ref.partition(":")
    if not mod or not attr:
        raise CardError(f"bad reference {ref!r} (expected 'module:attr')")
    return getattr(importlib.import_module(mod), attr)


@dataclass(frozen=True)
class ComponentSpec:
    """One weight-bearing (or processing) component of the artifact."""
    component_id: str
    kind: str        # dit | vae | text_encoder | tokenizer
    module: str      # loader reference, e.g. "diffusers:WanTransformer3DModel"
    subfolder: str   # subfolder in the diffusers checkpoint layout
    dtype: str = "bf16"  # bf16 | fp32 | "" (dtype-less, e.g. tokenizer)


@dataclass(frozen=True)
class LoopSpec:
    """One iterative computation the card can run.

    ``loop`` names the implementation class; ``params`` are plain JSON values
    passed to its constructor. The class carries a ``semantics`` id that
    provenance pins (see ``Provenance.assumes_loop``).
    """
    loop_id: str
    loop: str                                  # "fastvideo2.wan21.loop:WanDenoiseLoop"
    params: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SamplingDefaults:
    """Per-model generation defaults that are part of the trained artifact."""
    num_steps: int
    guidance_scale: float
    height: int
    width: int
    num_frames: int
    fps: int
    shift: float
    negative_prompt: str = ""


@dataclass(frozen=True)
class Provenance:
    """Where the weights came from and what they assume.

    ``assumes_loop`` is a *semantics id* (e.g. ``"wan.flow_euler.cfg/v1"``),
    not a loop_id: validation resolves every declared loop class and requires
    one whose ``semantics`` matches. A distilled student that requires a
    different sampler therefore cannot validate against a base card.
    ``substitution`` classifies this artifact relative to ``parents``:
    ``exact`` | ``bounded`` | ``quality-changing``.
    """
    method: str = "base"
    parents: tuple[str, ...] = ()
    assumes_loop: str = ""
    precision: str = "bf16"
    substitution: str = "exact"
    tolerances: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class ModelCard:
    model_id: str
    family: str
    weights: str                                # canonical source (HF repo id)
    components: dict[str, ComponentSpec]
    loops: dict[str, LoopSpec]
    capabilities: tuple[str, ...]
    provenance: Provenance
    sampling_defaults: SamplingDefaults
    determinism: str = "tolerance"              # bitwise | tolerance

    # --- identity ---------------------------------------------------------- #
    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, indent=2)

    def digest(self) -> str:
        """Content digest over the canonical JSON — the card's identity in the
        evidence ledger and every environment manifest."""
        canon = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canon.encode()).hexdigest()[:16]

    @classmethod
    def from_dict(cls, d: dict) -> "ModelCard":
        return cls(
            model_id=d["model_id"],
            family=d["family"],
            weights=d["weights"],
            components={k: ComponentSpec(**v) for k, v in d["components"].items()},
            loops={k: LoopSpec(**v) for k, v in d["loops"].items()},
            capabilities=tuple(d["capabilities"]),
            provenance=Provenance(**{**d["provenance"], "parents": tuple(d["provenance"]["parents"])}),
            sampling_defaults=SamplingDefaults(**d["sampling_defaults"]),
            determinism=d.get("determinism", "tolerance"),
        )

    # --- validation -------------------------------------------------------- #
    def validate(self) -> "ModelCard":
        errs: list[str] = []
        if not self.components:
            errs.append("card declares no components")
        if not self.loops:
            errs.append("card declares no loops")
        for cid, spec in self.components.items():
            if cid != spec.component_id:
                errs.append(f"component key {cid!r} != component_id {spec.component_id!r}")
        semantics_seen: list[str] = []
        for lid, spec in self.loops.items():
            if lid != spec.loop_id:
                errs.append(f"loop key {lid!r} != loop_id {spec.loop_id!r}")
            try:
                cls = resolve_ref(spec.loop)
            except Exception as e:  # unresolvable ref is a contract violation
                errs.append(f"loop {lid!r}: cannot resolve {spec.loop!r} ({e})")
                continue
            sem = getattr(cls, "semantics", None)
            if not sem:
                errs.append(f"loop {lid!r}: class {spec.loop!r} declares no `semantics` id")
            else:
                semantics_seen.append(sem)
            try:
                json.dumps(spec.params)
            except TypeError:
                errs.append(f"loop {lid!r}: params are not plain JSON values")
        # the teeth: weights may only be served under a loop whose semantics
        # they were trained for.
        if self.provenance.assumes_loop and self.provenance.assumes_loop not in semantics_seen:
            errs.append(
                f"provenance.assumes_loop={self.provenance.assumes_loop!r} matches no declared "
                f"loop semantics (have {semantics_seen}) — these weights cannot run on this card")
        if self.determinism not in ("bitwise", "tolerance"):
            errs.append(f"unknown determinism class {self.determinism!r}")
        if errs:
            raise CardError(f"ModelCard {self.model_id!r} failed validation:\n  - " + "\n  - ".join(errs))
        return self


def _merge_field(old: Any, patch: Any) -> Any:
    """One-level structural merge used by :func:`derive`.

    dict field + dict patch      -> merge by key (spec values replace; dict
                                    values patch the existing spec/dict)
    dataclass field + dict patch -> replace() with recursively merged fields
    anything else                -> the patch value wins
    """
    if is_dataclass(old) and isinstance(patch, dict):
        merged = {k: _merge_field(getattr(old, k), v) for k, v in patch.items()}
        return replace(old, **merged)
    if isinstance(old, dict) and isinstance(patch, dict):
        out = dict(old)
        for k, v in patch.items():
            out[k] = _merge_field(old[k], v) if k in old else v
        return out
    return patch


def derive(base: ModelCard, **delta: Any) -> ModelCard:
    """A variant as an explicit diff against a base card.

    Additive only: keys merge, nothing is deleted. A variant that must remove a
    component or loop is a different architecture — declare it fresh. The
    derived card re-validates, so an invalid diff fails at declaration.
    """
    valid = {f.name for f in fields(ModelCard)}
    unknown = set(delta) - valid
    if unknown:
        raise CardError(f"derive: unknown card fields {sorted(unknown)}")
    merged = {k: _merge_field(getattr(base, k), v) for k, v in delta.items()}
    return replace(base, **merged).validate()
