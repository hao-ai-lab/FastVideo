# SPDX-License-Identifier: Apache-2.0
"""FastH3-Omni (FastH3 OmniRef PDD student) support for the Studio.

A FastH3 OmniRef PDD export carries only what distillation changed
(``transformer_ref/``, the two schedulers and ``fastvideo_inference.json``);
the text encoder, processor and both VAEs come from the base MiniMax-H3
snapshot the export pins. This module:

* reads each export the server was started with (never a path from a request)
  and turns its ``fastvideo_inference.json`` into a *recipe*: the sampling
  settings the Studio locks for that model,
* composes export + base into one model directory per (export, base) snapshot
  pair, reusing the official example's composition code,
* checks the hardware the recipe needs before any weights load.

FastVideo itself re-reads and enforces the contract when the generator is
built (``MiniMaxH3PipelineConfig.resolve_checkpoint_settings``); the Studio
locks the same values so generic form defaults never reach that check.
"""

from __future__ import annotations

import argparse
import functools
import hashlib
import importlib.util
import json
import logging
import os
import re
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Any

logger = logging.getLogger("fastvideo.studio.fasth3_omni")

REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_PATH = REPO_ROOT / "examples" / "inference" / "basic" / "basic_fasth3_omniref_pdd.py"

RECIPE_KIND = "fasth3_omni_pdd"
CONTRACT_FILE = "fastvideo_inference.json"
REF2VA_PIPELINE = "MiniMaxH3Ref2VAModularPipeline"
# MiniMax-H3 samples at a fixed 24 fps; 15 s is its longest clip.
H3_FPS = 24
H3_MAX_FRAMES = 15 * H3_FPS + 1
# Sequence parallelism splits the 56 attention heads evenly across GPUs.
H3_NUM_HEADS = 56
VALID_NUM_GPUS = tuple(n for n in range(1, 9) if H3_NUM_HEADS % n == 0)
# Per-modality caps of fastvideo.pipelines.basic.minimax_h3.reference.validate_references.
REFERENCE_LIMITS = {"image": 9, "video": 3, "audio": 3}
MAX_REFERENCES = 12
# VSA-H3 tile sizes with only the data-center Blackwell CUDA kernel (no Triton fallback).
SM100A_ONLY_TILE_SIZES = frozenset({128})
SM100A_CAPABILITIES = frozenset({(10, 0), (10, 3)})

# Job fields the recipe owns. ``apply_recipe`` overwrites whatever the request
# carried, so the generic form's defaults (50 steps, CFG 5) never reach a run.
LOCKED_FIELDS = (
    "workload_type",
    "image_path",
    "last_image_path",
    "num_inference_steps",
    "guidance_scale",
    "guidance_rescale",
    "negative_prompt",
    "vsa_sparsity",
    "fps",
    "enable_torch_compile",
    "use_fsdp_inference",
    "tp_size",
    "sp_size",
    "dit_cpu_offload",
    "dit_layerwise_offload",
    "image_encoder_cpu_offload",
)
# What a new Omni job starts with, matching the official example.
RECIPE_DEFAULTS = {
    "height": 480,
    "width": 832,
    "num_frames": 124,
    "num_gpus": 1,
    "text_encoder_cpu_offload": True,
    "vae_cpu_offload": True,
}


class OmniWeightsError(RuntimeError):
    """An export or base component is missing, unreadable, or not an Omni export."""


class OmniHardwareError(RuntimeError):
    """This host cannot run the recipe's attention kernel."""


@functools.lru_cache(maxsize=1)
def load_example() -> ModuleType:
    """The official example, loaded by path the way its tests and benchmarks load it."""
    if not EXAMPLE_PATH.is_file():
        raise OmniWeightsError(f"FastH3-Omni composition needs {EXAMPLE_PATH}, which is missing from this checkout.")
    spec = importlib.util.spec_from_file_location("basic_fasth3_omniref_pdd", EXAMPLE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def split_source(spec: str) -> tuple[str, str | None]:
    """``"<repo id>@<revision>"`` -> (repo id, revision); an existing local directory is taken whole."""
    spec = spec.strip()
    if Path(spec).expanduser().is_dir() or "@" not in spec:
        return spec, None
    source, _, revision = spec.rpartition("@")
    return source, revision or None


def read_contract(export: str, revision: str | None) -> dict[str, Any]:
    """Read and fully validate an export's ``fastvideo_inference.json`` (local directory or repo id).

    FastVideo's own reader checks every field against the transformer and
    scheduler configs; anything that is not a Ref2VA PDD contract is rejected.
    """
    from fastvideo.configs.pipelines.minimax_h3 import _read_pdd_inference_file

    local = Path(export).expanduser()
    if not local.is_dir() and export.startswith(("/", "./", "../", "~")):
        raise OmniWeightsError(f"{export}: local export directory not found.")
    try:
        contract = _read_pdd_inference_file(str(local) if local.is_dir() else export, revision)
    except Exception as exc:  # unreadable, malformed, or a Hub error
        raise OmniWeightsError(f"{export}: {exc}") from exc
    if contract is None:
        raise OmniWeightsError(f"{export} has no Ref2VA PDD {CONTRACT_FILE}; it is not a FastH3-Omni export.")
    return contract


def contract_digest(contract: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest()


def hardware_status(tile_size: int) -> dict[str, Any]:
    """Whether this host can run VSA-H3 at ``tile_size``: {"supported", "reason"}.

    Tile 128 runs only on the sm_100a/sm_103a CUDA kernel; other tile sizes
    have a Triton route. ``supported`` is None when the GPUs cannot be read.
    """
    if tile_size not in SM100A_ONLY_TILE_SIZES:
        return {"supported": True, "reason": ""}
    try:
        from fastvideo_studio.gpu import _ensure_nvml

        pynvml = _ensure_nvml()
        devices = []
        for index in range(pynvml.nvmlDeviceGetCount()):
            handle = pynvml.nvmlDeviceGetHandleByIndex(index)
            name = pynvml.nvmlDeviceGetName(handle)
            capability = tuple(pynvml.nvmlDeviceGetCudaComputeCapability(handle))
            devices.append((name.decode() if isinstance(name, bytes) else name, capability))
    except Exception as exc:
        return {"supported": None, "reason": f"Could not read the GPUs: {exc}"}
    return _hardware_verdict(tile_size, devices, _sm100a_kernel_installed())


def _sm100a_kernel_installed() -> bool:
    try:
        return importlib.util.find_spec("fastvideo_kernel.block_sparse_attn_sm100a") is not None
    except (ImportError, ValueError):
        return False


def _hardware_verdict(tile_size: int, devices: list[tuple[str, tuple[int, ...]]],
                      kernel_installed: bool) -> dict[str, Any]:
    unsupported = sorted({
        f"{name} (sm_{major}{minor})"
        for name, (major, minor) in devices if (major, minor) not in SM100A_CAPABILITIES
    })
    if not devices or unsupported:
        found = ", ".join(unsupported) or "no GPU"
        return {
            "supported":
            False,
            "reason": (f"This recipe's {tile_size}-token VSA tiles run only on the sm_100a/sm_103a kernel "
                       f"(B200, B300, GB200, GB300); found {found}. There is no Triton fallback."),
        }
    if not kernel_installed:
        return {
            "supported":
            False,
            "reason": ("fastvideo_kernel.block_sparse_attn_sm100a is not installed; build fastvideo-kernel with the "
                       "Blackwell VSA extension."),
        }
    return {"supported": True, "reason": ""}


@dataclass
class OmniExport:
    """One FastH3-Omni export the server was configured with."""

    id: str
    export: str
    export_revision: str | None
    base: str | None
    base_revision: str | None
    contract: dict[str, Any] | None = None
    error: str | None = None
    hardware: dict[str, Any] = field(default_factory=lambda: {"supported": None, "reason": ""})

    @property
    def label(self) -> str:
        name = Path(self.export.rstrip("/")).name or self.export
        return f"FastH3-Omni · {name}" + (f"@{self.export_revision[:12]}" if self.export_revision else "")

    def recipe(self) -> dict[str, Any] | None:
        """The locked sampling recipe, with where each part came from."""
        if self.contract is None:
            return None
        # read_contract already checked the hf://<repo id>@<revision> form.
        pinned_repo, _, pinned_revision = self.contract["base_model_revision"].removeprefix("hf://").partition("@")
        return {
            "kind": RECIPE_KIND,
            "source": CONTRACT_FILE,
            "export": self.export,
            "export_revision": self.export_revision,
            "base": self.base or pinned_repo,
            "base_revision": self.base_revision if self.base else (self.base_revision or pinned_revision),
            "contract_sha256": contract_digest(self.contract),
            "contract": self.contract,
            "locked_fields": list(LOCKED_FIELDS),
        }


class FastH3OmniRegistry:
    """The Omni exports configured on the command line, and where composed models are cached."""

    def __init__(self, exports: list[str], base: str | None, cache_dir: str, check_hardware: bool = True):
        self.cache_dir = Path(cache_dir)
        self._compose_lock = threading.Lock()
        base_source, base_revision = split_source(base) if base else (None, None)
        self.exports: dict[str, OmniExport] = {}
        for spec in exports:
            source, revision = split_source(spec)
            entry = OmniExport(id=spec.strip(),
                               export=source,
                               export_revision=revision,
                               base=base_source,
                               base_revision=base_revision)
            try:
                entry.contract = read_contract(source, revision)
                if check_hardware:
                    entry.hardware = hardware_status(int(entry.contract["vsa_tile_size"]))
            except Exception as exc:
                entry.error = str(exc)
                logger.warning("FastH3-Omni export %s is unusable: %s", spec, exc)
            self.exports[entry.id] = entry

    def get(self, model_id: str) -> OmniExport | None:
        return self.exports.get(model_id)

    def prepare(self, model_id: str) -> tuple[Path, dict[str, Any]]:
        """Compose (or reuse) the model directory for an export; returns it and the effective recipe.

        The contract is re-read from the snapshot actually composed, so the
        returned recipe is what this run uses even if the export changed since
        the server started.
        """
        entry = self.exports.get(model_id)
        if entry is None:
            raise OmniWeightsError(f"{model_id} is not a FastH3-Omni export configured on this server.")
        example = load_example()
        with self._compose_lock:
            try:
                export_dir = example._snapshot(
                    entry.export, entry.export_revision,
                    [CONTRACT_FILE, *example.MANIFESTS, *(f"{name}/**" for name in example.EXPORT_COMPONENTS)])
                contract = example.load_contract(export_dir)
                base_source, base_revision = example.base_model_source(
                    argparse.Namespace(base_model_path=entry.base, base_revision=entry.base_revision), contract)
                base_dir = example._snapshot(base_source, base_revision,
                                             [*example.MANIFESTS, *(f"{name}/**" for name in example.BASE_COMPONENTS)])
                # One directory per (export snapshot, base snapshot): Hugging Face
                # snapshot paths carry their commit, so a new revision of either
                # gets a fresh directory and an old one is never silently reused.
                key = hashlib.sha256(f"{export_dir.resolve()}\n{base_dir.resolve()}".encode()).hexdigest()[:16]
                composed = example.compose_model_dir(export_dir, base_dir, self.cache_dir / key)
            except (OSError, ValueError) as exc:
                raise OmniWeightsError(str(exc)) from exc
            fresh = OmniExport(id=entry.id,
                               export=entry.export,
                               export_revision=entry.export_revision,
                               base=entry.base,
                               base_revision=entry.base_revision,
                               contract=read_contract(str(composed), None))
            recipe = fresh.recipe()
            assert recipe is not None
            recipe.update(base=base_source,
                          base_revision=base_revision,
                          export_dir=str(export_dir),
                          base_dir=str(base_dir),
                          model_dir=str(composed))
            provenance = self.cache_dir / f"{key}.json"
            provenance.write_text(json.dumps({**recipe, "composed_at": time.time()}, indent=2) + "\n", encoding="utf-8")
        return composed, recipe


def validate_omni_job(fields: dict[str, Any]) -> None:
    """Reject requests the Omni recipe cannot run. Raises ValueError with a user-facing message."""
    if (fields.get("job_type") or "inference") != "inference":
        raise ValueError("FastH3-Omni is an inference model; it cannot be trained from the Studio.")
    references = fields.get("references") or []
    if not references:
        raise ValueError("FastH3-Omni needs at least one ordered reference (image or video).")
    counts = {kind: 0 for kind in REFERENCE_LIMITS}
    for index, ref in enumerate(references):
        kind = (ref or {}).get("media_type")
        if kind not in counts:
            raise ValueError(f"Reference {index + 1} has media_type {kind!r}; expected image, video or audio.")
        if not (ref or {}).get("source"):
            raise ValueError(f"Reference {index + 1} has no source.")
        counts[kind] += 1
    if counts["image"] + counts["video"] == 0:
        raise ValueError("FastH3-Omni needs at least one image or video reference; audio alone is not enough.")
    for kind, limit in REFERENCE_LIMITS.items():
        if counts[kind] > limit:
            raise ValueError(f"At most {limit} {kind} references (have {counts[kind]}).")
    if len(references) > MAX_REFERENCES:
        raise ValueError(f"At most {MAX_REFERENCES} references (have {len(references)}).")
    if fields.get("image_path") or fields.get("last_image_path"):
        raise ValueError("FastH3-Omni takes ordered references, not a start or end frame.")
    num_gpus = int(fields.get("num_gpus") or 1)
    if num_gpus not in VALID_NUM_GPUS:
        raise ValueError(f"FastH3-Omni splits its {H3_NUM_HEADS} attention heads across GPUs; "
                         f"use {', '.join(map(str, VALID_NUM_GPUS))} GPUs (got {num_gpus}).")
    num_frames = int(fields.get("num_frames") or 0)
    if not 1 <= num_frames <= H3_MAX_FRAMES:
        raise ValueError(f"FastH3-Omni generates 1 to {H3_MAX_FRAMES} frames (15 s at {H3_FPS} fps); "
                         f"got {num_frames}.")


def apply_recipe(fields: dict[str, Any], recipe: dict[str, Any]) -> dict[str, Any]:
    """Return ``fields`` with every recipe-owned value set from the recipe, plus the recipe itself."""
    contract = recipe["contract"]
    num_gpus = int(fields.get("num_gpus") or 1)
    return {
        **fields,
        "workload_type": "i2v",
        "image_path": "",
        "last_image_path": "",
        # PDD counts fused blocks: one transformer forward each.
        "num_inference_steps": int(contract["num_inference_steps"]),
        "guidance_scale": float(contract["guidance_scale"]),
        "guidance_rescale": 0.0,
        # Unused without CFG; cleared so the stored config shows what ran.
        "negative_prompt": "",
        "vsa_sparsity": float(contract["vsa_sparsity"]),
        "fps": H3_FPS,
        # Tile 128 runs eagerly; regional compile needs 64-token tiles.
        "enable_torch_compile": False,
        # The example's layout: shard the DiT (FSDP) and split the sequence across the GPUs.
        "use_fsdp_inference": num_gpus > 1,
        "tp_size": 1,
        "sp_size": num_gpus,
        "dit_cpu_offload": False,
        "dit_layerwise_offload": False,
        "image_encoder_cpu_offload": False,
        "recipe": recipe,
    }


def generator_settings(recipe: dict[str, Any]) -> dict[str, Any]:
    """Generator overrides for an Omni run: the Ref2VA pipeline, and the contract's own VSA settings.

    ``VSA_sparsity=None`` lets FastVideo take the trained value from the
    contract rather than a Studio default; the attention backend and tile size
    are left unset for the same reason.
    """
    return {"override_pipeline_cls_name": REF2VA_PIPELINE, "vsa_sparsity": None}


def is_omni_recipe(recipe: Any) -> bool:
    return isinstance(recipe, dict) and recipe.get("kind") == RECIPE_KIND


_MEMORY_RE = re.compile(r"out of memory|OutOfMemoryError|CUDA error: out of memory|CUBLAS_STATUS_ALLOC_FAILED", re.I)
_KERNEL_RE = re.compile(
    r"sm_?100a?|sm_?103a?|block_sparse_attn|fastvideo_kernel|no Triton fallback|no kernel image|"
    r"VSA-H3|VIDEO_SPARSE_ATTN_H3|CUDA kernel|compute capability", re.I)
_WEIGHTS_RE = re.compile(
    r"Missing checkpoint component|not a FastH3|RepositoryNotFound|RevisionNotFound|EntryNotFound|"
    r"LocalEntryNotFound|snapshot_download|No such file or directory|safetensors|model_index\.json|"
    r"Diffusers model manifest", re.I)


def classify_error(exc: BaseException) -> str:
    """Bucket a failed run as weights / parameters / kernel / memory / runtime for the UI.

    Errors from worker processes arrive re-raised as generic exceptions, so the
    message is matched as well as the type.
    """
    text = f"{type(exc).__name__}: {exc}"
    if isinstance(exc, OmniHardwareError):
        return "kernel"
    if isinstance(exc, OmniWeightsError | FileNotFoundError):
        return "weights"
    if _MEMORY_RE.search(text):
        return "memory"
    if _KERNEL_RE.search(text):
        return "kernel"
    if _WEIGHTS_RE.search(text):
        return "weights"
    if isinstance(exc, ValueError) or "ValueError" in text:
        return "parameters"
    return "runtime"


def hardware_check_disabled() -> bool:
    """Set FASTVIDEO_STUDIO_SKIP_HARDWARE_CHECK=1 to let FastVideo itself report kernel problems."""
    return os.environ.get("FASTVIDEO_STUDIO_SKIP_HARDWARE_CHECK", "") in ("1", "true", "yes")
