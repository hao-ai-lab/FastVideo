# SPDX-License-Identifier: Apache-2.0
"""The part of a LoRA adapter that is not a low-rank product.

A LoRA states a weight delta as ``B @ A``. That form needs two things to be true: the
base checkpoint must already contain the parameter, and the delta must actually be low
rank. Distilled video checkpoints break both often enough that dropping whatever does
not fit silently loses real signal.

Two payload kinds cover the gap. The weight/bias spellings follow conventions used by
ComfyUI; the ``*_param`` spellings extend them to standalone FastVideo parameters:

``<module>.diff`` / ``<module>.diff_b`` / ``<parameter>.diff_param``
    An exact additive delta for a parameter the base model has. Used where a rank-``r``
    factorization buys nothing or cannot be formed at all -- RMSNorm vectors, biases,
    standalone parameters such as ``scale_shift_table``, and matrices whose smaller
    dimension is already at or below the rank that would be chosen. Factoring a
    length-``n`` vector into rank ``r`` costs ``r(1 + n) > n``.

``<module>.set_weight`` / ``<parameter>.set_param``
    A whole parameter the base model does not carry, so no delta is expressible. MiniMax
    H3's VSA ``to_gate_compress`` is the case that motivated this: it exists only under
    the sparse-attention backend, and :func:`load_model_from_full_model_state_dict`
    otherwise zero-initializes it, which is exactly the "gate contributes nothing"
    state a VSA-distilled student was trained away from.

Both kinds are applied to the *unsharded* tensor while the checkpoint is streaming in,
so FSDP and tensor-parallel placement come from the surrounding loader instead of being
reimplemented here. That ordering is not a preference: ``maybe_load_fsdp_model`` shards
the model on the meta device before any weight is read, so there is no earlier window,
and patching afterwards would mean gathering and redistributing every affected
parameter one at a time.

Because the adapter has to be known when the transformer loads, this path applies the
adapter a pipeline is constructed with. Swapping to a different adapter later still goes
through :meth:`LoRAPipeline.set_lora_adapter`, which handles the low-rank half only.
"""

from __future__ import annotations

import math
import os
import re
from collections.abc import Callable
from typing import Any

import torch
from safetensors import safe_open

from fastvideo.logger import init_logger

logger = init_logger(__name__)

# Suffix -> the parameter suffix it targets. An empty target suffix preserves the
# full parameter name for standalone nn.Parameters. Ordered longest-first so the
# generic spellings are tested before the shorter weight/bias spellings.
ADDITIVE_SUFFIXES: dict[str, str] = {".diff_param": "", ".diff_b": ".bias", ".diff": ".weight"}
REPLACEMENT_SUFFIXES: dict[str, str] = {".set_weight": ".weight", ".set_param": ""}

# One low-rank pair has many spellings. PEFT writes ``.lora_A.weight``, and interposes
# the adapter's name when it is not the default (``.lora_A.default.weight``); kohya and
# ComfyUI write ``.lora_down.weight`` with a bare ``.alpha``. Normalizing on the way in
# means an adapter published in any of those layouts loads without a conversion pass.
_ADAPTER_NAME_INFIX = re.compile(r"\.(lora_[AB])\.[^.]+\.weight$")
_LOW_RANK_ALIASES = ((".lora_down", ".lora_A"), (".lora_up", ".lora_B"))


def normalize_lora_key(name: str) -> str | None:
    """Rewrite an adapter key to the ``<module>.lora_A|lora_B|lora_alpha`` spelling.

    Returns ``None`` for keys the low-rank merge path deliberately does not handle --
    the dense payload above, which lands during checkpoint load, and bookkeeping like
    ``.dora_scale``. Callers use that to tell "not mine" apart from "mine and
    unmatched", which is the difference between a quiet skip and a warning.
    """
    if name.endswith(tuple(ADDITIVE_SUFFIXES) + tuple(REPLACEMENT_SUFFIXES)) or name.endswith(".dora_scale"):
        return None
    name = name.replace("diffusion_model.", "")
    name = _ADAPTER_NAME_INFIX.sub(r".\1.weight", name)
    for alias, canonical in _LOW_RANK_ALIASES:
        name = name.replace(alias + ".", canonical + ".")
        if name.endswith(alias):
            name = name[:-len(alias)] + canonical
    if name.endswith(".alpha"):
        name = name[:-len(".alpha")] + ".lora_alpha"
    return name


def _adapter_files(lora_path: str) -> list[str]:
    """Every safetensors shard belonging to an adapter given as a file or a directory."""
    if os.path.isfile(lora_path):
        return [lora_path]
    if os.path.isdir(lora_path):
        return sorted(
            os.path.join(lora_path, name) for name in os.listdir(lora_path) if name.endswith(".safetensors"))
    return []


class DenseLoRAPatch:
    """Adapter keys that address a whole parameter rather than a factor of one.

    Construction only reads the safetensors headers, so the tensors themselves stay on
    disk until the loader asks for one. That keeps peak host memory at a single
    parameter even for adapters whose dense half is several GiB, which the VSA gates
    alone are.
    """

    def __init__(self, files: list[str], additive: dict[str, tuple[str, str]],
                 replacement: dict[str, tuple[str, str]], strength: float = 1.0) -> None:
        if not math.isfinite(strength):
            raise ValueError(f"LoRA strength must be finite, got {strength}")
        self._files = files
        # target parameter name -> (file, adapter key)
        self._additive = additive
        self._replacement = replacement
        self._strength = float(strength)
        self._applied: set[str] = set()

    @classmethod
    def from_adapter(
        cls,
        lora_path: str | None,
        param_names_mapping: Callable[[str], tuple[str, Any, Any]] | None = None,
        *,
        lora_param_names_mapping: Callable[[str], tuple[str, Any, Any]] | None = None,
        strength: float = 1.0,
    ) -> DenseLoRAPatch | None:
        """Build a patch from an adapter, or ``None`` when it carries no dense payload.

        ``lora_param_names_mapping`` first translates adapter-specific official names
        into the published checkpoint layout. ``param_names_mapping`` then resolves that
        layout into the model's parameter names, matching the low-rank loader's order.
        """
        if not lora_path:
            return None
        # Deferred: fastvideo.utils pulls in enough of the package that importing it at
        # module scope would make this loader helper part of an import cycle.
        from fastvideo.utils import maybe_download_lora
        files = _adapter_files(maybe_download_lora(lora_path))
        if not files:
            logger.warning("LoRA path %s holds no safetensors file; no dense patch applied", lora_path)
            return None

        additive: dict[str, tuple[str, str]] = {}
        replacement: dict[str, tuple[str, str]] = {}
        for path in files:
            with safe_open(path, framework="pt") as handle:
                for key in handle.keys():
                    resolved = _resolve(key, lora_param_names_mapping, param_names_mapping)
                    if resolved is None:
                        continue
                    target, kind = resolved
                    table = additive if kind == "add" else replacement
                    if target in table:
                        raise ValueError(f"LoRA adapter {lora_path} maps two keys onto parameter {target}; "
                                         f"the second is {key}")
                    table[target] = (path, key)

        if not additive and not replacement:
            return None
        logger.info(
            "LoRA adapter %s carries a dense payload: %d additive, %d replacement parameters",
            lora_path, len(additive), len(replacement))
        return cls(files, additive, replacement, strength)

    def apply_to(self, param_name: str, tensor: torch.Tensor) -> torch.Tensor:
        """Add this parameter's ``.diff``/``.diff_b`` delta, if the adapter has one.

        The sum is taken in float32 whatever the operands are: both sides arrive in
        bfloat16, whose 8-bit significand would quantize away a delta three orders of
        magnitude below the base weight. The caller casts back to the target dtype.
        """
        entry = self._additive.get(param_name)
        if entry is None:
            return tensor
        delta = self._read(entry)
        if delta.shape != tensor.shape:
            raise ValueError(f"LoRA diff for {param_name} has shape {tuple(delta.shape)}, "
                             f"but the parameter is {tuple(tensor.shape)}")
        self._applied.add(param_name)
        return tensor.to(torch.float32) + delta.to(torch.float32) * self._strength

    def provides(self, param_name: str) -> bool:
        """Whether the adapter carries this parameter whole, without reading it."""
        return param_name in self._replacement

    @property
    def replacement_parameters(self) -> frozenset[str]:
        """Names of the parameters this adapter supplies outright.

        Lets a caller decide what the adapter needs from the runtime -- an H3 adapter
        carrying ``to_gate_compress`` only works under the VSA backend -- without
        reading any tensor or guessing from the file name.
        """
        return frozenset(self._replacement)

    def replacement_for(self, param_name: str) -> torch.Tensor | None:
        """The adapter's whole-tensor value for a parameter, or ``None``."""
        entry = self._replacement.get(param_name)
        if entry is None:
            return None
        self._applied.add(param_name)
        # Replacement payloads address parameters absent from the base checkpoint. The
        # loader initializes those parameters to zero, so scaling the supplied value is
        # the same interpolation contract as ``base + strength * delta``.
        return self._read(entry).to(torch.float32) * self._strength

    def report_unapplied(self) -> None:
        """Warn about dense keys that never reached a parameter.

        An adapter key the loader silently ignores is the failure mode this whole path
        exists to fix, so it is said out loud rather than left to be inferred from a
        model that merely generates worse.
        """
        pending = (set(self._additive) | set(self._replacement)) - self._applied
        if not pending:
            logger.info("LoRA dense payload fully applied: %d parameters", len(self._applied))
            return
        for target in sorted(pending):
            source = self._additive.get(target) or self._replacement[target]
            logger.warning("LoRA key not loaded: %s (targets %s, absent from the model)", source[1], target)
        logger.warning("LoRA dense payload: %d of %d parameters applied, %d keys unmatched", len(self._applied),
                       len(self._applied) + len(pending), len(pending))

    def _read(self, entry: tuple[str, str]) -> torch.Tensor:
        path, key = entry
        with safe_open(path, framework="pt") as handle:
            return handle.get_tensor(key)


def _resolve(
    key: str,
    lora_param_names_mapping: Callable[[str], tuple[str, Any, Any]] | None,
    param_names_mapping: Callable[[str], tuple[str, Any, Any]] | None,
) -> tuple[str, str] | None:
    """Map an adapter key to ``(model parameter name, "add" | "set")``.

    Returns ``None`` for anything that is not a dense payload key, which includes every
    low-rank factor -- those belong to ``LoRAPipeline``, not here.
    """
    # A terminal dense suffix is authoritative even when an ordinary module name
    # contains text such as ``.alpha`` or ``.lora_A``.
    for suffix, param_suffix in ADDITIVE_SUFFIXES.items():
        if key.endswith(suffix):
            return _map_name(
                key[:-len(suffix)] + param_suffix,
                lora_param_names_mapping,
                param_names_mapping,
                key,
            ), "add"
    for suffix, param_suffix in REPLACEMENT_SUFFIXES.items():
        if key.endswith(suffix):
            return _map_name(
                key[:-len(suffix)] + param_suffix,
                lora_param_names_mapping,
                param_names_mapping,
                key,
            ), "set"
    return None


def _map_name(
    param_name: str,
    lora_param_names_mapping: Callable[[str], tuple[str, Any, Any]] | None,
    param_names_mapping: Callable[[str], tuple[str, Any, Any]] | None,
    source_key: str,
) -> str:
    """Run the low-rank loader's two-stage renaming rules over a dense parameter."""
    param_name = param_name.replace("diffusion_model.", "")
    if lora_param_names_mapping is not None:
        param_name, merge_index, _ = lora_param_names_mapping(param_name)
        if merge_index is not None:
            raise NotImplementedError(f"LoRA dense key {source_key} resolves to a fused parameter during the "
                                      "adapter-specific mapping; whole-tensor payloads for fused parameters "
                                      "are not supported")
    if param_names_mapping is None:
        return param_name
    mapped, merge_index, _ = param_names_mapping(param_name)
    if merge_index is not None:
        # A fused target (stacked QKV and friends) takes its value from several
        # checkpoint tensors. Splitting a whole-tensor payload across that fusion is
        # not something we can infer, and guessing would corrupt the weight quietly.
        raise NotImplementedError(f"LoRA dense key {source_key} resolves to fused parameter {mapped}; "
                                  "whole-tensor payloads for fused parameters are not supported")
    return mapped
