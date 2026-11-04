# SPDX-License-Identifier: Apache-2.0
"""Torch-free validation of a FastH3 Ref2VA PDD ``fastvideo_inference.json``.

Ports the checks in ``fastvideo/configs/pipelines/minimax_h3.py``
(``_read_pdd_inference_file``), which cannot be imported without torch: the
exact field set, the fields FastVideo fixes in code, the ``base_model_revision``
format, the trained VSA settings, and the values that repeat a sibling
checkpoint file (``transformer_ref/config.json`` and the two scheduler configs).
The sibling files are checked when present, since a converted MLX root may be
trimmed; a present file that disagrees always fails.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastvideo.mlx_runtime.minimax_h3_vsa import VSA_H3_TILE_SHAPES

PDD_CONTRACT_SCHEMA = "fasth3-inference-contract-v1"
# Missing and unknown fields are both errors, so no field is silently ignored.
PDD_INFERENCE_FIELDS = frozenset({
    "schema_version",
    "schema",
    "model_type",
    "transformer_component",
    "conditioning",
    "base_model_revision",
    "pdd_steps",
    "pdd_step_indices",
    "num_inference_steps",
    "transformer_forwards",
    "grid_max_t",
    "video_scheduler_shift",
    "audio_scheduler_shift",
    "guidance_scale",
    "attention_backend",
    "vsa_sparsity",
    "vsa_tile_size",
    "vsa_ref_policy",
    "vsa_ref_keep_rate",
})
# Fields that repeat a value FastVideo fixes in code.
PDD_FIXED_FIELDS: dict[str, Any] = {
    "schema_version": PDD_CONTRACT_SCHEMA,
    "schema": PDD_CONTRACT_SCHEMA,
    "model_type": "ref2va",
    "transformer_component": "transformer_ref",
    "conditioning": "fixed_ordered_references_target_only_flow",
    "guidance_scale": 1.0,
    "attention_backend": "VIDEO_SPARSE_ATTN_H3",
    "vsa_ref_policy": "p2_multi_region",
}
_BASE_MODEL_REVISION_PREFIX = "hf://"
# (contract key, sibling file, key inside that file)
_SIBLING_CHECKS = (
    ("pdd_steps", "transformer_ref/config.json", "pdd_steps"),
    ("video_scheduler_shift", "scheduler/scheduler_config.json", "shift"),
    ("audio_scheduler_shift", "audio_scheduler/scheduler_config.json", "shift"),
)


def parse_base_model_revision(value: Any) -> tuple[str, str]:
    """Split ``hf://<repo id>@<revision>`` into repo id and revision."""
    if isinstance(value, str) and value.startswith(_BASE_MODEL_REVISION_PREFIX):
        repo, separator, revision = value[len(_BASE_MODEL_REVISION_PREFIX):].partition("@")
        if separator and repo.strip() and revision.strip() and "@" not in revision:
            return repo, revision
    raise ValueError(f"FastH3 base_model_revision={value!r} must be hf://<repo id>@<revision>.")


def _require_fraction(name: str, value: Any, *, allow_zero: bool) -> None:
    """Raise unless *value* is a real number in [0, 1) (``allow_zero``) or in (0, 1)."""
    is_real = isinstance(value, int | float) and not isinstance(value, bool)
    if not (is_real and (value >= 0.0 if allow_zero else value > 0.0) and value < 1.0):
        raise ValueError(f"{name} must be in {'[0, 1)' if allow_zero else '(0, 1)'}, got {value!r}.")


def _read_sibling(root: Path, relative: str) -> dict[str, Any] | None:
    path = root / relative
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Malformed {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"Malformed {path}: expected a JSON object.")
    return data


def validate_pdd_contract_fields(contract: dict[str, Any], path: Path) -> None:
    """Field set, code-fixed values, base revision and trained VSA settings."""
    missing = sorted(PDD_INFERENCE_FIELDS - contract.keys())
    unknown = sorted(contract.keys() - PDD_INFERENCE_FIELDS)
    if missing or unknown:
        raise ValueError(f"{path} must carry exactly the FastH3 PDD fields; missing {missing}, unknown {unknown}.")
    for key, expected in PDD_FIXED_FIELDS.items():
        if contract[key] != expected:
            raise ValueError(f"{path} {key}={contract[key]!r} is unsupported; expected {expected!r}.")
    parse_base_model_revision(contract["base_model_revision"])
    tile_size = contract["vsa_tile_size"]
    if type(tile_size) is not int or tile_size not in VSA_H3_TILE_SHAPES:
        raise ValueError(f"{path} vsa_tile_size={tile_size!r} must be one of {sorted(VSA_H3_TILE_SHAPES)}.")
    _require_fraction(f"{path} vsa_sparsity", contract["vsa_sparsity"], allow_zero=True)
    # A keep rate of 1 would leave reference-video attention dense.
    _require_fraction(f"{path} vsa_ref_keep_rate", contract["vsa_ref_keep_rate"], allow_zero=False)


def validate_pdd_contract_siblings(contract: dict[str, Any], path: Path) -> None:
    """Values the contract repeats from sibling checkpoint files must equal them."""
    root = path.parent
    for key, relative, sibling_key in _SIBLING_CHECKS:
        sibling = _read_sibling(root, relative)
        if sibling is None:
            continue
        expected = sibling.get(sibling_key)
        if contract[key] != expected:
            raise ValueError(f"{path} {key}={contract[key]!r} disagrees with {relative} {sibling_key}={expected!r}.")
