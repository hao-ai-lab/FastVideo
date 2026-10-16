# SPDX-License-Identifier: Apache-2.0
"""The model catalogue the Studio offers, with what each model accepts.

The frontend decides which inputs to show from ``capabilities`` (and locks
fields from ``recipe``) rather than from model names. Entries come from the
FastVideo registry and from FastH3-Omni exports configured on the server.
"""

from __future__ import annotations

from typing import Any

from fastvideo_studio.fasth3_omni import (H3_FPS, H3_MAX_FRAMES, MAX_REFERENCES, RECIPE_DEFAULTS, REFERENCE_LIMITS,
                                          VALID_NUM_GPUS, FastH3OmniRegistry)
from fastvideo_studio.models import model_label

# Registered checkpoints that ship MiniMax-H3's reference transformer
# (transformer_ref) and its first/last-frame path. The distilled FastH3
# preview and V2 checkpoints are T2AV-only.
MINIMAX_H3_REF2VA_MODELS = frozenset({"MiniMaxAI/MiniMax-H3"})


def _reference_capability(required: bool) -> dict[str, Any]:
    return {
        "media_types": list(REFERENCE_LIMITS),
        "limits": dict(REFERENCE_LIMITS),
        "max_total": MAX_REFERENCES,
        "requires_visual": True,
        "required": required,
    }


def registered_entry(model_id: str, workload_types: list[str] | None = None) -> dict[str, Any]:
    ref2va = model_id in MINIMAX_H3_REF2VA_MODELS
    return {
        "id": model_id,
        "label": model_label(model_id),
        "workload_types": workload_types or [],
        "config_source": "registry",
        "capabilities": {
            "references": _reference_capability(required=False) if ref2va else None,
            "start_frame": True,
            "end_frame": ref2va,
            "max_frames": H3_MAX_FRAMES if ref2va else None,
            "num_gpus": None,
        },
        "recipe": None,
        "recipe_defaults": None,
        "hardware": None,
        "error": None,
    }


def omni_entries(registry: FastH3OmniRegistry | None) -> list[dict[str, Any]]:
    if registry is None:
        return []
    entries = []
    for export in registry.exports.values():
        entries.append({
            "id": export.id,
            "label": export.label,
            "workload_types": ["i2v"],
            "config_source": "server",
            "capabilities": {
                "references": _reference_capability(required=True),
                "start_frame": False,
                "end_frame": False,
                "max_frames": H3_MAX_FRAMES,
                "num_gpus": list(VALID_NUM_GPUS),
                "fixed_fps": H3_FPS,
            },
            "recipe": export.recipe(),
            "recipe_defaults": dict(RECIPE_DEFAULTS),
            "hardware": export.hardware,
            "error": export.error,
        })
    return entries


def build_catalog(registered: list[dict[str, Any]], registry: FastH3OmniRegistry | None,
                  workload_type: str | None) -> list[dict[str, Any]]:
    """Registered models (``[{id, workload_types}]``) followed by the configured Omni exports."""
    entries = [registered_entry(m["id"], m.get("workload_types")) for m in registered]
    entries += omni_entries(registry)
    if workload_type:
        wanted = workload_type.lower()
        entries = [e for e in entries if wanted in e["workload_types"]]
    return entries
