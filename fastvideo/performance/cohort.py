# SPDX-License-Identifier: Apache-2.0
"""Canonical identity and display helpers for performance benchmark cohorts."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import Any

Record = Mapping[str, Any]

COMPARISON_IDENTITY_KEYS = (
    "workload_id",
    "variant_id",
    "benchmark_version",
    "recipe_fingerprint",
    "hardware_profile_id",
    "software_profile_id",
)


def cohort_value(value: Any) -> str:
    """Normalize one comparable identity value without losing numeric zero."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return str(value)


def comparison_identity_filters(record: Record) -> dict[str, str]:
    """Return the complete v2 comparison identity or raise for malformed data."""
    missing = [key for key in COMPARISON_IDENTITY_KEYS if not cohort_value(record.get(key))]
    if missing:
        raise ValueError("Performance record missing required comparison identity fields: " + ", ".join(missing))
    return {key: cohort_value(record.get(key)) for key in COMPARISON_IDENTITY_KEYS}


def record_uses_v2_identity(record: Record) -> bool:
    """Return whether a record declares or partially carries v2 identity."""
    return str(record.get("result_schema_version") or "") == "2" or any(key in record
                                                                        for key in COMPARISON_IDENTITY_KEYS)


def cohort_schema(record: Record) -> str:
    """Classify a record as a complete v2, malformed v2, or legacy cohort."""
    try:
        comparison_identity_filters(record)
    except ValueError:
        return "invalid_v2" if record_uses_v2_identity(record) else "legacy"
    return "v2"


def cohort_identity(record: Record) -> dict[str, Any]:
    """Return the canonical JSON-serializable identity payload for a record."""
    schema = cohort_schema(record)
    if schema == "v2":
        return {"schema": schema, **comparison_identity_filters(record)}

    identity: dict[str, Any] = {
        "schema": schema,
        "model_id": str(record.get("model_id") or "unknown"),
        "gpu_type": str(record.get("gpu_type") or "unknown"),
    }
    if schema == "invalid_v2":
        identity.update({key: cohort_value(record.get(key)) for key in COMPARISON_IDENTITY_KEYS})
    return identity


def _stable_key(prefix: str, payload: Mapping[str, Any]) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return f"{prefix}:{hashlib.sha256(canonical.encode('utf-8')).hexdigest()}"


def cohort_key(record: Record) -> str:
    """Return the stable opaque key used for grouping, filtering, and URL state."""
    identity = cohort_identity(record)
    return _stable_key(str(identity["schema"]), identity)


def gpu_key(record: Record) -> str:
    """Return a stable key for the high-level GPU configuration filter."""
    schema = cohort_schema(record)
    hardware_profile_id = cohort_value(record.get("hardware_profile_id"))
    if schema == "v2" and hardware_profile_id:
        payload = {"schema": "v2", "hardware_profile_id": hardware_profile_id}
    else:
        payload = {
            "schema": schema,
            "gpu_type": str(record.get("gpu_type") or "unknown"),
            "hardware_profile_id": hardware_profile_id,
        }
    return _stable_key("gpu", payload)


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _sequence(value: Any) -> Sequence[Any]:
    return value if isinstance(value, Sequence) and not isinstance(value, str | bytes) else ()


def _display_number(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    return str(int(number)) if number.is_integer() else f"{number:g}"


def gpu_configuration_label(record: Record) -> str:
    """Build a readable GPU count/model/memory/topology label."""
    profile = _mapping(record.get("hardware_profile"))
    gpus = [_mapping(gpu) for gpu in _sequence(profile.get("gpus"))]
    gpu_count = profile.get("gpu_count")
    if gpu_count is None:
        gpu_count = len(gpus) or None

    names = [str(gpu.get("name") or "unknown") for gpu in gpus]
    unique_names = list(dict.fromkeys(names))
    gpu_name = " + ".join(unique_names) if unique_names else str(record.get("gpu_type") or "unknown")
    primary = f"{gpu_count}× {gpu_name}" if gpu_count not in (None, "") else gpu_name

    memory_values = [gpu.get("memory_gb") for gpu in gpus if gpu.get("memory_gb") is not None]
    if memory_values and all(value == memory_values[0] for value in memory_values):
        primary += f" · {_display_number(memory_values[0])} GB"

    interconnect = cohort_value(profile.get("interconnect"))
    if interconnect and interconnect not in {"unknown", "single_gpu", "none"}:
        primary += f" · {interconnect.replace('_', ' ')}"
    return primary


def software_configuration_label(record: Record) -> str:
    """Build a concise readable software profile label."""
    profile = _mapping(record.get("software_profile"))
    parts = []
    for label, key in (("CUDA", "cuda"), ("PyTorch", "pytorch"), ("Python", "python")):
        value = cohort_value(profile.get(key))
        if value:
            parts.append(f"{label} {value}")
    attention_backend = cohort_value(profile.get("attention_backend"))
    if attention_backend and attention_backend != "auto":
        parts.append(attention_backend)
    return " · ".join(parts) or "Software profile unavailable"


def hardware_configuration_label(record: Record) -> str:
    """Build the hardware label, including host architecture when available."""
    parts = [gpu_configuration_label(record)]
    environment = _mapping(record.get("environment_metadata"))
    machine = cohort_value(_mapping(environment.get("platform")).get("machine"))
    if machine:
        parts.append(machine)
    return " · ".join(parts)


def recipe_configuration_label(record: Record) -> str:
    """Build a readable benchmark recipe label from stable identity metadata."""
    workload = cohort_value(record.get("workload_id"))
    variant = cohort_value(record.get("variant_id"))
    version = cohort_value(record.get("benchmark_version"))
    if not (workload or variant or version):
        return "Legacy benchmark recipe"
    parts = [part for part in (workload, variant) if part]
    if version:
        parts.append(f"v{version}")
    return " · ".join(parts)


def cohort_descriptor(record: Record) -> dict[str, Any]:
    """Return the reusable API/UI description for one canonical cohort."""
    schema = cohort_schema(record)
    raw_ids = {
        "hardware_profile_id": cohort_value(record.get("hardware_profile_id")),
        "software_profile_id": cohort_value(record.get("software_profile_id")),
        "recipe_fingerprint": cohort_value(record.get("recipe_fingerprint")),
    }
    return {
        "key": cohort_key(record),
        "schema": schema,
        "title": recipe_configuration_label(record),
        "gpu_key": gpu_key(record),
        "gpu_label": gpu_configuration_label(record),
        "hardware_label": hardware_configuration_label(record),
        "software_label": software_configuration_label(record),
        "recipe_label": recipe_configuration_label(record),
        "raw_ids": raw_ids,
    }


__all__ = [
    "COMPARISON_IDENTITY_KEYS",
    "cohort_descriptor",
    "cohort_identity",
    "cohort_key",
    "cohort_schema",
    "cohort_value",
    "comparison_identity_filters",
    "gpu_configuration_label",
    "gpu_key",
    "record_uses_v2_identity",
]
