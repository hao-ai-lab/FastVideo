# SPDX-License-Identifier: Apache-2.0
"""Metric policy for rolling performance baseline comparisons."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class MetricPolicy:
    key: str
    label: str
    precision: int
    lower_is_better: bool
    threshold_percent: float
    threshold_absolute: float
    gated: bool = True


@dataclass(frozen=True)
class MetricDelta:
    absolute: float
    percent: float
    threshold_exceeded: bool
    regressed: bool


DEFAULT_METRIC_POLICIES: tuple[MetricPolicy, ...] = (
    MetricPolicy("latency", "Latency", 3, True, 0.08, 0.5),
    MetricPolicy("throughput", "Throughput", 3, False, 0.08, 0.05),
    MetricPolicy("memory", "Memory", 1, True, 0.05, 256.0),
    MetricPolicy("text_encoder_time_s", "Text Enc", 3, True, 0.05, 0.25),
    MetricPolicy("dit_time_s", "DiT", 3, True, 0.05, 0.25),
    MetricPolicy("vae_decode_time_s", "VAE Decode", 3, True, 0.05, 0.25),
)

def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _optional_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
    return None


def resolve_metric_policies(
    threshold_overrides: Mapping[str, Any] | None,
) -> tuple[MetricPolicy, ...]:
    """Return default metric policies with optional per-metric overrides."""

    threshold_overrides = threshold_overrides or {}
    policies: list[MetricPolicy] = []
    for base_policy in DEFAULT_METRIC_POLICIES:
        raw_override = threshold_overrides.get(base_policy.key, {})
        if not isinstance(raw_override, Mapping):
            raw_override = {}

        threshold_percent = _optional_float(raw_override.get("threshold_percent"))
        threshold_absolute = _optional_float(raw_override.get("threshold_absolute"))
        gated = _optional_bool(raw_override.get("gated"))

        policies.append(
            MetricPolicy(
                key=base_policy.key,
                label=base_policy.label,
                precision=base_policy.precision,
                lower_is_better=base_policy.lower_is_better,
                threshold_percent=(
                    base_policy.threshold_percent
                    if threshold_percent is None
                    else threshold_percent
                ),
                threshold_absolute=(
                    base_policy.threshold_absolute
                    if threshold_absolute is None
                    else threshold_absolute
                ),
                gated=base_policy.gated if gated is None else gated,
            )
        )
    return tuple(policies)


def serialize_metric_thresholds(
    policies: tuple[MetricPolicy, ...],
) -> dict[str, dict[str, float | bool]]:
    return {
        policy.key: {
            "threshold_percent": policy.threshold_percent,
            "threshold_absolute": policy.threshold_absolute,
            "gated": policy.gated,
        }
        for policy in policies
    }


def regression_delta(
    policy: MetricPolicy,
    current: float,
    baseline: float,
) -> MetricDelta | None:
    if baseline <= 0:
        return None
    if policy.lower_is_better:
        absolute_delta = current - baseline
    else:
        absolute_delta = baseline - current
    percent_delta = absolute_delta / baseline
    threshold_exceeded = (
        percent_delta > policy.threshold_percent
        and absolute_delta > policy.threshold_absolute
    )
    return MetricDelta(
        absolute=absolute_delta,
        percent=percent_delta,
        threshold_exceeded=threshold_exceeded,
        regressed=policy.gated and threshold_exceeded,
    )
