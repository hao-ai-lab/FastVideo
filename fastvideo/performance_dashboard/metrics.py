# SPDX-License-Identifier: Apache-2.0
"""Metric definitions shared by the performance dashboard backend."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MetricDefinition:
    key: str
    label: str
    precision: int
    lower_is_better: bool


METRICS: tuple[MetricDefinition, ...] = (
    MetricDefinition("latency", "Latency", 3, True),
    MetricDefinition("throughput", "Throughput", 3, False),
    MetricDefinition("memory", "Memory", 1, True),
    MetricDefinition("text_encoder_time_s", "Text Encoder", 3, True),
    MetricDefinition("dit_time_s", "DiT", 3, True),
    MetricDefinition("vae_decode_time_s", "VAE Decode", 3, True),
)

METRIC_BY_KEY = {metric.key: metric for metric in METRICS}
