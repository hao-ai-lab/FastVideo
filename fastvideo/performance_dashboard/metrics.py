# SPDX-License-Identifier: Apache-2.0
"""Metric definitions shared by the performance dashboard backend."""

from fastvideo.performance.metric_policy import DEFAULT_METRIC_POLICIES

METRICS = DEFAULT_METRIC_POLICIES

METRIC_BY_KEY = {metric.key: metric for metric in METRICS}
