# SPDX-License-Identifier: Apache-2.0
"""Shared pytest rerun policy for Modal CI entrypoints."""

from __future__ import annotations

import shlex

PYTEST_RERUNS = 2
PYTEST_RERUNS_DELAY_SECONDS = 10

TRANSIENT_FAILURE_PATTERNS = (
    r"connection (?:reset|aborted|refused|timed out)",
    r"read timed out",
    r"remote end closed connection",
    r"temporarily unavailable",
    r"(?:HTTP|status code) 5\d\d",
    r"502 Bad Gateway",
    r"503 Service Unavailable",
    r"504 Gateway Timeout",
    r"NCCL.*(?:timeout|connection|remote error|unhandled system error|system error)",
    r"CUDA error: (?:unknown error|initialization error|operation not permitted|operation not supported)",
    r"Modal.*(?:preempt|interrupt|worker|container|timeout)",
)
TRANSIENT_FAILURE_REGEX = "(?i)(" + "|".join(TRANSIENT_FAILURE_PATTERNS) + ")"


def build_pytest_rerun_args() -> list[str]:
    return [
        "--reruns",
        str(PYTEST_RERUNS),
        "--reruns-delay",
        str(PYTEST_RERUNS_DELAY_SECONDS),
        "--rerun-show-tracebacks",
        "--only-rerun",
        TRANSIENT_FAILURE_REGEX,
    ]


def build_pytest_addopts(existing_addopts: str = "") -> str:
    retry_addopts = shlex.join(build_pytest_rerun_args())
    existing_addopts = existing_addopts.strip()
    if existing_addopts:
        return f"{existing_addopts} {retry_addopts}"
    return retry_addopts


def describe_pytest_reruns() -> str:
    return (
        "Pytest transient reruns enabled: "
        f"{PYTEST_RERUNS} reruns, {PYTEST_RERUNS_DELAY_SECONDS}s delay, "
        f"only-rerun={TRANSIENT_FAILURE_REGEX!r}"
    )
