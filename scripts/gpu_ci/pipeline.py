# SPDX-License-Identifier: Apache-2.0
"""Render an opt-in GPU backend without changing the canonical Slurm graph.

The caller is the operator-owned pipeline uploader. It must load this module
from its reviewed installation, never from a pull request checkout.
"""

from __future__ import annotations

import copy
import re
from typing import Any

BACKENDS = frozenset({"slurm", "modal", "vllm"})
STATUS_SUFFIXES = {
    "fastcheck": "fastcheck-passed",
    "full": "full-suite-passed",
    "merge": "full-suite-passed",
    "direct": "direct-test-completed",
    "scheduled": "scheduled-ssim-passed",
}


def render_pipeline(
    canonical: dict[str, Any],
    backend: str,
    scope: str,
    queue: str = "gpu-ci-dispatch",
) -> dict[str, Any]:
    """Return the original graph or one trusted, backend-specific suite step.

    Only enumerated backend and scope values reach the generated step. Request
    SHA, PR identity, direct lane, and merge plan remain build metadata and are
    validated independently by the trusted dispatcher. Arbitrary canonical
    graph commands, environment, hooks, and notifications are not copied into
    the new control plane.
    """
    backend = backend or "slurm"
    scope = scope or "fastcheck"
    if backend not in BACKENDS:
        raise ValueError(f"Unsupported GPU CI backend: {backend!r}")
    if scope not in STATUS_SUFFIXES:
        raise ValueError(f"Unsupported GPU CI scope: {scope!r}")
    if backend == "slurm":
        return copy.deepcopy(canonical)
    if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_-]{0,63}", queue):
        raise ValueError("GPU CI queue must be an operator-configured queue name")
    return {
        "notify": [{
            "github_commit_status": {
                "context": f"gpu-ci/{backend}/{STATUS_SUFFIXES[scope]}"
            }
        }],
        "steps": [{
            "label": f"GPU CI ({backend}, {scope})",
            "key": f"gpu-ci-{backend}-{scope}",
            "command": "/opt/fastvideo-gpu-ci/run",
            # Admission waiting happens inside this command and uses this
            # budget. An unscheduled Buildkite job has no command timeout yet.
            "timeout_in_minutes": 480,
            "env": {
                "CI_GPU_BACKEND": backend,
                "TEST_SCOPE": scope,
            },
            "agents": {
                "queue": queue
            },
        }],
    }
