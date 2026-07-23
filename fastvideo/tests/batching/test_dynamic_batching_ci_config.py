# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

_REPO_ROOT = Path(__file__).parents[3]
_PIPELINE_PATH = _REPO_ROOT / ".buildkite" / "pipeline.yml"
_TEST_TYPE_ENV = "TEST_TYPE=dynamic_batching_parity"
_REQUIRED_DEPENDENCY_PATTERNS = {
    ".buildkite/pipeline.yml",
    ".buildkite/scripts/pr_test.sh",
    "fastvideo/models/**",
    "fastvideo/pipelines/**",
    "fastvideo/tests/batching/**",
    "fastvideo/tests/modal/modal_image_utils.py",
    "fastvideo/tests/modal/pr_test.py",
    "fastvideo/worker/**",
}


def _load_pipeline() -> dict[str, Any]:
    return yaml.safe_load(_PIPELINE_PATH.read_text(encoding="utf-8"))


def _dynamic_parity_watchers(pipeline: dict[str, Any]) -> list[dict[str, Any]]:
    watchers = []
    for step in pipeline["steps"]:
        for plugin in step.get("plugins", []):
            for plugin_config in plugin.values():
                for watcher in plugin_config.get("watch", []):
                    if _TEST_TYPE_ENV in watcher.get("config", {}).get("env", []):
                        watchers.append(watcher)
    return watchers


def test_direct_dynamic_batching_parity_retries_fastcheck_context() -> None:
    pipeline = _load_pipeline()
    direct_steps = [
        step for step in pipeline["steps"]
        if "dynamic_batching_parity" in step.get("if", "")
    ]

    assert len(direct_steps) == 1
    assert direct_steps[0]["label"] == ":microscope: Dynamic Batching Parity"


def test_dynamic_batching_parity_lanes_watch_same_dependencies() -> None:
    watchers = _dynamic_parity_watchers(_load_pipeline())

    assert len(watchers) == 2
    assert {watcher["config"]["label"] for watcher in watchers} == {
        ":microscope: Dynamic Batching Parity",
        ":bar_chart: Dynamic Batching Parity",
    }
    watched_paths = [set(watcher["path"]) for watcher in watchers]
    assert watched_paths[0] == watched_paths[1]
    assert _REQUIRED_DEPENDENCY_PATTERNS <= watched_paths[0]
