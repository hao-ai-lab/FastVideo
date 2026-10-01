# SPDX-License-Identifier: Apache-2.0
"""CPU-only policy coverage for selectable Buildkite GPU backends."""

from __future__ import annotations

import importlib.util
import json
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
SPEC = importlib.util.spec_from_file_location("gpu_ci_pipeline", REPO_ROOT / "scripts/gpu_ci/pipeline.py")
assert SPEC is not None and SPEC.loader is not None
PIPELINE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PIPELINE)


@pytest.mark.parametrize("backend", ["", "slurm"])
@pytest.mark.parametrize("scope", ["", "fastcheck", "full", "merge", "direct", "scheduled"])
def test_default_preserves_complete_existing_pipeline(backend, scope):
    canonical = yaml.safe_load((REPO_ROOT / ".buildkite/pipeline.yml").read_text())
    rendered = PIPELINE.render_pipeline(canonical, backend, scope)
    assert rendered == canonical
    assert rendered is not canonical
    rendered["steps"][0]["env"]["TEST_TYPE"] = "changed"
    assert canonical["steps"][0]["env"]["TEST_TYPE"] != "changed"


@pytest.mark.parametrize("backend", ["modal", "vllm"])
@pytest.mark.parametrize("scope, context", [
    ("", "fastcheck-passed"),
    ("fastcheck", "fastcheck-passed"),
    ("full", "full-suite-passed"),
    ("merge", "full-suite-passed"),
    ("direct", "direct-test-completed"),
    ("scheduled", "scheduled-ssim-passed"),
])
def test_backend_uses_single_trusted_suite_step_and_distinct_status(backend, scope, context):
    # A PR can alter its pipeline file, but none of these fields may escape
    # into the operator-owned dispatcher pipeline.
    canonical = {
        "env": {"BASH_ENV": "/checkout/attack.sh", "CI_GPU_BACKEND": "slurm"},
        "steps": [{"command": "arbitrary PR command", "plugins": ["unsafe"]}],
        "notify": [{"github_commit_status": {"context": "full-suite-passed"}}],
    }
    rendered = PIPELINE.render_pipeline(canonical, backend, scope)
    assert rendered["notify"] == [{"github_commit_status": {"context": f"gpu-ci/{backend}/{context}"}}]
    assert set(rendered) == {"notify", "steps"}
    assert len(rendered["steps"]) == 1
    step = rendered["steps"][0]
    assert step["command"] == "/opt/fastvideo-gpu-ci/run"
    assert step["agents"] == {"queue": "gpu-ci-dispatch"}
    assert step["env"] == {"CI_GPU_BACKEND": backend, "TEST_SCOPE": scope or "fastcheck"}
    assert step["timeout_in_minutes"] == 480
    assert set(step) == {"label", "key", "command", "timeout_in_minutes", "env", "agents"}
    assert ":microscope:" not in step["label"]
    assert ":test_tube:" not in step["label"]
    assert ":bar_chart:" not in step["label"]


@pytest.mark.parametrize("backend", ["k8s", "VLLM", "vllm; touch /tmp/injected", "$(env)"])
def test_unknown_or_command_like_backend_fails_closed(backend):
    with pytest.raises(ValueError, match="backend"):
        PIPELINE.render_pipeline({}, backend, "fastcheck")


@pytest.mark.parametrize("scope", ["all", "full; true", "$(env)", "FULL"])
def test_unknown_or_command_like_scope_fails_closed(scope):
    with pytest.raises(ValueError, match="scope"):
        PIPELINE.render_pipeline({}, "vllm", scope)


@pytest.mark.parametrize("queue", ["", "queue with spaces", "${QUEUE}", "../ci", "q" * 65])
def test_invalid_dispatch_queue_fails_closed(queue):
    with pytest.raises(ValueError, match="queue"):
        PIPELINE.render_pipeline({}, "vllm", "fastcheck", queue)


def test_trusted_operator_can_choose_a_dedicated_queue():
    rendered = PIPELINE.render_pipeline({}, "vllm", "fastcheck", "gpu-ci-dispatch-canary")
    assert rendered["steps"][0]["agents"] == {"queue": "gpu-ci-dispatch-canary"}


def _promoted_statuses(statuses, backend="vllm"):
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is required to execute the GitHub status workflow's JavaScript")
    workflow = yaml.safe_load((REPO_ROOT / ".github/workflows/ci-gpu-backend-status.yml").read_text())
    script = workflow["jobs"]["promote"]["steps"][0]["with"]["script"]
    harness = """
const fs = require('node:fs');
const input = JSON.parse(fs.readFileSync(0, 'utf8'));
const updates = [];
process.env.SELECTED_BACKEND = input.backend;
const github = {
  paginate: async () => input.statuses,
  rest: {repos: {
    listCommitStatusesForRef: () => {},
    createCommitStatus: async update => {updates.push(update);},
  }},
};
const context = {repo: {owner: 'hao-ai-lab', repo: 'FastVideo'}, payload: {sha: 'a'.repeat(40)}};
const AsyncFunction = Object.getPrototypeOf(async function() {}).constructor;
new AsyncFunction('github', 'context', input.script)(github, context)
  .then(() => process.stdout.write(JSON.stringify(updates)))
  .catch(error => {process.stderr.write(String(error)); process.exitCode = 1;});
"""
    result = subprocess.run([node, "-e", harness], text=True, capture_output=True, check=True,
                            input=json.dumps({"script": script, "statuses": statuses, "backend": backend}))
    return {status["context"]: status["state"] for status in json.loads(result.stdout)}


def _status(context, state="success", second=0, identifier=1):
    return {"context": context, "state": state, "updated_at": f"2026-09-30T01:00:{second:02d}Z", "id": identifier}


def test_backend_statuses_cannot_combine_to_pass_merge_gate():
    statuses = [
        _status("gpu-ci/vllm/fastcheck-passed"),
        _status("gpu-ci/modal/full-suite-passed"),
        _status("full-suite-passed"),
        _status("gpu-ci/vllm/direct-test-completed"),
    ]
    assert _promoted_statuses(statuses) == {"fastcheck-passed": "success", "full-suite-passed": "pending"}


@pytest.mark.parametrize("state", ["pending", "failure", "error"])
def test_newer_selected_backend_non_success_replaces_older_success(state):
    statuses = [
        _status("gpu-ci/vllm/fastcheck-passed", second=1),
        _status("gpu-ci/vllm/full-suite-passed", second=1),
        _status("gpu-ci/vllm/full-suite-passed", state, second=2, identifier=2),
    ]
    assert _promoted_statuses(statuses) == {"fastcheck-passed": "success", "full-suite-passed": state}


def test_status_id_breaks_timestamp_ties_for_same_commit():
    statuses = [
        _status("gpu-ci/modal/fastcheck-passed", "success", identifier=1),
        _status("gpu-ci/modal/fastcheck-passed", "failure", identifier=2),
        _status("gpu-ci/modal/full-suite-passed", "success", identifier=3),
    ]
    assert _promoted_statuses(statuses, "modal") == {"fastcheck-passed": "failure", "full-suite-passed": "success"}


def test_missing_selected_backend_never_reuses_canonical_success():
    statuses = [_status("fastcheck-passed"), _status("full-suite-passed")]
    assert _promoted_statuses(statuses) == {"fastcheck-passed": "pending", "full-suite-passed": "pending"}
