"""CPU-only request, deployment-config, and trusted CLI boundary tests."""

import json
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from scripts.gpu_ci import __main__ as cli
from scripts.gpu_ci.policy import DEFAULT_REPOSITORY, LANES, build_request

ROOT = Path(__file__).resolve().parents[3]


def environment(**overrides):
    return {"CI_GPU_BACKEND": "vllm", "BUILDKITE_REPO": DEFAULT_REPOSITORY,
            "BUILDKITE_COMMIT": "a" * 40, "BUILDKITE_BUILD_ID": "build-1",
            "BUILDKITE_JOB_ID": "job-1", "BUILDKITE_PULL_REQUEST": "12", **overrides}


def test_lane_policy_matches_existing_complete_graph():
    pipeline = yaml.safe_load((ROOT / ".buildkite/pipeline.yml").read_text())
    assert {lane["key"] for lane in LANES} == {step["key"] for step in pipeline["steps"]}
    assert sum(lane["fastcheck"] for lane in LANES) == 6
    assert all((ROOT / ".buildkite/scripts" / lane["script"]).is_file() for lane in LANES)


@pytest.mark.parametrize("scope,count", [("fastcheck", 6), ("full", 20), ("scheduled", 1)])
def test_scope_selects_existing_payloads(scope, count):
    request = build_request(environment(TEST_SCOPE=scope), {}, ROOT)
    assert len(request["lanes"]) == count
    assert all(lane["script"].startswith(".buildkite/scripts/") for lane in request["lanes"])
    assert request["pr_key"] == DEFAULT_REPOSITORY + "#12"


@pytest.mark.parametrize("lane", LANES, ids=lambda lane: lane["key"])
def test_direct_public_and_compatibility_names(lane):
    for name in (lane["public_type"], lane["public_type"] + "_ci", lane["key"]):
        request = build_request(environment(TEST_SCOPE="direct", TEST_TYPE=name), {}, ROOT)
        assert [item["key"] for item in request["lanes"]] == [lane["key"]]


def test_docs_only_merge_plan_is_explicitly_empty():
    assert build_request(environment(TEST_SCOPE="merge", MERGE_TEST_PLAN=",none,"), {}, ROOT)["lanes"] == []


@pytest.mark.parametrize("plan", ["", ",", ",,", ",none,,", ",,golden-gate,,", ",ssim,ssim,",
                                 ",unit,", ",none,ssim,", ",unknown,", "ssim", ",../ssim,"])
def test_malformed_merge_plan_fails_closed(plan):
    with pytest.raises(ValueError):
        build_request(environment(TEST_SCOPE="merge", MERGE_TEST_PLAN=plan), {}, ROOT)


def test_focused_new_pr_test_can_be_validated_inside_worker():
    request = build_request(environment(TEST_SCOPE="merge", MERGE_TEST_PLAN=",golden-gate,ssim,",
                                       MERGE_GOLDEN_TESTS="test_new_model.py", MERGE_SSIM_TESTS="all"), {}, ROOT)
    assert request["env"]["FASTVIDEO_GOLDEN_TEST_FILES"] == "test_new_model.py"
    assert request["env"]["FASTVIDEO_SSIM_TEST_FILES"] == "all"


@pytest.mark.parametrize("files", ["", "../test_model.py", "test_a.py,test_a.py", "$(command)", "test_x.py/evil"])
def test_quality_selection_rejects_missing_or_unsafe_basename(files):
    with pytest.raises(ValueError):
        build_request(environment(TEST_SCOPE="merge", MERGE_TEST_PLAN=",ssim,", MERGE_SSIM_TESTS=files), {}, ROOT)


@pytest.mark.parametrize("field,value", [("BUILDKITE_REPO", "https://github.com/other/repo.git"),
                                       ("BUILDKITE_COMMIT", "main"), ("BUILDKITE_JOB_ID", "../escape"),
                                       ("BUILDKITE_PULL_REQUEST", "-1"), ("TEST_SCOPE", "unknown"),
                                       ("CI_GPU_BACKEND", "vllm;cmd"), ("FASTVIDEO_SSIM_BOOTSTRAP_MODE", "1")])
def test_untrusted_request_fields_are_rejected(field, value):
    with pytest.raises(ValueError):
        build_request(environment(**{field: value}), {}, ROOT)


def test_job_retry_has_distinct_attempt_but_shares_pr():
    first = build_request(environment(), {}, ROOT)
    retry = build_request(environment(BUILDKITE_JOB_ID="job-2"), {}, ROOT)
    assert first["build_id"] != retry["build_id"]
    assert first["pr_key"] == retry["pr_key"]


def test_credentials_and_shell_controls_never_enter_worker_request():
    request = build_request(environment(BUILDKITE_AGENT_TOKEN="secret", HF_TOKEN="secret", BASH_ENV="bad"), {}, ROOT)
    assert "secret" not in json.dumps(request)
    assert "BASH_ENV" not in request["env"]


def test_config_does_not_allow_raising_agreed_limits(tmp_path):
    config = {"state_path": str(tmp_path / "state"), "artifacts_dir": str(tmp_path / "runs")}
    path = tmp_path / "config.json"
    for key, value in (("max_active_prs", 3), ("max_gpus_per_pr", 5), ("max_gpus", 9)):
        path.write_text(json.dumps({**config, key: value}))
        with pytest.raises(ValueError, match="ceiling"):
            cli.load_config(path)


def test_promoted_backend_blocks_legacy_status_override(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"default_backend": "vllm", "state_path": str(tmp_path / "state"),
                                "artifacts_dir": str(tmp_path / "runs"), "slurm_uploader": ["/trusted/upload"]}))
    with patch.dict(cli.os.environ, {"CI_GPU_BACKEND": "slurm"}, clear=True), patch.object(cli.subprocess, "run") as run:
        assert cli.main(["upload", "--config", str(path)]) == 97
        run.assert_not_called()


def test_original_uploader_is_delegated_without_checkout_or_shell(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"state_path": str(tmp_path / "state"), "artifacts_dir": str(tmp_path / "runs"),
                                "slurm_uploader": ["/trusted/upload"]}))
    with patch.dict(cli.os.environ, {}, clear=True), patch.object(cli.subprocess, "run") as run:
        run.return_value.returncode = 0
        assert cli.main(["upload", "--config", str(path)]) == 0
        run.assert_called_once_with(["/trusted/upload"], check=False)
