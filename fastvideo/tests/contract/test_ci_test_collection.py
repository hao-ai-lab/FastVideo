# SPDX-License-Identifier: Apache-2.0
"""Guard: every test directory must be collected by some CI lane or be on
the explicit allowlist below.

Three separate incidents on 2026-07-05 found test files that no CI lane
ever collects (fastvideo/tests/stages/, tests/local_tests/ additions in
PR #1509, and this sweep found seven dark directories in total): the tests
pass review, merge, and then silently never run. This test makes going
dark an explicit, reviewed decision instead of an accident: adding a new
test directory fails CI until it is either wired into a lane or
allowlisted here with a reason.

Pure text analysis — no fastvideo imports, no GPU, no torch.
"""
import re
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
TESTS_ROOT = REPO_ROOT / "fastvideo" / "tests"

# Files whose text constitutes "a CI lane references this directory".
CI_SOURCES = [
    *sorted((REPO_ROOT / ".buildkite").rglob("*.yml")),
    *sorted((REPO_ROOT / ".buildkite").rglob("*.sh")),
    *sorted((REPO_ROOT / ".github/workflows").glob("ci-*.yml")),
]

# (slash name, compatibility slash name, public TEST_TYPE, runner TEST_TYPE,
#  Buildkite step key, fastcheck lane, pinned command)
SLURM_LANES = [
    ("encoder", "encoder-ci", "encoder", "encoder_ci", "encoder", True, "run-ci"),
    ("vae", "vae-ci", "vae", "vae_ci", "vae", True, "run-ci"),
    ("transformer", "transformer-ci", "transformer", "transformer_ci", "transformer", True, "run-ci"),
    ("kernel", "kernel-ci", "kernel_tests", "kernel_tests_ci", "kernel-tests", True, "run-ci"),
    ("unit", "unit-ci", "unit_test", "unit_test_ci", "unit", True, "run-unit"),
    ("dreamverse", "dreamverse-ci", "dreamverse_app", "dreamverse_app_ci", "dreamverse", True, "run-ci"),
    ("golden-gate", "golden-gate-ci", "golden_gate", "golden_gate_ci", "golden-gate", False, "run-ci"),
    ("ssim", "ssim-ci", "ssim", "ssim_ci", "ssim", False, "run-ci"),
    ("lora-inference", "lora-inference-ci", "inference_lora", "inference_lora_ci", "lora-inference", False,
     "run-ci"),
    ("lora-extraction", "lora-extraction-ci", "lora_extraction", "lora_extraction_ci", "lora-extraction", False,
     "run-ci"),
    ("training", "training-ci", "training", "training_ci", "training", False, "run-ci"),
    ("distillation", "distillation-ci", "distillation_dmd", "distillation_dmd_ci", "distillation", False,
     "run-ci"),
    ("self-forcing", "self-forcing-ci", "self_forcing", "self_forcing_ci", "self-forcing", False, "run-ci"),
    ("lora-training", "lora-training-ci", "training_lora", "training_lora_ci", "lora-training", False, "run-ci"),
    ("vsa", "vsa-ci", "training_vsa", "training_vsa_ci", "training-vsa", False, "run-ci"),
    ("vmoba", "vmoba-ci", "inference_vmoba", "inference_vmoba_ci", "inference-vmoba", False, "run-ci"),
    ("performance", "performance-ci", "performance", "performance_ci", "performance", False, "run-ci"),
    ("api", "api-ci", "api_server", "api_server_ci", "api-server", False, "run-ci"),
    ("train-framework", "train-framework-ci", "train_framework", "train_framework_ci", "train-framework", False,
     "run-ci"),
    ("eval", "eval-ci", "eval", "eval_ci", "eval", False, "run-ci"),
]

# Directories that intentionally have no CI lane today. Every entry needs a
# reason; remove the entry when the directory gets wired into a lane.
# State as found on 2026-07-05 — these SHOULD shrink over time, not grow.
ALLOWLIST = {
    "attention": "no lane yet — GPU attention-backend tests, run manually",
    "audio": "no lane yet — audio encoder tests, run manually",
    "distributed": "no lane yet — multi-GPU torchrun tests, run manually",
    "hooks": "no lane yet — run manually",
    "layers": "no lane yet — torchrun FSDP dispatch tests, run manually",
    "nightly": "by design: nightly cadence, not per-PR",
    "modal": "CI infrastructure itself, not a test suite",
}


def _dirs_with_tests() -> list[str]:
    dirs = []
    for child in sorted(TESTS_ROOT.iterdir()):
        if child.is_dir() and any(child.rglob("test_*.py")):
            dirs.append(child.name)
    return dirs


def _ci_text() -> str:
    return "\n".join(src.read_text(errors="replace") for src in CI_SOURCES if src.exists())


def test_every_test_directory_is_collected_or_allowlisted():
    ci_text = _ci_text()
    dark = [name for name in _dirs_with_tests() if f"tests/{name}" not in ci_text and name not in ALLOWLIST]
    assert not dark, (f"Test directories not referenced by any CI lane and not "
                      f"allowlisted: {dark}. Wire them into a lane in "
                      f"a .buildkite lane script, or add an "
                      f"allowlist entry with a reason in {__file__}.")


def test_local_tests_stays_out_of_ci():
    # tests/local_tests/ (repo root) is developer-local by design (author
    # decision, 2026-07-05): parity scaffolds and machine-specific checks
    # that must never gate CI. Fail if any CI source starts collecting it.
    assert "tests/local_tests" not in _ci_text(), ("tests/local_tests/ is local-only by design; remove the CI "
                                                   "reference or move the tests into a fastvideo/tests/ lane.")


def _pipeline_steps() -> list[dict[str, object]]:
    pipeline = yaml.safe_load((REPO_ROOT / ".buildkite/pipeline.yml").read_text())
    return pipeline["steps"]


def test_all_gpu_ci_routes_use_the_trusted_slurm_dispatcher():
    slash_commands = (REPO_ROOT / ".github/workflows/ci-slash-commands.yml").read_text()
    steps = _pipeline_steps()
    valid_line = next(line for line in slash_commands.splitlines() if line.strip().startswith("VALID="))
    valid_names = set(valid_line.split('"', maxsplit=2)[1].split())
    assert len(steps) == len(SLURM_LANES) == 20
    by_key = {step["key"]: step for step in steps}
    assert len(by_key) == len(steps)

    for slash_name, compatibility_name, public_type, runner_type, step_key, fastcheck, command in SLURM_LANES:
        assert slash_name in valid_names
        assert compatibility_name in valid_names
        assert f"[{slash_name}]={public_type}" in slash_commands
        assert f"[{compatibility_name}]={runner_type}" in slash_commands

        step = by_key[step_key]
        assert step["command"] == f"/opt/fastvideo-ci-runner/{command}"
        assert step["timeout_in_minutes"] == 90
        assert step["agents"] == {"queue": "ci-runner"}
        assert step["env"] == {"TEST_TYPE": runner_type}
        assert "soft_fail" not in step
        if step_key in {"ssim", "training"}:
            assert step["concurrency"] == 1
            assert step["concurrency_group"] == "fastvideo/slinky/whole-tray"
        else:
            assert "concurrency" not in step
            assert "concurrency_group" not in step
        condition = str(step["if"])
        assert 'build.env("TEST_SCOPE") == "full"' in condition
        assert 'build.env("TEST_SCOPE") == "merge"' in condition
        assert f'/,{step_key},/' in condition
        assert 'build.env("TEST_SCOPE") == "direct"' in condition
        assert f'build.env("TEST_TYPE") == "{public_type}"' in condition
        assert f'build.env("TEST_TYPE") == "{runner_type}"' in condition
        if fastcheck:
            assert 'build.env("TEST_SCOPE") == "fastcheck"' in condition
            assert 'build.env("TEST_SCOPE") == null' in condition


def test_merge_comment_has_one_change_aware_trigger_path():
    slash_commands = (REPO_ROOT / ".github/workflows/ci-slash-commands.yml").read_text()
    merge_job = slash_commands.split("  parse-command:", maxsplit=1)[0]
    ready_workflow = (REPO_ROOT / ".github/workflows/ci-trigger-full-suite.yml").read_text()

    assert "labels: ['ready']" in merge_job
    assert "api.buildkite.com" not in merge_job
    assert "BUILDKITE_API_TOKEN" not in merge_job
    assert "api.buildkite.com" in ready_workflow
    assert 'jq -r --arg pr_number "$PR_NUMBER"' in ready_workflow
    assert '.env.PR_NUMBER? == $pr_number' in ready_workflow
    assert 'TEST_SCOPE: "merge"' in ready_workflow
    assert 'FULL_SUITE: "true"' in ready_workflow
    assert "plan_merge_ci.py" in ready_workflow
    assert "github.event.pull_request.base.sha" in ready_workflow
    assert "MERGE_TEST_PLAN" in ready_workflow
    assert "MERGE_GOLDEN_TESTS" in ready_workflow
    assert "MERGE_SSIM_TESTS" in ready_workflow
    assert "__FASTVIDEO_CI_PLAN_ALL__" in ready_workflow


def test_full_ssim_has_a_weekly_slurm_schedule():
    workflow = (REPO_ROOT / ".github/workflows/ci-scheduled-ssim.yml").read_text()
    ssim_step = next(step for step in _pipeline_steps() if step["key"] == "ssim")

    assert 'cron: "0 5 * * 0"' in workflow
    assert 'TEST_SCOPE: "scheduled"' in workflow
    assert 'TEST_TYPE: "ssim"' in workflow
    assert "modal" not in workflow.lower()
    assert 'build.env("TEST_SCOPE") == "scheduled"' in str(ssim_step["if"])


def test_active_pipeline_has_no_modal_or_untrusted_compute_path():
    pipeline_text = (REPO_ROOT / ".buildkite/pipeline.yml").read_text()
    steps = _pipeline_steps()

    assert ".buildkite/scripts/pr_test.sh" not in pipeline_text
    assert "monorepo-diff" not in pipeline_text
    assert all(step.get("agents") == {"queue": "ci-runner"} for step in steps)
    assert all("plugins" not in step for step in steps)
    assert all(str(step.get("command", "")).startswith("/opt/fastvideo-ci-runner/") for step in steps)


def test_gpu_tests_preserve_a_launcher_assigned_rendezvous_port():
    hard_assignment = re.compile(r'''os\.environ\[\s*["']MASTER_PORT["']\s*\]\s*=''')
    overwrites = []
    for path in TESTS_ROOT.rglob("*.py"):
        if path.resolve() == Path(__file__).resolve():
            continue
        if hard_assignment.search(path.read_text(errors="replace")):
            overwrites.append(str(path.relative_to(REPO_ROOT)))
    assert not overwrites, f"Tests overwrite the CI runner's per-lease MASTER_PORT: {overwrites}"


def test_dreamverse_lane_keeps_arm64_browser_coverage_explicit():
    lane = (REPO_ROOT / ".buildkite/scripts/lanes/dreamverse.sh").read_text()

    assert "deb.nodesource.com" not in lane
    assert "node_version=v22.23.2" in lane
    assert "mktemp -d -t fastvideo-node.XXXXXX" in lane
    assert "sha256sum --check --status" in lane
    assert "aarch64|arm64" in lane
    assert "--project=firefox" in lane
    assert "--project=chromium" in lane
    assert "--project=mobile-chromium" in lane
    assert "--grep-invert=" in lane
    assert "--project=webkit" in lane
    assert "--project=mobile-safari" in lane


def test_training_lanes_keep_tracking_offline_and_secret_free():
    lane_root = REPO_ROOT / ".buildkite/scripts/lanes"
    for lane_name in ("training.sh", "self_forcing.sh", "training_lora.sh", "training_vsa.sh"):
        lane = (lane_root / lane_name).read_text()
        assert "export WANDB_MODE=offline" in lane
        assert "WANDB_API_KEY" not in lane
        assert "wandb login" not in lane


def test_slurm_lane_status_contexts_are_unique_and_aggregatable():
    labels = [str(step["label"]) for step in _pipeline_steps()]
    aggregate = (REPO_ROOT / ".github/workflows/ci-aggregate-status.yml").read_text()

    assert len(labels) == len(set(labels))
    assert sum(label.startswith(":microscope:") for label in labels) == 6
    assert all(label.startswith((":microscope:", ":test_tube:", ":bar_chart:")) for label in labels)
    assert "'buildkite/pr-fastcheck/microscope-'," in aggregate
    assert "'buildkite/ci/microscope-'," in aggregate
    assert "'buildkite/ci/test-tube-'," in aggregate
    assert "'buildkite/ci/bar-chart-'," in aggregate
    assert "fastcheck.size === 6" in aggregate
    assert "fullSuiteOnly.size === 14" in aggregate
    assert "fastcheckPassed" in aggregate


def test_ssim_lane_uses_the_local_four_gpu_scheduler():
    lane_script = (REPO_ROOT / ".buildkite/scripts/lanes/ssim.sh").read_text()
    scheduler = (TESTS_ROOT / "ssim/ci_runner.py").read_text()
    dockerfile = (REPO_ROOT / "docker/Dockerfile").read_text()

    assert "fastvideo/tests/ssim/ci_runner.py" in lane_script
    assert "libx11-dev" in lane_script
    assert "libx11-dev" in dockerfile
    assert "Skipping FA4 cute overlay on arm64" not in dockerfile
    assert "import flash_attn.cute" in dockerfile
    assert "import modal" not in scheduler
    assert "MAX_GPUS = 4" in scheduler
    assert "REQUIRED_GPUS" in scheduler
    assert "MODEL_TO_PARAMS" in scheduler
    assert "--test-file" in scheduler
    assert "FASTVIDEO_SSIM_TEST_FILES" in lane_script
    assert 'if [ "${TEST_SCOPE:-}" = merge ]; then' in lane_script
    assert "Missing FASTVIDEO_SSIM_TEST_FILES for merge scope" in lane_script


def test_golden_lane_accepts_only_focused_test_basenames():
    lane_script = (REPO_ROOT / ".buildkite/scripts/lanes/golden_gate.sh").read_text()

    assert "FASTVIDEO_GOLDEN_TEST_FILES" in lane_script
    assert "test_[a-z0-9_]+" in lane_script
    assert 'golden_root=./fastvideo/tests/golden_gate' in lane_script
    assert 'if [ "${TEST_SCOPE:-}" = merge ]; then' in lane_script
    assert "Missing FASTVIDEO_GOLDEN_TEST_FILES for merge scope" in lane_script


def test_allowlist_entries_are_still_real_directories():
    # A stale allowlist hides regressions; entries must track reality.
    missing = [name for name in ALLOWLIST if name != "modal" and not (TESTS_ROOT / name).is_dir()]
    assert not missing, (f"Allowlisted directories no longer exist — remove them: {missing}")
