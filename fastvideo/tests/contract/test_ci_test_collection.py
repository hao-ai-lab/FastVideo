# SPDX-License-Identifier: Apache-2.0
"""Guard: every test file must be collected by some CI lane or be on the
explicit allowlist below.

Three separate incidents on 2026-07-05 found test files that no CI lane
ever collects (fastvideo/tests/stages/, tests/local_tests/ additions in
PR #1509, and this sweep found seven dark directories in total): the tests
pass review, merge, and then silently never run. A directory-level check
misses the common case where a lane names a directory or a few files and a
new file lands beside them, so this guard resolves what each lane actually
passes to pytest and checks every test file: a new file fails CI until it
is wired into a lane or allowlisted here with a reason.

Pure text analysis: no fastvideo imports, no GPU, no torch. A lane that
selects tests through a variable, a glob or a generated list is not
resolved, so its files look dark; wire them by name or add an entry.
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

# Test files that intentionally have no CI lane today. Every entry needs a
# reason; delete the entry when the file is wired into a lane. This list only
# shrinks: test_dark_test_allowlist_stays_accurate fails when an entry stops
# being dark, so a fixed entry is removed instead of left stale.
# State as found on 2026-11-10.
DARK_TESTS = {
    # Audio
    "fastvideo/tests/audio/test_ltx2_bwe.py": "no lane: audio encoder test",
    # Multi-GPU collectives
    "fastvideo/tests/distributed/test_sp_hunyuanvideo.py": "no lane: multi-GPU torchrun test",
    "fastvideo/tests/distributed/test_sp_lingbot_video.py": "no lane: multi-GPU torchrun test",
    "fastvideo/tests/distributed/test_sp_ltx2.py": "no lane: multi-GPU torchrun test",
    "fastvideo/tests/distributed/test_sp_wan.py": "no lane: multi-GPU torchrun test",
    "fastvideo/tests/distributed/test_ulysses_a2a_parity.py": "no lane: multi-GPU torchrun test",
    "fastvideo/tests/distributed/test_ulysses_fault_injection.py": "no lane: multi-GPU torchrun test",
    "fastvideo/tests/distributed/test_ulysses_h3_policy.py": "no lane: multi-GPU torchrun test",
    # Runtime hooks
    "fastvideo/tests/hooks/test_activation_trace.py": "no lane: hook-system test",
    "fastvideo/tests/hooks/test_hooks_api.py": "no lane: hook-system test",
    "fastvideo/tests/hooks/test_layerwise_offload.py": "no lane: hook-system test",
    "fastvideo/tests/hooks/test_pinned_memory.py": "no lane: hook-system test",
    # Inference smoke and benchmarks
    "fastvideo/tests/inference/bsa/test_bsa_inference.py": "no lane: run manually",
    "fastvideo/tests/inference/lora/test_merge_lora_math.py": "no lane: run manually",
    "fastvideo/tests/inference/test_basic_fasth3_profile.py": "no lane: run manually",
    "fastvideo/tests/inference/test_benchmark_fasth3_spark_nvfp4.py": "no lane: run manually",
    # Tensor-parallel layers
    "fastvideo/tests/layers/test_linear_mlp_quant_gating.py": "no lane: FSDP or torchrun dispatch test",
    "fastvideo/tests/layers/test_lora_merge_precision.py": "no lane: FSDP or torchrun dispatch test",
    "fastvideo/tests/layers/test_mxfp8_optional_quack.py": "no lane: FSDP or torchrun dispatch test",
    "fastvideo/tests/layers/test_mxfp8_scaled_mm_scale_dtype.py": "no lane: FSDP or torchrun dispatch test",
    "fastvideo/tests/layers/test_nvfp4_mm_backend.py": "no lane: FSDP or torchrun dispatch test",
    "fastvideo/tests/layers/test_rmsnorm_forward_dispatch.py": "no lane: FSDP or torchrun dispatch test",
    # Apple MLX (the macOS smoke workflow names its files explicitly)
    "fastvideo/tests/mlx/test_import_laziness.py": "not in the macOS MLX smoke selection",
    "fastvideo/tests/mlx/test_mlx_diffusers_compile_dispatch.py": "not in the macOS MLX smoke selection",
    "fastvideo/tests/mlx/test_mlx_minimax_h3_8step_schedule.py": "not in the macOS MLX smoke selection",
    "fastvideo/tests/mlx/test_mlx_minimax_h3_bf16_checkpoint.py": "not in the macOS MLX smoke selection",
    "fastvideo/tests/mlx/test_mlx_minimax_h3_conditioner.py": "not in the macOS MLX smoke selection",
    "fastvideo/tests/mlx/test_mlx_minimax_h3_modulation.py": "not in the macOS MLX smoke selection",
    "fastvideo/tests/mlx/test_mlx_minimax_h3_vsa_ref2va.py": "not in the macOS MLX smoke selection",
    "fastvideo/tests/mlx/test_mlx_minimax_h3_vsa_simd.py": "not in the macOS MLX smoke selection",
    "fastvideo/tests/mlx/test_mlx_minimax_h3_weight_reads.py": "not in the macOS MLX smoke selection",
    "fastvideo/tests/mlx/test_mlx_taeh3.py": "not in the macOS MLX smoke selection",
    "fastvideo/tests/mlx/test_mlx_taehv_parity.py": "not in the macOS MLX smoke selection",
    "fastvideo/tests/mlx/test_mlx_wan22_memory_handoff.py": "not in the macOS MLX smoke selection",
    "fastvideo/tests/mlx/test_mlx_wan22_parity.py": "not in the macOS MLX smoke selection",
    "fastvideo/tests/mlx/test_mlx_wan22_real_weights_parity.py": "not in the macOS MLX smoke selection",
    "fastvideo/tests/mlx/test_mlx_wan_vae_denormalization.py": "not in the macOS MLX smoke selection",
    "fastvideo/tests/mlx/test_quant_backends.py": "not in the macOS MLX smoke selection",
    # Nightly cadence
    "fastvideo/tests/nightly/test_e2e_dmd_t2v_crush_smol.py": "by design: nightly cadence, not per-PR",
    "fastvideo/tests/nightly/test_e2e_i2v_overfit_single_sample.py": "by design: nightly cadence, not per-PR",
    "fastvideo/tests/nightly/test_e2e_kandinsky5_dmd_t2v_overfit.py": "by design: nightly cadence, not per-PR",
    "fastvideo/tests/nightly/test_e2e_ltx2_overfit_new_stack.py": "by design: nightly cadence, not per-PR",
    "fastvideo/tests/nightly/test_e2e_overfit_single_sample.py": "by design: nightly cadence, not per-PR",
    # Legacy training and distillation
    "fastvideo/tests/training/checkpoint/test_checkpointing_utils.py": "no lane: run manually",
    "fastvideo/tests/training/distill/test_anyflow_onpolicy.py": "no lane: run manually",
    "fastvideo/tests/training/distill/test_anyflow_pretrain.py": "no lane: run manually",
    "fastvideo/tests/training/distill/test_anyflow_smoke.py": "no lane: run manually",
    "fastvideo/tests/training/distill/test_wan_i2v_distillation.py": "no lane: run manually",
    # Package-root tests
    "fastvideo/tests/test_logger.py": "no lane: run manually",
    "fastvideo/tests/test_partial_hf_download.py": "no lane: run manually",
}

_PATH_RE = re.compile(r"(?:\./)?(?:fastvideo|apps)/[A-Za-z0-9_./-]+")
_ASSIGNMENT_RE = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_]*)=(\.?/?(?:fastvideo|apps)/[A-Za-z0-9_./-]+)\s*$")


def _ci_text() -> str:
    return "\n".join(src.read_text(errors="replace") for src in CI_SOURCES if src.exists())


def _normalise(path: str) -> str:
    return path[2:] if path.startswith("./") else path


def _pytest_invocations() -> list[tuple[set[str], set[str], set[str]]]:
    """Resolve what each CI ``pytest`` invocation collects.

    Returns one ``(files, directories, ignored)`` tuple per invocation. A shell
    variable assigned a repository path (``golden_root=./fastvideo/tests/...``)
    is substituted into the command, and ``--ignore=`` subtrees are recorded so
    they do not count as coverage for that invocation. ``ssim/ci_runner.py``
    discovers ``test_*.py`` under ``ssim/`` itself, so that directory counts as
    collected.
    """
    invocations: list[tuple[set[str], set[str], set[str]]] = []
    for source in CI_SOURCES:
        if not source.exists():
            continue
        # Shell and YAML continuations keep one pytest command on one line.
        text = re.sub(r"\\\n", " ", source.read_text(errors="replace"))
        variables: dict[str, str] = {}
        for line in text.splitlines():
            match = _ASSIGNMENT_RE.match(line)
            if match is not None:
                variables[match.group(1)] = match.group(2)
        for line in text.splitlines():
            if "pytest" not in line and "ci_runner.py" not in line:
                continue
            expanded = line
            for name, value in variables.items():
                expanded = expanded.replace("${" + name + "}", value).replace("$" + name, value)
            tokens = [token.strip("'\"()") for token in expanded.split()]
            ignored = {
                _normalise(token.split("=", 1)[1].rstrip("/"))
                for token in tokens
                if token.startswith("--ignore=")
            }
            files: set[str] = set()
            directories: set[str] = set()
            for token in tokens:
                token = token.strip("'\"),")
                if token.startswith("-"):
                    continue
                match = _PATH_RE.match(token)
                if match is None:
                    continue
                path = _normalise(match.group(0).rstrip("/"))
                absolute = REPO_ROOT / path
                if absolute.is_dir():
                    directories.add(path)
                elif path.endswith(".py") and absolute.is_file():
                    files.add(path)
            invocations.append((files, directories, ignored))
        if "ci_runner.py" in text:
            invocations.append((set(), {"fastvideo/tests/ssim"}, set()))
    return invocations


def _is_dark(path: str, invocations: list[tuple[set[str], set[str], set[str]]]) -> bool:
    for files, directories, ignored in invocations:
        if any(path == prefix or path.startswith(prefix + "/") for prefix in ignored):
            continue
        if path in files or any(path == directory or path.startswith(directory + "/") for directory in directories):
            return False
    return True


def _test_files() -> list[str]:
    return [
        path.relative_to(REPO_ROOT).as_posix() for path in sorted(TESTS_ROOT.rglob("test_*.py"))
        if "__pycache__" not in path.parts
    ]


def test_every_test_file_is_collected_or_allowlisted():
    invocations = _pytest_invocations()
    dark = [path for path in _test_files() if _is_dark(path, invocations) and path not in DARK_TESTS]
    assert not dark, (f"Test files not collected by any CI lane and not allowlisted: "
                      f"{dark}. Wire each file into a lane, or add an allowlist "
                      f"entry with a reason in {__file__}.")


def test_dark_test_allowlist_stays_accurate():
    # A stale allowlist hides regressions: every entry must still name a test
    # file that no CI lane collects, so fixing one forces its removal.
    invocations = _pytest_invocations()
    stale: list[str] = []
    for path in DARK_TESTS:
        absolute = REPO_ROOT / path
        if not absolute.is_file():
            stale.append(f"{path} (missing)")
        elif not _is_dark(path, invocations):
            stale.append(f"{path} (collected now)")
    assert not stale, f"Allowlist entries no longer dark; delete them from {__file__}: {stale}"


def test_local_tests_stays_out_of_ci():
    # tests/local_tests/ (repo root) is developer-local by design (author
    # decision, 2026-07-05): parity scaffolds and machine-specific checks
    # that must never gate CI. Fail if any CI source starts collecting it.
    assert "tests/local_tests" not in _ci_text(), ("tests/local_tests/ is local-only by design; remove the CI "
                                                   "reference or move the tests into a fastvideo/tests/ lane.")


SPARK_CAPABLE_LANES = frozenset({
    "unit", "dreamverse", "encoder", "vae", "transformer", "kernel-tests", "lora-inference", "lora-extraction",
    "inference-vmoba", "api-server", "eval", "train-framework", "golden-gate"
})
SPARK_BACKEND_TAG = "${FV_SPARK_BACKEND:-slurm}"


def _expected_agents(step_key: str) -> dict[str, str]:
    backend = SPARK_BACKEND_TAG if step_key in SPARK_CAPABLE_LANES else "slurm"
    return {"queue": "ci-runner", "backend": backend}


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
        assert step["agents"] == _expected_agents(step_key)
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
    merge_jobs = slash_commands.split("  parse-command:", maxsplit=1)[0]
    ready_workflow = (REPO_ROOT / ".github/workflows/ci-trigger-full-suite.yml").read_text()

    assert "labels: ['ready']" in merge_jobs
    assert "removeLabel" not in merge_jobs
    assert "continue-on-error: true" in merge_jobs
    assert "needs: handle-merge" in merge_jobs
    assert "if: needs.handle-merge.result == 'success'" in merge_jobs
    assert "uses: ./.github/workflows/ci-trigger-full-suite.yml" in merge_jobs
    assert "pr_number: ${{ github.event.issue.number }}" in merge_jobs
    assert "BUILDKITE_API_TOKEN: ${{ secrets.BUILDKITE_API_TOKEN }}" in merge_jobs
    assert "actions: write" not in merge_jobs
    assert "api.buildkite.com" not in merge_jobs
    assert "workflow_call:" in ready_workflow
    # Both entry points share one concurrency group, and the newest trigger wins:
    # a `/merge` call must not queue a second serialized gate run. The group is
    # job-level so an unrelated `labeled` event (which skips the guarded job)
    # cannot cancel an in-flight gate and then skip its replacement.
    workflow_header = ready_workflow.split("\njobs:", maxsplit=1)[0]
    assert "concurrency:" not in workflow_header
    trigger_job = ready_workflow.split("\n  trigger:", maxsplit=1)[1]
    assert "concurrency:" in trigger_job
    assert "group: merge-gate-${{ inputs.pr_number || github.event.pull_request.number }}" in trigger_job
    assert "cancel-in-progress: true" in trigger_job
    assert "BUILDKITE_API_TOKEN:\n        required: true" in ready_workflow
    assert "CALLED_PR_NUMBER: ${{ inputs.pr_number }}" in ready_workflow
    assert "Number.isSafeInteger(prNumber)" in ready_workflow
    assert "pull_number: prNumber" in ready_workflow
    assert "pr.state !== 'open'" in ready_workflow
    assert "pr.base.repo.full_name !== context.payload.repository.full_name" in ready_workflow
    assert "pr.base.ref !== context.payload.repository.default_branch" in ready_workflow
    trigger_if = yaml.safe_load(ready_workflow)["jobs"]["trigger"]["if"]
    assert "github.event.pull_request.base.ref == github.event.repository.default_branch" in trigger_if
    assert trigger_if.startswith("inputs.pr_number > 0 ||")
    assert "core.setOutput('head_sha', pr.head.sha)" in ready_workflow
    assert "core.setOutput('base_sha', pr.base.sha)" in ready_workflow
    assert "api.buildkite.com" in ready_workflow
    assert 'jq -r --arg pr_number "$PR_NUMBER"' in ready_workflow
    assert '.env.PR_NUMBER? == $pr_number' in ready_workflow
    assert 'TEST_SCOPE: "merge"' in ready_workflow
    assert 'FULL_SUITE: "true"' in ready_workflow
    assert "plan_merge_ci.py" in ready_workflow
    assert "ref: ${{ steps.check.outputs.base_sha }}" in ready_workflow
    assert "PR_SHA: ${{ steps.check.outputs.head_sha }}" in ready_workflow
    assert "PR_NUMBER: ${{ steps.check.outputs.pr_number }}" in ready_workflow
    assert "MERGE_TEST_PLAN" in ready_workflow
    assert "MERGE_GOLDEN_TESTS" in ready_workflow
    assert "MERGE_SSIM_TESTS" in ready_workflow
    assert "__FASTVIDEO_CI_PLAN_ALL__" in ready_workflow
    downstream_steps = ready_workflow.split("      - name: Cancel previous Buildkite builds", maxsplit=1)[1]
    assert "github.event.pull_request" not in downstream_steps


def test_merge_gate_buildkite_cancellation_is_best_effort_and_strict():
    workflow = yaml.safe_load((REPO_ROOT / ".github/workflows/ci-trigger-full-suite.yml").read_text())
    steps = workflow["jobs"]["trigger"]["steps"]
    cancel_step = next(step for step in steps if step.get("name") == "Cancel previous Buildkite builds")
    trigger_step = next(step for step in steps if step.get("name") == "Trigger Buildkite merge gate")
    cancel_script = cancel_step["run"]

    # Stale-build cleanup is best effort; creating the replacement gate remains
    # a hard failure so merge protection can never pass without a test build.
    assert cancel_step["continue-on-error"] is True
    assert cancel_step["timeout-minutes"] == 3
    assert trigger_step.get("continue-on-error") is not True
    assert "curl -sS --fail-with-body -X POST" in trigger_step["run"]

    assert cancel_script.splitlines()[0] == "set -euo pipefail"
    assert cancel_script.count("--connect-timeout 5") == 2
    assert cancel_script.count("--max-time 20") == 2
    assert "curl -sS --fail-with-body --connect-timeout 5 --max-time 20 --get" in cancel_script
    assert cancel_script.count('--data-urlencode "state[]=running"') == 1
    assert cancel_script.count('--data-urlencode "state[]=scheduled"') == 1
    assert cancel_script.count('--data-urlencode "state[]=failing"') == 1
    assert cancel_script.count('--data-urlencode "exclude_jobs=true"') == 1
    assert cancel_script.count('--data-urlencode "exclude_pipeline=true"') == 1
    assert 'state=running,scheduled' not in cancel_script

    # Only a validated array of build records can reach the URL construction.
    assert 'if type != "array" then false' in cancel_script
    assert 'if type != "object" then false' in cancel_script
    assert 'if type == "number" then . > 0 and floor == .' in cancel_script
    assert 'if ($env | type) != "object" then false' in cancel_script
    assert '$env.TEST_SCOPE?' in cancel_script
    assert '$env.PR_NUMBER?' in cancel_script

    # API bodies are kept out of annotations, every cancellation is checked,
    # and one failure does not stop attempts for the remaining build numbers.
    assert '--output "$response_file"' in cancel_script
    assert 'cat "$response_file"' not in cancel_script
    assert ("if ! curl -sS --fail-with-body --connect-timeout 5 --max-time 20 "
            "-o /dev/null -X PUT") in cancel_script
    assert "cancellation_failed=1" in cancel_script
    assert cancel_script.index("cancellation_failed=1") < cancel_script.index('done < "$builds_file"')
    assert cancel_script.index('done < "$builds_file"') < cancel_script.index("if (( cancellation_failed != 0 ))")


def test_full_suite_has_a_weekly_slurm_schedule():
    workflow = (REPO_ROOT / ".github/workflows/ci-scheduled-full-suite.yml").read_text()
    ssim_step = next(step for step in _pipeline_steps() if step["key"] == "ssim")

    assert not (REPO_ROOT / ".github/workflows/ci-scheduled-ssim.yml").exists()
    assert 'cron: "0 5 * * 0"' in workflow
    assert "workflow_dispatch:" in workflow
    assert 'TEST_SCOPE: "full"' in workflow
    assert 'FULL_SUITE: "true"' in workflow
    assert 'TEST_TYPE: "all"' in workflow
    assert 'PR_TITLE: "Scheduled full suite"' in workflow
    assert "Weekly full suite on main" in workflow
    assert "modal" not in workflow.lower()
    # The full suite includes the complete SSIM matrix.
    assert 'build.env("TEST_SCOPE") == "full"' in str(ssim_step["if"])


def test_performance_trigger_rejects_head_and_sends_a_pinned_direct_build():
    import json
    import os
    import subprocess
    import tempfile

    workflow = yaml.safe_load((REPO_ROOT / '.github/workflows/ci-scheduled-performance.yml').read_text())
    job = workflow['jobs']['trigger']
    assert "github.repository == 'hao-ai-lab/FastVideo'" in job['if']
    assert "github.ref == 'refs/heads/main'" in job['if']
    step = job['steps'][-1]
    assert step['env']['SOURCE_SHA'] == '${{ github.sha }}'
    with tempfile.TemporaryDirectory() as root:
        curl = Path(root) / 'curl'
        curl.write_text('#!/bin/bash\nwhile [[ $# -gt 0 ]]; do\n'
                        '  if [[ "$1" == --data-raw ]]; then printf "%s" "$2"; exit 0; fi\n'
                        '  shift\ndone\nexit 1\n')
        curl.chmod(0o755)
        env = {
            'PATH': root + os.pathsep + os.environ['PATH'],
            'BUILDKITE_API_TOKEN': 'test-only',
            'SOURCE_BRANCH': 'main',
            'BK_ORG': 'fastvideo',
            'BK_PIPELINE': step['env']['BK_PIPELINE'],
        }
        for commit in ('HEAD', 'abc123', 'g' * 40, 'a' * 40):
            env['SOURCE_SHA'] = commit
            result = subprocess.run(['bash', '-c', step['run']], env=env, text=True, capture_output=True)
            if commit != 'a' * 40:
                assert result.returncode != 0
                assert not result.stdout
                continue
            assert result.returncode == 0, result.stderr
            payload = json.loads(result.stdout)
            assert payload['commit'] == commit
            assert payload['branch'] == 'main'
            assert payload['env']['TEST_TYPE'] == 'performance'
            assert payload['env']['TEST_SCOPE'] == 'direct'
            assert payload['env']['FULL_SUITE'] == 'false'
            assert payload['env']['PR_NUMBER'] == 'false'


def test_weekly_performance_api_build_keeps_scheduled_baseline_policy():
    import os
    import subprocess
    import tempfile

    lane = (REPO_ROOT / '.buildkite/scripts/lanes/performance.sh').read_text()
    lane = lane.replace('export PERF_REPORTS_DIR=/workspace/artifacts/performance',
                        'export PERF_REPORTS_DIR="$TEST_REPORTS_DIR"')
    cases = (
        ('main', 'api', 'fastvideo-performance-lane', 'direct', 'false', 'scheduled_main', 'always'),
        ('main', 'schedule', 'fastvideo-performance-lane', 'direct', 'false', 'scheduled_main', 'always'),
        ('main', 'api', 'ci', 'full', 'false', 'scheduled_main', 'always'),
        ('main', 'api', 'ci', 'direct', 'false', 'unknown', 'pass'),
        ('feature', 'api', 'fastvideo-performance-lane', 'direct', 'false', 'unknown', 'pass'),
        ('main', 'api', 'fastvideo-performance-lane', 'direct', '1958', 'pr', 'never'),
    )
    with tempfile.TemporaryDirectory() as root:
        directory = Path(root)
        mocks = {
            'nvidia-smi': '#!/bin/sh\nexit 0\n',
            'pytest': '#!/bin/sh\nexit "$TEST_PYTEST_RC"\n',
            'python': '#!/bin/sh\nprintf "%s|%s|%s|%s\\n" "$1" "$PERF_RUN_SOURCE" '
            '"$PERF_UPLOAD_POLICY" "${PERF_PYTEST_RC:-}" >> "$TEST_CALLS_FILE"\n',
        }
        for name, contents in mocks.items():
            executable = directory / name
            executable.write_text(contents)
            executable.chmod(0o755)
        calls = directory / 'calls.txt'
        for branch, source, pipeline, scope, pr, run_source, policy in cases:
            for pytest_rc in (0, 1):
                calls.unlink(missing_ok=True)
                env = {
                    'PATH': root + os.pathsep + os.environ['PATH'],
                    'BUILDKITE_BRANCH': branch,
                    'BUILDKITE_SOURCE': source,
                    'BUILDKITE_PIPELINE_SLUG': pipeline,
                    'BUILDKITE_PULL_REQUEST': pr,
                    'TEST_SCOPE': scope,
                    'TEST_PYTEST_RC': str(pytest_rc),
                    'TEST_CALLS_FILE': str(calls),
                    'TEST_REPORTS_DIR': str(directory / 'reports'),
                }
                result = subprocess.run(['bash', '-c', lane], cwd=root, env=env, text=True, capture_output=True)
                assert result.returncode == pytest_rc, result.stderr
                records = [line.split('|') for line in calls.read_text().splitlines()]
                assert all(record[1:3] == [run_source, policy] for record in records)
                comparisons = [record for record in records if record[0].endswith('/compare_baseline.py')]
                if pytest_rc == 0 or policy == 'always':
                    assert len(comparisons) == 1
                    assert comparisons[0][3] == str(pytest_rc)
                else:
                    assert not comparisons


def test_spark_routing_covers_only_known_lanes():
    steps = _pipeline_steps()
    keys = {step["key"] for step in steps}
    assert SPARK_CAPABLE_LANES <= keys
    cluster_only = keys - SPARK_CAPABLE_LANES
    assert cluster_only == {
        "ssim", "training", "distillation", "self-forcing", "lora-training", "training-vsa", "performance"
    }
    for step in steps:
        assert step["agents"]["queue"] == "ci-runner"


def test_active_pipeline_has_no_modal_or_untrusted_compute_path():
    pipeline_text = (REPO_ROOT / ".buildkite/pipeline.yml").read_text()
    steps = _pipeline_steps()

    assert ".buildkite/scripts/pr_test.sh" not in pipeline_text
    assert "monorepo-diff" not in pipeline_text
    assert all(step.get("agents") == _expected_agents(step["key"]) for step in steps)
    assert all("plugins" not in step for step in steps)
    assert all(str(step.get("command", "")).startswith("/opt/fastvideo-ci-runner/") for step in steps)


def test_expensive_integration_lanes_wait_for_selected_goldens():
    # A skipped conditional golden step satisfies the dependency (direct
    # reruns/training-only plans remain available); a failed selected gate does
    # not. Never permit allow_failure or a soft-fail dependency here.
    for step in _pipeline_steps():
        key = step["key"]
        fastcheck = key in {"encoder", "vae", "transformer", "kernel-tests", "unit", "dreamverse"}
        if fastcheck or key == "golden-gate":
            assert "depends_on" not in step
        else:
            assert step["depends_on"] == "golden-gate"
        assert "allow_dependency_failure" not in step


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


def test_arm64_chromium_filters_generation_mode_playback_but_keeps_uploads():
    lane = (REPO_ROOT / '.buildkite/scripts/lanes/dreamverse.sh').read_text()
    pattern = re.search(r"--grep-invert='([^']+)'", lane).group(1)
    for mode in ('t2va', 'fl2va', 'ref2va'):
        title = f'{mode} sends validated assets and plays a clearly labeled sample'
        assert re.search(pattern, title), f'ARM64 Chromium cannot decode this playback test: {title}'
    assert not re.search(pattern, "proxies a media upload larger than Next's default 10 MiB body limit")
    assert not re.search(pattern, 'accepts a custom prompt and enters the streaming state')
    arm_branch = lane.rsplit('if [[ $machine_arch =~ ^(aarch64|arm64)$ ]]; then', 1)[1].split('\nelse')[0]
    firefox_command = re.search(r'npm run e2e(?:\\\n|[^\n])*', arm_branch).group(0)
    assert '--project=firefox' in firefox_command
    assert '--grep' not in firefox_command
