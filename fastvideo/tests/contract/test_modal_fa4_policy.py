# SPDX-License-Identifier: Apache-2.0
"""Guard Modal runtime policies that keep CI lanes on their intended backend.

Pure text/AST analysis: no fastvideo imports, no torch, no Modal client.
"""

from __future__ import annotations

import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
MODAL_ROOT = REPO_ROOT / "fastvideo" / "tests" / "modal"
PR_TEST = MODAL_ROOT / "pr_test.py"
LAUNCH_L40S_JOB = MODAL_ROOT / "launch_l40s_job.py"
SSIM_TEST = MODAL_ROOT / "ssim_test.py"


def _function_strings(path: Path, function_name: str) -> str:
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == function_name:
            return "\n".join(
                child.value
                for child in ast.walk(node)
                if isinstance(child, ast.Constant)
                and isinstance(child.value, str)
            )
    raise AssertionError(f"{function_name} not found in {path}")


def test_generic_l40s_launcher_defaults_fa4_off():
    source = LAUNCH_L40S_JOB.read_text(encoding="utf-8")
    assert '"FASTVIDEO_FA4": os.environ.get("FASTVIDEO_FA4", "0")' in source


def test_ssim_launcher_keeps_fa4_enabled_by_default():
    source = SSIM_TEST.read_text(encoding="utf-8")
    assert '"FASTVIDEO_FA4": os.environ.get("FASTVIDEO_FA4", "1")' in source


def test_performance_identity_env_reaches_modal_runtime():
    pr_source = PR_TEST.read_text(encoding="utf-8")
    launch_source = LAUNCH_L40S_JOB.read_text(encoding="utf-8")
    runtime_secret = pr_source.split("ci_env_secret =", 1)[1].split("hf_secret =", 1)[0]

    for key in ("FASTVIDEO_ATTENTION_BACKEND", "FASTVIDEO_PERFORMANCE_PROFILE_VERSION"):
        assert key in runtime_secret
    assert "FASTVIDEO_PERFORMANCE_PROFILE_VERSION" in launch_source.split(".env({", 1)[1]


def test_performance_lane_classifies_pull_requests_before_main():
    function_strings = _function_strings(PR_TEST, "run_performance_tests")
    assert function_strings.index(
        "BUILDKITE_PULL_REQUEST") < function_strings.index("BUILDKITE_BRANCH")


def test_pr_model_load_and_training_lanes_disable_fa4():
    lanes = {
        "run_transformer_tests": "pytest ./fastvideo/tests/transformers -vs",
        "run_training_tests": "pytest ./fastvideo/tests/training/Vanilla -srP",
        "run_training_lora_tests": "pytest ./fastvideo/tests/training/lora/test_lora_training.py -srP",
        "run_training_tests_VSA": "pytest ./fastvideo/tests/training/VSA -srP",
        "run_distill_dmd_tests": "pytest ./fastvideo/tests/training/distill/test_distill_dmd.py -vs",
        "run_self_forcing_tests": "pytest ./fastvideo/tests/training/self-forcing/test_self_forcing.py -vs",
        "run_train_framework_tests": "pytest ./fastvideo/tests/train/models ./fastvideo/tests/train/methods -vs",
        "seed_grad_norm_references": "pytest ./fastvideo/tests/train/methods -vs -rs",
    }

    for function_name, pytest_command in lanes.items():
        function_strings = _function_strings(PR_TEST, function_name)
        assert "FASTVIDEO_FA4=0" in function_strings
        assert pytest_command in function_strings
