# SPDX-License-Identifier: Apache-2.0
"""Contract and behavior checks for Modal pytest rerun wiring.

No fastvideo imports, GPU, or torch required.
"""
import importlib.util
import os
import subprocess
import sys
import textwrap
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
PYPROJECT = REPO_ROOT / "pyproject.toml"
MODAL_ROOT = REPO_ROOT / "fastvideo" / "tests" / "modal"
PYTEST_RETRY = MODAL_ROOT / "pytest_retry.py"
PR_TEST = MODAL_ROOT / "pr_test.py"
SSIM_TEST = MODAL_ROOT / "ssim_test.py"


def test_pytest_rerunfailures_is_in_test_extra():
    assert "pytest-rerunfailures>=16.3" in PYPROJECT.read_text(encoding="utf-8")


def test_modal_entrypoints_share_pytest_retry_policy():
    retry_text = PYTEST_RETRY.read_text(encoding="utf-8")
    pr_text = PR_TEST.read_text(encoding="utf-8")
    ssim_text = SSIM_TEST.read_text(encoding="utf-8")

    assert "TRANSIENT_FAILURE_REGEX" in retry_text
    assert "--only-rerun" in retry_text
    assert "--rerun-show-tracebacks" in retry_text
    assert "build_pytest_addopts" in pr_text
    assert "PYTEST_ADDOPTS" in pr_text
    assert "_build_pytest_rerun_args()" in ssim_text


def test_retry_policy_excludes_plain_assertion_failures():
    retry_text = PYTEST_RETRY.read_text(encoding="utf-8")

    assert "AssertionError" not in retry_text
    assert "assert " not in retry_text


def test_retry_policy_runtime_behavior(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("pytest_retry_under_test", PYTEST_RETRY)
    assert spec is not None
    assert spec.loader is not None
    pytest_retry = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pytest_retry)
    monkeypatch.setattr(pytest_retry, "PYTEST_RERUNS_DELAY_SECONDS", 0)

    sample_test = tmp_path / "test_retry_policy_sample.py"
    sample_test.write_text(
        textwrap.dedent("""
            from pathlib import Path

            TRANSIENT_ATTEMPTS = Path(__file__).with_name("transient_attempts.txt")
            ASSERTION_ATTEMPTS = Path(__file__).with_name("assertion_attempts.txt")


            def next_attempt(path):
                attempt = int(path.read_text(encoding="utf-8")) + 1 if path.exists() else 1
                path.write_text(str(attempt), encoding="utf-8")
                return attempt


            def test_matching_failure_is_rerun():
                if next_attempt(TRANSIENT_ATTEMPTS) == 1:
                    raise RuntimeError("503 Service Unavailable from synthetic test")


            def test_plain_assertion_is_not_rerun():
                next_attempt(ASSERTION_ATTEMPTS)
                assert False, "synthetic plain assertion"
        """),
        encoding="utf-8",
    )
    env = os.environ.copy()
    env.pop("PYTEST_ADDOPTS", None)
    result = subprocess.run(
        [sys.executable, "-m", "pytest", str(sample_test), "-q", *pytest_retry.build_pytest_rerun_args()],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    output = result.stdout + result.stderr

    assert result.returncode == 1, output
    assert (tmp_path / "transient_attempts.txt").read_text(encoding="utf-8") == "2"
    assert (tmp_path / "assertion_attempts.txt").read_text(encoding="utf-8") == "1"
    assert 'raise RuntimeError("503 Service Unavailable from synthetic test")' in output
    assert "RuntimeError: 503 Service Unavailable from synthetic test" in output
