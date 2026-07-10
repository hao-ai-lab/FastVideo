# SPDX-License-Identifier: Apache-2.0
"""Static checks for Modal pytest rerun wiring.

Pure text analysis: no fastvideo imports, no GPU, no torch.
"""
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
PYPROJECT = REPO_ROOT / "pyproject.toml"
MODAL_ROOT = REPO_ROOT / "fastvideo" / "tests" / "modal"
PYTEST_RETRY = MODAL_ROOT / "pytest_retry.py"
PR_TEST = MODAL_ROOT / "pr_test.py"
SSIM_TEST = MODAL_ROOT / "ssim_test.py"


def test_pytest_rerunfailures_is_in_test_extra():
    assert '"pytest-rerunfailures"' in PYPROJECT.read_text()


def test_modal_entrypoints_share_pytest_retry_policy():
    retry_text = PYTEST_RETRY.read_text()
    pr_text = PR_TEST.read_text()
    ssim_text = SSIM_TEST.read_text()

    assert "TRANSIENT_FAILURE_REGEX" in retry_text
    assert "--only-rerun" in retry_text
    assert "build_pytest_addopts" in pr_text
    assert "PYTEST_ADDOPTS" in pr_text
    assert "_build_pytest_rerun_args()" in ssim_text


def test_retry_policy_excludes_plain_assertion_failures():
    retry_text = PYTEST_RETRY.read_text()

    assert "AssertionError" not in retry_text
    assert "assert " not in retry_text
