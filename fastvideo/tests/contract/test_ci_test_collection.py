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
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
TESTS_ROOT = REPO_ROOT / "fastvideo" / "tests"

# Files whose text constitutes "a CI lane references this directory".
CI_SOURCES = [
    TESTS_ROOT / "modal" / "pr_test.py",
    TESTS_ROOT / "modal" / "ssim_test.py",
    *sorted((REPO_ROOT / ".buildkite").rglob("*.yml")),
    *sorted((REPO_ROOT / ".buildkite").rglob("*.sh")),
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
    "ops": "no lane yet — GPU op tests, run manually",
    "stages": "wiring in flight (PR #1518 review); remove when it lands",
    "modal": "CI infrastructure itself, not a test suite",
}


def _dirs_with_tests() -> list[str]:
    dirs = []
    for child in sorted(TESTS_ROOT.iterdir()):
        if child.is_dir() and any(child.rglob("test_*.py")):
            dirs.append(child.name)
    return dirs


def _ci_text() -> str:
    return "\n".join(
        src.read_text(errors="replace") for src in CI_SOURCES if src.exists())


def test_every_test_directory_is_collected_or_allowlisted():
    ci_text = _ci_text()
    dark = [
        name for name in _dirs_with_tests()
        if f"tests/{name}" not in ci_text and name not in ALLOWLIST
    ]
    assert not dark, (
        f"Test directories not referenced by any CI lane and not "
        f"allowlisted: {dark}. Wire them into a lane in "
        f"fastvideo/tests/modal/pr_test.py (or a Buildkite step), or add an "
        f"allowlist entry with a reason in {__file__}.")


def test_local_tests_stays_out_of_ci():
    # tests/local_tests/ (repo root) is developer-local by design (author
    # decision, 2026-07-05): parity scaffolds and machine-specific checks
    # that must never gate CI. Fail if any CI source starts collecting it.
    assert "tests/local_tests" not in _ci_text(), (
        "tests/local_tests/ is local-only by design; remove the CI "
        "reference or move the tests into a fastvideo/tests/ lane.")


def test_allowlist_entries_are_still_real_directories():
    # A stale allowlist hides regressions; entries must track reality.
    missing = [
        name for name in ALLOWLIST
        if name != "modal" and not (TESTS_ROOT / name).is_dir()
    ]
    assert not missing, (
        f"Allowlisted directories no longer exist — remove them: {missing}")
