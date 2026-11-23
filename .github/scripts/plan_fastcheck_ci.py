#!/usr/bin/env python3
"""Select the Fastcheck lanes needed by a PR diff.

Fastcheck is the universal six-lane baseline: the ``unit`` lane always runs and
every lane runs when the plan environment variable is absent.  This planner
narrows that baseline only for unambiguous diffs -- a changed path inside a
single domain test root selects that one extra lane, and documentation or
repository-metadata paths select nothing beyond ``unit``.  Any unrecognized
source or build path fails closed to all six lanes.
"""

from __future__ import annotations

import argparse
import fnmatch
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TextIO

FASTCHECK_LANES = (
    "unit",
    "kernel-tests",
    "dreamverse",
    "encoder",
    "vae",
    "transformer",
)

# A changed path that matches a single domain root selects just that lane.
DOMAIN_PATTERNS = (
    ("kernel-tests", "fastvideo-kernel/**"),
    ("dreamverse", "apps/dreamverse/**"),
    ("encoder", "fastvideo/tests/encoders/**"),
    ("vae", "fastvideo/tests/vaes/**"),
    ("transformer", "fastvideo/tests/transformers/**"),
)

SAFE_PATTERNS = (
    "*.md",
    "*.rst",
    "docs/**",
    "docs-site/**",
    ".agents/**",
    ".claude/**",
    ".codex/**",
    ".github/**",
    "assets/**",
    "examples/**",
    "scripts/**",
    "comfyui/**",
    "tests/local_tests/**",
    "AGENTS.md",
    "CLAUDE.md",
    "README.md",
    "LICENSE",
    "NOTICE",
    "CITATION.cff",
    "CODE_OF_CONDUCT.md",
    "CONTRIBUTING.md",
    "SECURITY.md",
    ".gitignore",
    ".gitattributes",
    ".git-blame-ignore-revs",
    ".pre-commit-config.yaml",
    "requirements-docs.in",
    "requirements-docs.txt",
    "__init__.py",
    "collect_env.py",
)


@dataclass
class FastcheckPlan:
    lanes: set[str] = field(default_factory=lambda: {"unit"})
    all_lanes: bool = False
    reasons: list[str] = field(default_factory=list)

    def add_lane(self, lane: str, reason: str) -> None:
        if lane not in FASTCHECK_LANES:
            raise ValueError(f"Unknown Fastcheck lane: {lane}")
        self.lanes.add(lane)
        self.reasons.append(reason)

    def require_all(self, reason: str) -> None:
        self.all_lanes = True
        self.lanes.update(FASTCHECK_LANES)
        self.reasons.append(reason)

    def ordered_lanes(self) -> tuple[str, ...]:
        return tuple(lane for lane in FASTCHECK_LANES if lane in self.lanes)

    def encoded_lanes(self) -> str:
        return "," + ",".join(self.ordered_lanes()) + ","


def _matches_any(path: str, patterns: tuple[str, ...]) -> bool:
    return any(fnmatch.fnmatchcase(path, pattern) for pattern in patterns)


def _domain_lane(path: str) -> str | None:
    for lane, pattern in DOMAIN_PATTERNS:
        if fnmatch.fnmatchcase(path, pattern):
            return lane
    return None


def _normalize_path(raw_path: str) -> str:
    path = raw_path.strip()
    while path.startswith("./"):
        path = path[2:]
    return path


def classify_fastcheck(paths: Iterable[str], removed_paths: Iterable[str] = ()) -> FastcheckPlan:
    """Plan Fastcheck lanes for ``paths``.

    ``removed_paths`` is accepted for symmetry with the merge planner and is
    deliberately ignored: a removed changed path classifies like any other.
    """
    plan = FastcheckPlan()
    normalized_paths = sorted({path for path in map(_normalize_path, paths) if path})
    if not normalized_paths:
        plan.require_all("changed-file list was empty; failing closed to every Fastcheck lane")
        return plan

    for path in normalized_paths:
        if path == "__FASTVIDEO_CI_PLAN_ALL__":
            plan.require_all("changed-file API failed; failing closed to every Fastcheck lane")
            continue

        domain = _domain_lane(path)
        if domain is not None:
            plan.add_lane(domain, reason=f"single-domain Fastcheck lane '{domain}': {path}")
            continue

        if _matches_any(path, SAFE_PATTERNS):
            plan.reasons.append(f"covered by the universal unit Fastcheck lane: {path}")
            continue

        plan.require_all(f"unclassified path; failing closed to every Fastcheck lane: {path}")

    return plan


def _write_github_output(output: TextIO, plan: FastcheckPlan) -> None:
    output.write(f"fastcheck_test_plan={plan.encoded_lanes()}\n")


def _write_summary(output: TextIO, plan: FastcheckPlan) -> None:
    output.write("## Change-aware Fastcheck test plan\n\n")
    output.write("| Selection | Value |\n|---|---|\n")
    output.write(f"| Fastcheck lanes | `{','.join(plan.ordered_lanes())}` |\n")
    output.write(f"| Encoded plan | `{plan.encoded_lanes()}` |\n\n")
    output.write("When `FASTCHECK_TEST_PLAN` is absent every lane still runs: the "
                 "universal six-lane default applies whenever the plan env is absent.\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paths-file", type=Path, required=True)
    parser.add_argument("--removed-paths-file", type=Path)
    parser.add_argument("--github-output", type=Path)
    parser.add_argument("--summary-file", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    paths = args.paths_file.read_text(encoding="utf-8").splitlines()
    removed_paths = (args.removed_paths_file.read_text(encoding="utf-8").splitlines()
                     if args.removed_paths_file else [])
    plan = classify_fastcheck(paths, removed_paths)
    print(f"FASTCHECK_TEST_PLAN={plan.encoded_lanes()}")
    for reason in plan.reasons:
        print(f"- {reason}")
    if args.github_output:
        with args.github_output.open("a", encoding="utf-8") as output:
            _write_github_output(output, plan)
    if args.summary_file:
        with args.summary_file.open("a", encoding="utf-8") as output:
            _write_summary(output, plan)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
