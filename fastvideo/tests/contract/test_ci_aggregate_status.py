# SPDX-License-Identifier: Apache-2.0
"""Keep direct CI reruns from manufacturing aggregate suite gates."""

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
WORKFLOW = REPO_ROOT / ".github" / "workflows" / "ci-aggregate-status.yml"
PIPELINE = REPO_ROOT / ".buildkite" / "pipeline.yml"


def test_direct_reruns_only_repair_failed_aggregates():
    source = WORKFLOW.read_text(encoding="utf-8")

    assert "s.context === context && s.state === 'failure'" in source
    assert source.count("failedAggregate(") == 2
    assert "failedAggregate('fastcheck-passed')" in source
    assert "failedAggregate('full-suite-passed')" in source


def test_aggregate_requires_the_complete_lane_set():
    source = WORKFLOW.read_text(encoding="utf-8")

    assert "fastcheck.size === 6" in source
    assert "fullSuiteOnly.size === 14" in source
    assert "'buildkite/pr-fastcheck/microscope-'" in source
    assert "'buildkite/ci/microscope-'" in source


def test_lane_counts_match_the_buildkite_pipeline():
    # Derive the lane counts from the Buildkite pipeline instead of trusting
    # the literals above, so constant drift between the workflow and
    # .buildkite/pipeline.yml fails CI. Namespace mapping follows
    # docs/contributing/ci_architecture.md (label emoji -> status namespace).
    labels = set(re.findall(r'label: "(:[a-z_]+: [^"]+)"', PIPELINE.read_text(encoding="utf-8")))
    fastcheck = {l for l in labels if l.startswith(":microscope:")}
    full_suite = {
        l for l in labels if l.split(" ", 1)[0] in (":test_tube:", ":bar_chart:")
    }
    source = WORKFLOW.read_text(encoding="utf-8")

    assert fastcheck, "no Fastcheck lanes found in .buildkite/pipeline.yml"
    assert full_suite, "no Full Suite lanes found in .buildkite/pipeline.yml"
    assert f"fastcheck.size === {len(fastcheck)}" in source
    assert f"fullSuiteOnly.size === {len(full_suite)}" in source
    assert f"All {len(fastcheck) + len(full_suite)} full suite tests passed" in source
