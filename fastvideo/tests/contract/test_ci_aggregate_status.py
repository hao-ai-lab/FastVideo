# SPDX-License-Identifier: Apache-2.0
"""Keep direct CI reruns from manufacturing aggregate suite gates."""

from pathlib import Path

WORKFLOW = Path(__file__).resolve().parents[3] / ".github" / "workflows" / "ci-aggregate-status.yml"


def test_direct_reruns_only_repair_failed_aggregates():
    source = WORKFLOW.read_text(encoding="utf-8")

    assert "s.context === context && s.state === 'failure'" in source
    assert source.count("failedAggregate(") == 2
    assert "failedAggregate('fastcheck-passed')" in source
    assert "failedAggregate('full-suite-passed')" in source
