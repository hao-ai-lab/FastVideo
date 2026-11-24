# SPDX-License-Identifier: Apache-2.0
"""CPU-only coverage for the change-aware Fastcheck test planner."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
PLANNER_PATH = REPO_ROOT / ".github/scripts/plan_fastcheck_ci.py"
SPEC = importlib.util.spec_from_file_location("plan_fastcheck_ci", PLANNER_PATH)
assert SPEC is not None and SPEC.loader is not None
PLAN_FASTCHECK_CI = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = PLAN_FASTCHECK_CI
SPEC.loader.exec_module(PLAN_FASTCHECK_CI)

ALL_LANES = ",unit,kernel-tests,dreamverse,encoder,vae,transformer,"


@pytest.mark.parametrize("paths, encoded", [
    (["docs/x.md", "README.md", "requirements-docs.txt"], ",unit,"),
    (["fastvideo/tests/encoders/test_t5.py"], ",unit,encoder,"),
    (["fastvideo/tests/vaes/test_x.py"], ",unit,vae,"),
    (["fastvideo/tests/transformers/test_y.py"], ",unit,transformer,"),
    (["fastvideo-kernel/tests/test_k.py"], ",unit,kernel-tests,"),
    (["apps/dreamverse/x.py"], ",unit,dreamverse,"),
    (["fastvideo/tests/encoders/a.py", "fastvideo/tests/vaes/b.py"], ",unit,encoder,vae,"),
])
def test_single_domain_or_docs_diffs_narrow_the_lane_set(paths, encoded):
    plan = PLAN_FASTCHECK_CI.classify_fastcheck(paths)

    assert plan.encoded_lanes() == encoded
    assert not plan.all_lanes
    assert "unit" in plan.lanes


def test_docs_only_diff_keeps_only_the_unit_lane():
    plan = PLAN_FASTCHECK_CI.classify_fastcheck(["docs/x.md", "README.md", "requirements-docs.txt"])

    assert plan.encoded_lanes() == ",unit,"
    assert plan.ordered_lanes() == ("unit", )


def test_mixed_domains_select_every_named_lane_in_canonical_order():
    plan = PLAN_FASTCHECK_CI.classify_fastcheck(["fastvideo/tests/encoders/a.py", "fastvideo/tests/vaes/b.py"])

    assert plan.encoded_lanes() == ",unit,encoder,vae,"
    assert plan.ordered_lanes() == ("unit", "encoder", "vae")


def test_shared_unmapped_path_fails_closed_to_all_six():
    plan = PLAN_FASTCHECK_CI.classify_fastcheck(["fastvideo/utils.py"])

    assert plan.all_lanes
    assert plan.encoded_lanes() == ALL_LANES
    assert plan.ordered_lanes() == PLAN_FASTCHECK_CI.FASTCHECK_LANES


def test_all_marker_fails_closed_to_all_six():
    plan = PLAN_FASTCHECK_CI.classify_fastcheck(["__FASTVIDEO_CI_PLAN_ALL__"])

    assert plan.all_lanes
    assert plan.encoded_lanes() == ALL_LANES


def test_empty_path_list_fails_closed_to_all_six():
    plan = PLAN_FASTCHECK_CI.classify_fastcheck([])

    assert plan.all_lanes
    assert plan.encoded_lanes() == ALL_LANES
