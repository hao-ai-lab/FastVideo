# SPDX-License-Identifier: Apache-2.0
"""Compare the final FastVideoArgs, PipelineConfig, and SamplingParam of every snapshot case with its golden file.

The cases and the serializer live in ``fastvideo/tests/api/config_snapshot.py``. After an intended behavior change,
regenerate the golden files with ``python -m fastvideo.tests.api.config_snapshot`` and review the JSON diff.
"""
import json

import pytest

from fastvideo.tests.api.config_snapshot import (
    OfflineResolutionError,
    collect_cases,
    describe_diff,
    dump_json,
    run_case,
)

SNAPSHOT_CASES = collect_cases()


@pytest.mark.parametrize("case", SNAPSHOT_CASES, ids=[case.case_id for case in SNAPSHOT_CASES])
def test_config_snapshot_matches_golden(case):
    try:
        current = json.loads(dump_json(run_case(case)))
    except OfflineResolutionError as error:
        pytest.skip(f"{case.case_id}: {error}")
    assert case.golden_path.exists(), (f"{case.case_id} has no golden file at {case.golden_path}. "
                                       "Run `python -m fastvideo.tests.api.config_snapshot`.")
    golden = json.loads(case.golden_path.read_text())
    if current != golden:
        pytest.fail(f"{case.case_id} differs from {case.golden_path.name}:\n"
                    f"{describe_diff(golden, current)}\n"
                    "If the change is intended, run `python -m fastvideo.tests.api.config_snapshot`.")
