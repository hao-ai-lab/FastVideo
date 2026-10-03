# SPDX-License-Identifier: Apache-2.0
"""Compare the final FastVideoArgs, PipelineConfig, and SamplingParam of every snapshot case with its golden file.

The cases and the serializer live in ``fastvideo/tests/api/config_snapshot.py``. Every entry of a snapshot must equal
the golden file, except ``resolution_decisions``: each decision that the golden file records must still be made with
the same value, by the same step, in the same step order, and resolution steps may decide further paths. After an
intended behavior change, regenerate the golden files with ``python -m fastvideo.tests.api.config_snapshot`` and
review the JSON diff.
"""
import json
from typing import Any

import pytest

from fastvideo.tests.api.config_snapshot import (
    OfflineResolutionError,
    collect_cases,
    describe_diff,
    dump_json,
    run_case,
)

SNAPSHOT_CASES = collect_cases()
DECISIONS = "resolution_decisions"


def _missing_decisions(golden: list[Any], current: list[Any]) -> list[str]:
    """Golden decisions that the current snapshot no longer makes, and a changed order of the deciding steps."""
    decided = {(source, path): value for source, values in current for path, value in values.items()}
    missing = [
        f"  {source}: {path}={value!r} (current={decided.get((source, path), '<absent>')!r})"
        for source, values in golden for path, value in values.items() if decided.get((source, path), ...) != value
    ]
    current_sources = iter(source for source, _ in current)
    if not all(source in current_sources for source, _ in golden):
        missing.append(f"  step order changed: golden={[s for s, _ in golden]} current={[s for s, _ in current]}")
    return missing


@pytest.mark.parametrize("case", SNAPSHOT_CASES, ids=[case.case_id for case in SNAPSHOT_CASES])
def test_config_snapshot_matches_golden(case):
    try:
        current = json.loads(dump_json(run_case(case)))
    except OfflineResolutionError as error:
        pytest.skip(f"{case.case_id}: {error}")
    assert case.golden_path.exists(), (f"{case.case_id} has no golden file at {case.golden_path}. "
                                       "Run `python -m fastvideo.tests.api.config_snapshot`.")
    golden = json.loads(case.golden_path.read_text())
    golden_values = {key: value for key, value in golden.items() if key != DECISIONS}
    current_values = {key: value for key, value in current.items() if key != DECISIONS}
    if current_values != golden_values:
        pytest.fail(f"{case.case_id} differs from {case.golden_path.name}:\n"
                    f"{describe_diff(golden_values, current_values)}\n"
                    "If the change is intended, run `python -m fastvideo.tests.api.config_snapshot`.")
    if (DECISIONS in golden) != (DECISIONS in current):
        pytest.fail(f"{case.case_id}: {DECISIONS} is present in only one of the golden file and the snapshot")
    missing = _missing_decisions(golden.get(DECISIONS, []), current.get(DECISIONS, []))
    if missing:
        pytest.fail(f"{case.case_id} no longer makes these decisions of {case.golden_path.name}:\n" +
                    "\n".join(missing) + "\nIf the change is intended, run `python -m fastvideo.tests.api.config_snapshot`.")
