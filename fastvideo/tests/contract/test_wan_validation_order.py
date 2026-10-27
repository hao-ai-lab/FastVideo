# SPDX-License-Identifier: Apache-2.0
"""Shell selection/order contracts; pytest is mocked, no GPU payload executes."""

import os
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]


@pytest.mark.parametrize("lane, expected", [
    ("vae", ["golden_gate/test_wan_vae.py", "tests/vaes"]),
    ("transformer", ["golden_gate/test_wan_t2v.py", "golden_gate/test_wan_causal.py", "tests/transformers"]),
])
@pytest.mark.parametrize("fail_first", [False, True])
def test_component_lane_stops_after_failed_golden(tmp_path, lane, expected, fail_first):
    log = tmp_path / "calls"
    stub = tmp_path / "pytest"
    stub.write_text('#!/bin/bash\nprintf "%s\\n" "$*" >> "$WAN_TEST_LOG"\nexit "${WAN_TEST_EXIT:-0}"\n')
    stub.chmod(0o755)
    env = {**os.environ, "PATH": f"{tmp_path}:{os.environ['PATH']}", "WAN_TEST_LOG": str(log),
           "WAN_TEST_EXIT": "1" if fail_first else "0"}
    result = subprocess.run(["bash", f".buildkite/scripts/lanes/{lane}.sh"], cwd=ROOT, env=env, check=False)
    calls = log.read_text().splitlines()
    assert len(calls) == (1 if fail_first else len(expected))
    assert all(path in call for path, call in zip(expected, calls))
    assert result.returncode == int(fail_first)
