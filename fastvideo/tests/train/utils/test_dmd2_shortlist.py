# SPDX-License-Identifier: Apache-2.0
"""CPU regressions for uniform-layout filtering and benchmark selection."""
import json
from pathlib import Path
import sys

import pytest

from examples.train.activation_memory import select_joint_plan as selector


def profile():
    blocks = []
    for role in ('student', 'critic'):
        for i in range(30):
            # Saving an even student block raises the forward peak.
            # Saving a critic block raises the backward peak.
            save = [4 if role == 'student' and i % 2 == 0 else 0, 1 if role == 'critic' else 0]
            blocks.append({'role': role, 'block': f'blocks.{i}', 'choices': [
                {'policy': 'save', 'device_bytes': save}, {'policy': 'recompute', 'device_bytes': [0, 0]}]})
    return {'fixed_device_bytes': [8, 4], 'blocks': blocks, 'metadata': {'world_size': 1, 'model': 'wan'}}


def test_filter_checks_uniform_positions_and_each_event():
    rows = {(r['student'], r['critic']): r for r in selector.candidates(profile(), 8)}
    assert len(rows) == 56
    assert rows[15, 4]['feasible']  # Odd student blocks, both events exactly at budget.
    assert rows[15, 4]['predicted_peak_bytes'] == 8
    assert not rows[30, 0]['feasible']
    assert not rows[0, 8]['feasible']


@pytest.mark.parametrize('wrong_position', [False, True])
def test_cli_selects_measured_feasible_layout(tmp_path, monkeypatch, wrong_position):
    monkeypatch.setattr(selector, 'STUDENT_COUNTS', (0, 15))
    monkeypatch.setattr(selector, 'CRITIC_COUNTS', (0,))
    source = tmp_path / 'profile.json'
    source.write_text(json.dumps(profile()))
    output = tmp_path / 'matrix.json'
    monkeypatch.setattr(sys, 'argv', ['select', '--model', 'wan', '--training-sample', 'cosmos.pt',
                                     '--data', 'parquet', '--profile', str(source), '--output', str(output),
                                     '--budget-gib', str(8 / 2**30)])

    def benchmark(cmd, **kwargs):
        assert cmd[cmd.index('--policy') + 1] == 'uniform'
        count = int(cmd[cmd.index('--save-count') + 1])
        kept = set(range(1, 30, 2)) if count else set()
        if count and wrong_position:
            kept.remove(1)
            kept.add(0)
        layouts = {role: {f'blocks.{i}': 'save' if role == 'student' and i in kept else 'recompute'
                          for i in range(30)} for role in ('student', 'critic')}
        # The faster 0+0 candidate exceeds the observed budget.
        record = {'manual_policies': layouts, 'steps': [{'step_ms': 2 if count else 1}] * 5,
                  'peak_allocated_bytes': 8 if count else 9}
        Path(cmd[cmd.index('--output') + 1]).write_text(json.dumps(record))

    monkeypatch.setattr(selector.subprocess, 'run', benchmark)
    if wrong_position:
        with pytest.raises(ValueError, match='block positions'):
            selector.main()
    else:
        selector.main()
        winner = json.loads(output.read_text())['winner']
        assert (winner['student'], winner['critic'], winner['mean_ms']) == (15, 0, 2)
