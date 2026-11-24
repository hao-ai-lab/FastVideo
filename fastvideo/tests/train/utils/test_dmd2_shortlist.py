# SPDX-License-Identifier: Apache-2.0
"""CPU regressions for uniform-layout filtering and benchmark selection."""
import json
from pathlib import Path
from subprocess import CompletedProcess
import sys

import pytest

from examples.train.activation_memory import select_joint_plan as selector


def profile():
    blocks = []
    for role in ('student', 'critic'):
        for i in range(30):
            forward_bytes = 4 if role == 'student' and i % 2 == 0 else 0
            backward_bytes = 1 if role == 'critic' else 0
            blocks.append({'role': role, 'block': f'blocks.{i}', 'choices': [
                {'policy': 'save', 'device_bytes': [forward_bytes, backward_bytes]}, {'policy': 'recompute', 'device_bytes': [0, 0]}]})
    return {'fixed_device_bytes': [8, 4], 'blocks': blocks, 'metadata': {'world_size': 1, 'model': 'wan'}}


def test_filter_checks_uniform_positions_and_each_event():
    rows = {(r['student'], r['critic']): r for r in selector.candidates(profile(), 8)}
    assert len(rows) == 56
    assert rows[15, 4]['feasible']
    assert rows[15, 4]['predicted_peak_bytes'] == 8
    assert not rows[30, 0]['feasible']
    assert not rows[0, 8]['feasible']


@pytest.fixture
def matrix_path(tmp_path, monkeypatch):
    monkeypatch.setattr(selector, 'STUDENT_COUNTS', (0, 15))
    monkeypatch.setattr(selector, 'CRITIC_COUNTS', (0,))
    source = tmp_path / 'profile.json'
    source.write_text(json.dumps(profile()))
    output = tmp_path / 'matrix.json'
    monkeypatch.setattr(sys, 'argv', ['select', '--model', 'wan', '--training-sample', 'cosmos.pt',
                                     '--data', 'parquet', '--profile', str(source), '--output', str(output),
                                     '--budget-gib', str(8 / 2**30)])

    return output


@pytest.mark.parametrize('wrong_position', [False, True])
def test_cli_selects_measured_feasible_layout(matrix_path, monkeypatch, wrong_position):
    def benchmark(cmd, **kwargs):
        assert cmd[cmd.index('--policy') + 1] == 'uniform'
        count = int(cmd[cmd.index('--save-count') + 1])
        kept = set(range(1, 30, 2)) if count else set()
        if count and wrong_position:
            kept.remove(1)
            kept.add(0)
        layouts = {role: {f'blocks.{i}': 'save' if role == 'student' and i in kept else 'recompute'
                          for i in range(30)} for role in ('student', 'critic')}
        record = {'manual_policies': layouts, 'steps': [{'step_ms': 2 if count else 1}] * 5,
                  'peak_allocated_bytes': 8 if count else 9}
        Path(cmd[cmd.index('--output') + 1]).write_text(json.dumps(record))
        return CompletedProcess(cmd, 0)

    monkeypatch.setattr(selector.subprocess, 'run', benchmark)
    if wrong_position:
        with pytest.raises(ValueError, match='block positions'):
            selector.main()
    else:
        selector.main()
        result = json.loads(matrix_path.read_text())
        over_budget, within_budget = result['matrix']
        assert over_budget['mean_ms'] < within_budget['mean_ms']
        assert over_budget['peak_bytes'] > result['budget_bytes']
        winner = result['winner']
        assert (winner['student'], winner['critic'], winner['mean_ms']) == (15, 0, 2)


@pytest.mark.parametrize('all_fail', [False, True])
def test_failed_benchmarks_are_recorded_and_remaining_layouts_run(matrix_path, monkeypatch, all_fail):
    def benchmark(cmd, stdout, **kwargs):
        student = int(cmd[cmd.index('--save-count') + 1])
        if student == 0 or all_fail:
            stdout.write('device out of memory\n')
            return CompletedProcess(cmd, 1)
        target = Path(cmd[cmd.index('--output') + 1])
        target.write_text(json.dumps({'manual_policies': selector.policies(student, 0),
                                      'steps': [{'step_ms': 2}] * 5, 'peak_allocated_bytes': 8}))
        return CompletedProcess(cmd, 0)

    monkeypatch.setattr(selector.subprocess, 'run', benchmark)
    if all_fail:
        with pytest.raises(SystemExit) as exc:
            selector.main()
        assert exc.value.code == 1
    else:
        selector.main()
    result = json.loads(matrix_path.read_text())
    failed = [row for row in result['matrix'] if row.get('status') == 'failed']
    assert [row['student'] for row in failed] == ([0, 15] if all_fail else [0])
    for row in failed:
        assert row['returncode'] == 1
        assert (matrix_path.parent / row['log']).read_text() == 'device out of memory\n'
        assert 'peak_bytes' not in row
    if all_fail:
        assert result['winner'] is None
    else:
        assert result['winner']['student'] == 15
