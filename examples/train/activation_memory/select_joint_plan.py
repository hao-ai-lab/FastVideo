# SPDX-License-Identifier: Apache-2.0
"""Benchmark memory-feasible uniform layouts for 30-block Wan DMD2 training."""
import argparse
import itertools
import json
import subprocess
import sys
from pathlib import Path
from statistics import mean

STUDENT_COUNTS = (0, 5, 10, 15, 20, 24, 30)
CRITIC_COUNTS = (0, 4, 8, 12, 16, 20, 24, 30)


def uniform_layout(saved_count):
    # profile_wan.policy_for(..., 'uniform') selects the midpoint of each interval.
    saved = {int((i + 0.5) * 30 / saved_count) for i in range(saved_count)}
    return {f'blocks.{i}': 'save' if i in saved else 'recompute' for i in range(30)}


def policies(student, critic):
    return {'student': uniform_layout(student), 'critic': uniform_layout(critic)}


def candidates(profile, budget):
    costs = {(block['role'], block['block']): {c['policy']: c['device_bytes'] for c in block['choices']}
             for block in profile['blocks']}
    if set(costs) != set(itertools.product(('student', 'critic'), uniform_layout(0))):
        raise ValueError('Expected a 30-block profile for both student and critic')
    rows = []
    for student, critic in itertools.product(STUDENT_COUNTS, CRITIC_COUNTS):
        vectors = [profile['fixed_device_bytes']]
        for role, layout in policies(student, critic).items():
            vectors.extend(costs[role, block][policy] for block, policy in layout.items())
        peak = max(map(sum, zip(*vectors, strict=True)))
        rows.append(dict(student=student, critic=critic, predicted_peak_bytes=peak, feasible=peak <= budget))
    return rows


def measure(args, row):
    student, critic = row['student'], row['critic']
    target = args.output.parent / f'candidate_{student}_{critic}.json'
    command = ['bash', args.launcher] if args.launcher else [sys.executable]
    with target.with_suffix('.log').open('w') as log:
        subprocess.run([*command, str(Path(__file__).with_name('profile_dmd2.py')), '--model', args.model,
                        '--training-sample', args.training_sample, '--data', args.data, '--policy', 'uniform',
                        '--save-count', str(student), '--critic-save-count', str(critic), '--warmup', '2',
                        '--steps', '5', '--output', str(target)], check=True, stdout=log, stderr=subprocess.STDOUT)
    run = json.loads(target.read_text())
    if run['manual_policies'] != policies(student, critic):
        raise ValueError(f'Measured block positions differ from the profiled layout in {target}')
    times = [step['step_ms'] for step in run['steps']]
    return dict(mean_ms=mean(times), min_ms=min(times), max_ms=max(times),
                peak_bytes=run['peak_allocated_bytes'], source=target.name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('model', 'training-sample', 'data'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--profile', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--launcher', help='Shell launcher accepting a Python script and its arguments')
    parser.add_argument('--budget-gib', type=float, default=54)
    args = parser.parse_args()
    profile = json.loads(args.profile.read_text())
    if profile['metadata'].get('world_size') != 1 or profile['metadata'].get('model') != args.model:
        parser.error('Use a single-rank profile collected with the same --model')
    budget = int(args.budget_gib * 2**30)
    matrix = candidates(profile, budget)
    feasible = [r for r in matrix if r['feasible']]
    if len(feasible) > 40:
        feasible = [r for r in feasible if r['student'] in (0, 15, 24, 30) and r['critic'] in (0, 8, 16, 24, 30)]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result = dict(budget_bytes=budget, matrix=matrix, winner=None)
    for row in feasible:
        row.update(measure(args, row))
        args.output.write_text(json.dumps(result, indent=2) + '\n')
    measured = [r for r in feasible if r['peak_bytes'] <= budget]
    if measured:
        result['winner'] = min(measured, key=lambda r: r['mean_ms'])
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    if not measured:
        parser.exit(1, 'No measured layout fits the allocated-memory budget\n')


if __name__ == '__main__':
    main()
