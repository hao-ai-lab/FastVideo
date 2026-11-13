# SPDX-License-Identifier: Apache-2.0
"""Fit additive role-policy costs from complete production DMD2 steps."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import replace
from pathlib import Path
from statistics import mean
from typing import Any

from fastvideo.train.utils.activation_memory import MemoryProfile


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--profile', required=True)
    parser.add_argument('--reference', required=True)
    for role in ('student', 'critic'):
        for policy in ('save', 'offload'):
            parser.add_argument(f'--{role}-{policy}', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    profile = MemoryProfile.load(args.profile)
    reference = json.loads(Path(args.reference).read_text())
    if any(policies is None or set(policies.values()) != {'recompute'} for policies in reference['manual_policies'].values()):
        raise ValueError('The reference must recompute both trainable roles')
    baseline_ms = mean(step['step_ms'] for step in reference['steps'])
    counts = Counter(block.role for block in profile.blocks)
    deltas: dict[str, dict[str, float]] = {}
    records: dict[str, dict[str, Any]] = {}
    for role in ('student', 'critic'):
        deltas[role] = {'recompute': 0.0}
        records[role] = {}
        for policy in ('save', 'offload'):
            path = getattr(args, f'{role}_{policy}')
            record = json.loads(Path(path).read_text())
            if record['training_sample'] != reference['training_sample']:
                raise ValueError('Role costs must use the same independent calibration sample')
            for name, policies in record['manual_policies'].items():
                expected = policy if name == role else 'recompute'
                if policies is None or set(policies.values()) != {expected}:
                    raise ValueError(f'{path} must change only {role} to {policy}')
            elapsed = mean(step['step_ms'] for step in record['steps'])
            deltas[role][policy] = elapsed - baseline_ms
            records[role][policy] = {'source': path, 'complete_step_mean_ms': elapsed,
                                     'delta_from_reference_ms': elapsed - baseline_ms}
    minimum = {role: min(costs.values()) for role, costs in deltas.items()}
    fixed_step_ms = baseline_ms + sum(minimum.values())
    blocks = tuple(replace(block, choices=tuple(
        replace(choice, time_ms=(deltas[block.role][choice.policy] - minimum[block.role]) / counts[block.role])
        for choice in block.choices)) for block in profile.blocks)
    metadata = {**profile.metadata,
                'execution_time_cost_model': 'independent_role_complete_step_deltas',
                'fixed_step_ms': fixed_step_ms,
                'time_model_reference_ms': baseline_ms,
                'time_model_reference': args.reference,
                'role_step_deltas_ms': deltas,
                'role_minimum_deltas_ms': minimum,
                'role_cost_measurements': records,
                'time_model_formula': 'predicted step = fixed_step_ms + sum(selected PolicyCost.time_ms)',
                'time_model_calibration_sample': reference['training_sample']}
    fitted = replace(profile, blocks=blocks, metadata=metadata)
    fitted.save(args.output)
    print(json.dumps({'profile': args.output, 'reference_ms': baseline_ms,
                      'fixed_step_ms': fixed_step_ms, 'role_deltas_ms': deltas}))


if __name__ == '__main__':
    main()
