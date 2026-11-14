# SPDX-License-Identifier: Apache-2.0
"""Pair-level bootstrap, kept separate from dense and aggregate throughput."""
import argparse
import json
from pathlib import Path
import statistics
from typing import Any

import numpy as np


def summarize(pairs: list[tuple[list[float], list[float]]]) -> dict[str, Any]:
    data = np.asarray([(statistics.median(a), statistics.median(b)) for a, b in pairs])
    rng = np.random.default_rng(1024)
    gains = []
    for _ in range(2000):
        sample = data[rng.integers(0, len(data), size=len(data))]
        gains.append(1 - np.median(sample[:, 1]) / np.median(sample[:, 0]))
    return {
        'pairs': len(data),
        'reference_ms': float(np.median(data[:, 0])),
        'candidate_ms': float(np.median(data[:, 1])),
        'elapsed_time_reduction': float(1 - np.median(data[:, 1]) / np.median(data[:, 0])),
        'paired95': np.percentile(gains, [2.5, 97.5]).tolist()
    }


def checked(rows: list[dict[str, Any]]) -> None:
    assert all(r['correctness'] and r['status'] == 'PASS' for r in rows)
    for r in rows:
        for key in ['paired', 'operator_paired']:
            for role in r.get(key, {}).values():
                assert role['stable'] and role['jit_count_before_after'][0] == role['jit_count_before_after'][1]
                assert len(role['warm_wall_ms_samples']) == 30


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--primary', type=Path, required=True)
    p.add_argument('--sp4', type=Path)
    p.add_argument('--replicas', type=Path)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    rows = [r for d in [0, 4, 5, 6] for r in json.loads((a.primary / f'device-{d}.json').read_text())]
    checked(rows)
    assert len(rows) == 12
    assert {(int(r['physical_device']), r['seed'])
            for r in rows} == {(d, s)
                               for d in [0, 4, 5, 6]
                               for s in [1024, 1025, 1026]}
    expected = json.loads(Path(__file__).with_name('primary_inputs.json').read_text())['input_sha256']
    assert all(r['input_hash'] == expected[str(r['seed'])] for r in rows)
    assert all(sum(r['order'][0] == 'reference' for r in rows if r['seed'] == s) == 2 for s in [1024, 1025, 1026])
    result = {
        key:
        summarize([(r[key]['reference']['warm_wall_ms_samples'], r[key]['candidate']['warm_wall_ms_samples'])
                   for r in rows])
        for key in ['paired', 'operator_paired']
    }
    result['dense_different_semantics'] = summarize([
        (r['dense']['warm_wall_ms_samples'], r['paired']['candidate']['warm_wall_ms_samples']) for r in rows
    ])
    result['threshold_pass'] = result['paired']['paired95'][0] > .03 and result['operator_paired']['paired95'][0] > .10
    if a.sp4:
        ranks = [json.loads((a.sp4 / f'rank-{rank}.json').read_text()) for rank in range(4)]
        for rs in ranks:
            checked(rs)
            assert len(rs) == 12 and all(r['world_size'] == 4 for r in rs)
        result['sp4'] = summarize([(r['paired']['reference']['max_rank_wall_ms_samples'],
                                    r['paired']['candidate']['max_rank_wall_ms_samples']) for r in ranks[0]])
        result['sp4']['single_group_scaling'] = result['paired']['candidate_ms'] / result['sp4']['candidate_ms']
    if a.replicas:
        rs = [
            json.loads((a.replicas / f'group-{g}/rank-{rank}.json').read_text()) for g in range(4) for rank in range(4)
        ]
        assert all(r['status'] == 'PASS' and r['correctness'] and r['finite'] for r in rs)
        elapsed = max(r['end'] for r in rs) - min(r['start'] for r in rs)
        result['four_sp4_replicas'] = {'calls': 120, 'elapsed_s': elapsed, 'calls_per_s': 120 / elapsed}
    a.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
