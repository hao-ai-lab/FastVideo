# SPDX-License-Identifier: Apache-2.0
"""Coordinate disjoint leased SP1 pairs or four actual SP4 replicas."""
import argparse
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--mode', choices=['primary', 'replicas'], required=True)
p.add_argument('--devices',
               required=True,
               help='Primary: exactly 0,4,5,6; replicas: 16 distinct allocated IDs in connected groups of four')
p.add_argument('--output', type=Path, required=True)
p.add_argument('--reference-dir', type=Path)
a = p.parse_args()
devices = [int(x) for x in a.devices.split(',')]
assert len(devices) == len(set(devices))
assert devices == [0, 4, 5, 6] if a.mode == 'primary' else len(devices) == 16
if a.mode == 'replicas' and a.reference_dir is None:
    p.error('replicas require locally generated --reference-dir')
out = a.output.resolve()
out.mkdir(parents=True, exist_ok=False)
bench = Path(__file__).resolve().parent
env = os.environ.copy()
env.update(FASTVIDEO_ATTENTION_BACKEND='VIDEO_SPARSE_ATTN',
           PYTHONPATH=str(bench.parents[1]) + os.pathsep + env.get('PYTHONPATH', ''))
barrier = out / 'ready'
if a.mode == 'primary':
    env.update(VSA_PHASE2_BARRIER=str(barrier), VSA_PHASE2_BARRIER_COUNT='4')
else:
    env['VSA_REPLICA_BARRIER'] = str(barrier)
children, logs, jobs = [], [], []


def interrupted(signum, frame):
    raise InterruptedError(f'Coordinator interrupted by signal {signum}')


signal.signal(signal.SIGTERM, interrupted)
signal.signal(signal.SIGINT, interrupted)
try:
    for group in range(4):
        ids = devices[group:group + 1] if a.mode == 'primary' else devices[4 * group:4 * group + 4]
        name = f'device-{ids[0]}' if a.mode == 'primary' else f'group-{group}'
        local_env = env.copy()
        with socket.socket() as sock:
            sock.bind(('127.0.0.1', 0))
            local_env.update(MASTER_ADDR='127.0.0.1', MASTER_PORT=str(sock.getsockname()[1]), HCCL_EXEC_TIMEOUT='60')
        args = [
            sys.executable,
            str(bench / 'lease.py'), '--devices', ','.join(map(str, ids)), '--run-dir',
            str(out / (name + '-lease')), '--timeout', '1800', '--', sys.executable
        ]
        if a.mode == 'primary':
            args += [str(bench / 'timing.py'), '--mode', 'primary', '--output', str(out / (name + '.json'))]
        else:
            args += [
                str(bench / 'launch_group.py'), '--program', 'replica', '--output',
                str(out / name), '--reference-dir',
                str(a.reference_dir.resolve()), '--timeout', '1680'
            ]
        log = (out / (name + '-coordinator.log')).open('w')
        logs.append(log)
        child = subprocess.Popen(args, env=local_env, stdout=log, stderr=subprocess.STDOUT)
        children.append(child)
        jobs.append({'devices': ids, 'pid': child.pid, 'command': args, 'started': time.time()})
    (out / 'jobs.json').write_text(json.dumps(jobs, indent=2) + '\n')
    while any(c.poll() is None for c in children):
        if any(c.poll() not in (None, 0) for c in children):
            raise RuntimeError('A leased worker failed; stop owned peers')
        time.sleep(.5)
finally:
    for child in children:
        if child.poll() is None:
            child.terminate()
    for child in children:
        try:
            child.wait(timeout=20)
        except subprocess.TimeoutExpired:
            child.kill()
            child.wait()
    for log in logs:
        log.close()
    (out / 'exit_codes.json').write_text(json.dumps([c.returncode for c in children]) + '\n')
sys.exit(any(c.returncode for c in children))
