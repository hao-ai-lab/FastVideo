# SPDX-License-Identifier: Apache-2.0
"""Health-check and exclusively lease explicit local 910B devices before imports."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import re
import signal
import socket
import subprocess
import sys
import time

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--devices', required=True, help='Allocated physical IDs, e.g. 0 or 0,1,2,3')
p.add_argument('--run-dir', type=Path, required=True, help='New directory for logs, cache and ownership')
p.add_argument('--cache-dir', type=Path, help='Explicit existing disk cache for startup measurements only')
p.add_argument('--lock-dir', type=Path, default=Path('/tmp/fastvideo-vsa-npu-leases'))
p.add_argument('--timeout', type=int, default=1200)
p.add_argument('command', nargs=argparse.REMAINDER)
a = p.parse_args()
devices = [int(x) for x in a.devices.split(',')]
if not devices or len(set(devices)) != len(devices) or any(d < 0 for d in devices):
    p.error('devices must be distinct nonnegative physical IDs')
cmd = a.command[1:] if a.command[:1] == ['--'] else a.command
if not cmd:
    p.error('a child command is required after --')
out = a.run_dir.resolve()
out.mkdir(parents=True, exist_ok=False)
a.lock_dir.mkdir(parents=True, exist_ok=True)
locks = []


def interrupted(signum, frame):
    raise InterruptedError(f'Lease interrupted by signal {signum}')


signal.signal(signal.SIGTERM, interrupted)
signal.signal(signal.SIGINT, interrupted)
for device in sorted(devices):
    handle = (a.lock_dir / f'{socket.gethostname()}-{device}.lock').open('a+')
    fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    locks.append(handle)
status = subprocess.run(['npu-smi', 'info'], capture_output=True, text=True, timeout=30)
(out / 'pre-npu.txt').write_text(status.stdout + status.stderr)
if status.returncode:
    raise RuntimeError('Cannot inspect live NPU status; no child was launched')
for device in devices:
    if not re.search(r'\|\s+' + str(device) + r'\s+910B\S*\s+\|\s+OK', status.stdout):
        raise RuntimeError(f'Physical NPU {device} is not a healthy 910B')
    if f'No running processes found in NPU {device} ' not in status.stdout:
        raise RuntimeError(f'Physical NPU {device} is occupied; no process was preempted')
env = os.environ.copy()
env.update(ASCEND_RT_VISIBLE_DEVICES=a.devices,
           OMP_NUM_THREADS='4',
           OPENBLAS_NUM_THREADS='4',
           TRITON_CACHE_DIR=str(a.cache_dir.resolve() if a.cache_dir else out / 'cache'))
record = {
    'hostname': socket.gethostname(),
    'devices': devices,
    'pid': os.getpid(),
    'command': cmd,
    'cache': env['TRITON_CACHE_DIR'],
    'start': time.time()
}
with (out / 'child.log').open('w') as log:
    child = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    record['child_pid'] = child.pid
    (out / 'ownership.json').write_text(json.dumps(record, indent=2) + '\n')
    try:
        record['returncode'] = child.wait(timeout=a.timeout)
    except BaseException:
        os.killpg(child.pid, signal.SIGTERM)
        try:
            child.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid, signal.SIGKILL)
            child.wait()
        record['returncode'] = child.returncode
    record['end'] = time.time()
    (out / 'ownership.json').write_text(json.dumps(record, indent=2) + '\n')
sys.exit(record['returncode'])
