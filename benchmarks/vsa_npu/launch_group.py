"""Launch explicitly bound ranks inside the caller's whole-group device lease."""
import argparse
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
from typing import Any

p = argparse.ArgumentParser()
p.add_argument('--output', required=True)
p.add_argument('--oracle-dir', default='.', help='Legacy compatibility argument; no external oracle is required')
p.add_argument('--suite', default='smoke')
p.add_argument('--timeout', type=int, default=600)
p.add_argument('--program', choices=['validate', 'timing', 'replica'], default='validate')
p.add_argument('--root', default='.')
p.add_argument('--reference-dir')
p.add_argument('--fail-rank', type=int)
p.add_argument('--timing-mode', choices=['sp4', 'sp4curve'], default='sp4')
a = p.parse_args()
out = Path(a.output)
out.mkdir(parents=True, exist_ok=False)
devices = os.environ['ASCEND_RT_VISIBLE_DEVICES'].split(',')
assert len(devices) == 4 and len(set(devices)) == 4
with socket.socket() as s:
    s.bind(('127.0.0.1', 0))
    port = s.getsockname()[1]
children = []
logs = []
result: dict[str, Any] = {
    'devices': devices,
    'rendezvous': f'127.0.0.1:{port}',
    'pid': os.getpid(),
    'start': time.time(),
    'children': []
}
try:
    for rank, device in enumerate(devices):
        env = os.environ.copy()
        env.update(ASCEND_RT_VISIBLE_DEVICES=device,
                   RANK=str(rank),
                   WORLD_SIZE='4',
                   LOCAL_RANK='0',
                   MASTER_ADDR='127.0.0.1',
                   MASTER_PORT=str(port),
                   HCCL_EXEC_TIMEOUT='60')
        env['TRITON_CACHE_DIR'] = str(Path(env['TRITON_CACHE_DIR']) / f'rank-{rank}')
        if a.fail_rank is not None:
            env['VSA_FAIL_RANK'] = str(a.fail_rank)
        log = (out / f'rank-{rank}.log').open('w')
        logs.append(log)
        cmd = [
            sys.executable,
            str(Path(__file__).with_name(a.program + '.py')), '--output',
            str(out / f'rank-{rank}.json')
        ]
        if a.program == 'validate':
            cmd += ['--oracle-dir', a.oracle_dir, '--suite', a.suite]
        else:
            cmd += ['--root', a.root]
        if a.reference_dir and a.program == 'replica':
            cmd += ['--reference-dir', a.reference_dir]
        if a.program == 'timing':
            cmd += ['--mode', a.timing_mode]
        child = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT)
        children.append(child)
        result['children'].append({
            'rank': rank,
            'physical': device,
            'pid': child.pid,
            'command': cmd,
            'cache': env['TRITON_CACHE_DIR']
        })
    (out / 'jobs.json').write_text(json.dumps(result, indent=2) + '\n')
    deadline = time.monotonic() + a.timeout
    while any(c.poll() is None for c in children):
        if any(c.poll() not in (None, 0) for c in children):
            raise RuntimeError('rank failed; terminate peers')
        if time.monotonic() > deadline:
            raise TimeoutError('whole SP4 group deadline')
        time.sleep(.5)
    result['status'] = 'PASS' if all(c.returncode == 0 for c in children) else 'FAILED'
except BaseException as exc:
    result['status'] = 'FAILED'
    result['failure'] = repr(exc)
finally:
    for child in children:
        if child.poll() is None:
            child.terminate()
    for child in children:
        try:
            child.wait(timeout=10)
        except subprocess.TimeoutExpired:
            child.kill()
            child.wait()
    for log in logs:
        log.close()
    result['exit_codes'] = [c.returncode for c in children]
    result['end'] = time.time()
    (out / 'jobs.json').write_text(json.dumps(result, indent=2) + '\n')
print(result['status'], result['exit_codes'])
sys.exit(result['status'] != 'PASS')
