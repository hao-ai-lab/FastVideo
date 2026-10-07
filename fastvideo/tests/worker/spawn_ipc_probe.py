# SPDX-License-Identifier: Apache-2.0
"""Standard-library-only spawn target for executor IPC tests."""

import os
from pathlib import Path
import tempfile
import time


# Spawn unpickles the target's module before its Queue arguments. When the
# parent test creates gate_dir(os.getppid()), hold that import until it
# releases the gate, so the test can delete semaphores in between.
def gate_dir(parent_pid: int) -> Path:
    return Path(tempfile.gettempdir()) / f"fastvideo_ipc_gate_{parent_pid}"


_gate = gate_dir(os.getppid())
if _gate.is_dir():
    _fd = os.open(_gate / f"{os.getpid()}.stderr", os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    os.dup2(_fd, 2)
    os.close(_fd)
    (_gate / f"{os.getpid()}.ready").touch()
    _deadline = time.monotonic() + 60
    while not (_gate / "release").exists():
        if time.monotonic() >= _deadline:
            raise TimeoutError("IPC fault-test import barrier expired")
        time.sleep(.01)


def probe(input_queue, output_queue, reply):
    if input_queue is None and output_queue is None:
        reply.send("disabled")
    else:
        output_queue.put(input_queue.get(timeout=10) + 1)
        reply.send("enabled")
    reply.close()
