"""Standard-library-only spawn target for executor IPC tests."""

import os
from pathlib import Path
import time


# Spawn unpickles the target's module before its Queue arguments. When the
# test sets FASTVIDEO_TEST_IPC_GATE, hold that import until the parent releases it.
_gate = os.environ.get("FASTVIDEO_TEST_IPC_GATE")
if _gate and str(os.getpid()) != os.environ.get("FASTVIDEO_TEST_IPC_PARENT"):
    _folder = Path(_gate)
    _fd = os.open(_folder / f"{os.getpid()}.stderr", os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    os.dup2(_fd, 2)
    os.close(_fd)
    (_folder / f"{os.getpid()}.ready").touch()
    _deadline = time.monotonic() + 60
    while not (_folder / "release").exists():
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
