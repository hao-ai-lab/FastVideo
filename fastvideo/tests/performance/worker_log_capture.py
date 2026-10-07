# SPDX-License-Identifier: Apache-2.0
"""Capture worker-process logs during performance benchmark runs.

Workers spawned by MultiprocExecutor forward records from the "fastvideo"
logger into a multiprocessing queue when one is passed to
``VideoGenerator.from_pretrained(..., log_queue=...)``. This module consumes
that queue into a size-capped per-benchmark log file so the log can be
attached to CI failures and uploaded as a build artifact.

Coverage caveats: only the "fastvideo" logger is forwarded (no torch/NCCL or
raw stderr output), and ranks > 0 suppress ``logger.info`` by default
(``local_main_process_only=True``), so the file contains rank-0 INFO plus
WARNING/ERROR from all ranks.
"""

import logging
import logging.handlers
import multiprocessing
import os

LOG_MAX_BYTES = 10 * 1024 * 1024
DEFAULT_TAIL_LINES = 200
LOG_DELIMITER = "=" * 78


class WorkerLogCapture:
    """Capture "fastvideo" logger output from the test process and all worker
    subprocesses into a size-capped per-benchmark log file."""

    def __init__(self, log_dir: str, benchmark_id: str, timestamp: str):
        os.makedirs(log_dir, exist_ok=True)
        self.log_path = os.path.join(log_dir, f"worker_{benchmark_id}_{timestamp}.log")
        # Manager queue required: with the spawn start method a plain
        # mp.Queue() cannot be shipped to workers through the executor RPC.
        self._manager = multiprocessing.Manager()
        self.log_queue = self._manager.Queue()
        self._file_handler = logging.handlers.RotatingFileHandler(
            self.log_path, maxBytes=LOG_MAX_BYTES, backupCount=1, encoding="utf-8")
        self._file_handler.setFormatter(
            logging.Formatter("%(asctime)s [%(levelname)s] %(processName)s %(name)s: %(message)s"))
        self._listener = logging.handlers.QueueListener(
            self.log_queue, self._file_handler, respect_handler_level=True)
        self._listener.start()
        # Also persist parent-process logs (run markers, load orchestration)
        # so the file reads as a single timeline.
        self._parent_logger = logging.getLogger("fastvideo")
        self._parent_logger.addHandler(self._file_handler)

    def close(self) -> None:
        """Drain the queue and release resources.

        Call after the executor is shut down (workers stop producing) and
        before any threshold assertion reads the log back.
        """
        try:
            self._parent_logger.removeHandler(self._file_handler)
            self._listener.stop()
            self._file_handler.close()
        finally:
            self._manager.shutdown()


def read_log_tail(log_path: str | None, max_lines: int = DEFAULT_TAIL_LINES) -> str | None:
    """Return the last ``max_lines`` of the capture, or None if unavailable.

    Concatenates the rotated ``.log.1`` backup (older) with the live file
    (newer) so the tail spans a rollover boundary.
    """
    if not log_path:
        return None
    lines: list[str] = []
    for path in (f"{log_path}.1", log_path):
        try:
            if os.path.isfile(path):
                with open(path, encoding="utf-8", errors="replace") as f:
                    lines.extend(f.readlines())
        except OSError:
            continue
    if not lines:
        return None
    tail = lines[-max_lines:]
    if len(lines) > max_lines:
        tail.insert(0, f"... truncated, showing last {max_lines} of {len(lines)} lines ...\n")
    return "".join(tail)


def format_worker_log_tail(benchmark_id: str, log_path: str | None) -> str:
    """Render a delimited worker-log tail block for CI failure output."""
    tail = read_log_tail(log_path)
    header = (f"{LOG_DELIMITER}\n"
              f"Worker log tail for {benchmark_id} ({log_path or 'no log file recorded'})\n"
              f"{LOG_DELIMITER}")
    if tail is None:
        return f"{header}\n<worker log unavailable: file missing or unreadable>\n{LOG_DELIMITER}"
    if not tail.endswith("\n"):
        tail += "\n"
    return f"{header}\n{tail}{LOG_DELIMITER}"
