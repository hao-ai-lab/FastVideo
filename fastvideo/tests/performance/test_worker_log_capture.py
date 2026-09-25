# SPDX-License-Identifier: Apache-2.0

import logging

from fastvideo.tests.performance.worker_log_capture import (
    DEFAULT_TAIL_LINES,
    LOG_DELIMITER,
    WorkerLogCapture,
    format_worker_log_tail,
    read_log_tail,
)


def _make_record(msg):
    return logging.LogRecord(
        name="fastvideo.worker",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg=msg,
        args=(),
        exc_info=None,
    )


def test_capture_round_trips_queue_and_parent_logger(tmp_path):
    capture = WorkerLogCapture(str(tmp_path), "bench", "20260807T000000Z")
    try:
        capture.log_queue.put(_make_record("from worker queue"))
        logging.getLogger("fastvideo").warning("from parent process")
    finally:
        capture.close()

    content = open(capture.log_path, encoding="utf-8").read()
    assert "from worker queue" in content
    assert "from parent process" in content


def test_close_drains_pending_queue_records(tmp_path):
    capture = WorkerLogCapture(str(tmp_path), "bench", "20260807T000000Z")
    for i in range(50):
        capture.log_queue.put(_make_record(f"pending record {i}"))
    capture.close()

    content = open(capture.log_path, encoding="utf-8").read()
    for i in range(50):
        assert f"pending record {i}" in content


def test_read_log_tail_truncates_to_last_lines(tmp_path):
    log_path = tmp_path / "worker_bench.log"
    total = DEFAULT_TAIL_LINES + 50
    log_path.write_text("".join(f"line {i}\n" for i in range(total)), encoding="utf-8")

    tail = read_log_tail(str(log_path))
    assert tail is not None
    assert f"showing last {DEFAULT_TAIL_LINES} of {total} lines" in tail
    assert f"line {total - 1}" in tail
    assert "line 0\n" not in tail


def test_read_log_tail_concatenates_rotated_backup(tmp_path):
    log_path = tmp_path / "worker_bench.log"
    (tmp_path / "worker_bench.log.1").write_text("older rotated line\n", encoding="utf-8")
    log_path.write_text("newer live line\n", encoding="utf-8")

    tail = read_log_tail(str(log_path))
    assert tail == "older rotated line\nnewer live line\n"


def test_read_log_tail_handles_missing_paths():
    assert read_log_tail(None) is None
    assert read_log_tail("/nonexistent/worker.log") is None


def test_format_worker_log_tail_renders_unavailable_block():
    block = format_worker_log_tail("bench", None)
    assert LOG_DELIMITER in block
    assert "no log file recorded" in block
    assert "worker log unavailable" in block

    block = format_worker_log_tail("bench", "/nonexistent/worker.log")
    assert "worker log unavailable" in block


def test_format_worker_log_tail_renders_content(tmp_path):
    log_path = tmp_path / "worker_bench.log"
    log_path.write_text("dit slowdown detected\n", encoding="utf-8")

    block = format_worker_log_tail("bench", str(log_path))
    assert "Worker log tail for bench" in block
    assert "dit slowdown detected" in block
    assert block.endswith(LOG_DELIMITER)
