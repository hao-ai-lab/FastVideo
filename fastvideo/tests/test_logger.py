# SPDX-License-Identifier: Apache-2.0
"""CPU-only unit tests for the process-aware logging in :mod:`fastvideo.logger`."""
import logging

import pytest

from fastvideo.logger import init_logger

_LOGGER_NAME = "fastvideo.tests.test_logger"


class _ListHandler(logging.Handler):
    """Collect records directly from the logger.

    ``fastvideo`` loggers are configured with ``propagate=False``, so the
    standard ``caplog`` fixture cannot observe them.
    """

    def __init__(self) -> None:
        super().__init__()
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)


def _info_records(msg: str = "hello %s") -> list[logging.LogRecord]:
    logger = init_logger(_LOGGER_NAME)
    handler = _ListHandler()
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    try:
        logger.info(msg, "world")
    finally:
        logger.removeHandler(handler)
    return [record for record in handler.records if record.levelno == logging.INFO]


@pytest.fixture()
def _non_main_rank(monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setenv("RANK", "1")
    monkeypatch.setenv("LOCAL_RANK", "1")
    monkeypatch.delenv("FASTVIDEO_LOG_ALL_PROCESSES", raising=False)


def test_non_main_rank_is_suppressed_by_default(_non_main_rank):
    assert _info_records() == []


def test_log_all_processes_enables_non_main_rank(_non_main_rank, monkeypatch):
    monkeypatch.setenv("FASTVIDEO_LOG_ALL_PROCESSES", "1")
    assert len(_info_records()) == 1


def test_main_rank_still_logs_when_env_var_unset(monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "1")
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.delenv("FASTVIDEO_LOG_ALL_PROCESSES", raising=False)
    assert len(_info_records()) == 1


@pytest.mark.parametrize("log_all_processes", [None, "1"])
def test_info_once_accepts_stacklevel(monkeypatch, log_all_processes):
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("LOCAL_RANK", "0")
    if log_all_processes is None:
        monkeypatch.delenv("FASTVIDEO_LOG_ALL_PROCESSES", raising=False)
    else:
        monkeypatch.setenv("FASTVIDEO_LOG_ALL_PROCESSES", log_all_processes)

    logger = init_logger(_LOGGER_NAME)
    handler = _ListHandler()
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    try:
        logger.info_once("info_once stacklevel check %s" % log_all_processes)
    finally:
        logger.removeHandler(handler)
    assert [record for record in handler.records if record.levelno == logging.INFO]
