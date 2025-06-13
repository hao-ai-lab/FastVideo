# SPDX-License-Identifier: Apache-2.0
# adapted from vllm: https://github.com/vllm-project/vllm/blob/v0.7.3/vllm/logger.py
"""Logging configuration for fastvideo.v1."""
import datetime
import json
import logging
import os
import sys
from functools import lru_cache, partial
from logging import Logger
from logging.config import dictConfig
from os import path
from types import MethodType
from typing import Any, Optional, cast

import fastvideo.v1.envs as envs

FASTVIDEO_CONFIGURE_LOGGING = envs.FASTVIDEO_CONFIGURE_LOGGING
FASTVIDEO_LOGGING_CONFIG_PATH = envs.FASTVIDEO_LOGGING_CONFIG_PATH
FASTVIDEO_LOGGING_LEVEL = envs.FASTVIDEO_LOGGING_LEVEL
FASTVIDEO_LOGGING_PREFIX = envs.FASTVIDEO_LOGGING_PREFIX

RED = '\033[91m'
GREEN = '\033[92m'
RESET = '\033[0;0m'

_warned_local_main_process = False
_warned_main_process = False

_FORMAT = (f"{FASTVIDEO_LOGGING_PREFIX}%(levelname)s %(asctime)s "
           "[%(filename)s:%(lineno)d] %(message)s")
_DATE_FORMAT = "%m-%d %H:%M:%S"

DEFAULT_LOGGING_CONFIG = {
    "formatters": {
        "fastvideo": {
            "class": "fastvideo.v1.logging_utils.NewLineFormatter",
            "datefmt": _DATE_FORMAT,
            "format": _FORMAT,
        },
    },
    "handlers": {
        "fastvideo": {
            "class": "logging.StreamHandler",
            "formatter": "fastvideo",
            "level": FASTVIDEO_LOGGING_LEVEL,
            "stream": "ext://sys.stdout",
        },
    },
    "loggers": {
        "fastvideo": {
            "handlers": ["fastvideo"],
            "level": "DEBUG",
            "propagate": False,
        },
    },
    "root": {
        "handlers": ["fastvideo"],
        "level": "DEBUG",
    },
    "version": 1,
    "disable_existing_loggers": False
}


@lru_cache
def _print_info_once(logger: Logger, msg: str) -> None:
    # Set the stacklevel to 2 to print the original caller's line info
    logger.info(msg, stacklevel=2)


@lru_cache
def _print_warning_once(logger: Logger, msg: str) -> None:
    # Set the stacklevel to 2 to print the original caller's line info
    logger.warning(msg, stacklevel=2)


# TODO(will): add env variable to control this process-aware logging behavior
def _info(logger: Logger,
          msg: object,
          *args: Any,
          main_process_only: bool = False,
          local_main_process_only: bool = True,
          **kwargs: Any) -> None:
    """Process-aware INFO level logging function.
    
    This function controls logging behavior based on the process rank, allowing for 
    selective logging from specific processes in a distributed environment.
    
    Args:
        logger: The logger instance to use for logging
        msg: The message format string to log
        *args: Format string arguments
        main_process_only: If True, only log if this is the global main process (RANK=0)
        local_main_process_only: If True, only log if this is the local main process (LOCAL_RANK=0)
        **kwargs: Additional keyword arguments to pass to the logger.log method
            - stacklevel: Defaults to 2 to show the original caller's location
    
    Note:
        - When both main_process_only and local_main_process_only are True, 
          the message will be logged only if both conditions are met
        - When both are False, the message will be logged from all processes
        - By default, only logs from processes with LOCAL_RANK=0
    """
    try:
        local_rank = int(os.environ["LOCAL_RANK"])
        rank = int(os.environ["RANK"])
    except Exception:
        local_rank = 0
        rank = 0

    is_main_process = rank == 0
    is_local_main_process = local_rank == 0

    if (main_process_only and is_main_process) or (local_main_process_only
                                                   and is_local_main_process):
        logger.log(logging.INFO, msg, *args, stacklevel=2, **kwargs)

    global _warned_local_main_process, _warned_main_process

    if not _warned_local_main_process and local_main_process_only:
        logger.warning(
            '%s is_local_main_process is set to True, logging only from the local main process.%s',
            GREEN,
            RESET,
        )
        _warned_local_main_process = True
    if not _warned_main_process and main_process_only:
        logger.warning(
            '%s is_main_process_only is set to True, logging only from the main process.%s',
            GREEN,
            RESET,
        )
        _warned_main_process = True

    if not main_process_only and not local_main_process_only:
        logger.log(logging.INFO, msg, *args, stacklevel=2, **kwargs)


class _FastvideoLogger(Logger):
    """
    Note:
        This class is just to provide type information.
        We actually patch the methods directly on the :class:`logging.Logger`
        instance to avoid conflicting with other libraries such as
        `intel_extension_for_pytorch.utils._logger`.
    """

    def info_once(self, msg: str) -> None:
        """
        As :meth:`info`, but subsequent calls with the same message
        are silently dropped.
        """
        _print_info_once(self, msg)

    def warning_once(self, msg: str) -> None:
        """
        As :meth:`warning`, but subsequent calls with the same message
        are silently dropped.
        """
        _print_warning_once(self, msg)

    def info(  # type: ignore[override]
            self,
            msg: object,
            *args: Any,
            main_process_only: bool = False,
            local_main_process_only: bool = True,
            **kwargs: Any) -> None:
        _info(self,
              msg,
              *args,
              main_process_only=main_process_only,
              local_main_process_only=local_main_process_only,
              **kwargs)


def _configure_fastvideo_root_logger() -> None:
    logging_config = dict[str, Any]()

    if not FASTVIDEO_CONFIGURE_LOGGING and FASTVIDEO_LOGGING_CONFIG_PATH:
        raise RuntimeError(
            "FASTVIDEO_CONFIGURE_LOGGING evaluated to false, but "
            "FASTVIDEO_LOGGING_CONFIG_PATH was given. FASTVIDEO_LOGGING_CONFIG_PATH "
            "implies FASTVIDEO_CONFIGURE_LOGGING. Please enable "
            "FASTVIDEO_CONFIGURE_LOGGING or unset FASTVIDEO_LOGGING_CONFIG_PATH."
        )

    if FASTVIDEO_CONFIGURE_LOGGING:
        logging_config = DEFAULT_LOGGING_CONFIG

    if FASTVIDEO_LOGGING_CONFIG_PATH:
        if not path.exists(FASTVIDEO_LOGGING_CONFIG_PATH):
            raise RuntimeError(
                "Could not load logging config. File does not exist: %s",
                FASTVIDEO_LOGGING_CONFIG_PATH)
        with open(FASTVIDEO_LOGGING_CONFIG_PATH, encoding="utf-8") as file:
            custom_config = json.loads(file.read())

        if not isinstance(custom_config, dict):
            raise ValueError("Invalid logging config. Expected Dict, got %s.",
                             type(custom_config).__name__)
        logging_config = custom_config

    for formatter in logging_config.get("formatters", {}).values():
        # This provides backwards compatibility after #10134.
        if formatter.get("class") == "fastvideo.v1.logging.NewLineFormatter":
            formatter["class"] = "fastvideo.v1.logging_utils.NewLineFormatter"

    if logging_config:
        dictConfig(logging_config)


def init_logger(name: str) -> _FastvideoLogger:
    """The main purpose of this function is to ensure that loggers are
    retrieved in such a way that we can be sure the root fastvideo logger has
    already been configured."""

    logger = logging.getLogger(name)

    methods_to_patch = {
        "info_once": _print_info_once,
        "warning_once": _print_warning_once,
        "info": _info,
    }

    for method_name, method in methods_to_patch.items():
        setattr(logger, method_name,
                MethodType(method, logger))  # type: ignore[arg-type]

    return cast(_FastvideoLogger, logger)


# The root logger is initialized when the module is imported.
# This is thread-safe as the module is only imported once,
# guaranteed by the Python GIL.
_configure_fastvideo_root_logger()

logger = init_logger(__name__)


def _trace_calls(log_path, root_dir, frame, event, arg=None):
    if event in ['call', 'return']:
        # Extract the filename, line number, function name, and the code object
        filename = frame.f_code.co_filename
        lineno = frame.f_lineno
        func_name = frame.f_code.co_name
        if not filename.startswith(root_dir):
            # only log the functions in the fastvideo root_dir
            return
        # Log every function call or return
        try:
            last_frame = frame.f_back
            if last_frame is not None:
                last_filename = last_frame.f_code.co_filename
                last_lineno = last_frame.f_lineno
                last_func_name = last_frame.f_code.co_name
            else:
                # initial frame
                last_filename = ""
                last_lineno = 0
                last_func_name = ""
            with open(log_path, 'a') as f:
                ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
                if event == 'call':
                    f.write(f"{ts} Call to"
                            f" {func_name} in {filename}:{lineno}"
                            f" from {last_func_name} in {last_filename}:"
                            f"{last_lineno}\n")
                else:
                    f.write(f"{ts} Return from"
                            f" {func_name} in {filename}:{lineno}"
                            f" to {last_func_name} in {last_filename}:"
                            f"{last_lineno}\n")
        except NameError:
            # modules are deleted during shutdown
            pass
    return partial(_trace_calls, log_path, root_dir)


def enable_trace_function_call(log_file_path: str,
                               root_dir: Optional[str] = None):
    """
    Enable tracing of every function call in code under `root_dir`.
    This is useful for debugging hangs or crashes.
    `log_file_path` is the path to the log file.
    `root_dir` is the root directory of the code to trace. If None, it is the
    fastvideo root directory.

    Note that this call is thread-level, any threads calling this function
    will have the trace enabled. Other threads will not be affected.
    """
    logger.warning(
        "FASTVIDEO_TRACE_FUNCTION is enabled. It will record every"
        " function executed by Python. This will slow down the code. It "
        "is suggested to be used for debugging hang or crashes only.")
    logger.info("Trace frame log is saved to %s", log_file_path)
    if root_dir is None:
        # by default, this is the fastvideo root directory
        root_dir = os.path.dirname(os.path.dirname(__file__))
    sys.settrace(partial(_trace_calls, log_file_path, root_dir))
