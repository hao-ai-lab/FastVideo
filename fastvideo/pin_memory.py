# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os

from functools import cache

import torch

from fastvideo.logger import init_logger
from fastvideo.platforms import current_platform

logger = init_logger(__name__)


def _probe_pin_memory() -> bool:
    if os.getenv("FASTVIDEO_DISABLE_PIN_MEMORY",
                 "") not in ("", "0", "false", "False"):
        return False
    if current_platform.is_cpu() or current_platform.is_mps():
        return False

    try:
        if torch.cuda.is_available():
            torch.cuda.current_device()
        _ = torch.empty(1024, device="cpu").pin_memory()
        _ = torch.empty(1024, device="cpu", pin_memory=True)
    except Exception as exc:
        logger.warning("Pinned memory is unavailable: %s", exc)
        return False

    return True


@cache
def _cached_pin_memory_available(pid: int) -> bool:
    return _probe_pin_memory()


def is_pin_memory_available() -> bool:
    return _cached_pin_memory_available(os.getpid())
