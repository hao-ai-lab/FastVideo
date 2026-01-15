# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from functools import lru_cache

import torch

from fastvideo.logger import init_logger
from fastvideo.platforms import current_platform

logger = init_logger(__name__)


@lru_cache(maxsize=1)
def is_pin_memory_available() -> bool:
    if current_platform.is_cpu() or current_platform.is_mps():
        return False

    try:
        torch.empty(1, device="cpu", pin_memory=True)
    except Exception as exc:
        logger.warning("Pinned memory is unavailable: %s", exc)
        return False

    return True
