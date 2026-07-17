# SPDX-License-Identifier: Apache-2.0
"""Make the fastvideo_studio package importable when running these tests
directly (pytest apps/fastvideo_studio/tests) without installing apps/."""

import sys
from pathlib import Path

_APPS_DIR = str(Path(__file__).resolve().parents[2])
if _APPS_DIR not in sys.path:
    sys.path.insert(0, _APPS_DIR)
