from __future__ import annotations

import sys
from pathlib import Path


TESTS_DIR = Path(__file__).resolve().parent
DREAMVERSE_PACKAGE_DIR = TESTS_DIR.parent
DREAMVERSE_APP_DIR = DREAMVERSE_PACKAGE_DIR.parent
BENCHMARKS_DIR = DREAMVERSE_PACKAGE_DIR / "benchmarks"

for path in (DREAMVERSE_APP_DIR, BENCHMARKS_DIR):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)
