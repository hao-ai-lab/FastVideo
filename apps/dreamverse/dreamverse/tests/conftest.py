from __future__ import annotations

import sys
from pathlib import Path


TESTS_DIR = Path(__file__).resolve().parent
SERVER_DIR = TESTS_DIR.parent
BENCHMARKS_DIR = SERVER_DIR / "benchmarks"

for path in (SERVER_DIR, BENCHMARKS_DIR):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)
