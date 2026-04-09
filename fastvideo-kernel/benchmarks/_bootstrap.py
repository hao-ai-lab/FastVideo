"""Make local benchmark scripts runnable from common repo entrypoints."""

from pathlib import Path
import sys


BENCHMARKS_DIR = Path(__file__).resolve().parent
KERNEL_ROOT = BENCHMARKS_DIR.parent
REPO_ROOT = KERNEL_ROOT.parent

for path in (KERNEL_ROOT, REPO_ROOT):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)
