"""Launch with python -I to exclude job-controlled PYTHONPATH and user packages."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.gpu_ci.__main__ import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
