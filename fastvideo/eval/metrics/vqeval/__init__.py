"""Bootstrap the pinned VQeval source tree.

VQeval ships as a nested package in the LongVideoSparseAttention repository.
FastVideo keeps that repository as a submodule and imports the package directly,
matching the existing third-party evaluation convention without installing the
upstream project's broad dependency list.
"""

from __future__ import annotations

import sys
from pathlib import Path

_UPSTREAM_ROOT = Path(__file__).resolve().parents[3] / "third_party" / "eval" / "vqeval"
_UPSTREAM_PACKAGE_ROOT = _UPSTREAM_ROOT / "vqeval"

if _UPSTREAM_PACKAGE_ROOT.is_dir() and str(_UPSTREAM_PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(_UPSTREAM_PACKAGE_ROOT))
