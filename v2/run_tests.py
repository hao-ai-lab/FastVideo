#!/usr/bin/env python3
"""Zero-dependency test runner for v2.

The v2 core is numpy-only and CPU-testable. This runner discovers
``test_*`` functions in ``v2/tests/test_*.py`` and runs them without
needing pytest installed (it is fully pytest-compatible, so ``pytest`` works too).

Usage:
    python3 v2/run_tests.py            # run all
    python3 v2/run_tests.py interleave # run files matching a substring
"""
from __future__ import annotations

import importlib.util
import sys
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
TESTS_DIR = HERE / "tests"
REPO_ROOT = HERE.parent  # so `import v2` works


def _load(path: Path):
    spec = importlib.util.spec_from_file_location(f"_mftest_{path.stem}", path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main(argv: list[str]) -> int:
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    needle = argv[0] if argv else ""
    files = sorted(p for p in TESTS_DIR.glob("test_*.py") if needle in p.name)
    if not files:
        print(f"no test files matching {needle!r} under {TESTS_DIR}")
        return 1

    passed = failed = 0
    failures: list[str] = []
    for f in files:
        mod = _load(f)
        fns = [(n, getattr(mod, n)) for n in sorted(dir(mod)) if n.startswith("test_") and callable(getattr(mod, n))]
        for name, fn in fns:
            label = f"{f.name}::{name}"
            try:
                fn()
                passed += 1
                print(f"  PASS  {label}")
            except Exception:  # noqa: BLE001 - test runner reports everything
                failed += 1
                tb = traceback.format_exc()
                failures.append(f"{label}\n{tb}")
                print(f"  FAIL  {label}")

    print(f"\n{passed} passed, {failed} failed")
    if failures:
        print("\n" + "=" * 70 + "\nFAILURES\n" + "=" * 70)
        for fail in failures:
            print("\n" + fail)
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
