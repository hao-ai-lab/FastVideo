# SPDX-License-Identifier: Apache-2.0
# Every test package under tests/ and tests/local_tests/ carries an
# __init__.py, so pytest's default "prepend" import mode walks up to the repo
# root as the package boundary but only actually prepends it to sys.path when
# collection happens to reach it via a parent __init__.py first (e.g. `pytest
# tests/`). Invoking pytest directly on a narrower path (e.g. `pytest
# tests/local_tests/<family>/`) skips that side effect, so repo-root-relative
# imports like `fastvideo.*` or `scripts.checkpoint_conversion.*` fail with
# ModuleNotFoundError. An explicit root conftest.py guarantees the repo root
# is always on sys.path regardless of invocation scope.
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
