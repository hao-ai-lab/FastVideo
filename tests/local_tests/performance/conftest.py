# SPDX-License-Identifier: Apache-2.0
import contextlib

import pytest


@pytest.fixture
def env_overrides():
    """Keep environment overrides until the end of the test.

    Same fixture as fastvideo/tests/conftest.py, which pytest does not load for
    tests under tests/local_tests/. Enter ``envs.NAME.override(...)`` or
    ``envs.override_external(...)`` with ``env_overrides.enter_context(...)``;
    the previous values come back at teardown. See docs/contributing/env_vars.md.
    """
    with contextlib.ExitStack() as stack:
        yield stack
