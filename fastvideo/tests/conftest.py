# SPDX-License-Identifier: Apache-2.0
import contextlib
import os

import pytest

import torch
import numpy as np

from fastvideo import envs
from fastvideo.distributed import (maybe_init_distributed_environment_and_model_parallel, cleanup_dist_env_and_memory)
from fastvideo.utils import get_open_port


@pytest.fixture(scope="function")
def distributed_setup():
    """
    Fixture to set up and tear down the distributed environment for tests.

    This ensures proper cleanup even if tests fail.
    """
    torch.manual_seed(42)
    np.random.seed(42)
    with contextlib.ExitStack() as stack:
        # The env:// rendezvous needs these outside the CI runner. Keep values the
        # launcher provides; otherwise use a fresh port per test and restore on exit.
        if os.environ.get("MASTER_ADDR") is None:
            stack.enter_context(envs.override_external("MASTER_ADDR", "127.0.0.1"))
        if os.environ.get("MASTER_PORT") is None:
            stack.enter_context(envs.override_external("MASTER_PORT", str(get_open_port())))
        maybe_init_distributed_environment_and_model_parallel(1, 1)
        try:
            yield
        finally:
            cleanup_dist_env_and_memory()


@pytest.fixture
def env_overrides():
    """Keep environment overrides until the end of the test.

    Enter ``envs.NAME.override(...)`` or ``envs.override_external(...)`` with
    ``env_overrides.enter_context(...)``; the previous values come back at
    teardown. See docs/contributing/env_vars.md.
    """
    with contextlib.ExitStack() as stack:
        yield stack
