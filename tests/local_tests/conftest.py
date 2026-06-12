# SPDX-License-Identifier: Apache-2.0
"""Pytest fixtures for local tests."""

import os
import pytest
import torch
import numpy as np

from fastvideo.distributed import (
    maybe_init_distributed_environment_and_model_parallel,
    cleanup_dist_env_and_memory,
)


@pytest.fixture(scope="function")
def distributed_setup():
    """
    Fixture to set up and tear down the distributed environment for tests.
    
    This ensures proper cleanup even if tests fail.
    """
    # Set environment variables for single-process distributed init
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "29500")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    
    torch.manual_seed(42)
    np.random.seed(42)
    maybe_init_distributed_environment_and_model_parallel(1, 1)
    yield
    cleanup_dist_env_and_memory()

