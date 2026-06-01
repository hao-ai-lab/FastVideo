# SPDX-License-Identifier: Apache-2.0
"""Single-rank ``distributed_setup`` fixture: the production component loaders
build TP layers that need the model-parallel groups initialized even on one GPU."""

import numpy as np
import pytest
import torch

from fastvideo.distributed import (
    cleanup_dist_env_and_memory,
    maybe_init_distributed_environment_and_model_parallel)


@pytest.fixture(scope="function")
def distributed_setup():
    """Set up (and tear down) a single-rank distributed + model-parallel env."""
    torch.manual_seed(42)
    np.random.seed(42)
    maybe_init_distributed_environment_and_model_parallel(1, 1)
    yield
    cleanup_dist_env_and_memory()
