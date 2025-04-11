# SPDX-License-Identifier: Apache-2.0
import pytest
import torch.distributed as dist

import pytest
import torch
import numpy as np

from fastvideo.v1.distributed import (init_distributed_environment,
                                      initialize_model_parallel,
                                      cleanup_dist_env_and_memory)


@pytest.fixture(scope="function")
def distributed_setup():
    """
    Fixture to set up and tear down the distributed environment for tests.

    This ensures proper cleanup even if tests fail.
    """
    torch.manual_seed(42)
    np.random.seed(42)

    init_distributed_environment(world_size=1,
                                 rank=0,
                                 distributed_init_method="env://",
                                 local_rank=0,
                                 backend="nccl")
    initialize_model_parallel(tensor_model_parallel_size=1,
                              sequence_model_parallel_size=1,
                              backend="nccl")
    yield

    cleanup_dist_env_and_memory()
