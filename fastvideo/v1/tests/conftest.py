import pytest
import torch.distributed as dist

from fastvideo.v1.distributed import (destroy_model_parallel,
                                      init_distributed_environment,
                                      initialize_model_parallel)


@pytest.fixture(scope="function")
def distributed_setup():
    """
    Fixture to set up and tear down the distributed environment for tests.

    This ensures proper cleanup even if tests fail.
    """
    init_distributed_environment(world_size=1,
                                 rank=0,
                                 distributed_init_method="env://",
                                 local_rank=0,
                                 backend="nccl")
    initialize_model_parallel(tensor_model_parallel_size=1,
                              sequence_model_parallel_size=1,
                              backend="nccl")
    yield

    if dist.is_initialized():
        destroy_model_parallel()
        dist.destroy_process_group()
