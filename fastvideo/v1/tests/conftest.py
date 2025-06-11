# SPDX-License-Identifier: Apache-2.0
import pytest
import torch.distributed as dist
import os
import subprocess
from pathlib import Path
from huggingface_hub import snapshot_download

import pytest
import torch
import numpy as np

from fastvideo.v1.distributed import (maybe_init_distributed_environment_and_model_parallel,
                                      cleanup_dist_env_and_memory)


@pytest.fixture(scope="function")
def distributed_setup():
    """
    Fixture to set up and tear down the distributed environment for tests.

    This ensures proper cleanup even if tests fail.
    """
    torch.manual_seed(42)
    np.random.seed(42)
    maybe_init_distributed_environment_and_model_parallel(1, 1)
    yield

    cleanup_dist_env_and_memory()


@pytest.fixture(scope="function")
def test_dataset_smol_crush():
    """
    Fixture to download and manage the test dataset.
    
    Downloads the crush-smol dataset from HuggingFace Hub if not already present.
    The dataset is downloaded once per test session and reused across tests.
    """
    data_dir = Path("data/crush-smol_parq")
    
    if not data_dir.exists():
        print(f"Downloading test dataset to {data_dir}...")
        snapshot_download(
            repo_id="PY007/crush-smol",
            local_dir=str(data_dir),
            repo_type="dataset",
            local_dir_use_symlinks=False
        )
    
    return str(data_dir)
