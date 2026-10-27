# SPDX-License-Identifier: Apache-2.0
"""Helpers for tests that depend on the CI runner backend or GPU architecture.

The Spark runner (NVIDIA DGX Spark, GB10, sm_121) exports
``FASTVIDEO_CI_BACKEND=spark4``. Cluster (GB200) runs leave it unset.
"""
import os

import pytest
import torch

CI_BACKEND_ENV = "FASTVIDEO_CI_BACKEND"
SPARK_BACKEND = "spark4"

CUTE_BLOCK_SPARSE_SM12X_REASON = ("FA4 CuTe block-sparse VSA kernels target sm100/sm103; they do not support "
                                  "sm_12x (RTX Blackwell / DGX Spark GB10)")


def on_spark_backend() -> bool:
    return os.environ.get(CI_BACKEND_ENV) == SPARK_BACKEND


def is_sm12x() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 12


def skip_cute_block_sparse_on_sm12x() -> None:
    """Skip the calling test on sm_12x, where CuTe block sparsity is unsupported."""
    if is_sm12x():
        pytest.skip(CUTE_BLOCK_SPARSE_SM12X_REASON)
