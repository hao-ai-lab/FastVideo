# SPDX-License-Identifier: Apache-2.0
"""Exact distributed primitives used by the MAGI-2 inference kernels."""

from fastvideo.models.dits.magi2_runtime.psm import initialize_expert_parallel, initialize_model_parallel, psm

__all__ = ["initialize_expert_parallel", "initialize_model_parallel", "psm"]
