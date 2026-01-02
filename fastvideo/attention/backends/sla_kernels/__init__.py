# SPDX-License-Identifier: Apache-2.0
# Adapted from TurboDiffusion SLA implementation
# Copyright (c) 2025 by SLA team.
#
# Citation:
# @article{zhang2025sla,
#   title={SLA: Beyond Sparsity in Diffusion Transformers via Fine-Tunable Sparse-Linear Attention},
#   author={Jintao Zhang and Haoxu Wang and Kai Jiang and Shuo Yang and Kaiwen Zheng and
#           Haocheng Xi and Ziteng Wang and Hongzhou Zhu and Min Zhao and Ion Stoica and
#           Joseph E. Gonzalez and Jun Zhu and Jianfei Chen},
#   journal={arXiv preprint arXiv:2509.24006},
#   year={2025}
# }

from .kernel import _attention
from .utils import get_block_map, get_cuda_arch, mean_pool

__all__ = ["_attention", "get_block_map", "get_cuda_arch", "mean_pool"]
