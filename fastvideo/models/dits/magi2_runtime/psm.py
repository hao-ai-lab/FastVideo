# Copyright (c) 2025-2026 SandAI. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Process-group management for MAGI-2 inference.

Inference uses three parallelism axes:

* **Context parallel (cp)** splits one sample's token sequence across ranks.
* **Data parallel (dp)** gives different samples to different ranks.
* **Expert parallel (ep)** shards MoE experts across ranks.

Ranks are numbered cp-major: a cp group is a contiguous block while a dp
group strides across blocks.  With ``world_size=8, cp_size=4`` the cp
groups are ``[0,1,2,3]`` and ``[4,5,6,7]``; dp groups are ``[0,4]``,
``[1,5]``, ``[2,6]``, ``[3,7]``.

All inference code accesses groups through the ``psm`` singleton below.
"""

from datetime import timedelta
from typing import List, Optional

import torch.distributed as dist

_CP_GROUP = None
_CP_RANKS: Optional[List[int]] = None
_DP_GROUP = None
_DP_RANKS: Optional[List[int]] = None
_EP_GROUP = None
_EP_RANKS: Optional[List[int]] = None


def bind_process_groups(
    cp_group: "dist.ProcessGroup",
    cp_ranks: list[int],
    dp_group: "dist.ProcessGroup",
    dp_ranks: list[int],
    ep_group: "dist.ProcessGroup",
    ep_ranks: list[int],
) -> None:
    """Bind MAGI-2 parallel axes to process groups owned by FastVideo."""
    global _CP_GROUP, _CP_RANKS, _DP_GROUP, _DP_RANKS, _EP_GROUP, _EP_RANKS

    if not dist.is_initialized():
        raise RuntimeError("torch.distributed must be initialized before binding MAGI-2 groups")
    rank = dist.get_rank()
    for group_name, group, ranks in (
        ("cp", cp_group, cp_ranks),
        ("dp", dp_group, dp_ranks),
        ("ep", ep_group, ep_ranks),
    ):
        if group is None or rank not in ranks:
            raise ValueError(f"Rank {rank} does not belong to the MAGI-2 {group_name} group {ranks}")

    _CP_GROUP, _CP_RANKS = cp_group, list(cp_ranks)
    _DP_GROUP, _DP_RANKS = dp_group, list(dp_ranks)
    _EP_GROUP, _EP_RANKS = ep_group, list(ep_ranks)


def initialize_model_parallel(cp_size: int = 1, distributed_timeout_minutes: int = 30) -> None:
    """Create cp and dp groups.  Must be called after ``dist.init_process_group``."""
    global _CP_GROUP, _CP_RANKS, _DP_GROUP, _DP_RANKS

    assert dist.is_initialized(), "torch.distributed must be initialized first"
    assert _CP_GROUP is None, "parallel state is already initialized"

    world_size = dist.get_world_size()
    assert world_size % cp_size == 0, f"world_size {world_size} not divisible by cp_size {cp_size}"
    dp_size = world_size // cp_size
    rank = dist.get_rank()
    timeout = timedelta(minutes=distributed_timeout_minutes)

    for i in range(dp_size):
        ranks = list(range(i * cp_size, (i + 1) * cp_size))
        group = dist.new_group(ranks, timeout=timeout)
        if rank in ranks:
            _CP_GROUP, _CP_RANKS = group, ranks

    for i in range(cp_size):
        ranks = list(range(i, world_size, cp_size))
        group = dist.new_group(ranks, timeout=timeout)
        if rank in ranks:
            _DP_GROUP, _DP_RANKS = group, ranks


def initialize_expert_parallel(ep_size: int) -> None:
    """Create ep groups.  Call after ``initialize_model_parallel``."""
    global _EP_GROUP, _EP_RANKS
    if ep_size <= 1:
        return
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    assert world_size % ep_size == 0
    for i in range(world_size // ep_size):
        ranks = list(range(i * ep_size, (i + 1) * ep_size))
        group = dist.new_group(ranks)
        if rank in ranks:
            _EP_GROUP, _EP_RANKS = group, ranks


class ParallelStateManager:
    """Minimal psm interface used by MAGI-2 inference."""

    @staticmethod
    def is_initialized() -> bool:
        return _CP_GROUP is not None

    @staticmethod
    def get_global_rank() -> int:
        if dist.is_available() and dist.is_initialized():
            return dist.get_rank()
        return 0

    def get_world_size(self, dim: str = "") -> int:
        if not dist.is_available() or not dist.is_initialized():
            return 1
        if dim == "cp":
            return dist.get_world_size(group=_CP_GROUP)
        elif dim == "dp":
            return dist.get_world_size(group=_DP_GROUP)
        elif dim == "ep":
            return dist.get_world_size(group=_EP_GROUP) if _EP_GROUP else dist.get_world_size()
        return dist.get_world_size()

    def get_local_rank(self, dim: str) -> int:
        if not dist.is_available() or not dist.is_initialized():
            return 0
        if dim == "cp":
            return dist.get_rank(group=_CP_GROUP)
        elif dim == "dp":
            return dist.get_rank(group=_DP_GROUP)
        elif dim == "ep":
            return dist.get_rank(group=_EP_GROUP) if _EP_GROUP else dist.get_rank()
        return 0

    def is_group_first_rank(self, dim: str) -> bool:
        return self.get_local_rank(dim) == 0

    def is_group_last_rank(self, dim: str) -> bool:
        return self.get_local_rank(dim) == (self.get_world_size(dim) - 1)

    def get_parallel_group(self, dim: str) -> Optional["dist.ProcessGroup"]:
        if not dist.is_available() or not dist.is_initialized():
            return None
        if dim == "cp":
            return _CP_GROUP
        elif dim == "dp":
            return _DP_GROUP
        elif dim == "ep":
            return _EP_GROUP
        return None

    def get_global_ranks(self, dim: str) -> List[int]:
        if dim == "cp":
            return _CP_RANKS or [self.get_global_rank()]
        elif dim == "dp":
            return _DP_RANKS or [self.get_global_rank()]
        elif dim == "ep":
            return _EP_RANKS or list(range(dist.get_world_size()))
        return [self.get_global_rank()]


psm = ParallelStateManager()
