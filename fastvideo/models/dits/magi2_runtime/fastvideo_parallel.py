# SPDX-License-Identifier: Apache-2.0
"""Connect MAGI-2 collectives to FastVideo-owned process groups."""

from fastvideo.distributed import get_dp_group, get_sp_group
from fastvideo.models.dits.magi2_runtime import psm as psm_manager
from fastvideo.models.dits.magi2_runtime.psm import bind_process_groups


MAGI2_PARALLEL_SIZE = 8


def bind_fastvideo_parallel_state() -> None:
    """Use FastVideo sequence parallelism for MAGI-2 context and experts."""
    sequence_group = get_sp_group()
    data_group = get_dp_group()
    if sequence_group.world_size != MAGI2_PARALLEL_SIZE:
        raise ValueError(
            "MAGI-2 Preview strict inference requires sp_size=8; "
            f"received sp_size={sequence_group.world_size}"
        )
    bind_process_groups(
        cp_group=sequence_group.device_group,
        cp_ranks=sequence_group.ranks,
        dp_group=data_group.device_group,
        dp_ranks=data_group.ranks,
        ep_group=sequence_group.device_group,
        ep_ranks=sequence_group.ranks,
    )
    if psm_manager.get_world_size("cp") != MAGI2_PARALLEL_SIZE:
        raise RuntimeError("MAGI-2 context-parallel binding did not preserve eight ranks")


__all__ = ["MAGI2_PARALLEL_SIZE", "bind_fastvideo_parallel_state"]
