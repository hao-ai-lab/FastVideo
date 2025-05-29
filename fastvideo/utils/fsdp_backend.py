import os
import torch
import torch.distributed as dist
from torch.distributed._composable.fsdp import (
    fully_shard,
    MixedPrecisionPolicy,
    CPUOffloadPolicy
)

class FSDPBackend:
    def __init__(self, world_size: int):
        self.world_size = world_size
        dist.init_process_group("nccl")
        self.device = torch.device("cuda", self.local_rank)
        
    @property
    def rank(self):
        return dist.get_rank()
        
    @property
    def local_rank(self):
        return int(os.environ.get("LOCAL_RANK", 0))
        
    @property
    def is_main_process(self):
        return self.rank == 0

    def apply_fsdp(
        self,
        model: torch.nn.Module,
        param_dtype: torch.dtype,
        reduce_dtype: torch.dtype,
        buffer_dtype: torch.dtype,
        cpu_offload: bool = False,
        *,
        reshard_after_forward: bool = True
    ):
        """Apply FSDP2 (composable FSDP) to model with mixed precision"""
        mp_policy = MixedPrecisionPolicy(
            param_dtype=param_dtype,
            reduce_dtype=reduce_dtype,
            buffer_dtype=buffer_dtype,
            cast_forward_inputs=True
        )

        fsdp_config = {
            "mp_policy": mp_policy,
            "reshard_after_forward": reshard_after_forward
        }

        if cpu_offload:
            fsdp_config["offload_policy"] = CPUOffloadPolicy(pin_memory=True)

        # Apply FSDP2 using fully_shard
        fully_shard(model, **fsdp_config)
        
        return model