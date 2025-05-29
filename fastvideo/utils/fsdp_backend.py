import os
import torch
import torch.distributed as dist
from torch.distributed._composable.fsdp import (
    fully_shard,
    MixedPrecisionPolicy,
    CPUOffloadPolicy,
    FSDPModule
)
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_model_state_dict,
    set_model_state_dict,
)
import logging
from typing import Dict, Any, List, Union
import functools
from typing import Optional

logger = logging.getLogger(__name__)

class ModelWrapper(torch.distributed.checkpoint.stateful.Stateful):
    def __init__(self, model: Union[torch.nn.Module, List[torch.nn.Module]]) -> None:
        self.model = [model] if isinstance(model, torch.nn.Module) else model

    def state_dict(self) -> Dict[str, Any]:
        return {k: v for sd in map(get_model_state_dict, self.model) for k, v in sd.items()}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        func = functools.partial(
            set_model_state_dict,
            model_state_dict=state_dict,
            options=StateDictOptions(strict=False),
        )
        list(map(func, self.model))

class FSDPBackend:
    def __init__(self, world_size: int):
        self.world_size = world_size
        # Ensure distributed environment is initialized
        if not dist.is_initialized():
            raise RuntimeError(
                "Distributed environment must be initialized before creating FSDPBackend. "
                "Please call init_distributed_environment() first."
            )
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
        cpu_offload: bool = False,
        *,
        reshard_after_forward: bool = True
    ):
        """Apply FSDP2 (composable FSDP) to model with mixed precision"""
        # Check if FSDP is already applied
        if any(isinstance(p, FSDPModule) for p in model.modules()):
            return model  # Return model unchanged if FSDP already applied
        
        mp_policy = MixedPrecisionPolicy(
            param_dtype=param_dtype,
            reduce_dtype=reduce_dtype,
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

    def gather_state_dict_on_cpu_rank0(
        self,
        model, 
        device: Optional[torch.device] = None, 
        *, 
        is_main_process: bool
    ) -> Dict[str, Any]:
        """Gather full state dict on CPU for rank 0"""
        cpu_state_dict = {}
        sharded_sd = model.state_dict()
        
        for param_name, param in sharded_sd.items():
            if param.is_cpu:
                param = param.to(device)
            if hasattr(param, "_local_tensor"):
                param = param.full_tensor()
            if is_main_process:
                cpu_state_dict[param_name] = param.cpu()
            torch.distributed.barrier()
            
        return cpu_state_dict