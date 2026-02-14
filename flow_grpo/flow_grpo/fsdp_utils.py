import os
import functools
import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.distributed.fsdp import ShardingStrategy, BackwardPrefetch, MixedPrecision, CPUOffload
from torch.distributed.fsdp.api import StateDictType, FullStateDictConfig
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    CheckpointImpl,
    apply_activation_checkpointing,
    checkpoint_wrapper,
)
from safetensors.torch import save_file


class FSDPConfig:
    def __init__(
        self,
        sharding_strategy="FULL_SHARD",
        backward_prefetch="BACKWARD_PRE", 
        cpu_offload=False,
        num_replicate=1,
        num_shard=8,
        mixed_precision_dtype=torch.bfloat16,
        use_activation_checkpointing=True,
        use_device_mesh=False,
    ):
        self.sharding_strategy = sharding_strategy
        self.backward_prefetch = backward_prefetch
        self.cpu_offload = cpu_offload
        self.num_replicate = num_replicate
        self.num_shard = num_shard
        self.mixed_precision_dtype = mixed_precision_dtype
        self.use_activation_checkpointing = use_activation_checkpointing
        self.use_device_mesh = use_device_mesh

def fsdp_wrapper(model, fsdp_config, get_transformer_layer_cls, ignored_modules=None):
    if ignored_modules is None:
        ignored_modules = []
    
    # Setup device mesh for hybrid sharding if needed
    device_mesh = None
    if fsdp_config.sharding_strategy == 'HYBRID_SHARD' and fsdp_config.use_device_mesh:
        device_mesh = init_device_mesh(
            "cuda", 
            mesh_shape=(fsdp_config.num_replicate, fsdp_config.num_shard),
            mesh_dim_names=("replicate", "shard")
        )
    
    # Create FSDP model
    fsdp_model = FSDP(
        model,
        auto_wrap_policy=functools.partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls=get_transformer_layer_cls(),
        ),
        ignored_modules=ignored_modules,
        mixed_precision=MixedPrecision(
            param_dtype=fsdp_config.mixed_precision_dtype,
            reduce_dtype=fsdp_config.mixed_precision_dtype,
            buffer_dtype=fsdp_config.mixed_precision_dtype,
        ),
        device_id=dist.get_rank() % torch.cuda.device_count(),
        sharding_strategy=ShardingStrategy[fsdp_config.sharding_strategy],
        backward_prefetch=BackwardPrefetch[fsdp_config.backward_prefetch],
        cpu_offload=CPUOffload(offload_params=fsdp_config.cpu_offload),
        device_mesh=device_mesh,
        use_orig_params=True,
    )
    
    # Apply activation checkpointing if enabled
    if fsdp_config.use_activation_checkpointing:
        def grad_checkpoint_check_fn(module):
            """Check function to determine which modules to checkpoint"""
            return isinstance(module, tuple(get_transformer_layer_cls()))
        
        apply_activation_checkpointing(
            fsdp_model, 
            checkpoint_wrapper_fn=functools.partial(
                checkpoint_wrapper, checkpoint_impl=CheckpointImpl.NO_REENTRANT
            ), 
            check_fn=grad_checkpoint_check_fn
        )
    
    return fsdp_model

    
def save_fsdp_checkpoint(save_dir, model, global_step, rank):
    save_path = os.path.join(save_dir, f"checkpoint-{global_step}")
    os.makedirs(save_path, exist_ok=True)
    
    # Save full state dict (rank 0 only)
    with FSDP.state_dict_type(
        model,
        StateDictType.FULL_STATE_DICT,
        FullStateDictConfig(rank0_only=True, offload_to_cpu=True),
    ):
        state_dict = model.state_dict()
        if rank == 0:
            save_file(state_dict, os.path.join(save_path, "model.safetensors"))
            print(f"Model saved as safetensors: {save_path}/model.safetensors")
        del state_dict
    
    dist.barrier()

class OptimizerOffloadHook:
    def __init__(self):
        self.cpu_states = {}

    def pre_step_hook(self, optimizer, args, kwargs):
        for group in optimizer.param_groups:
            for param in group['params']:
                if param in optimizer.state and param in self.cpu_states:
                    state = optimizer.state[param]
                    for state_key, cpu_tensor in self.cpu_states[param].items():
                        state[state_key] = cpu_tensor.to(param.device, non_blocking=True)
                    del self.cpu_states[param]

    def post_step_hook(self, optimizer, args, kwargs):
        for group in optimizer.param_groups:
            for param in group['params']:
                if param in optimizer.state:
                    state = optimizer.state[param]
                    if state:
                        self.cpu_states[param] = {}
                        for state_key, state_tensor in state.items():
                            if isinstance(state_tensor, torch.Tensor):
                                self.cpu_states[param][state_key] = state_tensor.to('cpu', non_blocking=True)
                                state[state_key] = torch.empty(0, device=param.device)
                                
def register_optimizer_offload_hooks(optimizer):
    hook = OptimizerOffloadHook()

    pre_handle = optimizer.register_step_pre_hook(hook.pre_step_hook)
    post_handle = optimizer.register_step_post_hook(hook.post_step_hook)

    return [pre_handle, post_handle], hook

def init_distributed():
    """Initialize distributed training"""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
    else:
        print('Not using distributed mode')
        return False, 0, 1, 0
        
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend='nccl', init_method='env://')
    
    # Setup device
    device = torch.device(f'cuda:{local_rank}')
    torch.cuda.set_device(device)
    
    return True, rank, world_size, local_rank