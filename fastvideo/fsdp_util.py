from sympy import use
import torch
import os
import torch.distributed as dist
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper,
    CheckpointImpl,
    apply_activation_checkpointing,
)
from peft.utils.other import fsdp_auto_wrap_policy

from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    StateDictType,
    FullStateDictConfig,  # general model non-sharded, non-flattened params
    LocalStateDictConfig,  # flattened params, usable only by FSDP
    # ShardedStateDictConfig, # un-flattened param but shards, usable by other parallel schemes.
)

from fastvideo.model.modeling_mochi import MochiTransformerBlock

from functools import partial

from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

from torch.distributed.fsdp import MixedPrecision, ShardingStrategy
import functools


non_reentrant_wrapper = partial(
    checkpoint_wrapper,
    checkpoint_impl=CheckpointImpl.NO_REENTRANT,
)

check_fn = lambda submodule: isinstance(submodule, MochiTransformerBlock)


def apply_fsdp_checkpointing(model, p=1):
    # https://github.com/foundation-model-stack/fms-fsdp/blob/408c7516d69ea9b6bcd4c0f5efab26c0f64b3c2d/fms_fsdp/policies/ac_handler.py#L16
    """apply activation checkpointing to model
    returns None as model is updated directly
    """
    print(f"--> applying fdsp activation checkpointing...")

    block_idx = 0
    cut_off = 1 / 2
    # when passing p as a fraction number (e.g. 1/3), it will be interpreted
    # as a string in argv, thus we need eval("1/3") here for fractions.
    p = eval(p) if isinstance(p, str) else p

    def selective_checkpointing(submodule):
        nonlocal block_idx
        nonlocal cut_off

        if isinstance(submodule, MochiTransformerBlock):
            block_idx += 1
            if block_idx * p >= cut_off:
                cut_off += 1
                return True
        return False
    
    
    apply_activation_checkpointing(
        model, checkpoint_wrapper_fn=non_reentrant_wrapper, check_fn=selective_checkpointing
    )


float32 = MixedPrecision(
    param_dtype=torch.float32,
    # Gradient communication precision.
    reduce_dtype=torch.float32,
    # Buffer precision.
    buffer_dtype=torch.float32,
    cast_forward_inputs=False
)



def get_dit_fsdp_kwargs(sharding_strategy, use_lora=False,  cpu_offload=False):
    if use_lora:
        auto_wrap_policy = fsdp_auto_wrap_policy
    else:
        auto_wrap_policy = functools.partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls={
                MochiTransformerBlock,
            },
        )
    
    # we use float32 for fsdp but autocast during training
    mixed_precision = float32
    
    if sharding_strategy == "full":
        sharding_strategy = ShardingStrategy.FULL_SHARD
    elif sharding_strategy == "hybrid_full":
        sharding_strategy = ShardingStrategy.HYBRID_SHARD
    elif sharding_strategy == "none":
        sharding_strategy = ShardingStrategy.NO_SHARD
        auto_wrap_policy = None
    elif sharding_strategy == "hybrid_zero2":
        sharding_strategy  = ShardingStrategy._HYBRID_SHARD_ZERO2
    
    device_id = torch.cuda.current_device()
    cpu_offload=torch.distributed.fsdp.CPUOffload(offload_params=True) if cpu_offload else None
    fsdp_kwargs = {
        "auto_wrap_policy": auto_wrap_policy,
        "mixed_precision": mixed_precision,
        "sharding_strategy": sharding_strategy,
        "device_id": device_id,
        "limit_all_gathers": True,
        "cpu_offload": cpu_offload,
    }
    
    # Add LoRA-specific settings when LoRA is enabled
    if use_lora:
        fsdp_kwargs.update({
            "use_orig_params": False,  # Required for LoRA memory savings
            "sync_module_states": True,
        })
    
    return fsdp_kwargs
    
    
        
        

def get_discriminator_fsdp_kwargs():

    auto_wrap_policy = None


    # Use existing mixed precision settings

    mixed_precision = float32
    sharding_strategy  = ShardingStrategy.NO_SHARD
    device_id = torch.cuda.current_device()
    fsdp_kwargs = {
        "auto_wrap_policy": auto_wrap_policy,
        "mixed_precision": mixed_precision,
        "sharding_strategy": sharding_strategy,
        "device_id": device_id,
        "limit_all_gathers": True,
    }
    
    return fsdp_kwargs
    
    
        