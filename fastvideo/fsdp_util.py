import torch
import os
import torch.distributed as dist
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper,
    CheckpointImpl,
    apply_activation_checkpointing,
)

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


def apply_fsdp_checkpointing(model):
    """apply activation checkpointing to model
    returns None as model is updated directly
    """
    print(f"--> applying fdsp activation checkpointing...")

    apply_activation_checkpointing(
        model, checkpoint_wrapper_fn=non_reentrant_wrapper, check_fn=check_fn
    )

bf16_mix = MixedPrecision(
    param_dtype=torch.bfloat16,
    # Gradient communication precision.
    reduce_dtype=torch.bfloat16,
    # Buffer precision.
    buffer_dtype=torch.bfloat16,
)



def get_fsdp_kwargs( sharding_strategy, cpu_offload=False):
    auto_wrap_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={
            MochiTransformerBlock,
        },
    )
    
    mixed_precision = bf16_mix
    
    if sharding_strategy == "full":
        sharding_strategy = ShardingStrategy.FULL_SHARD
    elif sharding_strategy == "none":
        sharding_strategy = ShardingStrategy.NO_SHARD
        auto_wrap_policy = None
    elif sharding_strategy == "hybrid_zero2":
        sharding_strategy  = ShardingStrategy._HYBRID_SHARD_ZERO2
    
    device_id = torch.cuda.current_device()
    cpu_offload=torch.distributed.fsdp.CPUOffload(offload_params=True) if cpu_offload else None
    return {
        "auto_wrap_policy": auto_wrap_policy,
        "mixed_precision": mixed_precision,
        "sharding_strategy": sharding_strategy,
        "device_id": device_id,
        "limit_all_gathers": True,
        "cpu_offload": cpu_offload,
    }
    
    
        