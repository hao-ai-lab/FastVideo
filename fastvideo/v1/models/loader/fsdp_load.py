# SPDX-License-Identifier: Apache-2.0

# Adapted from torchtune
# Copyright 2024 The TorchTune Authors.
# Copyright 2025 The FastVideo Authors.

import contextlib
import re
from collections import defaultdict
from itertools import chain
from typing import (Any, Callable, DefaultDict, Dict, Generator, Hashable, List,
                    Optional, Tuple, Type)

import torch
from torch import nn
from torch.distributed import DeviceMesh, init_device_mesh
from torch.distributed._composable.fsdp import CPUOffloadPolicy, fully_shard
from torch.distributed._tensor import distribute_tensor
from torch.nn.modules.module import _IncompatibleKeys

from fastvideo.v1.distributed.parallel_state import (
    get_sequence_model_parallel_world_size)
from fastvideo.v1.models.loader.weight_utils import safetensors_weights_iterator


# TODO(PY): move this to utils elsewhere
@contextlib.contextmanager
def set_default_dtype(dtype: torch.dtype) -> Generator[None, None, None]:
    """
    Context manager to set torch's default dtype.

    Args:
        dtype (torch.dtype): The desired default dtype inside the context manager.

    Returns:
        ContextManager: context manager for setting default dtype.

    Example:
        >>> with set_default_dtype(torch.bfloat16):
        >>>     x = torch.tensor([1, 2, 3])
        >>>     x.dtype
        torch.bfloat16


    """
    old_dtype = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        yield
    finally:
        torch.set_default_dtype(old_dtype)


def get_param_names_mapping(
        mapping_dict: Dict[str, str]) -> Callable[[str], tuple[str, Any, Any]]:
    """
    Creates a mapping function that transforms parameter names using regex patterns.
    
    Args:
        mapping_dict (Dict[str, str]): Dictionary mapping regex patterns to replacement patterns
        param_name (str): The parameter name to be transformed
        
    Returns:
        Callable[[str], str]: A function that maps parameter names from source to target format
    """

    def mapping_fn(name: str) -> tuple[str, Any, Any]:

        # Try to match and transform the name using the regex patterns in mapping_dict
        for pattern, replacement in mapping_dict.items():
            match = re.match(pattern, name)
            if match:
                merge_index = None
                total_splitted_params = None
                if isinstance(replacement, tuple):
                    merge_index = replacement[1]
                    total_splitted_params = replacement[2]
                    replacement = replacement[0]
                name = re.sub(pattern, replacement, name)
                return name, merge_index, total_splitted_params

        # If no pattern matches, return the original name
        return name, None, None

    return mapping_fn


# TODO(PY): add compile option
def load_fsdp_model(
    model_cls: Type[nn.Module],
    init_params: Dict[str, Any],
    weight_dir_list: List[str],
    device: torch.device,
    cpu_offload: bool = False,
    default_dtype: Optional[torch.dtype] = torch.bfloat16,
) -> torch.nn.Module:
    with set_default_dtype(default_dtype), torch.device("meta"):
        model = model_cls(**init_params)
    device_mesh = init_device_mesh(
        "cuda",
        mesh_shape=(get_sequence_model_parallel_world_size(), ),
        mesh_dim_names=("dp", ),
    )
    shard_model(model,
                cpu_offload=cpu_offload,
                reshard_after_forward=True,
                dp_mesh=device_mesh["dp"])
    weight_iterator = safetensors_weights_iterator(weight_dir_list)
    param_names_mapping_fn = get_param_names_mapping(model._param_names_mapping)
    load_fsdp_model_from_full_model_state_dict(
        model,
        weight_iterator,
        device,
        strict=True,
        cpu_offload=cpu_offload,
        param_names_mapping=param_names_mapping_fn,
    )
    for n, p in chain(model.named_parameters(), model.named_buffers()):
        if p.is_meta:
            raise RuntimeError(
                f"Unexpected param or buffer {n} on meta device.")
    for p in model.parameters():
        p.requires_grad = False
    return model


def shard_model(
    model,
    *,
    cpu_offload: bool,
    reshard_after_forward: bool = True,
    dp_mesh: Optional[DeviceMesh] = None,
) -> None:
    """
    Utility to shard a model with FSDP using the PyTorch Distributed fully_shard API.

    This method will over the model's named modules from the bottom-up and apply shard modules
    based on whether they meet any of the criteria from shard_conditions.

    Args:
        model (TransformerDecoder): Model to shard with FSDP.
        shard_conditions (List[Callable[[str, nn.Module], bool]]): A list of functions to determine
            which modules to shard with FSDP. Each function should take module name (relative to root)
            and the module itself, returning True if FSDP should shard the module and False otherwise.
            If any of shard_conditions return True for a given module, it will be sharded by FSDP.
        cpu_offload (bool): If set to True, FSDP will offload parameters, gradients, and optimizer
            states to CPU.
        reshard_after_forward (bool): Whether to reshard parameters and buffers after
            the forward pass. Setting this to True corresponds to the FULL_SHARD sharding strategy
            from FSDP1, while setting it to False corresponds to the SHARD_GRAD_OP sharding strategy.
        dp_mesh (Optional[DeviceMesh]): Device mesh to use for FSDP sharding under multiple parallelism.
            Default to None.

    Raises:
        ValueError: If no layer modules were sharded, indicating that no shard_condition was triggered.
    """
    fsdp_kwargs = {
        "reshard_after_forward": reshard_after_forward,
        "mesh": dp_mesh
    }
    if cpu_offload:
        fsdp_kwargs["offload_policy"] = CPUOffloadPolicy()

    # Shard the model with FSDP, iterating in reverse to start with
    # lowest-level modules first
    num_layers_sharded = 0
    for n, m in reversed(list(model.named_modules())):
        if any([
                shard_condition(n, m)
                for shard_condition in model._fsdp_shard_conditions
        ]):
            fully_shard(m, **fsdp_kwargs)
            num_layers_sharded += 1

    if num_layers_sharded == 0:
        raise ValueError(
            "No layer modules were sharded. Please check if shard conditions are working as expected."
        )

    # Finally shard the entire model to account for any stragglers
    fully_shard(model, **fsdp_kwargs)


# TODO(PY): device mesh for cfg parallel
def load_fsdp_model_from_full_model_state_dict(
    model: torch.nn.Module,
    full_sd_iterator: Generator[Tuple[str, torch.Tensor], None, None],
    device: torch.device,
    strict: bool = False,
    cpu_offload: bool = False,
    param_names_mapping: Optional[Callable[[str], tuple[str, Any, Any]]] = None,
) -> _IncompatibleKeys:
    """
    Converting full state dict into a sharded state dict
    and loading it into FSDP model
    Args:
        model (FSDPModule): Model to generate fully qualified names for cpu_state_dict
        full_sd_iterator (Generator): an iterator yielding (param_name, tensor) pairs
        device (torch.device): device used to move full state dict tensors
        strict (bool): flag to check if to load the model in strict mode
        cpu_offload (bool): flag to check if offload to CPU is enabled
        param_names_mapping (Optional[Callable[[str], str]]): a function that maps full param name to sharded param name

    Returns:
        ``NamedTuple`` with ``missing_keys`` and ``unexpected_keys`` fields:
            * **missing_keys** is a list of str containing the missing keys
            * **unexpected_keys** is a list of str containing the unexpected keys

    Raises:
        NotImplementedError: If got FSDP with more than 1D.
    """
    meta_sharded_sd = model.state_dict()

    sharded_sd = {}
    to_merge_params: DefaultDict[Hashable, Dict[Any, Any]] = defaultdict(dict)
    for source_param_name, full_tensor in full_sd_iterator:
        assert param_names_mapping is not None
        target_param_name, merge_index, num_params_to_merge = param_names_mapping(
            source_param_name)

        if merge_index is not None:
            to_merge_params[target_param_name][merge_index] = full_tensor
            if len(to_merge_params[target_param_name]) == num_params_to_merge:
                # cat at dim=1 according to the merge_index order
                sorted_tensors = [
                    to_merge_params[target_param_name][i]
                    for i in range(num_params_to_merge)
                ]
                full_tensor = torch.cat(sorted_tensors, dim=0)
                del to_merge_params[target_param_name]
            else:
                continue

        sharded_meta_param = meta_sharded_sd.get(target_param_name)
        if sharded_meta_param is None:
            raise ValueError(
                f"Parameter {source_param_name}-->{target_param_name} not found in meta sharded state dict"
            )
        full_tensor = full_tensor.to(sharded_meta_param.dtype).to(device)

        if not hasattr(sharded_meta_param, "device_mesh"):
            # In cases where parts of the model aren't sharded, some parameters will be plain tensors
            sharded_tensor = full_tensor
        else:
            sharded_tensor = distribute_tensor(
                full_tensor,
                sharded_meta_param.device_mesh,
                sharded_meta_param.placements,
            )
        if cpu_offload:
            sharded_tensor = sharded_tensor.cpu()
        sharded_sd[target_param_name] = nn.Parameter(sharded_tensor)
    # choose `assign=True` since we cannot call `copy_` on meta tensor
    return model.load_state_dict(sharded_sd, strict=strict, assign=True)
