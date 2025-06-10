# SPDX-License-Identifier: Apache-2.0
# Code adapted from SGLang https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/lora/layers.py

from typing import Dict, List, Tuple, Type, Union

import torch
from torch import nn
from torch.distributed.tensor import DTensor, distribute_tensor

from fastvideo.v1.distributed import (get_tp_rank, split_tensor_along_last_dim,
                                      tensor_model_parallel_all_gather,
                                      tensor_model_parallel_all_reduce)
from fastvideo.v1.layers.linear import (ColumnParallelLinear, LinearBase,
                                        MergedColumnParallelLinear,
                                        QKVParallelLinear, ReplicatedLinear,
                                        RowParallelLinear)
from fastvideo.v1.layers.vocab_parallel_embedding import VocabParallelEmbedding


class BaseLayerWithLoRA(nn.Module):

    def __init__(
        self,
        base_layer: nn.Module,
    ):
        super().__init__()
        self.base_layer: nn.Module = base_layer
        self.lora_A: torch.Tensor = None
        self.lora_B: torch.Tensor = None
        self.merged: bool = False
        self.weight = base_layer.weight
        self.cpu_weight = base_layer.weight.to("cpu")
        self.unmerge_count = 0
        # indicates adapter weights don't contain this layer
        # (which shouldn't normally happen, but we want to separate it from the case of erroneous merging)
        self.disable_lora: bool = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base_layer.forward(x)

    def slice_lora_a_weights(self, A: torch.Tensor) -> torch.Tensor:
        return A

    def slice_lora_b_weights(self, B: torch.Tensor) -> torch.Tensor:
        return B

    def set_lora_weights(self,
                         A: torch.Tensor,
                         B: torch.Tensor,
                         training_mode: bool = False) -> None:
        self.lora_A = A  # share storage with weights in the pipeline
        self.lora_B = B
        self.disable_lora = False
        if not training_mode:
            self.merge_lora_weights()

    @torch.no_grad()
    def merge_lora_weights(self) -> None:
        if self.disable_lora:
            return

        if self.merged:
            raise ValueError(
                "LoRA weights already merged. Please unmerge them first.")
        assert self.lora_A is not None and self.lora_B is not None, "LoRA weights not set. Please set them first."
        if isinstance(self.base_layer.weight, DTensor):
            mesh = self.base_layer.weight.data.device_mesh
            placements = self.base_layer.weight.data.placements
            current_device = self.base_layer.weight.data.device
            data = self.base_layer.weight.data.to(
                f"cuda:{torch.cuda.current_device()}").full_tensor()
            data += (self.slice_lora_b_weights(self.lora_B)
                     @ self.slice_lora_a_weights(self.lora_A)).to(data)
            self.base_layer.weight.data = distribute_tensor(
                data, mesh, placements=placements).to(current_device)
        else:
            current_device = self.base_layer.weight.data.device
            data = self.base_layer.weight.data.to(
                f"cuda:{torch.cuda.current_device()}")
            data += \
                (self.slice_lora_b_weights(self.lora_B) @ self.slice_lora_a_weights(self.lora_A)).to(data)
            self.base_layer.weight.data = data.to(current_device)
        self.merged = True

    @torch.no_grad()
    def unmerge_lora_weights(self) -> None:
        if self.disable_lora:
            return

        if not self.merged:
            raise ValueError(
                "LoRA weights not merged. Please merge them first before unmerging."
            )
        self.unmerge_count += 1

        # Avoid precision loss
        if self.unmerge_count % 3 == 0:
            self.base_layer.weight.data = self.cpu_weight.data.to(
                self.base_layer.weight)

        if isinstance(self.base_layer.weight, DTensor):
            mesh = self.base_layer.weight.data.device_mesh
            placement = self.base_layer.weight.data.placements
            device = self.base_layer.weight.data.device
            data = self.base_layer.weight.data.to(
                f"cuda:{torch.cuda.current_device()}").full_tensor()
            data -= self.slice_lora_b_weights(
                self.lora_B) @ self.slice_lora_a_weights(self.lora_A)
            self.base_layer.weight.data = distribute_tensor(
                data, mesh, placements=placement).to(device)
        else:
            self.base_layer.weight.data -= \
                self.slice_lora_b_weights(self.lora_B) @\
                self.slice_lora_a_weights(self.lora_A)

        self.merged = False


class VocabParallelEmbeddingWithLoRA(BaseLayerWithLoRA):
    """
    Vocab parallel embedding layer with support for LoRA (Low-Rank Adaptation).

    Note: The current version does not yet implement the LoRA functionality.
    This class behaves exactly the same as the base VocabParallelEmbedding.
    Future versions will integrate LoRA functionality to support efficient parameter fine-tuning.
    """

    def __init__(
        self,
        base_layer: VocabParallelEmbedding,
    ) -> None:
        super().__init__(base_layer)

    def forward(self, input_: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError(
            "We don't support VocabParallelEmbeddingWithLoRA yet.")


class ColumnParallelLinearWithLoRA(BaseLayerWithLoRA):

    def __init__(
        self,
        base_layer: ColumnParallelLinear,
    ) -> None:
        super().__init__(base_layer)

    def forward(self, input_: torch.Tensor) -> torch.Tensor:
        # duplicate the logic in ColumnParallelLinear
        bias = self.base_layer.bias if not self.base_layer.skip_bias_add else None
        output_parallel = self.base_layer.quant_method.apply(
            self.base_layer, input_, bias)
        if self.base_layer.gather_output:
            output = tensor_model_parallel_all_gather(output_parallel)
        else:
            output = output_parallel
        output_bias = self.base_layer.bias if self.base_layer.skip_bias_add else None
        return output, output_bias

    def slice_lora_a_weights(self, A: torch.Tensor) -> torch.Tensor:
        return A

    def slice_lora_b_weights(self, B: torch.Tensor) -> torch.Tensor:
        tp_rank = get_tp_rank()
        shard_size = self.base_layer.output_partition_sizes[0]
        start_idx = tp_rank * shard_size
        end_idx = (tp_rank + 1) * shard_size
        B = B[start_idx:end_idx, :]
        return B


class MergedColumnParallelLinearWithLoRA(ColumnParallelLinearWithLoRA):

    def __init__(
        self,
        base_layer: MergedColumnParallelLinear,
    ) -> None:
        super().__init__(base_layer)

    def slice_lora_a_weights(self, A: torch.Tensor) -> torch.Tensor:
        return A.to(self.base_layer.weight)

    def slice_lora_b_weights(self, B: torch.Tensor) -> torch.Tensor:
        tp_rank = get_tp_rank()
        # Since the outputs for both gate and up are identical, we use a random one.
        shard_size = self.base_layer.output_partition_sizes[0]
        start_idx = tp_rank * shard_size
        end_idx = (tp_rank + 1) * shard_size
        return B[:, start_idx:end_idx, :]


class QKVParallelLinearWithLoRA(ColumnParallelLinearWithLoRA):

    def __init__(
        self,
        base_layer: QKVParallelLinear,
    ) -> None:
        super().__init__(base_layer)

    def slice_lora_a_weights(self, A: torch.Tensor) -> torch.Tensor:
        return A

    def slice_lora_b_weights(
            self, B: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        tp_rank = get_tp_rank()
        B_q, B_kv = B
        base_layer = self.base_layer
        q_proj_shard_size = base_layer.q_proj_shard_size
        kv_proj_shard_size = base_layer.kv_proj_shard_size
        num_kv_head_replicas = base_layer.num_kv_head_replicas

        q_start_idx = q_proj_shard_size * tp_rank
        q_end_idx = q_start_idx + q_proj_shard_size

        kv_shard_id = tp_rank // num_kv_head_replicas
        kv_start_idx = kv_proj_shard_size * kv_shard_id
        kv_end_idx = kv_start_idx + kv_proj_shard_size

        return B_q[q_start_idx:q_end_idx, :], B_kv[:,
                                                   kv_start_idx:kv_end_idx, :]


class RowParallelLinearWithLoRA(BaseLayerWithLoRA):

    def __init__(
        self,
        base_layer: RowParallelLinear,
    ) -> None:
        super().__init__(base_layer)

    def forward(self, input_: torch.Tensor):
        # duplicate the logic in RowParallelLinear
        if self.base_layer.input_is_parallel:
            input_parallel = input_
        else:
            tp_rank = get_tp_rank()
            splitted_input = split_tensor_along_last_dim(
                input_, num_partitions=self.base_layer.tp_size)
            input_parallel = splitted_input[tp_rank].contiguous()
        output_parallel = self.base_layer.quant_method.apply(
            self.base_layer, input_parallel)

        if self.set_lora:
            output_parallel = self.apply_lora(output_parallel, input_parallel)

        if self.base_layer.reduce_results and self.base_layer.tp_size > 1:
            output_ = tensor_model_parallel_all_reduce(output_parallel)
        else:
            output_ = output_parallel

        if not self.base_layer.skip_bias_add:
            output = (output_ + self.base_layer.bias
                      if self.base_layer.bias is not None else output_)
            output_bias = None
        else:
            output = output_
            output_bias = self.base_layer.bias
        return output, output_bias

    def slice_lora_a_weights(self, A: torch.Tensor) -> torch.Tensor:
        tp_rank = get_tp_rank()
        shard_size = self.base_layer.input_size_per_partition
        start_idx = tp_rank * shard_size
        end_idx = (tp_rank + 1) * shard_size
        A = A[:, start_idx:end_idx].contiguous()
        return A

    def slice_lora_b_weights(self, B: torch.Tensor) -> torch.Tensor:
        return B


def get_lora_layer(layer: nn.Module) -> Union[BaseLayerWithLoRA, None]:
    supported_layer_types: Dict[Type[LinearBase], Type[BaseLayerWithLoRA]] = {
        # the order matters
        # VocabParallelEmbedding: VocabParallelEmbeddingWithLoRA,
        QKVParallelLinear: QKVParallelLinearWithLoRA,
        MergedColumnParallelLinear: MergedColumnParallelLinearWithLoRA,
        ColumnParallelLinear: ColumnParallelLinearWithLoRA,
        RowParallelLinear: RowParallelLinearWithLoRA,
        ReplicatedLinear: BaseLayerWithLoRA,
    }
    for src_layer_type, lora_layer_type in supported_layer_types.items():
        if isinstance(layer, src_layer_type):  # pylint: disable=unidiomatic-typecheck
            ret = lora_layer_type(layer)
            return ret
    return None


# source: https://github.com/vllm-project/vllm/blob/93b38bea5dd03e1b140ca997dfaadef86f8f1855/vllm/lora/utils.py#L9
def replace_submodule(model: nn.Module, module_name: str,
                      new_module: nn.Module) -> nn.Module:
    """Replace a submodule in a model with a new module."""
    parent = model.get_submodule(".".join(module_name.split(".")[:-1]))
    target_name = module_name.split(".")[-1]
    setattr(parent, target_name, new_module)
    return new_module
