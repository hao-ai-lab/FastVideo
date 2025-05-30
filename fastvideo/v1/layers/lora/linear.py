# Code adapted from SGLang https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/lora/layers.py

from typing import List, Tuple

import torch
from torch import nn

from fastvideo.v1.distributed import (
    get_tensor_model_parallel_rank,
    split_tensor_along_last_dim,
    tensor_model_parallel_all_gather,
    tensor_model_parallel_all_reduce,
)
from fastvideo.v1.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
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

    def forward(self, x: torch.Tensor):
        return self.base_layer.forward(x)


    def slice_lora_a_weights(self, A: torch.Tensor, tp_rank: int):
        return A

    def slice_lora_b_weights(self, B: torch.Tensor, tp_rank: int):
        return B

    def set_lora_weights(self, A: torch.Tensor, B: torch.Tensor, is_training: bool = False):
        self.lora_A = A # shares storage with weights in the pipeline 
        self.lora_B = B
        if not is_training:
            self.merge_lora_weights()

    def merge_lora_weights(self):
        if self.merged:
            raise ValueError("LoRA weights already merged. Please unmerge them first.")
        self.base_layer.weight.data += \
            self.slice_lora_a_weights(self.lora_A, get_tensor_model_parallel_rank()) @\
            self.slice_lora_b_weights(self.lora_B, get_tensor_model_parallel_rank())
        self.merged = True
        
    def unmerge_lora_weights(self):
        if not self.merged:
            raise ValueError("LoRA weights not merged. Please merge them first before unmerging.")
        self.base_layer.weight.data -= \
            self.slice_lora_a_weights(self.lora_A, get_tensor_model_parallel_rank()) @\
            self.slice_lora_b_weights(self.lora_B, get_tensor_model_parallel_rank())
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
        self.weight = base_layer.weight


class ColumnParallelLinearWithLoRA(BaseLayerWithLoRA):
    def __init__(
        self,
        base_layer: ColumnParallelLinear,
    ) -> None:
        super().__init__(base_layer)



    def forward(self, input_: torch.Tensor):
        # duplicate the logic in ColumnParallelLinear
        bias = self.base_layer.bias if not self.base_layer.skip_bias_add else None
        output_parallel = self.base_layer.quant_method.apply(
            self.base_layer, input_, bias
        )
        if self.base_layer.gather_output:
            output = tensor_model_parallel_all_gather(output_parallel)
        else:
            output = output_parallel
        output_bias = self.base_layer.bias if self.base_layer.skip_bias_add else None
        return output, output_bias

    def slice_lora_a_weights(self, A: torch.Tensor, tp_rank: int):
        return A

    def slice_lora_b_weights(self, B: torch.Tensor, tp_rank: int):
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

    def slice_lora_a_weights(self, A: torch.Tensor, tp_rank: int):
        return A

    def slice_lora_b_weights(self, B: torch.Tensor, tp_rank: int):
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

    def set_lora_info(
        self,
        A_buffer_qkv: torch.Tensor,
        B_buffer_q: torch.Tensor,
        B_buffer_kv: torch.Tensor,
    ):
        self.set_lora = True
        self.A_buffer_qkv = A_buffer_qkv

        if self.lora_backend.fuse_stacked_lora_b:
            assert (
                B_buffer_q.shape[-1] == B_buffer_kv.shape[-1]
            ), "The lora rank of q and kv should be the same when enabling fusion of qkv lora_b"
            output_dim_q, output_dim_kv = B_buffer_q.shape[-2], B_buffer_kv.shape[-2]

            # B_buffer_qkv: (num_lora, output_dim_q + 2 * output_dim_kv, r)
            if not hasattr(self, "B_buffer_qkv") or self.B_buffer_qkv is None:
                self.B_buffer_qkv = torch.empty(
                    (
                        B_buffer_q[0].shape[0],
                        output_dim_q + 2 * output_dim_kv,
                        B_buffer_q[0].shape[2],
                    ),
                    dtype=B_buffer_q[0].dtype,
                    device=B_buffer_q[0].device,
                )
            self.B_buffer_qkv[:, :output_dim_q, :].copy_(B_buffer_q[0])
            self.B_buffer_qkv[:, output_dim_q : output_dim_q + output_dim_kv, :].copy_(
                B_buffer_kv[0]
            )
            self.B_buffer_qkv[:, output_dim_q + output_dim_kv :, :].copy_(
                B_buffer_kv[1]
            )

            # Offsets of q/k/v in output dimension
            if not hasattr(self, "output_offset") or self.output_offset is None:
                self.output_offset = torch.empty(
                    4, dtype=torch.int32, device=B_buffer_q.device
                )
            self.output_offset[:4] = torch.tensor(
                [
                    0,
                    output_dim_q,
                    output_dim_q + output_dim_kv,
                    output_dim_q + 2 * output_dim_kv,
                ],
                dtype=torch.int32,
                device=B_buffer_q.device,
            )
            # For computing number of launched blocks
            self.max_qkv_out_dim = max(output_dim_q, output_dim_kv)
        else:
            self.B_buffer_qkv = (
                B_buffer_q,
                B_buffer_kv,
            )

    def slice_lora_a_weights(self, A: torch.Tensor, tp_rank: int):
        return A

    def slice_lora_b_weights(
        self, B: List[torch.Tensor], tp_rank: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
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

        return B_q[q_start_idx:q_end_idx, :], B_kv[:, kv_start_idx:kv_end_idx, :]


class RowParallelLinearWithLoRA(BaseLayerWithLoRA):
    def __init__(
        self,
        base_layer: RowParallelLinear,
    ) -> None:
        super().__init__(base_layer)

    def set_lora_info(self, A_buffer: torch.Tensor, B_buffer: torch.Tensor):
        self.set_lora = True
        self.A_buffer = A_buffer
        self.B_buffer = B_buffer


    def forward(self, input_: torch.Tensor):
        # duplicate the logic in RowParallelLinear
        if self.base_layer.input_is_parallel:
            input_parallel = input_
        else:
            tp_rank = get_tensor_model_parallel_rank()
            splitted_input = split_tensor_along_last_dim(
                input_, num_partitions=self.base_layer.tp_size
            )
            input_parallel = splitted_input[tp_rank].contiguous()
        output_parallel = self.base_layer.quant_method.apply(
            self.base_layer, input_parallel
        )

        if self.set_lora:
            output_parallel = self.apply_lora(output_parallel, input_parallel)

        if self.base_layer.reduce_results and self.base_layer.tp_size > 1:
            output_ = tensor_model_parallel_all_reduce(output_parallel)
        else:
            output_ = output_parallel

        if not self.base_layer.skip_bias_add:
            output = (
                output_ + self.base_layer.bias
                if self.base_layer.bias is not None
                else output_
            )
            output_bias = None
        else:
            output = output_
            output_bias = self.base_layer.bias
        return output, output_bias

    def slice_lora_a_weights(self, A: torch.Tensor, tp_rank: int):
        shard_size = self.base_layer.input_size_per_partition
        start_idx = tp_rank * shard_size
        end_idx = (tp_rank + 1) * shard_size
        A = A[:, start_idx:end_idx].contiguous()
        return A

    def slice_lora_b_weights(self, B: torch.Tensor, tp_rank: int):
        return B


def get_lora_layer(
    layer: nn.Module,
) -> BaseLayerWithLoRA:
    supported_layer_types = {
        # the order matters
        VocabParallelEmbedding: VocabParallelEmbeddingWithLoRA,
        QKVParallelLinear: QKVParallelLinearWithLoRA,
        MergedColumnParallelLinear: MergedColumnParallelLinearWithLoRA,
        ColumnParallelLinear: ColumnParallelLinearWithLoRA,
        RowParallelLinear: RowParallelLinearWithLoRA,
    }
    for src_layer_type, lora_layer_type in supported_layer_types.items():
        if isinstance(layer, src_layer_type):  # pylint: disable=unidiomatic-typecheck
            ret = lora_layer_type(layer)
            return ret
    return None