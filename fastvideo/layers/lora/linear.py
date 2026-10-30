# SPDX-License-Identifier: Apache-2.0
# Code adapted from SGLang https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/lora/layers.py

import math

import torch
from torch import nn
from torch.distributed._composable.fsdp import (CPUOffloadPolicy, OffloadPolicy, fully_shard)
from torch.distributed.tensor import DTensor

import fastvideo.envs as envs
from fastvideo.distributed import (get_local_torch_device, get_tp_rank, split_tensor_along_last_dim,
                                   tensor_model_parallel_all_gather, tensor_model_parallel_all_reduce)
from fastvideo.layers.linear import (ColumnParallelLinear, LinearBase, MergedColumnParallelLinear, QKVParallelLinear,
                                     ReplicatedLinear, RowParallelLinear)
from fastvideo.layers.vocab_parallel_embedding import VocabParallelEmbedding
from fastvideo.utils import get_mixed_precision_state

torch._dynamo.config.recompile_limit = 16

# A full-sequence LoRA delta is the same size as the packed layer output. On a 32 GB card the base output
# already occupies that space, so materializing the delta and then ``out + delta`` asks for the buffer a third time.
_SIDE_BRANCH_CHUNK_BYTES = 256 * 1024 * 1024


@torch.library.custom_op("fastvideo::lora_add_low_rank_inplace", mutates_args=("out", ))
def _add_low_rank_inplace(out: torch.Tensor, x: torch.Tensor, lora_a: torch.Tensor, lora_b: torch.Tensor,
                          strength: torch.Tensor, alpha_over_rank: float) -> None:
    """Add ``strength * alpha_over_rank * B(Ax)`` into ``out`` without a second full-size activation."""
    scale = float(strength) * alpha_over_rank
    if scale == 0.0:
        return
    out_rows = out.reshape(-1, out.shape[-1])
    x_rows = x.reshape(-1, x.shape[-1])
    bytes_per_row = max(out_rows.shape[-1] * out_rows.element_size(), 1)
    row_count = max(1, _SIDE_BRANCH_CHUNK_BYTES // bytes_per_row)
    for start in range(0, x_rows.shape[0], row_count):
        stop = start + row_count
        update = x_rows[start:stop] @ lora_a.transpose(0, 1) @ lora_b.transpose(0, 1)
        if scale != 1.0:
            update = update * scale
        if update.dtype != out_rows.dtype:
            update = update.to(dtype=out_rows.dtype)
        out_rows[start:stop].add_(update)


@_add_low_rank_inplace.register_fake
def _add_low_rank_inplace_fake(out: torch.Tensor, x: torch.Tensor, lora_a: torch.Tensor, lora_b: torch.Tensor,
                               strength: torch.Tensor, alpha_over_rank: float) -> None:
    return None


def requires_side_branch(layer: nn.Module) -> bool:
    """Return whether an adapter must run beside ``layer`` instead of being merged into it.

    A packed linear without its original dense weight cannot merge an adapter while preserving the export's
    calibration. Its adapter runs as ``y = base(x) + strength * (alpha / rank) * B(A x)`` in the activation dtype.
    When the original dense weight is retained, merge into it as before and let the pipeline repack the
    quantized buffers after the adapter changes.
    ``FASTVIDEO_LORA_SIDE_BRANCH=1`` forces the side branch for every layer, which makes the adapter strength
    changeable at runtime without touching the base weights.
    """
    if envs.FASTVIDEO_LORA_SIDE_BRANCH.get():
        return True
    return getattr(layer, "weight", None) is None


class BaseLayerWithLoRA(nn.Module):
    # Wrappers whose forward does not add the unmerged delta must keep merging.
    supports_side_branch: bool = True

    def __init__(
        self,
        base_layer: nn.Module,
        lora_rank: int | None = None,
        lora_alpha: int | None = None,
        training_mode: bool = False,
    ):
        super().__init__()
        self.base_layer: nn.Module = base_layer

        self.merged: bool = False
        self.side_branch: bool = (self.supports_side_branch and not training_mode and requires_side_branch(base_layer))
        # A frozen CPU snapshot is needed only to undo inference-time merging.
        # Training never merges adapters; retaining every H3 projection here
        # otherwise adds several GiB of host memory per HSDP rank.
        self.cpu_weight = (None if training_mode or self.side_branch else base_layer.weight.detach().to("cpu").clone())
        # indicates adapter weights don't contain this layer
        # (which shouldn't normally happen, but we want to separate it from the case of erroneous merging)
        self.disable_lora: bool = False
        self.lora_rank = lora_rank
        self.lora_alpha: float | None = lora_alpha
        self.training_mode = training_mode
        self.lora_path: str | None = None
        # The side branch reads the strength from this CPU scalar, so compiled regions take it as an input. A Python
        # float would be guarded by value, and each new strength would recompile until fullgraph hits the limit.
        with torch.inference_mode(False):
            self._side_branch_strength = torch.ones((), dtype=torch.float64)
        self._strength_is_zero = False
        self.lora_strength = 1.0

        if training_mode:
            assert self.lora_rank is not None, "LoRA rank  must be set for training mode"
            if self.lora_rank is None or self.lora_alpha is None:
                self.lora_alpha = lora_rank
            self.base_layer.requires_grad_(False)
            in_dim = self.base_layer.weight.shape[1]
            out_dim = self.base_layer.weight.shape[0]
            self.lora_A = nn.Parameter(
                torch.zeros(self.lora_rank,
                            in_dim,
                            device=self.base_layer.weight.device,
                            dtype=self.base_layer.weight.dtype))
            self.lora_B = nn.Parameter(
                torch.zeros(out_dim,
                            self.lora_rank,
                            device=self.base_layer.weight.device,
                            dtype=self.base_layer.weight.dtype))
            torch.nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            torch.nn.init.zeros_(self.lora_B)
        else:
            self.lora_A = None
            self.lora_B = None

    @property
    def lora_strength(self) -> float:
        return self._lora_strength

    @lora_strength.setter
    def lora_strength(self, strength: float) -> None:
        self._lora_strength = float(strength)
        self._side_branch_strength.fill_(self._lora_strength)
        self._strength_is_zero = self._lora_strength == 0.0

    @property
    def weight(self) -> torch.Tensor:
        """The wrapped layer's weight.

        Model code reads ``layer.weight`` for perfectly ordinary reasons -- MiniMax H3's
        AdaLN modulation casts its input with ``self.linear.weight.dtype``, and its
        ``proj_in`` reads the tensor itself. Wrapping a layer should not change what
        reading it looks like from outside, and without this the wrap turns those into
        ``AttributeError`` at the first forward. Forcing every model to be listed in
        ``lora_target_modules`` around its own attribute access is the wrong fix: the
        list would have to be maintained against code it does not own, and getting it
        wrong fails at generation time rather than at load.

        Resolved through ``base_layer`` on each access rather than cached, because
        merging replaces that module outright.
        """
        return self.base_layer.weight

    @property
    def bias(self) -> torch.Tensor | None:
        """The wrapped layer's bias, or ``None`` when it has none."""
        return getattr(self.base_layer, "bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # The wrapper adds no compiled region around the base layer, so it runs exactly as it would unwrapped: eagerly,
        # or inside a caller's compiled block. Compiling a packed NVFP4/FP8 linear on its own changes its numerics, and
        # a wrapped layer with no adapter or strength 0 must equal the unwrapped layer exactly.
        if self.lora_A is None or (self.side_branch and self._strength_is_zero):
            # No adapter, or strength 0 on a side branch, is exactly the unadapted layer, down to the output dtype.
            # A merge-mode wrapper reaches this branch while its first adapter is still loading or after a load
            # failed: the pipeline wraps every linear before it reads the weights, so an unadapted request must
            # not depend on an adapter that is not there.
            return self.base_layer(x)

        out, output_bias = self.base_layer(x)
        if self.merged or self.disable_lora:
            return out.to(x), output_bias
        return self._add_adapter(out, x), output_bias

    @torch.compile()
    def _add_adapter(self, out: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        lora_A = self.lora_A
        lora_B = self.lora_B
        if isinstance(lora_B, DTensor):
            lora_B = lora_B.to_local()
            lora_A = lora_A.to_local()
        lora_A_sliced = self.slice_lora_a_weights(lora_A.to(x, non_blocking=True))
        lora_B_sliced = self.slice_lora_b_weights(lora_B.to(x, non_blocking=True))
        alpha_over_rank = 1.0
        if self.lora_alpha != self.lora_rank:
            alpha_over_rank = float(self.lora_alpha) / float(self.lora_rank)  # type: ignore[arg-type]
        if self.side_branch:
            # Keep chunk temporaries opaque to Inductor so fullgraph capture cannot retain a full-size delta.
            _add_low_rank_inplace(out, x, lora_A_sliced, lora_B_sliced, self._side_branch_strength, alpha_over_rank)
            return out
        scale = float(self.lora_strength) * alpha_over_rank
        delta = x @ lora_A_sliced.transpose(0, 1) @ lora_B_sliced.transpose(0, 1)
        if scale != 1.0:
            delta = delta * scale
        return out + delta

    def slice_lora_a_weights(self, A: torch.Tensor) -> torch.Tensor:
        return A

    def slice_lora_b_weights(self, B: torch.Tensor) -> torch.Tensor:
        return B

    def set_lora_weights(self,
                         A: torch.Tensor,
                         B: torch.Tensor,
                         lora_alpha: float | None = None,
                         training_mode: bool = False,
                         lora_path: str | None = None,
                         strength: float = 1.0,
                         accumulate: bool = False) -> None:
        if self.side_branch:
            if accumulate and self.lora_A is not None:
                raise ValueError("Accumulating adapters is unsupported on a side-branch LoRA layer (packed or "
                                 "forced side branch); load one adapter at a time.")
            # Kept in bf16 so the branch matches the activation dtype of a packed base layer.
            A, B = A.to(torch.bfloat16), B.to(torch.bfloat16)
        self.lora_A = torch.nn.Parameter(A)  # share storage with weights in the pipeline
        self.lora_B = torch.nn.Parameter(B)
        self.disable_lora = False

        # Store rank and alpha directly
        rank = A.shape[0]  # rank is the first dimension of A
        self.lora_rank = rank
        self.lora_alpha = lora_alpha if lora_alpha is not None else rank
        self.lora_strength = float(strength)

        if self.side_branch:
            self.merged = False
            self.lora_path = lora_path
            return
        if not training_mode:
            self.merge_lora_weights(accumulate=accumulate)
        self.lora_path = lora_path

    @torch.no_grad()
    def merge_lora_weights(self, accumulate: bool = False) -> None:
        if self.disable_lora or self.side_branch:
            return

        if self.merged and not accumulate:
            self.unmerge_lora_weights()
        assert self.lora_A is not None and self.lora_B is not None, "LoRA weights not set. Please set them first."
        if isinstance(self.base_layer.weight, DTensor):
            mesh = self.base_layer.weight.data.device_mesh
            unsharded_base_layer = ReplicatedLinear(
                input_size=self.base_layer.input_size,
                output_size=self.base_layer.output_size,
                bias=getattr(self.base_layer, "bias", None) is not None,
                skip_bias_add=self.base_layer.skip_bias_add,
                params_dtype=self.base_layer.params_dtype,
                quant_config=self.base_layer.quant_config,
                prefix=self.base_layer.prefix,
            )
            # Using offload param is on CPU, so current_device is for "CPU -> GPU -> merge -> CPU"
            current_device = self.base_layer.weight.data.device
            data = self.base_layer.weight.data.to(get_local_torch_device()).full_tensor()

            unsharded_base_layer.weight = nn.Parameter(self._merged_weight(data).to(current_device))
            if isinstance(getattr(self.base_layer, "bias", None), DTensor):
                unsharded_base_layer.bias = nn.Parameter(
                    self.base_layer.bias.to(get_local_torch_device(),
                                            non_blocking=True).full_tensor().to(current_device))

            offload_policy = CPUOffloadPolicy() if "cpu" in str(current_device) else OffloadPolicy()
            mp_policy = get_mixed_precision_state().mp_policy

            self.base_layer = fully_shard(unsharded_base_layer,
                                          mesh=mesh,
                                          mp_policy=mp_policy,
                                          offload_policy=offload_policy)
        else:
            current_device = self.base_layer.weight.data.device
            data = self.base_layer.weight.data.to(get_local_torch_device())

            self.base_layer.weight.data = self._merged_weight(data).to(current_device, non_blocking=True)

        self.merged = True

    def _merged_weight(self, weight: torch.Tensor) -> torch.Tensor:
        """``weight`` plus the scaled adapter delta, in ``weight``'s dtype.

        An adapter stored wider than the base (an FP32 adapter on a BF16 DiT) is multiplied and added at its own
        precision and rounded once, as a merge-then-cast reference does; with FSDP or without.
        """
        accumulate_dtype = torch.promote_types(weight.dtype, self.lora_A.dtype)
        lora_B = self.slice_lora_b_weights(self.lora_B.to(device=weight.device, dtype=accumulate_dtype))
        lora_A = self.slice_lora_a_weights(self.lora_A.to(device=weight.device, dtype=accumulate_dtype))
        scale = ((self.lora_alpha / self.lora_rank) if (self.lora_alpha and self.lora_rank) else 1.0)
        return (weight.to(accumulate_dtype) + (lora_B @ lora_A) * (scale * self.lora_strength)).to(weight.dtype)

    @torch.no_grad()
    # @torch.compile(dynamic=True)
    def unmerge_lora_weights(self, *, if_merged_only: bool = True) -> None:
        if self.disable_lora or self.side_branch:
            return

        if not self.merged:
            if if_merged_only:
                return
            raise ValueError("unmerge_lora_weights called but no LoRA is currently merged")

        # avoid precision loss
        if self.cpu_weight is None:
            raise RuntimeError("Training-mode LoRA layers do not retain an inference unmerge snapshot")
        if isinstance(self.base_layer.weight, DTensor):
            device = self.base_layer.weight.data.device
            self.base_layer.weight = nn.Parameter(self.cpu_weight.to(device, non_blocking=True))
        else:
            self.base_layer.weight.data = self.cpu_weight.data.to(self.base_layer.weight, non_blocking=True)

        self.merged = False


class VocabParallelEmbeddingWithLoRA(BaseLayerWithLoRA):
    """
    Vocab parallel embedding layer with support for LoRA (Low-Rank Adaptation).

    Note: The current version does not yet implement the LoRA functionality.
    This class behaves exactly the same as the base VocabParallelEmbedding.
    Future versions will integrate LoRA functionality to support efficient parameter fine-tuning.
    """
    supports_side_branch = False

    def __init__(
        self,
        base_layer: VocabParallelEmbedding,
    ) -> None:
        super().__init__(base_layer)

    def forward(self, input_: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("We don't support VocabParallelEmbeddingWithLoRA yet.")


class ColumnParallelLinearWithLoRA(BaseLayerWithLoRA):
    supports_side_branch = False

    def __init__(
        self,
        base_layer: ColumnParallelLinear,
        lora_rank: int | None = None,
        lora_alpha: int | None = None,
        training_mode: bool = False,
    ) -> None:
        super().__init__(base_layer, lora_rank, lora_alpha, training_mode)

    def forward(self, input_: torch.Tensor) -> torch.Tensor:
        # duplicate the logic in ColumnParallelLinear
        bias = self.base_layer.bias if not self.base_layer.skip_bias_add else None
        output_parallel = self.base_layer.quant_method.apply(self.base_layer, input_, bias)
        output = tensor_model_parallel_all_gather(output_parallel) if self.base_layer.gather_output else output_parallel
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
        lora_rank: int | None = None,
        lora_alpha: int | None = None,
        training_mode: bool = False,
    ) -> None:
        super().__init__(base_layer, lora_rank, lora_alpha, training_mode)

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
        lora_rank: int | None = None,
        lora_alpha: int | None = None,
        training_mode: bool = False,
    ) -> None:
        super().__init__(base_layer, lora_rank, lora_alpha, training_mode)

    def slice_lora_a_weights(self, A: torch.Tensor) -> torch.Tensor:
        return A

    def slice_lora_b_weights(self, B: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
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

        return B_q[q_start_idx:q_end_idx, :], B_kv[:, kv_start_idx:kv_end_idx, :]


class RowParallelLinearWithLoRA(BaseLayerWithLoRA):
    supports_side_branch = False

    def __init__(
        self,
        base_layer: RowParallelLinear,
        lora_rank: int | None = None,
        lora_alpha: int | None = None,
        training_mode: bool = False,
    ) -> None:
        super().__init__(base_layer, lora_rank, lora_alpha, training_mode)

    def forward(self, input_: torch.Tensor):
        # duplicate the logic in RowParallelLinear
        if self.base_layer.input_is_parallel:
            input_parallel = input_
        else:
            tp_rank = get_tp_rank()
            splitted_input = split_tensor_along_last_dim(input_, num_partitions=self.base_layer.tp_size)
            input_parallel = splitted_input[tp_rank].contiguous()
        output_parallel = self.base_layer.quant_method.apply(self.base_layer, input_parallel)

        if self.set_lora:
            output_parallel = self.apply_lora(output_parallel, input_parallel)

        if self.base_layer.reduce_results and self.base_layer.tp_size > 1:
            output_ = tensor_model_parallel_all_reduce(output_parallel)
        else:
            output_ = output_parallel

        if not self.base_layer.skip_bias_add:
            output = (output_ + self.base_layer.bias if self.base_layer.bias is not None else output_)
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


def get_lora_layer(layer: nn.Module,
                   lora_rank: int | None = None,
                   lora_alpha: int | None = None,
                   training_mode: bool = False) -> BaseLayerWithLoRA | None:
    supported_layer_types: dict[type[LinearBase], type[BaseLayerWithLoRA]] = {
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
            ret = lora_layer_type(layer, lora_rank=lora_rank, lora_alpha=lora_alpha, training_mode=training_mode)
            return ret
    return None


# source: https://github.com/vllm-project/vllm/blob/93b38bea5dd03e1b140ca997dfaadef86f8f1855/vllm/lora/utils.py#L9
def replace_submodule(model: nn.Module, module_name: str, new_module: nn.Module) -> nn.Module:
    """Replace a submodule in a model with a new module."""
    parent = model.get_submodule(".".join(module_name.split(".")[:-1]))
    target_name = module_name.split(".")[-1]
    setattr(parent, target_name, new_module)
    return new_module
