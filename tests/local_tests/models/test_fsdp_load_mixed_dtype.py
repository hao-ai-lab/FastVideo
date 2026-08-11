# SPDX-License-Identifier: Apache-2.0
"""Mixed-precision parameter loading coverage for native model dtype policies."""

import os

import pytest
import torch
from torch.distributed._tensor import DTensor
from torch.distributed.fsdp import MixedPrecisionPolicy

from fastvideo.distributed.parallel_state import init_distributed_environment
from fastvideo.models.dits.lingbot_video import LingBotVideoRouter
from fastvideo.models.loader.fsdp_load import (
    load_model_from_full_model_state_dict,
    maybe_load_fsdp_model,
    shard_model,
)


class _MixedDtypeModel(torch.nn.Module):
    """Tiny model that keeps one checkpoint parameter in fp32."""

    def __init__(self) -> None:
        super().__init__()
        self.bulk = torch.nn.Parameter(torch.zeros(2, dtype=torch.bfloat16))
        self.sensitive = torch.nn.Parameter(torch.zeros(2, dtype=torch.bfloat16))

    def _get_parameter_dtype(self, name: str, default_dtype: torch.dtype) -> torch.dtype:
        """Keep the sensitive test tensor in fp32."""
        return torch.float32 if name == "sensitive" else default_dtype


class _MixedDtypeBlock(torch.nn.Module):
    """Small FSDP child with one managed and one replicated parameter."""

    def __init__(self) -> None:
        super().__init__()
        self.bulk = torch.nn.Parameter(torch.eye(2, dtype=torch.bfloat16))
        self.sensitive = torch.nn.Parameter(torch.ones(2, dtype=torch.float32))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Apply both parameters so FSDP must initialize the mixed child."""
        return torch.nn.functional.linear(hidden_states, self.bulk) + self.sensitive.to(hidden_states.dtype)


class _NestedMixedDtypeModel(torch.nn.Module):
    """Nested model exercising both child and root ignored-parameter wiring."""

    def __init__(self) -> None:
        super().__init__()
        self.blocks = torch.nn.ModuleList([_MixedDtypeBlock()])
        self.root_bulk = torch.nn.Parameter(torch.ones(2, dtype=torch.bfloat16))
        self.root_sensitive = torch.nn.Parameter(torch.ones(2, dtype=torch.float32))

    def _get_parameter_dtype(self, name: str, default_dtype: torch.dtype) -> torch.dtype:
        """Keep every sensitive test parameter replicated in fp32."""
        return torch.float32 if "sensitive" in name else default_dtype

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Run the sharded child and consume both root parameters."""
        hidden_states = self.blocks[0](hidden_states)
        return hidden_states * self.root_bulk + self.root_sensitive.to(hidden_states.dtype)


class _RouterBufferModel(torch.nn.Module):
    """Wrap the released router to exercise its persistent correction buffer."""

    def __init__(self) -> None:
        super().__init__()
        self.router = LingBotVideoRouter(2, 3, 1, "sigmoid", True, None, None, 1.0)

    def _get_parameter_dtype(self, name: str, default_dtype: torch.dtype) -> torch.dtype:
        """Keep the released router state in fp32."""
        return torch.float32 if "router" in name else default_dtype


def test_full_state_dict_loader_honors_model_parameter_dtypes() -> None:
    """Load exact fp32 values for selected tensors while casting ordinary weights."""
    model = _MixedDtypeModel()
    bulk = torch.tensor([1.001, -2.003], dtype=torch.float32)
    sensitive = torch.tensor([3.001, -4.003], dtype=torch.float32)
    incompatible = load_model_from_full_model_state_dict(
        model,
        iter((("bulk", bulk), ("sensitive", sensitive))),
        device=torch.device("cpu"),
        param_dtype=torch.bfloat16,
        strict=True,
        param_names_mapping=lambda name: (name, None, None),
    )
    assert incompatible.missing_keys == []
    assert incompatible.unexpected_keys == []
    assert model.bulk.dtype == torch.bfloat16
    assert model.sensitive.dtype == torch.float32
    torch.testing.assert_close(model.bulk, bulk.to(torch.bfloat16))
    torch.testing.assert_close(model.sensitive, sensitive)


def test_full_state_dict_loader_preserves_router_bias_buffer() -> None:
    """Keep the MoE correction bias registered as a non-trainable buffer."""
    model = _RouterBufferModel()
    weight = torch.arange(6, dtype=torch.float32).reshape(3, 2)
    bias = torch.tensor([0.1, 0.2, 0.3], dtype=torch.float32)

    incompatible = load_model_from_full_model_state_dict(
        model,
        iter((("router.weight", weight), ("router.e_score_correction_bias", bias))),
        device=torch.device("cpu"),
        param_dtype=torch.bfloat16,
        strict=True,
        param_names_mapping=lambda name: (name, None, None),
    )

    assert incompatible.missing_keys == []
    assert incompatible.unexpected_keys == []
    assert "router.e_score_correction_bias" in dict(model.named_buffers())
    assert "router.e_score_correction_bias" not in dict(model.named_parameters())
    torch.testing.assert_close(model.router.e_score_correction_bias, bias)


def test_training_rejection_uses_fsdp_parameter_dtype() -> None:
    """Reject replicated fp32 training state when construction defaults to fp32."""
    with pytest.raises(NotImplementedError, match="separate gradient synchronization"):
        maybe_load_fsdp_model(
            model_cls=_MixedDtypeModel,
            init_params={},
            weight_dir_list=[],
            device=torch.device("cpu"),
            hsdp_replicate_dim=1,
            hsdp_shard_dim=1,
            default_dtype=torch.float32,
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.float32,
            training_mode=True,
            pin_cpu_memory=False,
        )


def test_nested_fsdp_ignores_selected_fp32_parameters() -> None:
    """Run a CUDA forward with bf16 DTensors and replicated fp32 parameters."""
    if os.environ.get("LINGBOT_VIDEO_RUN_GPU_TESTS") != "1":
        pytest.skip("set LINGBOT_VIDEO_RUN_GPU_TESTS=1 on an allocated GPU")
    if not torch.cuda.is_available():
        raise RuntimeError("mixed-dtype FSDP coverage requires CUDA")
    if not torch.distributed.is_initialized():
        init_distributed_environment(world_size=1, rank=0, local_rank=0)
    model = _NestedMixedDtypeModel().cuda()
    mesh = torch.distributed.init_device_mesh(
        "cuda",
        mesh_shape=(1, 1),
        mesh_dim_names=("replicate", "shard"),
    )
    shard_model(
        model,
        cpu_offload=False,
        mp_policy=MixedPrecisionPolicy(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.float32,
            output_dtype=None,
            cast_forward_inputs=False,
        ),
        mesh=mesh,
        fsdp_shard_conditions=[lambda name, module: isinstance(module, _MixedDtypeBlock)],
        pin_cpu_memory=False,
    )
    output = model(torch.ones(1, 2, device="cuda", dtype=torch.bfloat16))
    assert torch.isfinite(output).all()
    assert isinstance(model.blocks[0].bulk, DTensor)
    assert not isinstance(model.blocks[0].sensitive, DTensor)
    assert model.blocks[0].sensitive.dtype == torch.float32
    assert isinstance(model.root_bulk, DTensor)
    assert not isinstance(model.root_sensitive, DTensor)
    assert model.root_sensitive.dtype == torch.float32
