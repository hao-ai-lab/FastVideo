# SPDX-License-Identifier: Apache-2.0
"""FP8 must refuse FSDP sharding instead of silently reading a local shard.

``_maybe_quantize_model`` runs *after* ``shard_model``, so by the time
``convert_model_to_fp8`` walks the tree every weight is a sharded ``DTensor``.
The converter reads ``weight.to_local()``, registers that shard as a plain
``_fp8_weight`` buffer and pops the FSDP-managed parameter. Nothing is left for
FSDP to all-gather, so the forward multiplies a shard by a full-width
activation and either raises a shape error or returns wrong numbers.

Two ranks are enough to show it and neither needs a GPU: ``DTensor`` sharding
is a placement, not a device, and ``gloo`` runs on CPU. That is also how the
behaviour was first reported.
"""
from __future__ import annotations

import contextlib
import os
import socket

import pytest
import torch
import torch.distributed as dist
from torch import nn

from fastvideo.layers.linear import ReplicatedLinear
from fastvideo.layers.quantization.fp8_config import (
    FP8Config,
    FP8QuantizeMethod,
    convert_model_to_fp8,
)
from fastvideo.models.loader.fsdp_load import (
    _has_fp8_convertible_layers,
    maybe_load_fsdp_model,
)

# Matches ``_FP8_EXACT_SUFFIXES``. A prefix that does not match leaves the
# layer with an ``FP8Config`` but no ``FP8QuantizeMethod``, which is a
# different state and is asserted separately below.
MATCHED_PREFIX = "minimax_h3.transformer_blocks.0.ff.fc_in"
UNMATCHED_PREFIX = "minimax_h3.proj_out"
WORLD_SIZE = 2


def _free_port() -> int:
    with contextlib.closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


class _FP8Linear(nn.Module):
    """One linear whose prefix matches, so ``FP8QuantizeMethod`` attaches."""

    def __init__(self, prefix: str = MATCHED_PREFIX) -> None:
        super().__init__()
        self.fc_in = ReplicatedLinear(8, 4, bias=False, quant_config=FP8Config(), prefix=prefix)


def test_helper_reports_only_layers_the_converter_would_touch() -> None:
    """The guard keys on what converts, not on what merely asked to convert.

    A model carrying an ``FP8Config`` whose prefix matched nothing converts
    nothing, so sharding it is safe and rejecting it would be a regression on
    the 0%-coverage path.
    """
    matched = _FP8Linear()
    assert isinstance(matched.fc_in.quant_method, FP8QuantizeMethod)
    assert _has_fp8_convertible_layers(matched) is True

    unmatched = _FP8Linear(prefix=UNMATCHED_PREFIX)
    assert not isinstance(unmatched.fc_in.quant_method, FP8QuantizeMethod)
    assert _has_fp8_convertible_layers(unmatched) is False

    plain = nn.Sequential(nn.Linear(8, 4, bias=False))
    assert _has_fp8_convertible_layers(plain) is False


def test_loader_refuses_fp8_when_the_shard_dim_is_greater_than_one() -> None:
    """The guard must fire before the loader builds a device mesh.

    ``maybe_load_fsdp_model`` hard-codes a CUDA mesh for every non-NPU
    platform, so a guard placed after mesh creation could not be reached on a
    CPU host at all, and on a GPU host would fire only after sharding had
    already started.
    """
    with pytest.raises(NotImplementedError) as excinfo:
        maybe_load_fsdp_model(
            model_cls=_FP8Linear,
            init_params={},
            weight_dir_list=[],
            device=torch.device("cpu"),
            hsdp_replicate_dim=1,
            hsdp_shard_dim=WORLD_SIZE,
            default_dtype=torch.bfloat16,
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.float32,
            fsdp_inference=True,
            training_mode=False,
        )
    message = str(excinfo.value)
    assert "hsdp_shard_dim=2" in message
    assert "FP8" in message


def _sharded_conversion_worker(rank: int, world_size: int, port: int) -> None:
    from torch.distributed.tensor import Shard, distribute_tensor, init_device_mesh

    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    try:
        mesh = init_device_mesh("cpu", (world_size, ))
        model = _FP8Linear()
        full = torch.arange(32, dtype=torch.float32).reshape(model.fc_in.weight.shape)
        global_shape = tuple(full.shape)
        model.fc_in.weight = nn.Parameter(distribute_tensor(full, mesh, [Shard(0)]))

        convert_model_to_fp8(model)

        local_shape = tuple(model.fc_in._fp8_weight.shape)
        # The recorded failure: the buffer is the rank's slice, not the layer.
        assert local_shape[0] == global_shape[0] // world_size, (
            f"rank {rank}: expected the local shard {global_shape[0] // world_size} rows, got {local_shape}")
        assert local_shape != global_shape, (
            f"rank {rank}: conversion kept the full {global_shape}; if FP8 became "
            "FSDP-aware, drop the guard in maybe_load_fsdp_model and update this test")
        # Nothing is left for FSDP to all-gather.
        assert "weight" not in model.fc_in._parameters
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_available() or not dist.is_gloo_available(), reason="needs torch.distributed with gloo")
def test_conversion_on_a_sharded_dtensor_keeps_only_the_local_shard() -> None:
    """Why the guard exists, on two real ranks.

    If this starts failing because the shapes now match, FP8 has become
    FSDP-aware and ``maybe_load_fsdp_model`` should stop refusing.
    """
    torch.multiprocessing.spawn(
        _sharded_conversion_worker,
        args=(WORLD_SIZE, _free_port()),
        nprocs=WORLD_SIZE,
        join=True,
    )
