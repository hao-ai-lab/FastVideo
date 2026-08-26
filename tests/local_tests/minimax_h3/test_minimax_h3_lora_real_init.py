# SPDX-License-Identifier: Apache-2.0
"""Opt-in real-checkpoint MiniMax H3 Ref2VA LoRA ownership gate.

This gate loads only ``transformer_ref``. It does not build a dataloader, run a
forward pass, gather a state dict, export, or retain a CPU model snapshot.
Expected GB10 unified-memory working set is approximately 70--80 GiB: about
62 GiB of BF16 transformer weights, at most one roughly 5 GiB input shard, the
rank-32 adapters/FSDP metadata, and loader/runtime overhead.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch
from torch.distributed.fsdp import FSDPModule
from torch.distributed.tensor import DTensor, Shard

from fastvideo.distributed import (
    cleanup_dist_env_and_memory,
    maybe_init_distributed_environment_and_model_parallel,
)
from fastvideo.layers.lora.linear import BaseLayerWithLoRA
from fastvideo.train.models.minimax_h3.minimax_h3_ref2va import MiniMaxH3Ref2VALoraModel
from fastvideo.train.utils.config import load_run_config

_TARGETS = [
    "attn.to_q",
    "attn.to_k",
    "attn.to_v",
    "attn.to_out",
    "ff.fc_in",
    "ff.fc_out",
]


def test_minimax_h3_ref2va_real_checkpoint_lora_fsdp_init() -> None:
    if os.environ.get("MINIMAX_H3_RUN_LORA_REAL_INIT") != "1":
        pytest.skip("set MINIMAX_H3_RUN_LORA_REAL_INIT=1 to run the real-checkpoint init gate")
    if not torch.cuda.is_available():
        pytest.fail("MINIMAX_H3_RUN_LORA_REAL_INIT=1 requires CUDA", pytrace=False)
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        pytest.fail("the scoped real-checkpoint init gate requires WORLD_SIZE=1", pytrace=False)

    model_root = Path(os.environ.get("MINIMAX_H3_MODEL_ROOT", "/home/will/models/MiniMax-H3")).resolve()
    component = model_root / "transformer_ref"
    if not (model_root / "model_index.json").is_file() or not (component / "config.json").is_file():
        pytest.fail(f"complete MiniMax H3 transformer_ref checkpoint missing under {model_root}", pytrace=False)
    if len(list(component.glob("diffusion_pytorch_model-*.safetensors"))) != 14:
        pytest.fail(f"expected 14 transformer_ref safetensors shards under {component}", pytrace=False)

    run_config = load_run_config("examples/train/configs/overfit_minimax_h3_ref2va_lora.yaml")
    training_config = run_config.training
    training_config.distributed.num_gpus = 1
    training_config.distributed.tp_size = 1
    training_config.distributed.sp_size = 1
    training_config.distributed.hsdp_replicate_dim = 1
    training_config.distributed.hsdp_shard_dim = 1
    training_config.distributed.pin_cpu_memory = False
    training_config.model.enable_gradient_checkpointing_type = None

    maybe_init_distributed_environment_and_model_parallel(tp_size=1, sp_size=1)
    model = None
    try:
        # The production loader is strict for H3. Reaching the assertions below
        # proves every base shard loaded with no missing or unexpected keys.
        model = MiniMaxH3Ref2VALoraModel(
            init_from=str(model_root),
            training_config=training_config,
            trainable=True,
            enable_gradient_checkpointing_type=None,
            lora={
                "enable": True,
                "rank": 32,
                "alpha": 32,
                "target_modules": _TARGETS,
            },
            expected_lora_layers=312,
        )
        transformer = model.transformer
        wrappers = [module for module in transformer.modules() if isinstance(module, BaseLayerWithLoRA)]
        trainable = [(name, parameter) for name, parameter in transformer.named_parameters()
                     if parameter.requires_grad]

        assert isinstance(transformer, FSDPModule)
        assert len(wrappers) == 312
        assert len(trainable) == 624
        assert all(name.endswith((".lora_A", ".lora_B")) for name, _ in trainable)
        assert sum(parameter.numel() for _, parameter in trainable) == 172_949_504
        assert not any(hasattr(module, "_checkpoint_wrapped_module") for module in transformer.modules())
        assert all(wrapper.cpu_weight is None for wrapper in wrappers)
        assert all(not parameter.is_meta for parameter in transformer.parameters())
        assert all(parameter.device.type == "cuda" for _, parameter in trainable)

        for wrapper in wrappers:
            assert isinstance(wrapper.base_layer.weight, DTensor)
            for adapter in (wrapper.lora_A, wrapper.lora_B):
                assert isinstance(adapter, DTensor)
                assert adapter.device_mesh == wrapper.base_layer.weight.device_mesh
                assert adapter.placements == wrapper.base_layer.weight.placements
                assert any(isinstance(placement, Shard) for placement in adapter.placements)

        print({
            "component": str(component),
            "wrappers": len(wrappers),
            "trainable_adapter_parameters": len(trainable),
            "trainable_adapter_elements": sum(parameter.numel() for _, parameter in trainable),
            "cuda_peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
            "cuda_peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
        })
    finally:
        del model
        cleanup_dist_env_and_memory()
