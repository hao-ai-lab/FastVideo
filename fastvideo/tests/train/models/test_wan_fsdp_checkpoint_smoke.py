# SPDX-License-Identifier: Apache-2.0
"""GPU training smoke for Wan's pre-FSDP activation checkpointing.

``WanModel`` wraps each transformer block in a ``CheckpointWrapper`` before
the loader shards it, so training runs ``fully_shard(CheckpointWrapper(block))``
with ``reshard_after_forward=True``. The backward pass recomputes each block
inside the unshard window that FSDP opens for that block's gradients.

The test writes a tiny random-init Wan checkpoint in the Diffusers layout,
loads it through ``WanModel`` and the real FSDP loader on one or two GPUs, and
takes a few optimizer steps, both eager and with regional compile. Next to it,
the same checkpoint is loaded without activation checkpointing. The test checks
the wrapper composition, that every checkpointed block recomputes once per
backward, and that losses, gradients and updated weights match the reference.
"""

from __future__ import annotations

import json
import math
import socket
from contextlib import ExitStack
from pathlib import Path

import pytest
import torch
import torch.multiprocessing as mp

import fastvideo.envs as envs

_TINY_WAN = {
    "patch_size": (1, 2, 2),
    "num_attention_heads": 4,
    "attention_head_dim": 16,
    "in_channels": 16,
    "out_channels": 16,
    "text_dim": 64,
    "freq_dim": 64,
    "ffn_dim": 128,
    "num_layers": 2,
    "cross_attn_norm": True,
    "qk_norm": "rms_norm_across_heads",
    "eps": 1e-6,
    "rope_max_seq_len": 64,
}
_SEED = 1718
_STEPS = 3
_LEARNING_RATE = 1e-2
# Checkpointing must not change the math. Only nondeterministic attention
# backward kernels separate the two runs.
_MAX_RELATIVE_ERROR = 2e-2


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _canonical_name(name: str) -> str:
    return name.replace("._checkpoint_wrapped_module.", ".")


def _write_tiny_wan_checkpoint(root: Path) -> Path:
    """Write a seeded random-init Wan T2V model in the Diffusers layout."""
    import diffusers
    from diffusers import WanTransformer3DModel as DiffusersWanTransformer3DModel

    model_dir = root / "tiny-wan-t2v-diffusers"
    torch.manual_seed(_SEED)
    DiffusersWanTransformer3DModel(**_TINY_WAN).save_pretrained(model_dir / "transformer", safe_serialization=True)
    (model_dir / "model_index.json").write_text(
        json.dumps({
            "_class_name": "WanPipeline",
            "_diffusers_version": diffusers.__version__,
            "transformer": ["diffusers", "WanTransformer3DModel"],
        }))
    return model_dir


def _training_config(world_size: int, *, checkpointing: str | None, compile_blocks: bool):
    from fastvideo.models.wan.pipeline_config import WanT2V480PConfig
    from fastvideo.train.utils.training_config import (
        DistributedConfig,
        ModelTrainingConfig,
        TrainingConfig,
    )

    return TrainingConfig(
        distributed=DistributedConfig(num_gpus=world_size, hsdp_replicate_dim=1, hsdp_shard_dim=world_size),
        model=ModelTrainingConfig(
            enable_gradient_checkpointing_type=checkpointing,
            enable_torch_compile=compile_blocks,
        ),
        pipeline_config=WanT2V480PConfig(),
        dit_precision="fp32",
    )


def _training_inputs(rank: int, step: int, device: torch.device):
    from fastvideo.pipelines import TrainingBatch

    generator = torch.Generator(device="cpu").manual_seed(_SEED + 1000 * rank + step)
    # (batch, frames, channels, height, width): 2 x 4 x 4 = 32 patch tokens.
    latent_shape = (1, 2, _TINY_WAN["in_channels"], 8, 8)
    noisy_latents = torch.randn(latent_shape, generator=generator).to(device)
    target = torch.randn(latent_shape, generator=generator).to(device)
    text_embeddings = torch.randn(1, 8, _TINY_WAN["text_dim"], generator=generator)

    batch = TrainingBatch()
    batch.timesteps = torch.full((1, ), 100.0 + 300.0 * step, device=device)
    batch.attn_metadata = None
    batch.conditional_dict = {
        "encoder_hidden_states": text_embeddings.to(device=device, dtype=torch.bfloat16),
        "encoder_attention_mask": torch.ones(1, 8, device=device),
    }
    return batch, noisy_latents, target


def _local(tensor: torch.Tensor) -> torch.Tensor:
    from torch.distributed.tensor import DTensor

    return tensor.to_local() if isinstance(tensor, DTensor) else tensor


def _global_sum(value: torch.Tensor) -> float:
    value = value.detach().float().clone()
    torch.distributed.all_reduce(value)
    return float(value.item())


def _relative_error(actual: dict[str, torch.Tensor], expected: dict[str, torch.Tensor]) -> float:
    device = next(iter(expected.values())).device
    error = torch.zeros((), device=device)
    norm = torch.zeros((), device=device)
    for name, expected_value in expected.items():
        error += (actual[name].float() - expected_value.float()).pow(2).sum()
        norm += expected_value.float().pow(2).sum()
    return math.sqrt(_global_sum(error) / _global_sum(norm))


def _run_smoke(rank: int, world_size: int, model_dir: str, compile_blocks: bool) -> None:
    from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import CheckpointWrapper
    from torch.distributed.fsdp import FSDPModule
    from torch.distributed.tensor import DTensor

    from fastvideo.train.models.wan import WanModel

    def load(checkpointing: str | None) -> WanModel:
        return WanModel(
            init_from=model_dir,
            training_config=_training_config(world_size, checkpointing=checkpointing, compile_blocks=compile_blocks),
            trainable=True,
            attention_backend="TORCH_SDPA",
        )

    student = load("full")
    reference = load(None)
    assert student._activation_checkpointing_applied
    assert not reference._activation_checkpointing_applied

    # fully_shard swaps the class of the module it wraps, so the FSDP unit is
    # the CheckpointWrapper itself and the block inside it is not an FSDP unit.
    for block in student.transformer.blocks:
        assert isinstance(block, CheckpointWrapper) and isinstance(block, FSDPModule)
        assert not isinstance(block._checkpoint_wrapped_module, FSDPModule)
    for block in reference.transformer.blocks:
        assert isinstance(block, FSDPModule) and not isinstance(block, CheckpointWrapper)

    student_params = {_canonical_name(name): param for name, param in student.transformer.named_parameters()}
    reference_params = dict(reference.transformer.named_parameters())
    assert student_params.keys() == reference_params.keys()
    for name, param in student_params.items():
        assert isinstance(param, DTensor) and param.requires_grad, name
    # Parameters are really sharded: with two ranks, each owns half of the rows.
    to_q = student_params["blocks.0.to_q.weight"]
    assert _local(to_q).shape[0] * world_size == to_q.shape[0]

    if compile_blocks:
        # Regional compile replaces the forward of the block inside each
        # wrapper; the checkpoint wrapper and FSDP hooks stay eager.
        for block in student.transformer.blocks:
            assert "forward" in vars(block._checkpoint_wrapped_module)
        for block in reference.transformer.blocks:
            assert "forward" in vars(block)

    block_calls = {"student": 0, "reference": 0}

    def count(role: str):

        def hook(module: torch.nn.Module, args: tuple) -> None:
            block_calls[role] += 1

        return hook

    student.transformer.blocks[0]._checkpoint_wrapped_module.register_forward_pre_hook(count("student"))
    reference.transformer.blocks[0].register_forward_pre_hook(count("reference"))

    initial_block0 = {
        name: _local(param).detach().clone()
        for name, param in student_params.items() if name.startswith("blocks.0.")
    }
    optimizers = {
        "student": torch.optim.SGD(student_params.values(), lr=_LEARNING_RATE),
        "reference": torch.optim.SGD(reference_params.values(), lr=_LEARNING_RATE),
    }
    device = student.device
    for step in range(_STEPS):
        batch, noisy_latents, target = _training_inputs(rank, step, device)
        losses = {}
        for role, model in (("student", student), ("reference", reference)):
            prediction = model.predict_noise(noisy_latents, batch.timesteps, batch, conditional=True)
            assert prediction.shape == noisy_latents.shape
            loss = torch.nn.functional.mse_loss(prediction.float(), target)
            model.backward(loss, (batch.timesteps, batch.attn_metadata), grad_accum_rounds=1)
            losses[role] = loss.detach()

        assert math.isfinite(_global_sum(losses["student"])), f"step {step}: non-finite loss"
        torch.testing.assert_close(losses["student"], losses["reference"], rtol=_MAX_RELATIVE_ERROR, atol=1e-6)
        # One forward plus one recompute per checkpointed block.
        assert block_calls == {"student": 2 * (step + 1), "reference": step + 1}, block_calls

        student_grads = {name: _local(param.grad) for name, param in student_params.items()}
        reference_grads = {name: _local(param.grad) for name, param in reference_params.items()}
        for name, grad in student_grads.items():
            assert grad is not None and torch.isfinite(grad).all(), f"step {step}: bad gradient for {name}"
            assert reference_grads[name] is not None, f"step {step}: no reference gradient for {name}"
        block0_norm = sum(grad.float().pow(2).sum() for name, grad in student_grads.items()
                          if name.startswith("blocks.0."))
        assert _global_sum(block0_norm) > 0, f"step {step}: backward did not reach blocks.0"
        grad_error = _relative_error(student_grads, reference_grads)
        assert grad_error < _MAX_RELATIVE_ERROR, f"step {step}: gradient relative error {grad_error}"

        for optimizer in optimizers.values():
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

    student_weights = {name: _local(param).detach() for name, param in student_params.items()}
    reference_weights = {name: _local(param).detach() for name, param in reference_params.items()}
    moved = sum((student_weights[name].float() - value.float()).pow(2).sum() for name, value in initial_block0.items())
    assert _global_sum(moved) > 0, "optimizer steps did not update blocks.0"
    weight_error = _relative_error(student_weights, reference_weights)
    assert weight_error < _MAX_RELATIVE_ERROR, f"trained weights diverged: relative error {weight_error}"


def _smoke_worker(rank: int, world_size: int, port: int, model_dir: str, compile_blocks: bool) -> None:
    from fastvideo.distributed import (
        cleanup_dist_env_and_memory,
        maybe_init_distributed_environment_and_model_parallel,
    )

    with ExitStack() as stack:
        stack.enter_context(envs.override_external("RANK", str(rank)))
        stack.enter_context(envs.override_external("LOCAL_RANK", str(rank)))
        stack.enter_context(envs.override_external("WORLD_SIZE", str(world_size)))
        maybe_init_distributed_environment_and_model_parallel(
            1,
            1,
            distributed_init_method=f"tcp://127.0.0.1:{port}",
        )
        stack.callback(cleanup_dist_env_and_memory)
        _run_smoke(rank, world_size, model_dir, compile_blocks)


@pytest.mark.parametrize("compile_blocks", [False, True], ids=["eager", "regional_compile"])
@pytest.mark.parametrize("world_size", [1, 2], ids=["1gpu", "2gpu"])
def test_wan_fsdp_checkpoint_wrapper_trains(tmp_path: Path, world_size: int, compile_blocks: bool) -> None:
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    if torch.cuda.device_count() < world_size:
        pytest.skip(f"requires {world_size} GPUs")

    model_dir = _write_tiny_wan_checkpoint(tmp_path)
    mp.start_processes(
        _smoke_worker,
        args=(world_size, _free_port(), str(model_dir), compile_blocks),
        nprocs=world_size,
        join=True,
        start_method="spawn",
    )
