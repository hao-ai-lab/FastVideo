# SPDX-License-Identifier: Apache-2.0
"""Token-sharded sequence parallelism for the causal Wan training forward.

One torchrun worker runs a tiny CausalWanTransformer3DModel on a single rank.
A second worker runs it with sp_size=2, once on the default path, which
replicates the sequence and splits attention heads, and once with
FASTVIDEO_CAUSAL_WAN_TOKEN_SP, which shards the tokens. Outputs and gradients
averaged over the SP group must match the single rank in every case.

Each latent frame has 16 tokens and a block has 3 frames. Teacher forcing with
6 frames gives each rank 6 whole frames, once without and once with full
activation checkpointing, so backward also recomputes the all-to-alls. Diffusion
forcing with 9 frames puts the shard boundary inside frame 4, across 3 causal
blocks. The fallback case has 45 tokens, which do not divide by 2.

Requires two CUDA devices.
"""

from __future__ import annotations

import argparse
import os
import socket
import subprocess
from pathlib import Path
from typing import NamedTuple

import pytest
import torch
import torch.distributed as dist

import fastvideo.envs as envs
from fastvideo.distributed import (
    cleanup_dist_env_and_memory,
    maybe_init_distributed_environment_and_model_parallel,
)
from fastvideo.forward_context import set_forward_context
from fastvideo.models.wan.causal_transformer import CausalWanTransformer3DModel
from fastvideo.models.wan.config import WanVideoArchConfig, WanVideoConfig
from fastvideo.pipelines.pipeline_batch_info import ForwardBatch
from fastvideo.train.utils.activation_checkpoint import apply_activation_checkpointing

SP_WORLD_SIZE = 2
SEED = 2026


class Case(NamedTuple):
    teacher_forcing: bool
    frames: int
    height: int
    width: int
    checkpointing: bool = False


CASES = {
    "teacher_forcing": Case(True, 6, 8, 8),
    "teacher_forcing_checkpointed": Case(True, 6, 8, 8, checkpointing=True),
    "diffusion_forcing": Case(False, 9, 8, 8),
    "fallback": Case(False, 3, 6, 10),
}


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _build_model(device: torch.device) -> CausalWanTransformer3DModel:
    config = WanVideoConfig(arch_config=WanVideoArchConfig(
        num_attention_heads=4,
        # Compiled flex attention needs a head dim of at least 16.
        attention_head_dim=16,
        in_channels=4,
        out_channels=4,
        text_dim=16,
        freq_dim=16,
        ffn_dim=64,
        num_layers=2,
        patch_size=(1, 2, 2),
        num_frames_per_block=3,
    ))
    model = CausalWanTransformer3DModel(config=config, hf_config={})
    model = model.to(device=device, dtype=torch.float32)
    # ReplicatedLinear allocates with torch.empty, so give every parameter a value.
    torch.manual_seed(SEED + 3)
    with torch.no_grad():
        for name, param in model.named_parameters():
            if param.ndim >= 2:
                torch.nn.init.xavier_uniform_(param)
            elif name.endswith("bias"):
                param.normal_(0.0, 0.02)
            else:
                param.fill_(1.0)
    # FastVideo linear layers create their weights with requires_grad=False; full fine-tuning
    # turns them on, so do the same to check the gradients of every weight.
    return model.train().requires_grad_(True)


def _build_inputs(case: str, device: torch.device) -> dict[str, torch.Tensor | None]:
    teacher_forcing, frames, height, width, _ = CASES[case]
    generator = torch.Generator(device="cpu").manual_seed(SEED + 1)
    shape = (1, 4, frames, height, width)
    hidden_states = torch.randn(shape, generator=generator)
    clean_x = torch.randn(shape, generator=generator) if teacher_forcing else None
    encoder_hidden_states = torch.randn(1, 8, 16, generator=generator)
    # A different timestep per frame, so a misaligned modulation row would show.
    timestep = torch.randint(0, 1000, (1, frames), generator=generator)
    inputs = {
        "hidden_states": hidden_states,
        "encoder_hidden_states": encoder_hidden_states,
        "timestep": timestep,
        "clean_x": clean_x,
    }
    return {k: None if v is None else v.to(device) for k, v in inputs.items()}


def _run_case(case: str, device: torch.device, sp_size: int) -> dict[str, object]:
    model = _build_model(device)
    if CASES[case].checkpointing:
        model = apply_activation_checkpointing(model, checkpointing_type="full")
    inputs = _build_inputs(case, device)
    shard_calls = []

    def record_shard(*args):
        shard_calls.append(args[0].shape)
        return CausalWanTransformer3DModel._shard_for_token_sp(*args)

    model._shard_for_token_sp = record_shard
    with set_forward_context(current_timestep=0, attn_metadata=None, forward_batch=ForwardBatch(data_type="dummy")):
        output = model(**inputs)
        loss = output.float().square().mean()
        loss.backward()
    grads = {}
    missing = []
    for name, param in model.named_parameters():
        if param.grad is None:
            missing.append(name)
        grad = torch.zeros_like(param) if param.grad is None else param.grad
        if sp_size > 1:
            # FSDP averages gradients over the ranks of the SP group.
            dist.all_reduce(grad, op=dist.ReduceOp.AVG)
        grads[name] = grad.detach().cpu()
    return {
        "output": output.detach().cpu(),
        "loss": loss.item(),
        "grads": grads,
        "token_sharded": bool(shard_calls),
        "params_without_grad": missing,
    }


def _run_worker(mode: str, output_path: Path) -> None:
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    sp_size = 1 if mode == "single" else SP_WORLD_SIZE
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    try:
        maybe_init_distributed_environment_and_model_parallel(1, sp_size)
        results = {}
        for case in CASES:
            if mode == "single":
                results[(case, "single")] = _run_case(case, device, sp_size)
                continue
            for token_sp in (False, True):
                with envs.FASTVIDEO_CAUSAL_WAN_TOKEN_SP.override(token_sp):
                    results[(case, "token" if token_sp else "replicated")] = _run_case(case, device, sp_size)
        if int(os.environ.get("RANK", "0")) == 0:
            torch.save(results, output_path)
        if sp_size > 1:
            dist.barrier()
    finally:
        cleanup_dist_env_and_memory()


def _run_torchrun(mode: str, nproc_per_node: int, output_path: Path) -> None:
    cmd = [
        "torchrun", "--nnodes", "1", "--nproc_per_node",
        str(nproc_per_node), "--master_port",
        str(_free_port()),
        str(Path(__file__).resolve()), "--worker", "--mode", mode, "--output",
        str(output_path)
    ]
    with envs.FASTVIDEO_ATTENTION_BACKEND.override("TORCH_SDPA"):
        process = subprocess.run(cmd, capture_output=True, text=True)
    if process.returncode != 0:
        raise RuntimeError(f"{mode} worker failed with code {process.returncode}\n"
                           f"STDOUT:\n{process.stdout}\nSTDERR:\n{process.stderr}")


@pytest.fixture(scope="module")
def results(tmp_path_factory: pytest.TempPathFactory) -> dict:
    if not torch.cuda.is_available():
        pytest.skip("This test requires CUDA.")
    if torch.cuda.device_count() < SP_WORLD_SIZE:
        pytest.skip(f"This test requires at least {SP_WORLD_SIZE} CUDA devices.")
    out_dir = tmp_path_factory.mktemp("sp_causal_wan")
    _run_torchrun("single", 1, out_dir / "single.pt")
    _run_torchrun("sp", SP_WORLD_SIZE, out_dir / "sp.pt")
    merged = torch.load(out_dir / "single.pt")
    merged.update(torch.load(out_dir / "sp.pt"))
    return merged


@pytest.mark.parametrize("case", list(CASES))
@pytest.mark.parametrize("sp_mode", ["replicated", "token"])
def test_sp_causal_wan_matches_single_rank(results: dict, case: str, sp_mode: str) -> None:
    ref, got = results[(case, "single")], results[(case, sp_mode)]
    assert got["token_sharded"] == (sp_mode == "token" and case != "fallback")
    assert not ref["params_without_grad"] and not got["params_without_grad"], got["params_without_grad"]
    torch.testing.assert_close(got["output"], ref["output"], rtol=1e-4, atol=1e-5)
    assert got["loss"] == pytest.approx(ref["loss"], rel=1e-5)
    mismatched = []
    for name, ref_grad in ref["grads"].items():
        grad = got["grads"][name]
        # Normalize by the largest reference gradient: a few biases have zero
        # gradient in exact math, so a per-tensor relative error is meaningless.
        if (grad - ref_grad).abs().max() > 1e-4 * ref_grad.abs().max().clamp_min(1e-12) + 1e-7:
            mismatched.append(f"{name}: max abs diff {(grad - ref_grad).abs().max().item():.3e}")
    assert not mismatched, "\n".join(mismatched)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--mode", choices=["single", "sp"])
    parser.add_argument("--output", type=str)
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    if not args.worker:
        raise SystemExit("This module is intended to be run by pytest.")
    _run_worker(mode=args.mode, output_path=Path(args.output))
