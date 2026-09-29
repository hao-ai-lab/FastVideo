# SPDX-License-Identifier: Apache-2.0
"""Deterministic forward parity for Wan-VACE sequence parallelism."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import socket
import subprocess
import sys

import pytest
import torch
import torch.distributed as dist
from torch.testing import assert_close

from fastvideo import envs
from fastvideo.attention.selector import _component_attention_backend_scope
from fastvideo.distributed import (
    cleanup_dist_env_and_memory,
    maybe_init_distributed_environment_and_model_parallel,
)
from fastvideo.forward_context import set_forward_context
from fastvideo.models.wan.vace_config import WanVACEArchConfig, WanVACEVideoConfig
from fastvideo.models.wan.vace_transformer import WanVACETransformer3DModel
from fastvideo.platforms import AttentionBackendEnum

SP_WORLD_SIZE = 2
SEED = 20260925


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _seed_everything() -> None:
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.enable_cudnn_sdp(False)
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)


def _build_config() -> WanVACEVideoConfig:
    arch = WanVACEArchConfig(
        patch_size=(1, 2, 2),
        num_attention_heads=4,
        attention_head_dim=8,
        in_channels=4,
        out_channels=4,
        vace_in_channels=4,
        text_dim=16,
        freq_dim=16,
        ffn_dim=64,
        num_layers=2,
        vace_layers=[0],
    )
    config = WanVACEVideoConfig(arch_config=arch)
    config._resolved_attention_backend = AttentionBackendEnum.TORCH_SDPA
    return config


def _initialize_parameters(model: torch.nn.Module) -> None:
    torch.manual_seed(SEED + 1)
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if parameter.ndim <= 1:
                if name.endswith("weight") and "norm" in name:
                    parameter.fill_(1.0)
                else:
                    parameter.zero_()
            else:
                torch.nn.init.xavier_uniform_(parameter)


def _build_inputs(device: torch.device) -> dict[str, torch.Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(SEED + 2)
    hidden = torch.randn(1, 4, 5, 6, 6, generator=generator)
    control = torch.randn(1, 4, 4, 6, 6, generator=generator)
    text = torch.randn(1, 3, 16, generator=generator)
    assert 5 * 3 * 3 == 45  # Odd sequence requires SP pad/gather/unpad.
    assert 4 * 3 * 3 == 36  # Control is shorter and requires VACE padding.
    return {
        "hidden_states": hidden.to(device),
        "control_hidden_states": control.to(device),
        "encoder_hidden_states": text.to(device),
        "timestep": torch.tensor([500], device=device),
    }


def _worker(mode: str, output_path: Path) -> None:
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank = int(os.environ.get("RANK", "0"))
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    _seed_everything()
    try:
        maybe_init_distributed_environment_and_model_parallel(1, 1 if mode == "single" else SP_WORLD_SIZE)
        config = _build_config()
        with _component_attention_backend_scope(AttentionBackendEnum.TORCH_SDPA, component="transformer"):
            model = WanVACETransformer3DModel(config=config, hf_config={}).to(device=device, dtype=torch.float32)
        _initialize_parameters(model)
        model.eval()
        with torch.inference_mode(), set_forward_context(current_timestep=0, attn_metadata=None):
            output = model(**_build_inputs(device))
        assert output.shape == (1, 4, 5, 6, 6)
        assert torch.isfinite(output).all()
        torch.save({"output": output.detach().cpu()}, output_path.with_name(f"{output_path.stem}_rank{rank}.pt"))
        dist.barrier()
    finally:
        cleanup_dist_env_and_memory()


def _torchrun(mode: str, processes: int, output_path: Path) -> None:
    command = [
        sys.executable, "-m", "torch.distributed.run", "--nnodes", "1", "--nproc_per_node", str(processes),
        "--master_port", str(_free_port()), str(Path(__file__).resolve()), "--sp-worker", "--mode", mode,
        "--output", str(output_path),
    ]
    with envs.FASTVIDEO_ATTENTION_BACKEND.override("TORCH_SDPA"):
        process = subprocess.run(command, capture_output=True, text=True)
    if process.returncode:
        raise RuntimeError(f"{mode} worker exited {process.returncode}\n{process.stdout}\n{process.stderr}")


def test_sp_forward_matches_single_rank(tmp_path: Path) -> None:
    if not torch.cuda.is_available() or torch.cuda.device_count() < SP_WORLD_SIZE:
        pytest.skip("Wan-VACE SP parity requires two CUDA devices.")
    _torchrun("single", 1, tmp_path / "single.pt")
    _torchrun("sp", SP_WORLD_SIZE, tmp_path / "sp.pt")
    single = torch.load(tmp_path / "single_rank0.pt", map_location="cpu", weights_only=True)["output"]
    for rank in range(SP_WORLD_SIZE):
        parallel = torch.load(tmp_path / f"sp_rank{rank}.pt", map_location="cpu", weights_only=True)["output"]
        assert_close(parallel, single, atol=1e-5, rtol=1e-5)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sp-worker", action="store_true")
    parser.add_argument("--mode", choices=["single", "sp"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.sp_worker:
        raise SystemExit("This module is intended to be run by pytest.")
    _worker(args.mode, args.output)
