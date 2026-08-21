# SPDX-License-Identifier: Apache-2.0
"""Forward parity coverage for LingBot-Video sequence parallelism."""

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

from fastvideo.configs.models.dits.lingbot_video import LingBotVideoArchConfig, LingBotVideoConfig
from fastvideo.distributed import (
    cleanup_dist_env_and_memory,
    maybe_init_distributed_environment_and_model_parallel,
)
from fastvideo.models.dits.lingbot_video import LingBotVideoTransformer3DModel

SP_WORLD_SIZE = 2
SEED = 20260711


def _free_port() -> int:
    """Reserve a currently unused localhost port for one torchrun invocation."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _seed_everything(seed: int) -> None:
    """Make model initialization and raw SDPA deterministic across runs."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.enable_cudnn_sdp(False)
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)


def _build_tiny_config() -> LingBotVideoConfig:
    """Build a two-block Dense model whose four heads divide over two ranks."""
    arch_config = LingBotVideoArchConfig(
        patch_size=(1, 1, 1),
        in_channels=4,
        out_channels=4,
        hidden_size=32,
        num_attention_heads=4,
        depth=2,
        intermediate_size=64,
        text_dim=12,
        freq_dim=16,
        axes_dims=(2, 2, 4),
        axes_lens=(16, 8, 8),
    )
    return LingBotVideoConfig(arch_config=arch_config)


def _initialize_model_parameters(model: torch.nn.Module) -> None:
    """Initialize empty replicated-linear parameters identically in each run."""
    torch.manual_seed(SEED + 1)
    torch.cuda.manual_seed_all(SEED + 1)
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if parameter.ndim <= 1:
                if name.endswith("weight") and "norm" in name:
                    parameter.fill_(1.0)
                else:
                    parameter.zero_()
                continue
            torch.nn.init.xavier_uniform_(parameter)


def _build_inputs(config: LingBotVideoConfig, device: torch.device) -> dict[str, torch.Tensor]:
    """Create B=2 inputs with unequal masks and a seven-token joint sequence."""
    generator = torch.Generator(device="cpu").manual_seed(SEED + 2)
    hidden_states = torch.randn(
        2,
        config.in_channels,
        1,
        2,
        2,
        generator=generator,
        dtype=torch.float32,
    )
    encoder_hidden_states = torch.randn(
        2,
        3,
        config.text_dim,
        generator=generator,
        dtype=torch.float32,
    )
    encoder_attention_mask = torch.tensor([[1, 1, 1], [1, 0, 0]], dtype=torch.long)
    joint_length = 1 * 2 * 2 + encoder_hidden_states.shape[1]
    assert joint_length == 7 and joint_length % SP_WORLD_SIZE != 0
    assert encoder_attention_mask.sum(dim=1).tolist() == [3, 1]
    return {
        "hidden_states": hidden_states.to(device),
        "encoder_hidden_states": encoder_hidden_states.to(device),
        "encoder_attention_mask": encoder_attention_mask.to(device),
        "timestep": torch.tensor([10.0, 20.0], device=device),
    }


def _run_worker(mode: str, output_path: Path) -> None:
    """Run one deterministic single-rank or sequence-parallel model forward."""
    if mode not in {"single", "sp"}:
        raise ValueError(f"Unsupported mode: {mode}")

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank = int(os.environ.get("RANK", "0"))
    sp_size = 1 if mode == "single" else SP_WORLD_SIZE
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    _seed_everything(SEED)

    try:
        maybe_init_distributed_environment_and_model_parallel(1, sp_size)
        config = _build_tiny_config()
        model = LingBotVideoTransformer3DModel(config=config, hf_config={}).to(device=device, dtype=torch.float32)
        _initialize_model_parameters(model)
        model.eval()

        with torch.inference_mode():
            output = model(**_build_inputs(config, device)).sample
        assert torch.isfinite(output).all()
        if rank == 0:
            torch.save({"output": output.detach().cpu()}, output_path)
        dist.barrier()
    finally:
        cleanup_dist_env_and_memory()


def _run_torchrun(script_path: Path, mode: str, nproc_per_node: int, output_path: Path) -> None:
    """Launch one isolated worker group and surface its complete failure output."""
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--nnodes",
        "1",
        "--nproc_per_node",
        str(nproc_per_node),
        "--master_port",
        str(_free_port()),
        str(script_path),
        "--sp-worker",
        "--mode",
        mode,
        "--output",
        str(output_path),
    ]
    environment = os.environ.copy()
    environment["FASTVIDEO_ATTENTION_BACKEND"] = "TORCH_SDPA"
    process = subprocess.run(command, capture_output=True, text=True, env=environment)
    if process.returncode != 0:
        raise RuntimeError(f"{mode} worker failed with code {process.returncode}\n"
                           f"STDOUT:\n{process.stdout}\n"
                           f"STDERR:\n{process.stderr}")


def test_sp_forward_matches_single_rank(tmp_path: Path) -> None:
    """Compare full model outputs from one rank and a padded two-rank shard."""
    if not torch.cuda.is_available():
        pytest.skip("This test requires CUDA.")
    if torch.cuda.device_count() < SP_WORLD_SIZE:
        pytest.skip(f"This test requires at least {SP_WORLD_SIZE} CUDA devices.")

    script_path = Path(__file__).resolve()
    single_path = tmp_path / "single_rank_output.pt"
    sp_path = tmp_path / f"sp{SP_WORLD_SIZE}_output.pt"
    _run_torchrun(script_path, "single", 1, single_path)
    _run_torchrun(script_path, "sp", SP_WORLD_SIZE, sp_path)

    single_output = torch.load(single_path, map_location="cpu", weights_only=True)["output"]
    sp_output = torch.load(sp_path, map_location="cpu", weights_only=True)["output"]
    assert single_output.shape == (2, 4, 1, 2, 2)
    assert_close(sp_output, single_output, atol=1e-5, rtol=1e-5)


def _parse_args() -> argparse.Namespace:
    """Parse the private worker interface used by the parent pytest process."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--sp-worker", action="store_true")
    parser.add_argument("--mode", choices=["single", "sp"], default=None)
    parser.add_argument("--output", type=str, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    if not args.sp_worker:
        raise SystemExit("This module is intended to be run by pytest.")
    if args.mode is None or args.output is None:
        raise SystemExit("--mode and --output are required in worker mode.")
    _run_worker(mode=args.mode, output_path=Path(args.output))
