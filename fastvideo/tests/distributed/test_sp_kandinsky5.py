# SPDX-License-Identifier: Apache-2.0
"""Kandinsky5 SP parity: requires two GPUs, with optional four-GPU and FlashAttention cases.

Synthetic visual conditioning checks the DiT input layout, not a complete I2V pipeline.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
from pathlib import Path
import socket
import subprocess
import sys

import pytest
import torch
import torch.distributed as dist
from torch.testing import assert_close

from fastvideo.configs.models.dits.kandinsky5 import Kandinsky5ArchConfig, Kandinsky5VideoConfig
from fastvideo.distributed import (
    cleanup_dist_env_and_memory,
    maybe_init_distributed_environment_and_model_parallel,
)
from fastvideo.models.dits.kandinsky5 import Kandinsky5Transformer3DModel
from fastvideo.forward_context import set_forward_context
from fastvideo.models.loader.fsdp_load import _compile_model_regions, _enable_regional_attention_compile
from fastvideo.platforms import AttentionBackendEnum
from fastvideo.utils import set_mixed_precision_policy

SEED = 20260711


def _free_port() -> int:
    """Reuse the runner's lease port; fall back to an ephemeral local port."""
    if "MASTER_PORT" in os.environ:
        return int(os.environ["MASTER_PORT"])
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


def _build_tiny_config(visual_cond=False) -> Kandinsky5VideoConfig:
    return Kandinsky5VideoConfig(arch_config=Kandinsky5ArchConfig(
        patch_size=(1, 1, 1), in_visual_dim=4, out_visual_dim=4,
        model_dim=128, ff_dim=64, time_dim=16, in_text_dim=12, in_text_dim2=8,
        axes_dims=(8, 8, 16), num_text_blocks=1, num_visual_blocks=2, visual_cond=visual_cond))


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


def _build_inputs(config, device, length=7):
    generator = torch.Generator(device="cpu").manual_seed(SEED + 2)
    grid = (2, 2, 2) if length == 8 else (1, 1, length)
    return {
        "hidden_states": torch.randn(2, *grid, 9 if config.arch_config.visual_cond else 4, generator=generator).to(device),
        "encoder_hidden_states": torch.randn(2, 3, 12, generator=generator).to(device),
        "pooled_projections": torch.randn(2, 8, generator=generator).to(device),
        "timestep": torch.tensor([10.0, 20.0], device=device),
        "visual_rope_pos": [torch.arange(size, device=device) for size in grid],
        "text_rope_pos": torch.arange(3, device=device),
    }


def _run_worker(mode: str, output_path: Path, sp_size=2, dtype="float32", visual_cond=False) -> None:
    """Run one deterministic single-rank or sequence-parallel model forward."""
    if mode not in {"single", "sp", "compiled"}:
        raise ValueError(f"Unsupported mode: {mode}")

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank = int(os.environ.get("RANK", "0"))
    sp_size = 1 if mode == "single" else sp_size
    compute_dtype = getattr(torch, dtype)
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    _seed_everything(SEED)

    try:
        maybe_init_distributed_environment_and_model_parallel(1, sp_size)
        set_mixed_precision_policy(param_dtype=compute_dtype, reduce_dtype=torch.float32, output_dtype=None)
        config = _build_tiny_config(visual_cond)
        model = Kandinsky5Transformer3DModel(config=config, hf_config={}).to(device=device, dtype=compute_dtype)
        _initialize_model_parameters(model)
        model.eval()

        if mode == "compiled":
            _enable_regional_attention_compile(model)
            _compile_model_regions(model, {})
        expected_backend = AttentionBackendEnum[os.environ["FASTVIDEO_ATTENTION_BACKEND"]]
        for block in model.visual_transformer_blocks:
            attention = block.self_attention
            assert attention.local_attention.backend == expected_backend
            if sp_size > 1:
                assert attention.distributed_attention is not None
                assert attention.distributed_attention.backend == expected_backend
        outputs = {"state_keys": tuple(model.state_dict())}
        if mode == "sp":
            bad = _build_tiny_config()
            bad.arch_config.attention_type = "nabla"
            with pytest.raises(ValueError, match="dense checkpoints"):
                Kandinsky5Transformer3DModel(config=bad, hf_config={})
            bad = _build_tiny_config()
            bad.arch_config.model_dim = 96
            with pytest.raises(ValueError, match="divisible"):
                Kandinsky5Transformer3DModel(config=bad, hf_config={})
            with torch.enable_grad(), set_forward_context(0, None), torch.autocast(
                    "cuda", dtype=compute_dtype, enabled=compute_dtype != torch.float32):
                with pytest.raises(RuntimeError, match="inference only"):
                    model(**_build_inputs(config, device))
        with torch.inference_mode(), set_forward_context(0, None), torch.autocast(
                "cuda", dtype=compute_dtype, enabled=compute_dtype != torch.float32):
            for length in (1, 7, 8):
                inputs = _build_inputs(config, device, length)
                inputs = {k: v.to(compute_dtype) if isinstance(v, torch.Tensor) and v.is_floating_point() else v
                          for k, v in inputs.items()}
                output = model(**inputs)
                assert torch.isfinite(output).all()
                outputs[length] = output.detach().cpu()
        if rank == 0:
            torch.save(outputs, output_path)
        dist.barrier()
    finally:
        cleanup_dist_env_and_memory()


def _run_torchrun(script_path: Path, mode: str, nproc_per_node: int, output_path: Path,
                  dtype="float32", visual_cond=False, backend="TORCH_SDPA") -> None:
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
        "--sp-size", str(nproc_per_node), "--dtype", dtype,
    ]
    if visual_cond:
        command.append("--visual-cond")
    environment = os.environ.copy()
    environment["FASTVIDEO_ATTENTION_BACKEND"] = backend
    process = subprocess.run(command, capture_output=True, text=True, env=environment, timeout=300)
    if process.returncode != 0:
        raise RuntimeError(f"{mode} worker failed with code {process.returncode}\n"
                           f"STDOUT:\n{process.stdout}\n"
                           f"STDERR:\n{process.stderr}")


@pytest.mark.parametrize("sp_size,dtype,visual_cond,backend", [
    (2, "float32", False, "TORCH_SDPA"),
    (2, "bfloat16", True, "TORCH_SDPA"),
    (4, "float32", True, "TORCH_SDPA"),
    (2, "bfloat16", True, "FLASH_ATTN"),
])
def test_sp_forward_matches_single_rank(tmp_path: Path, sp_size, dtype, visual_cond, backend) -> None:
    """Compare full model outputs from one rank and a padded two-rank shard."""
    if not torch.cuda.is_available():
        pytest.skip("This test requires CUDA.")
    if torch.cuda.device_count() < sp_size:
        pytest.skip(f"This test requires at least {sp_size} CUDA devices.")

    if backend == "FLASH_ATTN" and not any(importlib.util.find_spec(name) is not None
                                           for name in ("flash_attn", "flash_attn_interface")):
        pytest.skip("FlashAttention is not installed; SDPA fallback is not backend coverage.")
    script_path = Path(__file__).resolve()
    single_path = tmp_path / "single_rank_output.pt"
    _run_torchrun(script_path, "single", 1, single_path, dtype, visual_cond, backend)
    expected = torch.load(single_path, weights_only=True)
    for mode in ("sp", "compiled"):
        path = tmp_path / f"{mode}.pt"
        _run_torchrun(script_path, mode, sp_size, path, dtype, visual_cond, backend)
        actual = torch.load(path, weights_only=True)
        assert actual["state_keys"] == expected["state_keys"]
        for length in (1, 7, 8):
            grid = (2, 2, 2) if length == 8 else (1, 1, length)
            assert actual[length].shape == (2, *grid, 4)
            tolerance = 2e-5 if dtype == "float32" else 4e-2
            assert_close(actual[length], expected[length], atol=tolerance, rtol=tolerance)


def _parse_args() -> argparse.Namespace:
    """Parse the private worker interface used by the parent pytest process."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--sp-worker", action="store_true")
    parser.add_argument("--mode", choices=["single", "sp", "compiled"], default=None)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--sp-size", type=int, default=2)
    parser.add_argument("--dtype", choices=["float32", "bfloat16"], default="float32")
    parser.add_argument("--visual-cond", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    if not args.sp_worker:
        raise SystemExit("This module is intended to be run by pytest.")
    if args.mode is None or args.output is None:
        raise SystemExit("--mode and --output are required in worker mode.")
    _run_worker(mode=args.mode, output_path=Path(args.output), sp_size=args.sp_size,
                dtype=args.dtype, visual_cond=args.visual_cond)
