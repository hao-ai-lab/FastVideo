# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import os
import socket
import subprocess
from pathlib import Path

import pytest
import torch
import torch.distributed as dist

from fastvideo.configs.models.dits.wanvideo import (
    WanVideoArchConfig,
    WanVideoConfig,
)
from fastvideo.distributed import (
    cleanup_dist_env_and_memory,
    maybe_init_distributed_environment_and_model_parallel,
)
from fastvideo.forward_context import set_forward_context
from fastvideo.models.dits.wanvideo import WanTransformer3DModel
from fastvideo.pipelines.pipeline_batch_info import ForwardBatch

SP_WORLD_SIZE = 2
SEED = 2026


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _seed_everything(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def _build_tiny_wan_config() -> WanVideoConfig:
    arch_config = WanVideoArchConfig(
        num_attention_heads=4,
        attention_head_dim=8,
        in_channels=4,
        out_channels=4,
        text_dim=16,
        freq_dim=16,
        ffn_dim=64,
        num_layers=2,
        patch_size=(1, 2, 2),
        rope_max_seq_len=64,
    )
    return WanVideoConfig(arch_config=arch_config)


def _build_hf_config(config: WanVideoConfig) -> dict[str, object]:
    return {
        "num_attention_heads": config.num_attention_heads,
        "attention_head_dim": config.attention_head_dim,
        "in_channels": config.in_channels,
        "out_channels": config.out_channels,
        "text_dim": config.text_dim,
        "freq_dim": config.freq_dim,
        "ffn_dim": config.ffn_dim,
        "num_layers": config.num_layers,
        "patch_size": config.patch_size,
        "text_len": config.text_len,
        "rope_max_seq_len": config.rope_max_seq_len,
    }


def _build_inputs(
    config: WanVideoConfig,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(SEED + 1)

    hidden_states = torch.randn(
        1,
        config.in_channels,
        4,
        8,
        8,
        generator=generator,
        dtype=torch.float32,
    ).to(device)
    encoder_hidden_states = torch.randn(
        1,
        8,
        config.text_dim,
        generator=generator,
        dtype=torch.float32,
    ).to(device)
    timestep = torch.tensor([10], device=device, dtype=torch.long)
    return hidden_states, encoder_hidden_states, timestep


def _collect_named_parameter_grads(
    model: torch.nn.Module,
) -> dict[str, torch.Tensor]:
    named_grads: dict[str, torch.Tensor] = {}
    for name, param in model.named_parameters():
        grad = param.grad
        if grad is None:
            grad = torch.zeros_like(param)
        named_grads[name] = grad.detach().float().cpu()
    return named_grads


def _initialize_model_parameters(model: torch.nn.Module) -> None:
    # ReplicatedLinear parameters are allocated with torch.empty and need an
    # explicit initialization in tests to avoid undefined values.
    torch.manual_seed(SEED + 3)
    torch.cuda.manual_seed_all(SEED + 3)
    with torch.no_grad():
        for name, param in model.named_parameters():
            if param.ndim <= 1:
                if name.endswith("weight") and "norm" in name:
                    param.fill_(1.0)
                else:
                    param.zero_()
                continue
            torch.nn.init.xavier_uniform_(param)


def _assert_finite(name: str, tensor: torch.Tensor) -> None:
    if torch.isfinite(tensor).all():
        return
    nan_count = int(torch.isnan(tensor).sum().item())
    inf_count = int(torch.isinf(tensor).sum().item())
    raise RuntimeError(
        f"{name} contains non-finite values (nan={nan_count}, inf={inf_count})"
    )


def _run_worker(mode: str, output_path: Path) -> None:
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

        config = _build_tiny_wan_config()
        hf_config = _build_hf_config(config)
        model = WanTransformer3DModel(config=config, hf_config=hf_config)
        model = model.to(device=device, dtype=torch.float32)
        _initialize_model_parameters(model)
        model.train()
        model.zero_grad(set_to_none=True)

        hidden_states, encoder_hidden_states, timestep = _build_inputs(
            config,
            device,
        )

        forward_batch = ForwardBatch(data_type="dummy")
        with set_forward_context(
            current_timestep=0,
            attn_metadata=None,
            forward_batch=forward_batch,
        ):
            output = model(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                timestep=timestep,
            )

        _assert_finite("output", output)
        loss = output.float().square().mean()
        _assert_finite("loss", loss)
        loss.backward()
        for name, param in model.named_parameters():
            if param.grad is not None:
                _assert_finite(f"grad[{name}]", param.grad)
        if mode == "sp":
            for param in model.parameters():
                if param.grad is not None:
                    dist.all_reduce(param.grad, op=dist.ReduceOp.AVG)

        grads = _collect_named_parameter_grads(model)
        if rank == 0:
            torch.save({"grads": grads}, output_path)

        dist.barrier()
    finally:
        cleanup_dist_env_and_memory()


def _run_torchrun(
    script_path: Path,
    mode: str,
    nproc_per_node: int,
    output_path: Path,
) -> None:
    cmd = [
        "torchrun",
        "--nnodes",
        "1",
        "--nproc_per_node",
        str(nproc_per_node),
        "--master_port",
        str(_free_port()),
        str(script_path),
        "--sp-grad-worker",
        "--mode",
        mode,
        "--output",
        str(output_path),
    ]
    env = os.environ.copy()
    env["FASTVIDEO_ATTENTION_BACKEND"] = "TORCH_SDPA"
    process = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if process.returncode != 0:
        raise RuntimeError(
            f"{mode} worker failed with code {process.returncode}\n"
            f"STDOUT:\n{process.stdout}\n"
            f"STDERR:\n{process.stderr}"
        )


def test_sp_gradient_matches_single_rank(tmp_path: Path) -> None:
    if not torch.cuda.is_available():
        pytest.skip("This test requires CUDA.")
    if torch.cuda.device_count() < SP_WORLD_SIZE:
        pytest.skip(
            f"This test requires at least {SP_WORLD_SIZE} CUDA devices."
        )

    script_path = Path(__file__).resolve()
    single_path = tmp_path / "single_rank_grads.pt"
    sp_path = tmp_path / f"sp{SP_WORLD_SIZE}_grads.pt"

    _run_torchrun(
        script_path=script_path,
        mode="single",
        nproc_per_node=1,
        output_path=single_path,
    )
    _run_torchrun(
        script_path=script_path,
        mode="sp",
        nproc_per_node=SP_WORLD_SIZE,
        output_path=sp_path,
    )

    single_grads: dict[str, torch.Tensor] = torch.load(
        single_path,
        map_location="cpu",
    )["grads"]
    sp_grads: dict[str, torch.Tensor] = torch.load(
        sp_path,
        map_location="cpu",
    )["grads"]
    single_names = set(single_grads)
    sp_names = set(sp_grads)
    single_only = sorted(single_names - sp_names)
    sp_only = sorted(sp_names - single_names)
    same_params: list[str] = []
    different_params: list[str] = []
    for name in sorted(single_names & sp_names):
        single_grad = single_grads[name]
        sp_grad = sp_grads[name]
        if torch.allclose(single_grad, sp_grad, rtol=1e-4, atol=1e-5):
            same_params.append(name)
            continue

        max_abs_diff = (single_grad - sp_grad).abs().max().item()
        different_params.append(f"{name} (max_abs_diff={max_abs_diff:.3e})")

    report = (
        f"single-only params ({len(single_only)}):\n"
        + "\n".join(single_only)
        + f"\n\nsp-only params ({len(sp_only)}):\n"
        + "\n".join(sp_only)
        + "\n\n"
        f"same params ({len(same_params)}):\n"
        + "\n".join(same_params)
        + f"\n\ndifferent params ({len(different_params)}):\n"
        + "\n".join(different_params)
    )
    print(report)
    assert not single_only and not sp_only and not different_params, report


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sp-grad-worker", action="store_true")
    parser.add_argument("--mode", choices=["single", "sp"], default=None)
    parser.add_argument("--output", type=str, default=None)
    return parser.parse_args()

# pytest -sv fastvideo/tests/distributed/test_sp_wan.py 
if __name__ == "__main__":
    args = _parse_args()
    if not args.sp_grad_worker:
        raise SystemExit("This module is intended to be run by pytest.")
    if args.mode is None or args.output is None:
        raise SystemExit("--mode and --output are required in worker mode.")
    _run_worker(mode=args.mode, output_path=Path(args.output))
