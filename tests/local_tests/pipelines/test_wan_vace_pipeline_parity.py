# SPDX-License-Identifier: Apache-2.0
"""Wan-VACE transformer single-step forward parity against Diffusers."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
import torch
from torch.testing import assert_close

os.environ.setdefault("DISABLE_SP", "1")
os.environ.setdefault("FASTVIDEO_ATTENTION_BACKEND", "TORCH_SDPA")
os.environ.setdefault("MASTER_ADDR", "localhost")
os.environ.setdefault("MASTER_PORT", "29520")

REPO_ROOT = Path(__file__).resolve().parents[3]
MODEL_DIR = Path(
    os.getenv(
        "WAN_VACE_MODEL_DIR",
        REPO_ROOT / "official_weights" / "Wan2.1-VACE-1.3B-diffusers",
    ))


def _resolve_model_dir() -> Path:
    if MODEL_DIR.is_dir() and (MODEL_DIR / "model_index.json").is_file():
        return MODEL_DIR
    hub_root = Path(
        "/home/wenbo.ji/egocentric/datastorage/public_models/huggingface/hub/models--Wan-AI--Wan2.1-VACE-1.3B-diffusers/snapshots"
    )
    if hub_root.is_dir():
        for candidate in sorted(hub_root.iterdir()):
            if (candidate / "model_index.json").is_file():
                return candidate
    pytest.skip(f"Wan-VACE weights not found under {MODEL_DIR}")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Wan-VACE parity requires CUDA")
def test_wan_vace_transformer_forward_parity() -> None:
    from fastvideo.distributed import (
        cleanup_dist_env_and_memory,
        maybe_init_distributed_environment_and_model_parallel,
    )

    maybe_init_distributed_environment_and_model_parallel(1, 1)

    model_dir = _resolve_model_dir()
    transformer_dir = model_dir / "transformer"
    if not (transformer_dir / "config.json").is_file():
        pytest.skip(f"transformer config missing under {transformer_dir}")

    from diffusers.models.transformers.transformer_wan_vace import (
        WanVACETransformer3DModel as DiffusersWanVACETransformer3DModel,
    )
    from fastvideo.forward_context import set_forward_context
    from fastvideo.models.loader.fsdp_load import maybe_load_fsdp_model
    from fastvideo.models.wan.pipeline_config import WanVACE1_3B_Config
    from fastvideo.models.wan.vace_transformer import WanVACETransformer3DModel
    from fastvideo.pipelines.pipeline_batch_info import ForwardBatch

    device = torch.device("cuda")
    dtype = torch.float32

    official = DiffusersWanVACETransformer3DModel.from_pretrained(
        str(model_dir),
        subfolder="transformer",
        torch_dtype=dtype,
    ).to(device=device).eval()

    with (transformer_dir / "config.json").open(encoding="utf-8") as file:
        hf_config = json.load(file)
    config = WanVACE1_3B_Config().dit_config
    weight_files = sorted(transformer_dir.glob("diffusion_pytorch_model*.safetensors"))
    if not weight_files:
        pytest.skip(f"No transformer weights under {transformer_dir}")

    fastvideo = maybe_load_fsdp_model(
        model_cls=WanVACETransformer3DModel,
        init_params={"config": config, "hf_config": hf_config},
        weight_dir_list=[str(path) for path in weight_files],
        device=device,
        hsdp_replicate_dim=1,
        hsdp_shard_dim=1,
        default_dtype=dtype,
        param_dtype=dtype,
        reduce_dtype=torch.float32,
        strict=True,
        cpu_offload=False,
        fsdp_inference=False,
        training_mode=False,
        pin_cpu_memory=False,
    ).eval()

    generator = torch.Generator(device=device).manual_seed(42)
    hidden_states = torch.randn(1, 16, 2, 8, 8, generator=generator, dtype=dtype, device=device)
    control_hidden_states = torch.randn(1, 96, 2, 8, 8, generator=generator, dtype=dtype, device=device)
    encoder_hidden_states = torch.randn(1, 22, 4096, generator=generator, dtype=dtype, device=device)
    timestep = torch.tensor([500], device=device, dtype=torch.long)

    num_vace_layers = len(config.arch_config.vace_layers)
    scale_tensor = torch.ones(num_vace_layers, device=device, dtype=dtype)
    with torch.inference_mode():
        official_out = official(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            timestep=timestep,
            control_hidden_states=control_hidden_states,
            control_hidden_states_scale=scale_tensor,
        ).sample
        with set_forward_context(
                current_timestep=int(timestep.item()),
                attn_metadata=None,
                forward_batch=ForwardBatch(data_type="dummy"),
        ):
            fastvideo_out = fastvideo(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                timestep=timestep,
                control_hidden_states=control_hidden_states,
                control_hidden_states_scale=scale_tensor.tolist(),
            )

    assert_close(fastvideo_out, official_out, rtol=0.02, atol=0.02)
    cleanup_dist_env_and_memory()
