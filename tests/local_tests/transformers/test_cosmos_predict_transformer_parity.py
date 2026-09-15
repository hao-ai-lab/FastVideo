# SPDX-License-Identifier: Apache-2.0
"""Component parity scaffold for cosmos_predict transformer.

This file is intended to be created early in a port. It may skip until the
official reference, FastVideo class, and real weights are available, but it must
never become an unconditional skip or shape-only test.

Fill every TODO before considering this test active.
"""
from __future__ import annotations

import importlib
import os
from pathlib import Path
import sys

import pytest
import torch
from torch.testing import assert_close

os.environ.setdefault("MASTER_ADDR", "localhost")
os.environ.setdefault("MASTER_PORT", "29519")
os.environ.setdefault("DISABLE_SP", "1")
os.environ.setdefault("FASTVIDEO_ATTENTION_BACKEND", "TORCH_SDPA")

REPO_ROOT = Path(__file__).resolve().parents[3]
FAMILY = "cosmos_predict"  # TODO: snake_case family name.
COMPONENT = "transformer"  # TODO: transformer | vae | encoder | conditioner | ...
PARITY_SCOPE = "implementation_subcomponent"  # TODO: production_loader | implementation_subcomponent | both
OFFICIAL_MODULE = "diffusers.models.transformers.transformer_cosmos"
OFFICIAL_CLASS = "CosmosTransformer3DModel"
FASTVIDEO_CONFIG_MODULE = "fastvideo.configs.models.dits.cosmos2_5"
FASTVIDEO_CONFIG_CLASS = "Cosmos25VideoConfig"
FASTVIDEO_MODEL_MODULE = "fastvideo.models.dits.cosmos2_5"
FASTVIDEO_MODEL_CLASS = "Cosmos25Transformer3DModel"

OFFICIAL_REF_DIR = Path(os.getenv("COSMOS_PREDICT_OFFICIAL_REF_DIR", REPO_ROOT / "Cosmos"))
LOCAL_WEIGHTS_DIR = Path(os.getenv("COSMOS_PREDICT_LOCAL_WEIGHTS_DIR", REPO_ROOT / "official_weights" / FAMILY))
CONVERTED_WEIGHTS_DIR = Path(os.getenv("COSMOS_PREDICT_CONVERTED_WEIGHTS_DIR",
                                       REPO_ROOT / "converted_weights" / FAMILY))


def _resolve_hf_token() -> str | None:
    for key in ("HF_TOKEN", "HUGGINGFACE_HUB_TOKEN", "HF_API_KEY"):
        value = os.environ.get(key)
        if value:
            return value
    return None


def _add_official_to_path() -> None:
    """Add the official source path before importing upstream modules."""
    # TODO: adjust for the official repo layout. Common examples:
    # OFFICIAL_REF_DIR / "src"
    # OFFICIAL_REF_DIR / "packages" / "<pkg>" / "src"
    # OFFICIAL_REF_DIR
    official_src = OFFICIAL_REF_DIR / "src"
    if not official_src.exists():
        official_src = OFFICIAL_REF_DIR
    if official_src.exists() and str(official_src) not in sys.path:
        sys.path.insert(0, str(official_src))


def _import_or_skip(module_name: str, attr_name: str | None = None):
    if "<" in module_name or (attr_name is not None and "<" in attr_name):
        pytest.skip(f"Template import placeholder not filled: {module_name}.{attr_name}")
    try:
        module = importlib.import_module(module_name)
    except Exception as exc:  # noqa: BLE001 - local parity should skip missing refs.
        pytest.skip(f"Cannot import {module_name}: {exc}")
    if attr_name is None:
        return module
    try:
        return getattr(module, attr_name)
    except AttributeError:
        pytest.skip(f"{module_name} has no attribute {attr_name}")


def _load_official_model(device: torch.device, dtype: torch.dtype) -> torch.nn.Module:
    """Load the official component with real weights."""
    _add_official_to_path()
    if not OFFICIAL_REF_DIR.exists():
        pytest.skip(f"Official reference missing: {OFFICIAL_REF_DIR}")
    if not LOCAL_WEIGHTS_DIR.exists():
        pytest.skip(f"Local weights missing: {LOCAL_WEIGHTS_DIR}")

    OfficialClass = _import_or_skip(OFFICIAL_MODULE, OFFICIAL_CLASS)
    # Instantiate with small dimensions
    model = OfficialClass(
        in_channels=16,
        out_channels=16,
        num_attention_heads=2,
        attention_head_dim=32,
        num_layers=2,
        text_embed_dim=128,
        adaln_lora_dim=64,
        max_size=(16, 32, 32),
        patch_size=(1, 2, 2),
    )
    return model.to(device=device, dtype=dtype).eval()


def _load_fastvideo_model(device: torch.device, dtype: torch.dtype) -> torch.nn.Module:
    """Load the FastVideo component with the same tensor content."""
    if not CONVERTED_WEIGHTS_DIR.exists() and not LOCAL_WEIGHTS_DIR.exists():
        pytest.skip(f"No FastVideo loadable weights: {CONVERTED_WEIGHTS_DIR} or {LOCAL_WEIGHTS_DIR}")

    FastVideoConfig = _import_or_skip(FASTVIDEO_CONFIG_MODULE, FASTVIDEO_CONFIG_CLASS)
    FastVideoModel = _import_or_skip(FASTVIDEO_MODEL_MODULE, FASTVIDEO_MODEL_CLASS)
    FastVideoArchConfig = _import_or_skip(FASTVIDEO_CONFIG_MODULE, "Cosmos25ArchConfig")

    arch_config = FastVideoArchConfig(
        in_channels=16,
        out_channels=16,
        num_attention_heads=2,
        attention_head_dim=32,
        num_layers=2,
        text_embed_dim=128,
        adaln_lora_dim=64,
        max_size=(16, 32, 32),
        patch_size=(1, 2, 2),
        use_condition_mask=False,
    )
    config = FastVideoConfig(arch_config=arch_config)
    model = FastVideoModel(config=config, hf_config={})
    return model.to(device=device, dtype=dtype).eval()


def _make_inputs(device: torch.device, dtype: torch.dtype) -> dict[str, torch.Tensor]:
    """Create deterministic inputs matching the official component call."""
    torch.manual_seed(0)
    # Inputs: hidden_states (B, C, T, H, W)
    # encoder_hidden_states (B, L, D)
    return {
        "hidden_states": torch.randn(1, 16, 16, 32, 32, device=device, dtype=dtype),
        "encoder_hidden_states": torch.randn(1, 64, 128, device=device, dtype=dtype),
        "attention_mask": torch.ones(1, 64, device=device, dtype=torch.bool),
        "timestep": torch.tensor([10.0], device=device, dtype=dtype),
        "image_rotary_emb": None,
        "padding_mask": torch.ones(1, 1, 32, 32, device=device, dtype=dtype),
    }


def _run_official(model: torch.nn.Module, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
    """Run official component and return the tensor to compare."""
    with torch.inference_mode():
        output = model(
            hidden_states=inputs["hidden_states"],
            encoder_hidden_states=inputs["encoder_hidden_states"],
            timestep=inputs["timestep"],
            attention_mask=inputs["attention_mask"],
            padding_mask=inputs["padding_mask"],
            return_dict=True,
        )
    return output.sample.detach().float().cpu()


def _run_fastvideo(model: torch.nn.Module, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
    """Run FastVideo component and return the tensor to compare."""
    with torch.inference_mode():
        output = model(
            hidden_states=inputs["hidden_states"],
            encoder_hidden_states=inputs["encoder_hidden_states"],
            timestep=inputs["timestep"],
            attention_mask=inputs["attention_mask"],
            padding_mask=inputs["padding_mask"],
        )
    if isinstance(output, dict):
        sample = output.get("sample")
        output = sample if sample is not None else output.get("x")
    elif hasattr(output, "sample"):
        output = output.sample
    elif isinstance(output, tuple):
        output = output[0]
    assert torch.is_tensor(output), f"FastVideo output is not tensor: {type(output)}"
    return output.detach().float().cpu()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for this parity test.")
def test_component_parity():
    """Compare official and FastVideo outputs on identical inputs."""
    device = torch.device("cuda:0")
    dtype = torch.bfloat16

    import torch.distributed as dist
    from fastvideo.distributed.parallel_state import init_distributed_environment, initialize_model_parallel
    if not dist.is_initialized():
        init_distributed_environment(backend="nccl")
        initialize_model_parallel(tensor_model_parallel_size=1, sequence_model_parallel_size=1)
        
    official = _load_official_model(device, dtype)
    fastvideo = _load_fastvideo_model(device, dtype)

    # Sync random weights, mapping diffusers names to fastvideo names
    import re
    state_dict = official.state_dict()
    mapped_state_dict = {}
    for k, v in state_dict.items():
        k = re.sub(r"^transformer_blocks\.(\d+)\.ff\.net\.0\.proj\.(.*)$", r"transformer_blocks.\1.mlp.fc_in.\2", k)
        k = re.sub(r"^transformer_blocks\.(\d+)\.ff\.net\.2\.(.*)$", r"transformer_blocks.\1.mlp.fc_out.\2", k)
        
        # AdaLN modulations in transformer blocks
        k = re.sub(r"^transformer_blocks\.(\d+)\.norm1\.linear_1\.(.*)$", r"transformer_blocks.\1.adaln_modulation_self_attn.1.\2", k)
        k = re.sub(r"^transformer_blocks\.(\d+)\.norm1\.linear_2\.(.*)$", r"transformer_blocks.\1.adaln_modulation_self_attn.2.\2", k)
        k = re.sub(r"^transformer_blocks\.(\d+)\.norm2\.linear_1\.(.*)$", r"transformer_blocks.\1.adaln_modulation_cross_attn.1.\2", k)
        k = re.sub(r"^transformer_blocks\.(\d+)\.norm2\.linear_2\.(.*)$", r"transformer_blocks.\1.adaln_modulation_cross_attn.2.\2", k)
        k = re.sub(r"^transformer_blocks\.(\d+)\.norm3\.linear_1\.(.*)$", r"transformer_blocks.\1.adaln_modulation_mlp.1.\2", k)
        k = re.sub(r"^transformer_blocks\.(\d+)\.norm3\.linear_2\.(.*)$", r"transformer_blocks.\1.adaln_modulation_mlp.2.\2", k)

        # Norm weights
        k = re.sub(r"^transformer_blocks\.(\d+)\.norm1\.norm\.(.*)$", r"transformer_blocks.\1.norm1.norm.\2", k)
        k = re.sub(r"^transformer_blocks\.(\d+)\.norm2\.norm\.(.*)$", r"transformer_blocks.\1.norm2.norm.\2", k)
        k = re.sub(r"^transformer_blocks\.(\d+)\.norm3\.norm\.(.*)$", r"transformer_blocks.\1.norm3.norm.\2", k)
        
        k = re.sub(r"^transformer_blocks\.(\d+)\.attn1\.to_out\.0\.(.*)$", r"transformer_blocks.\1.attn1.to_out.\2", k)
        k = re.sub(r"^transformer_blocks\.(\d+)\.attn2\.to_out\.0\.(.*)$", r"transformer_blocks.\1.attn2.to_out.\2", k)
        
        # Final layer
        k = re.sub(r"^norm_out\.linear_1\.(.*)$", r"final_layer.linear_1.\1", k)
        k = re.sub(r"^norm_out\.linear_2\.(.*)$", r"final_layer.linear_2.\1", k)
        k = re.sub(r"^norm_out\.norm\.(.*)$", r"final_layer.norm.norm.\1", k)
        k = re.sub(r"^proj_out\.(.*)$", r"final_layer.proj_out.\1", k)
        
        # In diffusers, they use learnable_pos_embed.pos_emb_*, which doesn't exist in our state dict (we generate it)
        if "learnable_pos_embed" in k:
            continue
            
        mapped_state_dict[k] = v
        
    fastvideo.load_state_dict(mapped_state_dict, strict=True)
    
    inputs = _make_inputs(device, dtype)

    official_out = _run_official(official, inputs)
    
    from fastvideo.forward_context import set_forward_context
    with set_forward_context(current_timestep=0, attn_metadata=None):
        fastvideo_out = _run_fastvideo(fastvideo, inputs)

    print("Diffs:")
    assert official_out.shape == fastvideo_out.shape
    diff = (official_out - fastvideo_out).abs()
    print(f"official abs_mean={official_out.abs().mean().item():.6f} "
          f"fastvideo abs_mean={fastvideo_out.abs().mean().item():.6f} "
          f"diff_max={diff.max().item():.6f} diff_mean={diff.mean().item():.6f}")

    # TODO: pick tolerance by scope:
    # - single block / same kernel: 1e-4
    # - full DiT aligned kernels: 1e-2
    # - full DiT cross-kernel bf16: 1e-1 + abs_mean drift check
    # - VAE decode fp32: 5e-2 after normalization alignment
    assert_close(fastvideo_out, official_out, atol=2e-2, rtol=2e-2)
