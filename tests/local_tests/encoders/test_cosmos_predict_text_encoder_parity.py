# SPDX-License-Identifier: Apache-2.0
"""Component parity scaffold for cosmos_predict text_encoder.

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
COMPONENT = "text_encoder"  # TODO: transformer | vae | encoder | conditioner | ...
PARITY_SCOPE = "implementation_subcomponent"  # TODO: production_loader | implementation_subcomponent | both
OFFICIAL_MODULE = "transformers"
OFFICIAL_CLASS = "Qwen2_5_VLForConditionalGeneration"
FASTVIDEO_CONFIG_MODULE = "fastvideo.configs.models.encoders"
FASTVIDEO_CONFIG_CLASS = "Qwen2_5_VLConfig"
FASTVIDEO_MODEL_MODULE = "fastvideo.models.encoders.cosmos_predict_text_encoder"
FASTVIDEO_MODEL_CLASS = "CosmosPredictTextEncoder"

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

    # Since we are doing component parity, we can instantiate the HF model with a dummy config
    # to avoid loading massive weights.
    from transformers import Qwen2_5_VLConfig
    from transformers.models.qwen2_5_vl.configuration_qwen2_5_vl import Qwen2_5_VLVisionConfig
    
    # Create a small dummy config for fast testing
    vision_config = Qwen2_5_VLVisionConfig(
        depth=2,
        hidden_size=64,
        intermediate_size=128,
        num_heads=2,
    )
    config = Qwen2_5_VLConfig(
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        rope_scaling={"type": "mrope", "mrope_section": [8, 4, 4]},
        vision_config=vision_config.to_dict(),
    )
    OfficialClass = _import_or_skip(OFFICIAL_MODULE, OFFICIAL_CLASS)
    model = OfficialClass(config)
    return model.to(device=device, dtype=dtype).eval()


def _load_fastvideo_model(device: torch.device, dtype: torch.dtype) -> torch.nn.Module:
    """Load the FastVideo component with the same tensor content."""
    if not CONVERTED_WEIGHTS_DIR.exists() and not LOCAL_WEIGHTS_DIR.exists():
        pytest.skip(f"No FastVideo loadable weights: {CONVERTED_WEIGHTS_DIR} or {LOCAL_WEIGHTS_DIR}")

    FastVideoModel = _import_or_skip(FASTVIDEO_MODEL_MODULE, FASTVIDEO_MODEL_CLASS)

    from transformers import Qwen2_5_VLConfig
    from transformers.models.qwen2_5_vl.configuration_qwen2_5_vl import Qwen2_5_VLVisionConfig
    
    vision_config = Qwen2_5_VLVisionConfig(
        depth=2,
        hidden_size=64,
        intermediate_size=128,
        num_heads=2,
    )
    config = Qwen2_5_VLConfig(
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        rope_scaling={"type": "mrope", "mrope_section": [8, 4, 4]},
        vision_config=vision_config.to_dict(),
    )
    model = FastVideoModel(config=config)
    
    return model.to(device=device, dtype=dtype).eval()


def _make_inputs(device: torch.device, dtype: torch.dtype) -> dict[str, torch.Tensor]:
    """Create deterministic inputs matching the official component call."""
    torch.manual_seed(0)
    return {
        "input_ids": torch.randint(0, 1000, (1, 32), device=device, dtype=torch.long),
    }


def _run_official(model: torch.nn.Module, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
    """Run official component and return the tensor to compare."""
    with torch.inference_mode():
        outputs = model(
            input_ids=inputs["input_ids"],
            output_hidden_states=True,
            return_dict=True,
        )
        hidden_states = outputs.hidden_states

        normalized_hidden_states = []
        for layer_idx in range(1, len(hidden_states)):
            normalized_state = (hidden_states[layer_idx] - hidden_states[layer_idx].mean(dim=-1, keepdim=True)) / (
                hidden_states[layer_idx].std(dim=-1, keepdim=True) + 1e-8
            )
            normalized_hidden_states.append(normalized_state)

        prompt_embeds = torch.cat(normalized_hidden_states, dim=-1)
    
    return prompt_embeds.detach().float().cpu()


def _run_fastvideo(model: torch.nn.Module, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
    """Run FastVideo component and return the tensor to compare."""
    with torch.inference_mode():
        output = model(**inputs)  # TODO: adapt FastVideo call signature.
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

    official = _load_official_model(device, dtype)
    fastvideo = _load_fastvideo_model(device, dtype)
    # Sync weights since we randomly initialized them!
    fastvideo.model.load_state_dict(official.state_dict(), strict=True)
    inputs = _make_inputs(device, dtype)

    official_out = _run_official(official, inputs)
    fastvideo_out = _run_fastvideo(fastvideo, inputs)

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
    assert_close(fastvideo_out, official_out, atol=1e-4, rtol=1e-4)
