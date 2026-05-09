# SPDX-License-Identifier: Apache-2.0
"""Component parity scaffold for <FAMILY> <COMPONENT>.

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
FAMILY = "<family>"  # TODO: snake_case family name.
COMPONENT = "<component>"  # TODO: transformer | vae | encoder | conditioner | ...
PARITY_SCOPE = "implementation_subcomponent"  # TODO: production_loader | implementation_subcomponent | both
OFFICIAL_MODULE = "<official.module>"  # TODO: e.g. "ltx_core.model.transformer".
OFFICIAL_CLASS = "<OfficialClass>"  # TODO: official class/factory name.
FASTVIDEO_CONFIG_MODULE = "fastvideo.configs.models.<bucket>"  # TODO.
FASTVIDEO_CONFIG_CLASS = "<FastVideoConfig>"  # TODO.
FASTVIDEO_MODEL_MODULE = "fastvideo.models.<bucket>.<module>"  # TODO.
FASTVIDEO_MODEL_CLASS = "<FastVideoModel>"  # TODO.

OFFICIAL_REF_DIR = Path(
    os.getenv("<FAMILY_UPPER>_OFFICIAL_REF_DIR", REPO_ROOT / "<ReferenceDir>")
)
LOCAL_WEIGHTS_DIR = Path(
    os.getenv("<FAMILY_UPPER>_LOCAL_WEIGHTS_DIR", REPO_ROOT / "official_weights" / FAMILY)
)
CONVERTED_WEIGHTS_DIR = Path(
    os.getenv("<FAMILY_UPPER>_CONVERTED_WEIGHTS_DIR", REPO_ROOT / "converted_weights" / FAMILY)
)


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

    # TODO: import official class/factory and load real weights strictly.
    # Examples in-tree:
    # - LTX2: SingleGPUModelBuilder(...).build(device=device, dtype=dtype)
    # - GameCraft: torch.load(...)["module"] -> official_model.load_state_dict(...)
    # - Oobleck: create_model_from_config(config) + ckpt state_dict
    OfficialClass = _import_or_skip(OFFICIAL_MODULE, OFFICIAL_CLASS)
    model = OfficialClass()  # TODO: pass official config kwargs.
    state_dict = {}  # TODO: load official state dict from LOCAL_WEIGHTS_DIR.
    missing, unexpected = model.load_state_dict(state_dict, strict=True)
    assert not missing and not unexpected, (
        f"official load mismatch missing={missing[:5]} unexpected={unexpected[:5]}"
    )
    return model.to(device=device, dtype=dtype).eval()


def _load_fastvideo_model(device: torch.device, dtype: torch.dtype) -> torch.nn.Module:
    """Load the FastVideo component with the same tensor content."""
    if not CONVERTED_WEIGHTS_DIR.exists() and not LOCAL_WEIGHTS_DIR.exists():
        pytest.skip(
            f"No FastVideo loadable weights: {CONVERTED_WEIGHTS_DIR} or {LOCAL_WEIGHTS_DIR}"
        )

    # TODO: replace with the bucket-specific FastVideo config/class/loader.
    # DiT examples:
    #   from fastvideo.configs.models.dits import <Config>
    #   from fastvideo.models.dits.<module> import <Model>
    # VAE examples:
    #   from fastvideo.models.vaes.<module> import <VAE>
    #   model = <VAE>.from_pretrained(...)
    FastVideoConfig = _import_or_skip(FASTVIDEO_CONFIG_MODULE, FASTVIDEO_CONFIG_CLASS)
    FastVideoModel = _import_or_skip(FASTVIDEO_MODEL_MODULE, FASTVIDEO_MODEL_CLASS)

    config = FastVideoConfig()
    model = FastVideoModel(config=config)
    state_dict = {}  # TODO: load converted or directly mapped state dict.
    missing, unexpected = model.load_state_dict(state_dict, strict=True)
    assert not missing and not unexpected, (
        f"FastVideo load mismatch missing={missing[:5]} unexpected={unexpected[:5]}"
    )
    return model.to(device=device, dtype=dtype).eval()


def _make_inputs(device: torch.device, dtype: torch.dtype) -> dict[str, torch.Tensor]:
    """Create deterministic inputs matching the official component call."""
    torch.manual_seed(0)
    # TODO: replace with component-specific tensors and metadata.
    return {
        "hidden_states": torch.randn(1, 4, 16, device=device, dtype=dtype),
        "timestep": torch.tensor([10], device=device),
    }


def _run_official(model: torch.nn.Module, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
    """Run official component and return the tensor to compare."""
    with torch.inference_mode():
        output = model(**inputs)  # TODO: adapt official call signature.
    if isinstance(output, dict):
        sample = output.get("sample")
        output = sample if sample is not None else output.get("x")
    elif hasattr(output, "sample"):
        output = output.sample
    elif isinstance(output, tuple):
        output = output[0]
    assert torch.is_tensor(output), f"official output is not tensor: {type(output)}"
    return output.detach().float().cpu()


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
    inputs = _make_inputs(device, dtype)

    official_out = _run_official(official, inputs)
    fastvideo_out = _run_fastvideo(fastvideo, inputs)

    assert official_out.shape == fastvideo_out.shape
    diff = (official_out - fastvideo_out).abs()
    print(
        f"official abs_mean={official_out.abs().mean().item():.6f} "
        f"fastvideo abs_mean={fastvideo_out.abs().mean().item():.6f} "
        f"diff_max={diff.max().item():.6f} diff_mean={diff.mean().item():.6f}"
    )

    # TODO: pick tolerance by scope:
    # - single block / same kernel: 1e-4
    # - full DiT aligned kernels: 1e-2
    # - full DiT cross-kernel bf16: 1e-1 + abs_mean drift check
    # - VAE decode fp32: 5e-2 after normalization alignment
    assert_close(fastvideo_out, official_out, atol=1e-4, rtol=1e-4)
