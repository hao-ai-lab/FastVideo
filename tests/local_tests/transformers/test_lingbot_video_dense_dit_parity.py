# SPDX-License-Identifier: Apache-2.0
"""Dense LingBot-Video DiT parity scaffold.

Coverage scope: implementation_subcomponent. The official side loads the real
Dense 1.3B transformer checkpoint. The FastVideo side uses the planned native
DiT config/class path and strict-loads the same checkpoint tensor surface.
"""

from __future__ import annotations

import importlib
import json
import os
from pathlib import Path
import sys
from typing import Any

import pytest
import torch
from torch.testing import assert_close

from fastvideo.models.loader.fsdp_load import maybe_load_fsdp_model


os.environ.setdefault("MASTER_ADDR", "localhost")
os.environ.setdefault("MASTER_PORT", "29519")
os.environ.setdefault("DISABLE_SP", "1")
os.environ.setdefault("FASTVIDEO_ATTENTION_BACKEND", "TORCH_SDPA")
os.environ.setdefault("DIFFUSERS_ATTN_BACKEND", "native")

REPO_ROOT = Path(__file__).resolve().parents[3]
OFFICIAL_REF_DIR = Path(
    os.getenv(
        "LINGBOT_VIDEO_OFFICIAL_REF_DIR",
        "/mnt/weka/shrd/wm/junda/fv-hub/lingbot-video-reference",
    )
)
OFFICIAL_MODEL_DIR = Path(
    os.getenv(
        "LINGBOT_VIDEO_DENSE_MODEL_DIR",
        REPO_ROOT / "checkpoints" / "lingbot-video" / "official" / "dense-1.3b",
    )
)
TRANSFORMER_DIR = OFFICIAL_MODEL_DIR / "transformer"
CONVERTED_TRANSFORMER_DIR = REPO_ROOT / "checkpoints" / "lingbot-video" / "converted" / "dense-1.3b" / "transformer"
FASTVIDEO_CONFIG_MODULE = "fastvideo.configs.models.dits.lingbot_video"
FASTVIDEO_CONFIG_CLASS = "LingBotVideoConfig"
FASTVIDEO_MODEL_MODULE = "fastvideo.models.dits.lingbot_video"
FASTVIDEO_MODEL_CLASS = "LingBotVideoTransformer3DModel"
FASTVIDEO_MODEL_FILE = REPO_ROOT / "fastvideo" / "models" / "dits" / "lingbot_video.py"
PARITY_SCOPE = "implementation_subcomponent"


def _load_fastvideo_types_or_skip():
    """Resolve the planned native class/config, skipping only until the class exists."""
    if not FASTVIDEO_MODEL_FILE.is_file():
        pytest.skip(f"FastVideo Dense DiT class missing: {FASTVIDEO_MODEL_MODULE}.{FASTVIDEO_MODEL_CLASS}")
    model_module = importlib.import_module(FASTVIDEO_MODEL_MODULE)
    try:
        model_class = getattr(model_module, FASTVIDEO_MODEL_CLASS)
    except AttributeError:
        pytest.skip(f"FastVideo Dense DiT class missing: {FASTVIDEO_MODEL_MODULE}.{FASTVIDEO_MODEL_CLASS}")

    config_module = importlib.import_module(FASTVIDEO_CONFIG_MODULE)
    return getattr(config_module, FASTVIDEO_CONFIG_CLASS), model_class


def _require_official_assets() -> None:
    """Fail clearly when the pinned reference checkout or Dense checkpoint is incomplete."""
    required = (
        OFFICIAL_REF_DIR / "lingbot_video" / "transformer_lingbot_video.py",
        TRANSFORMER_DIR / "config.json",
        TRANSFORMER_DIR / "diffusion_pytorch_model.safetensors",
    )
    missing = [str(path) for path in required if not path.is_file()]
    assert not missing, f"Missing required official Dense DiT assets: {missing}"


def _load_official_model(device: torch.device, dtype: torch.dtype) -> torch.nn.Module:
    """Load the official Dense transformer through its production Diffusers path."""
    _require_official_assets()
    if str(OFFICIAL_REF_DIR) not in sys.path:
        sys.path.insert(0, str(OFFICIAL_REF_DIR))
    from lingbot_video.transformer_lingbot_video import LingBotVideoTransformer3DModel

    model = LingBotVideoTransformer3DModel.from_pretrained(
        str(OFFICIAL_MODEL_DIR),
        subfolder="transformer",
        torch_dtype=dtype,
    )
    return model.to(device=device).eval()


def _load_fastvideo_model(device: torch.device, dtype: torch.dtype) -> torch.nn.Module:
    """Load the native Dense DiT through the production mixed-precision loader."""
    FastVideoConfig, FastVideoModel = _load_fastvideo_types_or_skip()
    _require_official_assets()
    with (CONVERTED_TRANSFORMER_DIR / "config.json").open(encoding="utf-8") as file:
        hf_config = json.load(file)

    config = FastVideoConfig()
    model = maybe_load_fsdp_model(
        model_cls=FastVideoModel,
        init_params={"config": config, "hf_config": hf_config},
        weight_dir_list=[str(CONVERTED_TRANSFORMER_DIR / "diffusion_pytorch_model.safetensors")],
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
    assert model.patch_embedder.weight.dtype == torch.bfloat16
    assert model.blocks[0].scale_shift_table.dtype == torch.float32
    assert model.blocks[0].attn.to_q.weight.dtype == torch.bfloat16
    return model


def _make_inputs(device: torch.device, dtype: torch.dtype) -> dict[str, torch.Tensor]:
    """Create a small deterministic upstream-format latent and text condition."""
    generator = torch.Generator(device="cpu").manual_seed(20260711)
    hidden_states = torch.randn(1, 16, 1, 4, 4, generator=generator, dtype=torch.float32)
    encoder_hidden_states = torch.randn(1, 22, 2560, generator=generator, dtype=torch.float32)
    return {
        "hidden_states": hidden_states.to(device=device),
        "timestep": torch.tensor([500.0], device=device, dtype=torch.float32),
        "encoder_hidden_states": encoder_hidden_states.to(device=device, dtype=dtype),
        "encoder_attention_mask": torch.ones((1, 22), device=device, dtype=torch.long),
        "return_dict": False,
    }


def _run_model(model: torch.nn.Module, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
    """Execute the full Dense DiT and extract its denoised latent tensor."""
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        output = model(**inputs)
    if isinstance(output, tuple):
        output = output[0]
    elif hasattr(output, "sample"):
        output = output.sample
    assert torch.is_tensor(output), f"Dense DiT output is not a tensor: {type(output)}"
    return output.detach().float().cpu()


def _capture_intermediates(model: torch.nn.Module) -> tuple[dict[str, torch.Tensor], list[Any]]:
    """Capture ordered parity checkpoints without changing either implementation."""
    captured: dict[str, torch.Tensor] = {}
    handles: list[Any] = []
    suffixes = (
        "patch_embedder",
        "time_proj",
        "time_embedder",
        "time_modulation",
        "text_embedder",
        "norm1",
        "to_q",
        "norm_q",
        "to_k",
        "norm_k",
        "to_v",
        "to_out",
        "norm_post_attn",
        "norm2",
        "gate_proj",
        "up_proj",
        "down_proj",
        "norm_post_ffn",
        "norm_out",
        "norm_out_modulation",
        "proj_out",
    )

    def save(name: str):
        def hook(_module, _inputs, output):
            tensor = output[0] if isinstance(output, tuple) else output
            if torch.is_tensor(tensor):
                captured[name] = tensor.detach().float().cpu()

        return hook

    for name, module in model.named_modules():
        if name == "" or not (name.startswith("blocks.") or name in suffixes):
            continue
        if name.startswith("blocks.") and not name.endswith(suffixes):
            parts = name.split(".")
            if len(parts) != 2:
                continue
        handles.append(module.register_forward_hook(save(name)))
    return captured, handles


def _report_first_intermediate_drift(official: dict[str, torch.Tensor], fastvideo: dict[str, torch.Tensor]) -> None:
    """Print the first execution-ordered checkpoint whose values diverge materially."""
    for name, expected in official.items():
        actual = fastvideo.get(name)
        if actual is None or actual.shape != expected.shape:
            print(f"first_intermediate_drift={name} shape={getattr(actual, 'shape', None)} vs {expected.shape}")
            return
        drift = (actual - expected).abs()
        if drift.max().item() > 1e-3:
            print(
                f"first_intermediate_drift={name} max_abs={drift.max().item():.8f} mean_abs={drift.mean().item():.8f}"
            )
            return
    print("first_intermediate_drift=not_found")


def test_lingbot_video_dense_dit_outputs_match() -> None:
    """Compare real official and native Dense DiT outputs for identical tensors."""
    if os.environ.get("LINGBOT_VIDEO_RUN_GPU_TESTS") != "1":
        pytest.skip("Set LINGBOT_VIDEO_RUN_GPU_TESTS=1 on a scheduled GPU node.")
    if not torch.cuda.is_available():
        pytest.skip("LingBot-Video Dense DiT parity requires CUDA.")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    fastvideo = _load_fastvideo_model(device, dtype)
    official = _load_official_model(device, dtype)
    inputs = _make_inputs(device, dtype)
    official_intermediates, official_handles = _capture_intermediates(official)
    fastvideo_intermediates, fastvideo_handles = _capture_intermediates(fastvideo)
    official_output = _run_model(official, inputs)
    fastvideo_output = _run_model(fastvideo, inputs)
    for handle in official_handles + fastvideo_handles:
        handle.remove()

    assert official_output.shape == fastvideo_output.shape
    difference = (official_output - fastvideo_output).abs()
    official_abs_mean = official_output.abs().mean()
    abs_mean_drift = (fastvideo_output.abs().mean() - official_abs_mean).abs()
    print(
        f"official_abs_mean={official_abs_mean.item():.6f} "
        f"fastvideo_abs_mean={fastvideo_output.abs().mean().item():.6f} "
        f"diff_max={difference.max().item():.6f} diff_mean={difference.mean().item():.6f}"
    )
    _report_first_intermediate_drift(official_intermediates, fastvideo_intermediates)

    assert abs_mean_drift <= official_abs_mean * 0.05
    assert_close(fastvideo_output, official_output, atol=1e-2, rtol=1e-2)
