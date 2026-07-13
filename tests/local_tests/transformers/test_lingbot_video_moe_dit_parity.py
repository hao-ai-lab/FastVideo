# SPDX-License-Identifier: Apache-2.0
"""Sequential real-checkpoint parity for the released 30B-A3B MoE DiT."""

from __future__ import annotations

import gc
import json
import os
from pathlib import Path
import sys
from typing import Any

import pytest
import torch
from torch.testing import assert_close

from fastvideo.configs.models.dits.lingbot_video import LingBotVideoConfig
from fastvideo.models.dits.lingbot_video import LingBotVideoTransformer3DModel
from fastvideo.models.loader.fsdp_load import maybe_load_fsdp_model


os.environ.setdefault("MASTER_ADDR", "localhost")
os.environ.setdefault("MASTER_PORT", "29529")
os.environ.setdefault("DISABLE_SP", "1")
os.environ.setdefault("FASTVIDEO_ATTENTION_BACKEND", "TORCH_SDPA")
os.environ.setdefault("DIFFUSERS_ATTN_BACKEND", "native")
os.environ.setdefault("LINGBOT_MOE_EXPERT_BACKEND", "grouped_mm")
os.environ.setdefault("LINGBOT_MOE_PAD_BACKEND", "loop")
os.environ.setdefault("LINGBOT_MOE_REORDER_BACKEND", "sort")
os.environ.setdefault("LINGBOT_MOE_RESTORE_BACKEND", "scatter")

WORKTREE = Path("/mnt/weka/shrd/wm/junda/fv-hub/fastvideo-port-lingbot-video")
OFFICIAL_REF_DIR = Path("/mnt/weka/shrd/wm/junda/fv-hub/lingbot-video-reference")
OFFICIAL_MODEL_DIR = WORKTREE / "checkpoints/lingbot-video/official/moe-30b-a3b"
PARITY_VARIANT = os.environ.get("LINGBOT_VIDEO_MOE_PARITY_VARIANT", "base")
if PARITY_VARIANT not in {"base", "refiner"}:
    raise ValueError(f"Unsupported LingBot-Video MoE parity variant: {PARITY_VARIANT}")
OFFICIAL_SUBFOLDER = "transformer" if PARITY_VARIANT == "base" else "refiner"
NATIVE_SUBFOLDER = "transformer" if PARITY_VARIANT == "base" else "transformer_2"
NATIVE_TRANSFORMER_DIR = WORKTREE / "checkpoints/lingbot-video/converted/moe-30b-a3b" / NATIVE_SUBFOLDER


def _make_inputs(device: torch.device) -> dict[str, torch.Tensor | bool]:
    """Create a deterministic small latent and single-branch text condition."""
    generator = torch.Generator(device="cpu").manual_seed(20260711)
    return {
        "hidden_states": torch.randn(1, 16, 1, 4, 4, generator=generator).to(device),
        "timestep": torch.tensor([500.0], device=device),
        "encoder_hidden_states": torch.randn(1, 22, 2560, generator=generator).to(
            device=device, dtype=torch.bfloat16
        ),
        "encoder_attention_mask": torch.ones(1, 22, device=device, dtype=torch.long),
        "return_dict": False,
    }


def _capture_blocks(model: torch.nn.Module) -> tuple[dict[str, torch.Tensor], list[Any]]:
    """Capture each block output so failures identify the first divergent layer."""
    captured: dict[str, torch.Tensor] = {}
    handles: list[Any] = []

    def save(name: str):
        def hook(_module, _inputs, output):
            tensor = output[0] if isinstance(output, tuple) else output
            captured[name] = tensor.detach().float().cpu()

        return hook

    for name, module in model.named_modules():
        if name.startswith("blocks.") and len(name.split(".")) == 2:
            handles.append(module.register_forward_hook(save(name)))
    return captured, handles


def _run_model(model: torch.nn.Module, device: torch.device) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Run one full MoE DiT call and return final plus per-block CPU tensors."""
    captured, handles = _capture_blocks(model)
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        output = model(**_make_inputs(device))
    for handle in handles:
        handle.remove()
    if isinstance(output, tuple):
        output = output[0]
    elif hasattr(output, "sample"):
        output = output.sample
    return output.detach().float().cpu(), captured


def _load_official(device: torch.device) -> torch.nn.Module:
    """Load the released MoE transformer through its official Diffusers class."""
    if str(OFFICIAL_REF_DIR) not in sys.path:
        sys.path.insert(0, str(OFFICIAL_REF_DIR))
    from lingbot_video.transformer_lingbot_video import LingBotVideoTransformer3DModel as OfficialModel

    model = OfficialModel.from_pretrained(
        str(OFFICIAL_MODEL_DIR),
        subfolder=OFFICIAL_SUBFOLDER,
        torch_dtype=torch.bfloat16,
    )
    return model.to(device).eval()


def _load_native(device: torch.device) -> torch.nn.Module:
    """Load the native MoE transformer with FastVideo's mixed-dtype production loader."""
    hf_config = json.loads((NATIVE_TRANSFORMER_DIR / "config.json").read_text())
    weight_files = sorted(str(path) for path in NATIVE_TRANSFORMER_DIR.glob("*.safetensors"))
    model = maybe_load_fsdp_model(
        model_cls=LingBotVideoTransformer3DModel,
        init_params={"config": LingBotVideoConfig(), "hf_config": hf_config},
        weight_dir_list=weight_files,
        device=device,
        hsdp_replicate_dim=1,
        hsdp_shard_dim=1,
        default_dtype=torch.bfloat16,
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.float32,
        strict=True,
        cpu_offload=False,
        fsdp_inference=False,
        training_mode=False,
        pin_cpu_memory=False,
    ).eval()
    assert model.blocks[0].ffn.router.weight.dtype == torch.float32
    assert model.blocks[0].ffn.experts.w1.dtype == torch.bfloat16
    return model


def _report_first_block_drift(
    official: dict[str, torch.Tensor], native: dict[str, torch.Tensor]
) -> None:
    """Print the first block with nonzero drift in execution order."""
    for name, expected in official.items():
        actual = native[name]
        difference = (actual - expected).abs()
        if difference.max().item() != 0.0:
            print(
                f"first_block_drift={name} max_abs={difference.max().item():.8f} "
                f"mean_abs={difference.mean().item():.8f}"
            )
            return
    print("first_block_drift=not_found")


def test_lingbot_video_moe_dit_matches_official_sequentially() -> None:
    """Run one selected official/native 30B DiT pair sequentially on an H200."""
    if os.environ.get("LINGBOT_VIDEO_RUN_FULL_MOE_TESTS") != "1":
        pytest.skip("Set LINGBOT_VIDEO_RUN_FULL_MOE_TESTS=1 on a scheduled H200.")
    if not torch.cuda.is_available():
        pytest.skip("LingBot-Video MoE DiT parity requires CUDA.")
    device = torch.device("cuda:0")
    torch.backends.cuda.enable_cudnn_sdp(False)

    official = _load_official(device)
    torch.cuda.reset_peak_memory_stats(device)
    official_output, official_blocks = _run_model(official, device)
    official_peak = torch.cuda.max_memory_allocated(device)
    del official
    gc.collect()
    torch.cuda.empty_cache()

    native = _load_native(device)
    torch.cuda.reset_peak_memory_stats(device)
    native_output, native_blocks = _run_model(native, device)
    native_peak = torch.cuda.max_memory_allocated(device)
    difference = (native_output - official_output).abs()
    print(
        f"variant={PARITY_VARIANT} diff_max={difference.max().item():.8f} "
        f"diff_mean={difference.mean().item():.8f} "
        f"official_peak_bytes={official_peak} native_peak_bytes={native_peak}"
    )
    _report_first_block_drift(official_blocks, native_blocks)
    assert_close(native_output, official_output, atol=0.0, rtol=0.0)
