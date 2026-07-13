# SPDX-License-Identifier: Apache-2.0
"""Real-checkpoint CUDA parity for one released LingBot-Video MoE block."""

from __future__ import annotations

import gc
import json
import os
from pathlib import Path
import sys

import pytest
from safetensors import safe_open
import torch
from torch.testing import assert_close

from fastvideo.models.loader.fsdp_load import set_default_dtype


os.environ.setdefault("DIFFUSERS_ATTN_BACKEND", "native")
os.environ.setdefault("LINGBOT_MOE_EXPERT_BACKEND", "grouped_mm")
os.environ.setdefault("LINGBOT_MOE_PAD_BACKEND", "loop")
os.environ.setdefault("LINGBOT_MOE_REORDER_BACKEND", "sort")
os.environ.setdefault("LINGBOT_MOE_RESTORE_BACKEND", "scatter")

WORKTREE = Path("/mnt/weka/shrd/wm/junda/fv-hub/fastvideo-port-lingbot-video")
OFFICIAL_REF_DIR = Path("/mnt/weka/shrd/wm/junda/fv-hub/lingbot-video-reference")
TRANSFORMER_DIR = WORKTREE / "checkpoints/lingbot-video/official/moe-30b-a3b/transformer"


def _block_kwargs(config: dict, layer_index: int = 0) -> dict:
    """Select the shared official/native constructor arguments for one block."""
    return {
        "hidden_size": config["hidden_size"],
        "num_attention_heads": config["num_attention_heads"],
        "intermediate_size": config["intermediate_size"],
        "norm_eps": config["norm_eps"],
        "qkv_bias": config["qkv_bias"],
        "out_bias": config["out_bias"],
        "num_experts": config["num_experts"],
        "num_experts_per_tok": config["num_experts_per_tok"],
        "moe_intermediate_size": config["moe_intermediate_size"],
        "decoder_sparse_step": config["decoder_sparse_step"],
        "mlp_only_layers": config["mlp_only_layers"],
        "n_shared_experts": config["n_shared_experts"],
        "score_func": config["score_func"],
        "norm_topk_prob": config["norm_topk_prob"],
        "n_group": config["n_group"],
        "topk_group": config["topk_group"],
        "routed_scaling_factor": config["routed_scaling_factor"],
        "layer_idx": layer_index,
    }


def _load_block_state(layer_index: int = 0) -> dict[str, torch.Tensor]:
    """Read only one block's tensors from the indexed 30B checkpoint shards."""
    index = json.loads((TRANSFORMER_DIR / "diffusion_pytorch_model.safetensors.index.json").read_text())
    prefix = f"blocks.{layer_index}."
    keys_by_shard: dict[str, list[str]] = {}
    for name, shard_name in index["weight_map"].items():
        if name.startswith(prefix):
            keys_by_shard.setdefault(shard_name, []).append(name)
    state: dict[str, torch.Tensor] = {}
    for shard_name, names in keys_by_shard.items():
        with safe_open(TRANSFORMER_DIR / shard_name, framework="pt", device="cpu") as shard:
            for name in names:
                state[name.removeprefix(prefix)] = shard.get_tensor(name)
    if not state:
        raise RuntimeError(f"No checkpoint tensors found for {prefix}")
    return state


def _make_block_inputs(device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create deterministic mixed-precision block inputs and complex rotary phases."""
    generator = torch.Generator(device="cpu").manual_seed(20260711)
    hidden_states = torch.randn(1, 17, 2048, generator=generator, dtype=torch.float32).to(
        device=device, dtype=torch.bfloat16
    )
    temb6 = torch.randn(17, 6 * 2048, generator=generator, dtype=torch.float32).to(device)
    phases = torch.randn(1, 17, 64, generator=generator, dtype=torch.float32).to(device)
    rotary = torch.polar(torch.ones_like(phases), phases).to(torch.complex64)
    return hidden_states, temb6, rotary


def _run_block(block: torch.nn.Module, device: torch.device) -> torch.Tensor:
    """Execute a block with the same Torch SDPA and grouped-mm runtime policy."""
    hidden_states, temb6, rotary = _make_block_inputs(device)
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        output = block(hidden_states, temb6, rotary)
    return output.detach().float().cpu()


def test_lingbot_video_moe_block_matches_official_checkpoint() -> None:
    """Compare official and native block zero sequentially on one scheduled GPU."""
    if os.environ.get("LINGBOT_VIDEO_RUN_GPU_TESTS") != "1":
        pytest.skip("Set LINGBOT_VIDEO_RUN_GPU_TESTS=1 on a scheduled GPU node.")
    if not torch.cuda.is_available():
        pytest.skip("LingBot-Video MoE block parity requires CUDA.")
    if str(OFFICIAL_REF_DIR) not in sys.path:
        sys.path.insert(0, str(OFFICIAL_REF_DIR))
    from lingbot_video.transformer_lingbot_video import LingBotVideoBlock as OfficialBlock
    from fastvideo.models.dits.lingbot_video import LingBotVideoBlock as NativeBlock

    config = json.loads((TRANSFORMER_DIR / "config.json").read_text())
    state = _load_block_state()
    device = torch.device("cuda:0")
    torch.backends.cuda.enable_cudnn_sdp(False)

    with set_default_dtype(torch.bfloat16), torch.device("meta"):
        official = OfficialBlock(**_block_kwargs(config))
    official.load_state_dict(state, strict=True, assign=True)
    official = official.to(device).eval()
    official_output = _run_block(official, device)
    del official
    gc.collect()
    torch.cuda.empty_cache()

    with set_default_dtype(torch.bfloat16), torch.device("meta"):
        native = NativeBlock(**_block_kwargs(config))
    native.load_state_dict(state, strict=True, assign=True)
    native = native.to(device).eval()
    native_output = _run_block(native, device)

    difference = (native_output - official_output).abs()
    print(f"diff_max={difference.max().item():.8f} diff_mean={difference.mean().item():.8f}")
    assert_close(native_output, official_output, atol=0.0, rtol=0.0)
