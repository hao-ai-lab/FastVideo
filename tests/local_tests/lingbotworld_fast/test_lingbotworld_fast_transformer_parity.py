# SPDX-License-Identifier: Apache-2.0
"""Official-vs-FastVideo full-DiT parity for LingBot-World-Fast.

Coverage scope: both. The official ``WanModelFast`` implementation and the
FastVideo production ``TransformerLoader`` load the same released transformer
weights, then run one real cached causal forward with the released global
attention and three-frame chunk settings.
"""

from __future__ import annotations

import gc
import importlib
import json
import os
from pathlib import Path
import sys

import pytest
import torch
from torch.testing import assert_close

os.environ.setdefault("FASTVIDEO_ATTENTION_BACKEND", "TORCH_SDPA")
os.environ.setdefault("MASTER_ADDR", "localhost")
os.environ.setdefault("MASTER_PORT", "29531")

DEFAULT_MODEL_DIR = Path("official_weights/FastVideo/LingBot-World-Fast-Diffusers")
DEFAULT_REFERENCE_DIR = Path("official_repos/lingbot-world")


def _model_dir() -> Path:
    return Path(os.getenv("LINGBOTWORLD_FAST_MODEL_DIR", str(DEFAULT_MODEL_DIR))).expanduser().resolve()


def _reference_dir() -> Path:
    return Path(os.getenv("LINGBOTWORLD_FAST_REFERENCE_DIR", str(DEFAULT_REFERENCE_DIR))).expanduser().resolve()


def _transformer_dir() -> Path:
    return _model_dir() / "transformer"


def _has_real_weights() -> bool:
    transformer_dir = _transformer_dir()
    return transformer_dir.exists() and any(transformer_dir.glob("*.safetensors"))


def _official_constructor_kwargs() -> dict:
    return {
        "model_type": "i2v",
        "control_type": "cam",
        "patch_size": (1, 2, 2),
        "text_len": 512,
        "in_dim": 36,
        "dim": 5120,
        "ffn_dim": 13824,
        "freq_dim": 256,
        "text_dim": 4096,
        "out_dim": 16,
        "num_heads": 40,
        "num_layers": 40,
        "local_attn_size": -1,
        "sink_size": 9,
        "qk_norm": True,
        "cross_attn_norm": True,
        "eps": 1e-6,
    }


def _deterministic_inputs(dtype: torch.dtype) -> dict:
    generator = torch.Generator(device="cpu").manual_seed(20260407)
    chunk_size = 3
    latent_height = 4
    latent_width = 4
    frame_seqlen = latent_height * latent_width // 4
    token_count = chunk_size * frame_seqlen

    def randn(*shape: int) -> torch.Tensor:
        return torch.randn(shape, dtype=dtype, generator=generator)

    return {
        "x": [randn(16, chunk_size, latent_height, latent_width)],
        "t": torch.tensor([679.0], dtype=dtype),
        "context": [randn(8, 4096)],
        "seq_len": token_count,
        "y": [randn(20, chunk_size, latent_height, latent_width)],
        # The released pipeline packs each six-channel Plucker ray over the
        # VAE's 8x8 spatial stride, yielding 6 * 8 * 8 = 384 channels.
        "dit_cond_dict": {
            "c2ws_plucker_emb": [randn(1, 384, chunk_size, latent_height, latent_width)],
        },
        "current_start": 0,
        "max_attention_size": token_count,
        "frame_seqlen": frame_seqlen,
        "cross_attn_first_call": True,
    }


def _to_device(value, device: torch.device):
    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, list):
        return [_to_device(item, device) for item in value]
    if isinstance(value, dict):
        return {key: _to_device(item, device) for key, item in value.items()}
    return value


def _new_caches(device: torch.device, dtype: torch.dtype, token_count: int) -> tuple[list[dict], list[dict]]:
    self_cache = [{
        "k": torch.zeros((1, token_count, 40, 128), device=device, dtype=dtype),
        "v": torch.zeros((1, token_count, 40, 128), device=device, dtype=dtype),
        "global_end_index": torch.tensor([0], device=device, dtype=torch.long),
        "local_end_index": torch.tensor([0], device=device, dtype=torch.long),
    } for _ in range(40)]
    cross_cache = [{
        "k": torch.zeros((1, 512, 40, 128), device=device, dtype=dtype),
        "v": torch.zeros((1, 512, 40, 128), device=device, dtype=dtype),
        "is_init": torch.tensor(0, device=device, dtype=torch.int32),
    } for _ in range(40)]
    return self_cache, cross_cache


def _run_official(inputs: dict, dtype: torch.dtype) -> torch.Tensor:
    reference_dir = _reference_dir()
    sys.path.insert(0, str(reference_dir))
    try:
        model_module = importlib.import_module("wan.modules.model_fast")
        # The official cross-attention calls its FlashAttention-only helper
        # directly. Route it through the official module's own SDPA-capable
        # dispatcher so both implementations execute the same public fallback.
        model_module.flash_attention = model_module.attention
        model = model_module.WanModelFast.from_pretrained(
            _transformer_dir(),
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            device_map={"": "cuda"},
            **_official_constructor_kwargs(),
        ).eval()
        device_inputs = _to_device(inputs, torch.device("cuda"))
        self_cache, cross_cache = _new_caches(torch.device("cuda"), dtype, inputs["seq_len"])
        with (
            torch.inference_mode(),
            torch.amp.autocast("cuda", dtype=dtype),
            torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH),
        ):
            output = model(**device_inputs, kv_cache=self_cache, crossattn_cache=cross_cache)[0]
        return output.detach().float().cpu()
    finally:
        sys.path.remove(str(reference_dir))
        if "model" in locals():
            del model
        gc.collect()
        torch.cuda.empty_cache()


def _run_fastvideo(inputs: dict, dtype: torch.dtype) -> torch.Tensor:
    from fastvideo.configs.models.dits.lingbotworld_fast import LingBotWorldFastVideoConfig
    from fastvideo.configs.pipelines.base import PipelineConfig
    from fastvideo.distributed import cleanup_dist_env_and_memory, maybe_init_distributed_environment_and_model_parallel
    from fastvideo.fastvideo_args import FastVideoArgs
    from fastvideo.models.dits.lingbotworld2.causal_fast import LingBotWorld2CausalFastTransformer3DModel
    from fastvideo.models.loader.component_loader import TransformerLoader

    precision = "bf16" if dtype == torch.bfloat16 else "fp16"
    maybe_init_distributed_environment_and_model_parallel(1, 1)
    args = FastVideoArgs(
        model_path=str(_model_dir()),
        dit_cpu_offload=False,
        dit_layerwise_offload=False,
        use_fsdp_inference=False,
        pipeline_config=PipelineConfig(
            dit_config=LingBotWorldFastVideoConfig(),
            dit_precision=precision,
        ),
    )
    args.device = torch.device("cuda")
    model = TransformerLoader().load(str(_transformer_dir()), args).eval()
    assert isinstance(model, LingBotWorld2CausalFastTransformer3DModel)
    device_inputs = _to_device(inputs, torch.device("cuda"))
    self_cache, cross_cache = _new_caches(torch.device("cuda"), dtype, inputs["seq_len"])
    try:
        with (
            torch.inference_mode(),
            torch.amp.autocast("cuda", dtype=dtype),
            torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH),
        ):
            output = model(**device_inputs, kv_cache=self_cache, crossattn_cache=cross_cache)[0]
        return output.detach().float().cpu()
    finally:
        del model
        gc.collect()
        cleanup_dist_env_and_memory()


def test_lingbotworld_fast_checkpoint_config_matches_released_variant() -> None:
    config_path = _transformer_dir() / "config.json"
    if not config_path.exists():
        pytest.skip(f"LingBot-World-Fast transformer config not found at {config_path}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    assert config["_class_name"] == "CausalLingBotWorldTransformer3DModel"
    assert config["local_attn_size"] == -1
    assert config["num_frames_per_block"] == 3
    assert config["num_layers"] == 40
    assert config["num_attention_heads"] == 40
    assert config["attention_head_dim"] == 128


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Full LingBot-World-Fast DiT parity requires CUDA")
def test_lingbotworld_fast_full_transformer_matches_official() -> None:
    if not _reference_dir().exists():
        pytest.skip(
            "Official LingBot source is absent; set LINGBOTWORLD_FAST_REFERENCE_DIR "
            "to a Robbyant/lingbot-world checkout.")
    if not _has_real_weights():
        pytest.skip(
            "Released transformer weights are absent; set LINGBOTWORLD_FAST_MODEL_DIR "
            "to FastVideo/LingBot-World-Fast-Diffusers.")

    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    inputs = _deterministic_inputs(dtype)
    official = _run_official(inputs, dtype)
    fastvideo = _run_fastvideo(inputs, dtype)

    difference = (fastvideo - official).abs()
    relative_mean_drift = difference.mean() / official.abs().mean().clamp_min(1e-6)
    print(
        "LingBot-World-Fast full-DiT parity: "
        f"max={difference.max():.6f}, mean={difference.mean():.6f}, "
        f"relative_mean={relative_mean_drift:.6f}")
    assert relative_mean_drift < 0.05
    assert_close(fastvideo, official, atol=0.1, rtol=0.1)
