# SPDX-License-Identifier: Apache-2.0
"""Helios tokenizer and exact-checkpoint UMT5-XXL reuse parity.

Coverage scope: both. The tokenizer test exercises FastVideo's production
third-party loader boundary. The encoder test runs the official and native
implementations sequentially so two UMT5-XXL copies do not occupy one GPU.
"""

from __future__ import annotations

import gc
import json
from pathlib import Path
import subprocess
import sys

import pytest
import torch
from torch.testing import assert_close
from transformers import AutoTokenizer, UMT5EncoderModel as OfficialUMT5EncoderModel

from fastvideo.configs.models.encoders import T5Config
from fastvideo.configs.models.encoders.t5 import T5ArchConfig
from fastvideo.models.encoders.t5 import UMT5EncoderModel
from fastvideo.models.loader.weight_utils import (
    resolve_safetensors_files,
    safetensors_weights_iterator,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
MODEL_DIR = REPO_ROOT / "official_weights" / "helios"
TEXT_ENCODER_DIR = MODEL_DIR / "text_encoder"
TOKENIZER_DIR = MODEL_DIR / "tokenizer"
PARITY_SCOPE = "both"


def _require_assets() -> None:
    required = (
        TEXT_ENCODER_DIR / "model.safetensors.index.json",
        TOKENIZER_DIR / "tokenizer.json",
    )
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        pytest.skip(f"Helios text assets missing: {missing}")


def _native_config() -> T5Config:
    return T5Config(
        arch_config=T5ArchConfig(
            architectures=["UMT5EncoderModel"],
            vocab_size=256384,
            d_model=4096,
            d_kv=64,
            d_ff=10240,
            num_layers=24,
            num_decoder_layers=24,
            num_heads=64,
            relative_attention_num_buckets=32,
            relative_attention_max_distance=128,
            dropout_rate=0.1,
            layer_norm_epsilon=1e-6,
            feed_forward_proj="gated-gelu",
            is_encoder_decoder=True,
            use_cache=True,
            text_len=512,
        ),
        prefix="umt5",
    )


def _patch_single_process_text_parallel(monkeypatch) -> None:
    import fastvideo.layers.linear as fastvideo_linear
    import fastvideo.layers.vocab_parallel_embedding as fastvideo_embedding
    import fastvideo.models.encoders.t5 as fastvideo_t5

    for module in (fastvideo_t5, fastvideo_embedding, fastvideo_linear):
        if hasattr(module, "get_tp_rank"):
            monkeypatch.setattr(module, "get_tp_rank", lambda: 0)
        if hasattr(module, "get_tp_world_size"):
            monkeypatch.setattr(module, "get_tp_world_size", lambda: 1)
    monkeypatch.setattr(fastvideo_embedding, "tensor_model_parallel_all_reduce", lambda value: value)


def test_helios_tokenizer_loader_matches_official_assets() -> None:
    _require_assets()
    official = AutoTokenizer.from_pretrained(str(TOKENIZER_DIR), local_files_only=True)
    prompts = [
        "A glass sculpture turning slowly in a quiet studio.",
        "海面上缓慢升起的清晨薄雾。",
    ]
    kwargs = {
        "padding": "max_length",
        "truncation": True,
        "max_length": 32,
        "return_tensors": "pt",
    }
    official_batch = official(prompts, **kwargs)
    script = f"""
import json
from types import SimpleNamespace
from fastvideo.models.loader.component_loader import TokenizerLoader

args = SimpleNamespace(
    pipeline_config=SimpleNamespace(text_encoder_configs=()),
    trust_remote_code=False,
)
tokenizer = TokenizerLoader().load({str(TOKENIZER_DIR)!r}, args)
batch = tokenizer({prompts!r}, padding='max_length', truncation=True,
                  max_length=32, return_tensors='pt')
print(json.dumps({{
    'input_ids': batch.input_ids.tolist(),
    'attention_mask': batch.attention_mask.tolist(),
}}))
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    fastvideo_batch = json.loads(result.stdout.strip().splitlines()[-1])
    assert fastvideo_batch["input_ids"] == official_batch.input_ids.tolist()
    assert fastvideo_batch["attention_mask"] == official_batch.attention_mask.tolist()


def test_helios_text_encoder_config_matches_umt5_xxl() -> None:
    config = _native_config()
    assert config.vocab_size == 256384
    assert config.d_model == 4096
    assert config.d_kv == 64
    assert config.d_ff == 10240
    assert config.num_layers == config.num_decoder_layers == 24
    assert config.num_heads == 64
    assert config.feed_forward_proj == "gated-gelu"
    assert config.text_len == 512


def test_helios_pipeline_config_reuses_verified_components() -> None:
    try:
        from fastvideo.configs.pipelines.helios import HeliosPipelineConfig
    except ImportError as exc:
        raise AssertionError("HeliosPipelineConfig has not been implemented yet") from exc

    config = HeliosPipelineConfig()
    assert config.dit_config.__class__.__name__ == "HeliosConfig"
    assert config.vae_config.__class__.__name__ == "WanVAEConfig"
    assert config.vae_config.load_encoder is False
    assert config.vae_config.load_decoder is True
    assert config.text_encoder_precisions == ("bf16", )
    assert config.dit_precision == "bf16"
    assert config.vae_decode_precision == "fp32"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for UMT5 parity.")
@pytest.mark.parametrize(
    ("dtype", "atol", "rtol", "mean_limit"),
    [
        pytest.param(torch.float32, 1e-4, 1e-4, 1e-5, id="fp32-math"),
        pytest.param(torch.bfloat16, 3e-2, 3e-2, 3e-3, id="bf16-runtime"),
    ],
)
def test_helios_umt5_hidden_state_parity(
    monkeypatch,
    dtype: torch.dtype,
    atol: float,
    rtol: float,
    mean_limit: float,
) -> None:
    _require_assets()
    device = torch.device("cuda:0")
    tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER_DIR), local_files_only=True)
    batch = tokenizer(
        ["A precise macro shot of frost forming on a red leaf."],
        padding="max_length",
        truncation=True,
        max_length=32,
        return_tensors="pt",
    )
    input_ids = batch.input_ids.to(device)
    attention_mask = batch.attention_mask.to(device)

    official = (OfficialUMT5EncoderModel.from_pretrained(str(TEXT_ENCODER_DIR),
                                                         local_files_only=True,
                                                         torch_dtype=dtype).to(device).eval())
    with torch.inference_mode():
        official_hidden = official(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state.float().cpu()
    del official
    gc.collect()
    torch.cuda.empty_cache()

    _patch_single_process_text_parallel(monkeypatch)
    native = UMT5EncoderModel(_native_config()).to(device=device, dtype=dtype)
    files = resolve_safetensors_files(str(TEXT_ENCODER_DIR))
    loaded = native.load_weights(safetensors_weights_iterator(files, to_cpu=True))
    missing = {name for name, _ in native.named_parameters()} - loaded
    assert missing == set()
    native.eval()
    with torch.inference_mode():
        fastvideo_hidden = native(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state.float().cpu()

    assert official_hidden.shape == fastvideo_hidden.shape
    diff = (official_hidden - fastvideo_hidden).abs()
    print(f"UMT5 dtype={dtype} diff_max={diff.max().item():.8f} diff_mean={diff.mean().item():.8f}")
    assert diff.mean().item() < mean_limit
    assert_close(fastvideo_hidden, official_hidden, atol=atol, rtol=rtol)
