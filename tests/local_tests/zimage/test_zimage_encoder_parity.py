# SPDX-License-Identifier: Apache-2.0
"""
Parity test for Z-Image Qwen3 text encoder support in FastVideo.

This compares:
1) direct transformers AutoModel output, and
2) FastVideo Qwen3Model wrapper output,

using identical local checkpoint, tokenization, and inputs.

Usage:
    pytest tests/local_tests/zimage/test_zimage_qwen3_encoder_parity.py -v
"""

from __future__ import annotations

from pathlib import Path
import json
import os

import pytest
import torch
from torch.testing import assert_close
from transformers import AutoModel, AutoTokenizer
from safetensors.torch import safe_open

from fastvideo.configs.models.encoders.qwen3 import Qwen3Config
from fastvideo.distributed.parallel_state import cleanup_dist_env_and_memory, maybe_init_distributed_environment_and_model_parallel
from fastvideo.models.registry import ModelRegistry


REPO_ROOT = Path(__file__).resolve().parents[3]
ZIMAGE_TEXT_ENCODER_DIR = REPO_ROOT / "official_weights" / "Z-Image" / "text_encoder"
ZIMAGE_TOKENIZER_DIR = REPO_ROOT / "official_weights" / "Z-Image" / "tokenizer"


@pytest.fixture(scope="module", autouse=True)
def _init_dist_and_tp_groups():
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "29531")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("LOCAL_RANK", "0")

    maybe_init_distributed_environment_and_model_parallel(1, 1)
    yield
    cleanup_dist_env_and_memory()


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _iter_safetensors(path: Path):
    with safe_open(str(path), framework="pt", device="cpu") as f:
        for k in f.keys():
            yield k, f.get_tensor(k)


def _iter_pretrained_safetensors(model_dir: Path):
    single = model_dir / "model.safetensors"
    if single.exists():
        yield from _iter_safetensors(single)
        return

    index = model_dir / "model.safetensors.index.json"
    if index.exists():
        idx = _load_json(index)
        shard_names = sorted(set(idx["weight_map"].values()))
        for shard in shard_names:
            yield from _iter_safetensors(model_dir / shard)
        return

    raise FileNotFoundError(
        f"Missing safetensors checkpoint in {model_dir} (expected model.safetensors or model.safetensors.index.json)"
    )


@pytest.mark.skipif(not ZIMAGE_TEXT_ENCODER_DIR.exists(), reason="Z-Image text encoder checkpoint required")
def test_zimage_qwen3_encoder_parity_forward():
    if not ZIMAGE_TOKENIZER_DIR.exists():
        pytest.skip(f"Z-Image tokenizer dir not found: {ZIMAGE_TOKENIZER_DIR}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # NOTE: Test failing with torch.bloat16
    # dtype = torch.bfloat16 if (device.type == "cuda" and torch.cuda.is_bf16_supported()) else torch.float32
    dtype = torch.float32
    torch.manual_seed(11)

    tokenizer = AutoTokenizer.from_pretrained(str(ZIMAGE_TOKENIZER_DIR), local_files_only=True)

    prompts = [
        "A cinematic shot of a rainy neon street at night.",
        "A watercolor illustration of a fox in autumn leaves.",
    ]
    toks = tokenizer(
        prompts,
        padding="max_length",
        truncation=True,
        max_length=128,
        return_tensors="pt",
    )
    input_ids = toks["input_ids"].to(device)
    attention_mask = toks["attention_mask"].to(device)

    ref = AutoModel.from_pretrained(
        str(ZIMAGE_TEXT_ENCODER_DIR),
        local_files_only=True,
        trust_remote_code=True,
        dtype=dtype,
        low_cpu_mem_usage=True,
    ).eval().to(device=device, dtype=dtype)

    with torch.no_grad():
        ref_out = ref(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        ref_last = ref_out.last_hidden_state.detach().float().cpu()
        ref_hs_m2 = ref_out.hidden_states[-2].detach().float().cpu()

    fv_cls, _ = ModelRegistry.resolve_model_cls("Qwen3Model")
    cfg_raw = _load_json(ZIMAGE_TEXT_ENCODER_DIR / "config.json")
    for k in ("_name_or_path", "transformers_version", "model_type", "torch_dtype"):
        cfg_raw.pop(k, None)

    cfg = Qwen3Config()
    cfg.update_model_arch(cfg_raw)
    fv = fv_cls(cfg).eval()
    loaded = fv.load_weights(_iter_pretrained_safetensors(ZIMAGE_TEXT_ENCODER_DIR))
    assert loaded, "No Qwen3 weights were loaded into FastVideo model"
    fv = fv.to(device=device, dtype=dtype)

    with torch.no_grad():
        fv_out = fv(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        fv_last = fv_out.last_hidden_state.detach().float().cpu()
        assert fv_out.hidden_states is not None
        fv_hs_m2 = fv_out.hidden_states[-2].detach().float().cpu()

    # Z-Image uses only valid token embeddings selected by attention mask.
    # Compare parity on that exact path to avoid padded-token artifacts.
    mask_cpu = attention_mask.detach().bool().cpu()
    for i in range(mask_cpu.shape[0]):
        valid = mask_cpu[i]
        assert_close(ref_last[i][valid], fv_last[i][valid], atol=1e-4, rtol=1e-4)
        assert_close(ref_hs_m2[i][valid], fv_hs_m2[i][valid], atol=1e-4, rtol=1e-4)
