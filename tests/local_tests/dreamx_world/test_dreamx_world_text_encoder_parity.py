# SPDX-License-Identifier: Apache-2.0
"""DreamX-World Wan T5 encoder reuse parity scaffold.

Coverage scope: implementation_subcomponent. It records the official
WanT5EncoderModel loading path and FastVideo T5 target for later activation
with staged Wan2.2 base text encoder/tokenizer weights.
"""
from __future__ import annotations

import os
from pathlib import Path
import sys
import types

import pytest
import torch
from omegaconf import OmegaConf
from torch.testing import assert_close
from transformers import AutoTokenizer

from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.configs.pipelines.dreamx_world import (
    DreamXWorld5BCamPipelineConfig,
    make_dreamx_world_5b_cam_text_encoder_config,
)
from fastvideo.models.loader.component_loader import TextEncoderLoader

REPO_ROOT = Path(__file__).resolve().parents[3]
OFFICIAL_REF_DIR = Path(os.getenv("DREAMX_WORLD_OFFICIAL_REF_DIR", REPO_ROOT / "DreamX-World"))
WAN_BASE_DIR = Path(os.getenv("DREAMX_WORLD_WAN_BASE_DIR", REPO_ROOT / "official_weights" / "Wan2.2-TI2V-5B"))
WAN_DIFFUSERS_DIR = Path(os.getenv("DREAMX_WORLD_WAN_DIFFUSERS_DIR", REPO_ROOT / "official_weights" / "Wan2.2-TI2V-5B-Diffusers"))
PARITY_SCOPE = "implementation_subcomponent"


def _add_official_to_path():
    if not OFFICIAL_REF_DIR.exists():
        pytest.skip(f"Official reference missing: {OFFICIAL_REF_DIR}")
    if str(OFFICIAL_REF_DIR) not in sys.path:
        sys.path.insert(0, str(OFFICIAL_REF_DIR))


def _install_xfuser_stub() -> None:
    if "xfuser" in sys.modules:
        return
    xfuser = types.ModuleType("xfuser")
    core = types.ModuleType("xfuser.core")
    distributed = types.ModuleType("xfuser.core.distributed")
    long_ctx = types.ModuleType("xfuser.core.long_ctx_attention")
    distributed.get_sequence_parallel_rank = lambda: 0
    distributed.get_sequence_parallel_world_size = lambda: 1
    distributed.get_sp_group = lambda: None
    distributed.get_world_group = lambda: types.SimpleNamespace(local_rank=0, rank=0)
    distributed.init_distributed_environment = lambda *args, **kwargs: None
    distributed.initialize_model_parallel = lambda *args, **kwargs: None
    distributed.model_parallel_is_initialized = lambda: False

    class XFuserLongContextAttention:
        def __call__(self, *args, **kwargs):
            raise RuntimeError("xfuser stub cannot execute attention")

    long_ctx.xFuserLongContextAttention = XFuserLongContextAttention
    sys.modules.update({
        "xfuser": xfuser,
        "xfuser.core": core,
        "xfuser.core.distributed": distributed,
        "xfuser.core.long_ctx_attention": long_ctx,
    })


def _text_kwargs():
    config = OmegaConf.load(OFFICIAL_REF_DIR / "configs" / "wan2.2" / "wan_ti2v_5b.yaml")
    return OmegaConf.to_container(config["text_encoder_kwargs"])


def _patch_single_process_text_parallel(monkeypatch):
    import fastvideo.layers.linear as fastvideo_linear
    import fastvideo.layers.vocab_parallel_embedding as fastvideo_embedding
    import fastvideo.models.encoders.t5 as fastvideo_t5

    for module in (fastvideo_t5, fastvideo_embedding, fastvideo_linear):
        if hasattr(module, "get_tp_rank"):
            monkeypatch.setattr(module, "get_tp_rank", lambda: 0)
        if hasattr(module, "get_tp_world_size"):
            monkeypatch.setattr(module, "get_tp_world_size", lambda: 1)
    monkeypatch.setattr(fastvideo_embedding, "tensor_model_parallel_all_reduce", lambda x: x)


def _load_official_text_encoder(device, dtype):
    _add_official_to_path()
    _install_xfuser_stub()
    text_path = WAN_BASE_DIR / "models_t5_umt5-xxl-enc-bf16.pth"
    if not text_path.exists():
        pytest.skip(f"Wan2.2 text encoder weights missing: {text_path}")
    try:
        from models import WanT5EncoderModel
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"Cannot import official DreamX text encoder: {exc}")
    model = WanT5EncoderModel.from_pretrained(
        str(text_path), additional_kwargs=_text_kwargs(), low_cpu_mem_usage=True, torch_dtype=dtype
    )
    return model.to(device=device, dtype=dtype).eval()


def _load_fastvideo_text_encoder(device, dtype, monkeypatch):
    text_encoder_path = WAN_DIFFUSERS_DIR / "text_encoder"
    if not text_encoder_path.exists():
        pytest.skip(f"Wan2.2 Diffusers text encoder missing: {text_encoder_path}")
    _patch_single_process_text_parallel(monkeypatch)
    pipeline_config = DreamXWorld5BCamPipelineConfig()
    pipeline_config.text_encoder_configs[0]._fsdp_shard_conditions = []
    args = FastVideoArgs(
        model_path=str(text_encoder_path),
        pipeline_config=pipeline_config,
        text_encoder_cpu_offload=(device.type == "cpu"),
    )
    args.model_paths = {}
    return TextEncoderLoader().load(str(text_encoder_path), args).to(device=device, dtype=dtype).eval()


def test_dreamx_world_text_encoder_config_matches_umt5_xxl_shape():
    config = make_dreamx_world_5b_cam_text_encoder_config()
    assert config.vocab_size == 256384
    assert config.d_model == 4096
    assert config.d_kv == 64
    assert config.d_ff == 10240
    assert config.num_heads == 64
    assert config.num_layers == 24
    assert config.relative_attention_num_buckets == 32
    assert config.dropout_rate == 0.0
    assert config.text_len == 512
    assert config.prefix == "umt5"


def test_dreamx_world_fastvideo_text_encoder_loads_staged_weights(monkeypatch):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = _load_fastvideo_text_encoder(device, torch.bfloat16, monkeypatch)
    assert model.__class__.__name__ == "UMT5EncoderModel"
    assert next(model.parameters()).device.type == device.type
    assert next(model.parameters()).dtype == torch.bfloat16


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for text encoder parity.")
def test_dreamx_world_text_encoder_parity_scaffold(monkeypatch):
    device = torch.device("cuda:0")
    dtype = torch.bfloat16
    official = _load_official_text_encoder(device, dtype)
    fastvideo = _load_fastvideo_text_encoder(device, dtype, monkeypatch)
    tokenizer_path = WAN_BASE_DIR / "google" / "umt5-xxl"
    if not tokenizer_path.exists():
        pytest.skip(f"Wan2.2 tokenizer missing: {tokenizer_path}")
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path))
    batch = tokenizer(["A quiet forest trail at sunrise."], padding="max_length", max_length=512, return_tensors="pt")
    input_ids = batch.input_ids.to(device)
    attention_mask = batch.attention_mask.to(device)
    with torch.inference_mode():
        official_hidden = official(input_ids, attention_mask=attention_mask)[0].float().cpu()
        fastvideo_hidden = fastvideo(input_ids, attention_mask=attention_mask).last_hidden_state.float().cpu()
    assert official_hidden.shape == fastvideo_hidden.shape
    assert_close(fastvideo_hidden, official_hidden, atol=1e-3, rtol=1e-3)
