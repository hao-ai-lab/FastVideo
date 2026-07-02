# SPDX-License-Identifier: Apache-2.0
"""DreamX-World-5B autoregressive transformer parity.

Coverage scope: both. The tiny forward parity compares FastVideo's native AR DiT
against the official DreamX ``CausalWanModel`` implementation with identical
weights. The real 5B checkpoint gate strict-loads the downloaded safetensors
through the FastVideo model class.
"""
from __future__ import annotations

import os
from pathlib import Path
import sys

import pytest
import torch
from torch.testing import assert_close

from fastvideo.configs.models.dits.dreamx_world import DreamXWorldARArchConfig, DreamXWorldARConfig
from fastvideo.configs.pipelines.dreamx_world import make_dreamx_world_5b_ar_dit_config
from fastvideo.models.dits.dreamx_world_ar import DreamXWorldARTransformer3DModel
from fastvideo.models.loader.fsdp_load import load_model_from_full_model_state_dict
from fastvideo.models.loader.utils import get_param_names_mapping
from fastvideo.models.loader.weight_utils import resolve_safetensors_files, safetensors_weights_iterator

REPO_ROOT = Path(__file__).resolve().parents[3]
OFFICIAL_REF_DIR = Path(os.getenv("DREAMX_WORLD_OFFICIAL_REF_DIR", REPO_ROOT / "DreamX-World"))
CONVERTED_AR_DIR = Path(os.getenv("DREAMX_WORLD_AR_CONVERTED_DIR", "/tmp/converted_dreamx_world_ar"))
PARITY_SCOPE = "both"


def _tiny_config() -> DreamXWorldARConfig:
    return DreamXWorldARConfig(
        arch_config=DreamXWorldARArchConfig(
            num_attention_heads=1,
            attention_head_dim=8,
            in_channels=4,
            out_channels=4,
            ffn_dim=16,
            num_layers=1,
            text_dim=8,
            freq_dim=8,
            text_len=4,
            local_attn_size=2,
            sink_size=1,
            attn_compress=1,
            cam_self_attn_layers=(0,),
        ))


def _load_official_tiny():
    if not OFFICIAL_REF_DIR.exists():
        pytest.skip(f"Official DreamX reference missing: {OFFICIAL_REF_DIR}")
    sys.path.insert(0, str(OFFICIAL_REF_DIR))
    try:
        from wan.modules import attention as official_attention
        from wan.modules import causal_camera_model_2_2_prope_infinity as causal_module
        from wan.modules import model_2_2 as official_model_2_2
        from wan.modules.causal_camera_model_2_2_prope_infinity import CausalWanModel
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"Cannot import official AR transformer: {exc}")
    official_attention.FLASH_ATTN_2_AVAILABLE = False
    official_attention.FLASH_ATTN_3_AVAILABLE = False

    def _sdpa_same_dtype(q, k, v, **kwargs):
        del kwargs
        out = torch.nn.functional.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), dropout_p=0.0)
        return out.transpose(1, 2).contiguous()

    official_attention.attention = _sdpa_same_dtype
    official_attention.flash_attention = _sdpa_same_dtype
    official_model_2_2.flash_attention = _sdpa_same_dtype
    causal_module.attention = _sdpa_same_dtype
    return CausalWanModel(
        model_type="ti2v",
        patch_size=(1, 2, 2),
        text_len=4,
        in_dim=4,
        dim=8,
        ffn_dim=16,
        freq_dim=8,
        text_dim=8,
        out_dim=4,
        num_heads=1,
        num_layers=1,
        local_attn_size=2,
        sink_size=1,
        qk_norm=True,
        cross_attn_norm=True,
        add_control_adapter=True,
        cam_method="prope",
        attn_compress=1,
        cam_self_attn_layers=(0,),
    ).eval()


def _make_inputs():
    torch.manual_seed(123)
    x = [torch.randn(4, 1, 4, 4)]
    t = torch.zeros(1, 4, dtype=torch.long)
    context = [torch.randn(2, 8)]
    camera = {
        "viewmats": torch.eye(4).reshape(1, 1, 4, 4).repeat(1, 4, 1, 1),
        "K": torch.eye(3).reshape(1, 1, 3, 3).repeat(1, 4, 1, 1),
    }
    kv_cache = [{
        "k": torch.zeros(1, 8, 1, 8),
        "v": torch.zeros(1, 8, 1, 8),
        "global_end_index": torch.tensor([0]),
        "local_end_index": torch.tensor([0]),
        "prope_k": torch.zeros(1, 8, 1, 8),
        "prope_v": torch.zeros(1, 8, 1, 8),
        "prope_global_end_index": torch.tensor([0]),
        "prope_local_end_index": torch.tensor([0]),
    }]
    cross_cache = [{
        "k": torch.zeros(1, 4, 1, 8),
        "v": torch.zeros(1, 4, 1, 8),
        "is_init": False,
    }]
    return x, t, context, camera, kv_cache, cross_cache


def test_dreamx_world_ar_tiny_forward_matches_official():
    official = _load_official_tiny()
    fastvideo = DreamXWorldARTransformer3DModel(_tiny_config(), {}).eval()
    fastvideo.load_state_dict(official.state_dict(), strict=True)

    x, t, context, camera, kv_cache, cross_cache = _make_inputs()
    official_out = official(x=x, t=t, context=context, seq_len=4, y_camera=camera,
                            kv_cache=kv_cache, crossattn_cache=cross_cache).detach()
    x, t, context, camera, kv_cache, cross_cache = _make_inputs()
    fastvideo_out = fastvideo(x=x, t=t, context=context, seq_len=4, y_camera=camera,
                              kv_cache=kv_cache, crossattn_cache=cross_cache).detach()
    assert_close(fastvideo_out, official_out, atol=1e-5, rtol=1e-5)


def test_dreamx_world_ar_5b_config_matches_official_shape():
    config = make_dreamx_world_5b_ar_dit_config()
    assert config.num_layers == 30
    assert config.num_attention_heads == 24
    assert config.attention_head_dim == 128
    assert config.hidden_size == 3072
    assert config.ffn_dim == 14336
    assert config.local_attn_size == 12
    assert config.sink_size == 3
    assert config.attn_compress == 4
    assert config.cam_self_attn_layers == tuple(range(30))


def test_dreamx_world_ar_converted_5b_transformer_strict_loads():
    transformer_dir = CONVERTED_AR_DIR / "transformer"
    if not transformer_dir.exists():
        pytest.skip(f"Converted AR transformer missing: {transformer_dir}")
    with torch.device("meta"):
        model = DreamXWorldARTransformer3DModel(make_dreamx_world_5b_ar_dit_config(), {})
    incompatible = load_model_from_full_model_state_dict(
        model,
        safetensors_weights_iterator(resolve_safetensors_files(str(transformer_dir)), to_cpu=True),
        device=torch.device("cpu"),
        param_dtype=torch.bfloat16,
        strict=True,
        param_names_mapping=get_param_names_mapping(model.param_names_mapping),
        training_mode=False,
    )
    assert incompatible.missing_keys == []
    assert incompatible.unexpected_keys == []
    assert not any(param.is_meta for param in model.parameters())
