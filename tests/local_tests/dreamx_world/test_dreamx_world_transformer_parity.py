# SPDX-License-Identifier: Apache-2.0
"""DreamX-World transformer parity scaffold.

Coverage scope: both. The official side loads DreamX-World-5B-Cam through
Wan2_2Transformer3DModel.from_pretrained with PRoPE camera control enabled.
The FastVideo side strict-loads the converted DreamX transformer weights into
the native DreamX-World DiT implementation.
"""
from __future__ import annotations

import math
import os
from pathlib import Path
import sys
import types

import pytest
import torch
from omegaconf import OmegaConf
from torch.testing import assert_close

from fastvideo.forward_context import set_forward_context
from fastvideo.configs.models.dits.dreamx_world import (
    DreamXWorldArchConfig, DreamXWorldConfig)
from fastvideo.pipelines.basic.dreamx_world.camera_conditioning import build_dreamx_camera_condition
from fastvideo.configs.pipelines.dreamx_world import make_dreamx_world_5b_cam_dit_config
from fastvideo.platforms import AttentionBackendEnum
from fastvideo.models.dits.dreamx_world import (
    DreamXPropeSelfAttention, DreamXWorldTransformer3DModel,
    DreamXWorldTransformerBlock)
from fastvideo.models.loader.fsdp_load import load_model_from_full_model_state_dict
from fastvideo.models.loader.utils import get_param_names_mapping
from fastvideo.models.loader.weight_utils import resolve_safetensors_files, safetensors_weights_iterator
from scripts.checkpoint_conversion.dreamx_world_to_diffusers import map_transformer_key

REPO_ROOT = Path(__file__).resolve().parents[3]
OFFICIAL_REF_DIR = Path(os.getenv("DREAMX_WORLD_OFFICIAL_REF_DIR", REPO_ROOT / "DreamX-World"))
LOCAL_WEIGHTS_DIR = Path(os.getenv("DREAMX_WORLD_LOCAL_WEIGHTS_DIR", REPO_ROOT / "official_weights" / "dreamx_world"))
WAN_BASE_DIR = Path(os.getenv("DREAMX_WORLD_WAN_BASE_DIR", REPO_ROOT / "official_weights" / "Wan2.2-TI2V-5B"))
CONVERTED_WEIGHTS_DIR = Path(os.getenv("DREAMX_WORLD_CONVERTED_WEIGHTS_DIR", REPO_ROOT / "converted_weights" / "dreamx_world"))
PARITY_SCOPE = "both"


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


def _make_tiny_dreamx_config() -> DreamXWorldConfig:
    return DreamXWorldConfig(
        arch_config=DreamXWorldArchConfig(
            num_attention_heads=1,
            attention_head_dim=8,
            in_channels=16,
            out_channels=16,
            ffn_dim=32,
            num_layers=1,
            cross_attn_norm=True,
            qk_norm="rms_norm_across_heads",
            add_control_adapter=True,
            cam_method="prope",
            attn_compress=1,
            cam_self_attn_layers=None,
        ))


def _add_official_to_path() -> None:
    if not OFFICIAL_REF_DIR.exists():
        pytest.skip(f"Official reference missing: {OFFICIAL_REF_DIR}")
    if str(OFFICIAL_REF_DIR) not in sys.path:
        sys.path.insert(0, str(OFFICIAL_REF_DIR))


def _official_transformer_kwargs() -> dict:
    config_path = OFFICIAL_REF_DIR / "configs" / "wan2.2" / "wan_ti2v_5b.yaml"
    if not config_path.exists():
        pytest.skip(f"DreamX Wan config missing: {config_path}")
    config = OmegaConf.load(config_path)
    kwargs = OmegaConf.to_container(config["transformer_additional_kwargs"])
    kwargs["cam_method"] = "prope"
    kwargs["add_control_adapter"] = True
    return kwargs


def _load_official_transformer(device: torch.device, dtype: torch.dtype) -> torch.nn.Module:
    _add_official_to_path()
    _install_xfuser_stub()
    if not LOCAL_WEIGHTS_DIR.exists():
        pytest.skip(f"DreamX transformer weights missing: {LOCAL_WEIGHTS_DIR}")
    try:
        from models import Wan2_2Transformer3DModel
    except Exception as exc:  # noqa: BLE001 - local parity should skip missing refs.
        pytest.skip(f"Cannot import official DreamX transformer: {exc}")
    model = Wan2_2Transformer3DModel.from_pretrained(
        str(LOCAL_WEIGHTS_DIR),
        transformer_additional_kwargs=_official_transformer_kwargs(),
        torch_dtype=dtype,
    )
    return model.to(device=device, dtype=dtype).eval()


def _load_fastvideo_transformer(device: torch.device, dtype: torch.dtype) -> torch.nn.Module:
    model = _load_fastvideo_transformer_strict(torch.device("cpu"), dtype)
    return model.to(device=device, dtype=dtype).eval()


def _load_fastvideo_transformer_strict(device: torch.device, dtype: torch.dtype) -> torch.nn.Module:
    transformer_dir = CONVERTED_WEIGHTS_DIR / "transformer"
    if not transformer_dir.exists():
        pytest.skip(f"Converted DreamX transformer missing: {transformer_dir}")
    safetensors_files = resolve_safetensors_files(str(transformer_dir))
    config = make_dreamx_world_5b_cam_dit_config()
    import fastvideo.models.dits.dreamx_world as fastvideo_dreamx
    original_get_sp_world_size = fastvideo_dreamx.get_sp_world_size
    fastvideo_dreamx.get_sp_world_size = lambda: 1
    try:
        with torch.device("meta"):
            model = DreamXWorldTransformer3DModel(config=config, hf_config={})
    finally:
        fastvideo_dreamx.get_sp_world_size = original_get_sp_world_size
    incompatible = load_model_from_full_model_state_dict(
        model,
        safetensors_weights_iterator(safetensors_files, to_cpu=True),
        device=device,
        param_dtype=dtype,
        strict=True,
        param_names_mapping=get_param_names_mapping(model.param_names_mapping),
        training_mode=False,
    )
    assert incompatible.missing_keys == []
    assert incompatible.unexpected_keys == []
    assert not any(param.is_meta for param in model.parameters())
    return model.to(device=device, dtype=dtype).eval()


def _make_inputs(device: torch.device, dtype: torch.dtype):
    torch.manual_seed(1234)
    num_frames = 5
    height = 64
    width = 64
    latent_frames = (num_frames - 1) // 4 + 1
    latent_h = height // 16
    latent_w = width // 16
    x = torch.randn(1, 48, latent_frames, latent_h, latent_w, device=device, dtype=dtype)
    context = [torch.randn(16, 4096, device=device, dtype=dtype)]
    seq_len = math.ceil((latent_h * latent_w) / 4 * latent_frames)
    timestep = torch.full((1, seq_len), 250, device=device, dtype=torch.long)
    camera = build_dreamx_camera_condition(
        ["w"], [4], num_frames=num_frames, height=height, width=width, dtype=dtype, device=device
    )
    camera = {key: value.unsqueeze(0) for key, value in camera.items()}
    return {"x": [x[0]], "context": context, "t": timestep, "seq_len": seq_len, "y_camera": camera}


def _run_official(model: torch.nn.Module, inputs: dict) -> torch.Tensor:
    with torch.inference_mode():
        output = model(**inputs)
    if isinstance(output, list):
        output = torch.stack(output, dim=0)
    assert torch.is_tensor(output), f"official output is not a tensor: {type(output)}"
    return output.detach().float().cpu()


def _run_fastvideo(model: torch.nn.Module, inputs: dict) -> torch.Tensor:
    hidden_states = torch.stack(inputs["x"], dim=0)
    encoder_hidden_states = torch.stack([
        torch.cat([inputs["context"][0], inputs["context"][0].new_zeros(512 - inputs["context"][0].shape[0], 4096)])
    ])
    with torch.inference_mode(), set_forward_context(current_timestep=0, attn_metadata=None):
        output = model(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            timestep=inputs["t"],
            y_camera=inputs["y_camera"],
        )
    assert torch.is_tensor(output), f"FastVideo output is not a tensor: {type(output)}"
    return output.detach().float().cpu()


def test_dreamx_world_conversion_mapping_strict_load_smoke(monkeypatch):
    _add_official_to_path()
    _install_xfuser_stub()
    try:
        from models import Wan2_2Transformer3DModel
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"Cannot import official DreamX transformer: {exc}")

    official = Wan2_2Transformer3DModel(
        dim=8,
        ffn_dim=32,
        num_heads=1,
        num_layers=1,
        add_control_adapter=True,
        cam_method="prope",
    )
    official_state = official.state_dict()
    diffusers_like_state = {
        map_transformer_key(key): value.detach().clone()
        for key, value in official_state.items()
    }

    import fastvideo.models.dits.dreamx_world as fastvideo_dreamx
    monkeypatch.setattr(fastvideo_dreamx, "get_sp_world_size", lambda: 1)
    with torch.device("meta"):
        fastvideo = DreamXWorldTransformer3DModel(config=_make_tiny_dreamx_config(), hf_config={})

    incompatible = load_model_from_full_model_state_dict(
        fastvideo,
        iter(diffusers_like_state.items()),
        device=torch.device("cpu"),
        param_dtype=torch.float32,
        strict=True,
        param_names_mapping=get_param_names_mapping(fastvideo.param_names_mapping),
        training_mode=False,
    )
    assert incompatible.missing_keys == []
    assert incompatible.unexpected_keys == []
    assert not any(param.is_meta for param in fastvideo.parameters())


def test_dreamx_world_5b_cam_dit_config_matches_official_shape():
    config = make_dreamx_world_5b_cam_dit_config()
    assert config.num_layers == 30
    assert config.num_attention_heads == 24
    assert config.attention_head_dim == 128
    assert config.hidden_size == 3072
    assert config.ffn_dim == 14336
    assert config.add_control_adapter is True
    assert config.cam_method == "prope"
    assert config.attn_compress == 1


def test_dreamx_world_converted_5b_transformer_strict_loads():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    _load_fastvideo_transformer_strict(device, torch.bfloat16)


def test_dreamx_world_fastvideo_prope_branch_smoke():
    block = DreamXWorldTransformerBlock(
        8,
        32,
        1,
        cross_attn_norm=True,
        add_control_adapter=True,
        cam_method="prope",
        attn_compress=1,
        layer_idx=0,
        supported_attention_backends=(AttentionBackendEnum.TORCH_SDPA,),
    )
    assert block.cam_self_attn is not None
    assert [
        name for name, _ in block.named_parameters()
        if name.startswith("cam_self_attn.")
    ][:8] == [
        "cam_self_attn.q_proj.weight",
        "cam_self_attn.q_proj.bias",
        "cam_self_attn.k_proj.weight",
        "cam_self_attn.k_proj.bias",
        "cam_self_attn.v_proj.weight",
        "cam_self_attn.v_proj.bias",
        "cam_self_attn.out_proj.weight",
        "cam_self_attn.out_proj.bias",
    ]

    module = DreamXPropeSelfAttention(
        dim=8,
        attn_dim=8,
        num_heads=1,
        qk_norm="rms_norm_across_heads",
    ).eval()
    assert module.num_heads == 1
    assert module.head_dim == 8
    assert tuple(module.out_proj.weight.shape) == (8, 8)
    assert torch.count_nonzero(module.out_proj.weight) == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for transformer parity.")
def test_dreamx_world_transformer_parity_scaffold(monkeypatch):
    device = torch.device("cuda:0")
    dtype = torch.float32
    inputs = _make_inputs(device, dtype)

    official = _load_official_transformer(device, dtype)
    official_out = _run_official(official, inputs)
    del official
    torch.cuda.empty_cache()

    import fastvideo.attention.layer as attention_layer
    import fastvideo.distributed.communication_op as communication_op
    import fastvideo.models.dits.dreamx_world as fastvideo_dreamx
    monkeypatch.setattr(attention_layer, "get_sp_parallel_rank", lambda: 0)
    monkeypatch.setattr(attention_layer, "get_sp_world_size", lambda: 1)
    monkeypatch.setattr(attention_layer, "sequence_model_parallel_all_to_all_4D", lambda tensor, scatter_dim=2, gather_dim=1: tensor)
    monkeypatch.setattr(attention_layer, "sequence_model_parallel_all_gather", lambda tensor, dim=-1: tensor)
    monkeypatch.setattr(fastvideo_dreamx, "get_sp_world_size", lambda: 1)
    monkeypatch.setattr(communication_op, "get_sp_world_size", lambda: 1)
    monkeypatch.setattr(fastvideo_dreamx, "sequence_model_parallel_shard", lambda tensor, dim=1: (tensor, tensor.shape[dim]))
    monkeypatch.setattr(
        fastvideo_dreamx,
        "sequence_model_parallel_all_gather_with_unpad",
        lambda tensor, original_seq_len, dim=1: tensor.narrow(dim, 0, original_seq_len),
    )
    fastvideo = _load_fastvideo_transformer(device, dtype)
    fastvideo_out = _run_fastvideo(fastvideo, inputs)
    assert official_out.shape == fastvideo_out.shape
    diff = (official_out - fastvideo_out).abs()
    print(f"diff_max={diff.max().item():.6f} diff_mean={diff.mean().item():.6f}")
    assert_close(fastvideo_out, official_out, atol=1e-1, rtol=1e-1)
