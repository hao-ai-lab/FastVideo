# SPDX-License-Identifier: Apache-2.0
"""DreamX-World Wan2.2 VAE reuse parity scaffold.

Coverage scope: implementation_subcomponent. The official side uses
AutoencoderKLWan3_8 from DreamX, while the FastVideo side targets the native
Wan VAE. This remains a scaffold until Wan2.2 base VAE weights are staged.
"""
from __future__ import annotations

import os
from pathlib import Path
import sys
import re
import types

import pytest
import torch
from omegaconf import OmegaConf
from torch.testing import assert_close

from fastvideo.configs.pipelines.dreamx_world import make_dreamx_world_5b_cam_vae_config
from fastvideo.models.vaes.wanvae import AutoencoderKLWan

REPO_ROOT = Path(__file__).resolve().parents[3]
OFFICIAL_REF_DIR = Path(os.getenv("DREAMX_WORLD_OFFICIAL_REF_DIR", REPO_ROOT / "DreamX-World"))
WAN_BASE_DIR = Path(os.getenv("DREAMX_WORLD_WAN_BASE_DIR", REPO_ROOT / "official_weights" / "Wan2.2-TI2V-5B"))
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


def _map_residual_subkey(prefix: str, sub: str) -> str | None:
    if sub == "residual.0.gamma":
        return f"{prefix}.norm1.gamma"
    match = re.match(r"^residual\.2\.(weight|bias)$", sub)
    if match:
        return f"{prefix}.conv1.{match.group(1)}"
    if sub == "residual.3.gamma":
        return f"{prefix}.norm2.gamma"
    match = re.match(r"^residual\.6\.(weight|bias)$", sub)
    if match:
        return f"{prefix}.conv2.{match.group(1)}"
    match = re.match(r"^shortcut\.(weight|bias)$", sub)
    if match:
        return f"{prefix}.conv_shortcut.{match.group(1)}"
    return None


def _map_attention_subkey(prefix: str, sub: str) -> str | None:
    if sub == "norm.gamma":
        return f"{prefix}.norm.gamma"
    match = re.match(r"^(to_qkv|proj)\.(weight|bias)$", sub)
    if match:
        return f"{prefix}.{match.group(1)}.{match.group(2)}"
    return None


def _map_resample_subkey(prefix: str, sub: str) -> str | None:
    match = re.match(r"^resample\.1\.(weight|bias)$", sub)
    if match:
        return f"{prefix}.resample.1.{match.group(1)}"
    match = re.match(r"^time_conv\.(weight|bias)$", sub)
    if match:
        return f"{prefix}.time_conv.{match.group(1)}"
    return None


def _map_dreamx_raw_vae_key(key: str) -> str | None:
    match = re.match(r"^(conv1|conv2)\.(weight|bias)$", key)
    if match:
        prefix = "quant_conv" if match.group(1) == "conv1" else "post_quant_conv"
        return f"{prefix}.{match.group(2)}"
    match = re.match(r"^(encoder|decoder)\.conv1\.(weight|bias)$", key)
    if match:
        return f"{match.group(1)}.conv_in.{match.group(2)}"
    match = re.match(r"^(encoder|decoder)\.head\.0\.gamma$", key)
    if match:
        return f"{match.group(1)}.norm_out.gamma"
    match = re.match(r"^(encoder|decoder)\.head\.2\.(weight|bias)$", key)
    if match:
        return f"{match.group(1)}.conv_out.{match.group(2)}"
    match = re.match(r"^(encoder|decoder)\.middle\.0\.(.*)$", key)
    if match:
        return _map_residual_subkey(f"{match.group(1)}.mid_block.resnets.0", match.group(2))
    match = re.match(r"^(encoder|decoder)\.middle\.1\.(.*)$", key)
    if match:
        return _map_attention_subkey(f"{match.group(1)}.mid_block.attentions.0", match.group(2))
    match = re.match(r"^(encoder|decoder)\.middle\.2\.(.*)$", key)
    if match:
        return _map_residual_subkey(f"{match.group(1)}.mid_block.resnets.1", match.group(2))
    match = re.match(r"^encoder\.downsamples\.(\d+)\.downsamples\.(\d+)\.(.*)$", key)
    if match:
        stage = int(match.group(1))
        block = int(match.group(2))
        sub = match.group(3)
        if block in (0, 1):
            return _map_residual_subkey(f"encoder.down_blocks.{stage}.resnets.{block}", sub)
        if block == 2:
            return _map_resample_subkey(f"encoder.down_blocks.{stage}.downsampler", sub)
        return None
    match = re.match(r"^decoder\.upsamples\.(\d+)\.upsamples\.(\d+)\.(.*)$", key)
    if match:
        stage = int(match.group(1))
        block = int(match.group(2))
        sub = match.group(3)
        if block in (0, 1, 2):
            return _map_residual_subkey(f"decoder.up_blocks.{stage}.resnets.{block}", sub)
        if block == 3:
            return _map_resample_subkey(f"decoder.up_blocks.{stage}.upsampler", sub)
        return None
    return None


def _vae_kwargs():
    config = OmegaConf.load(OFFICIAL_REF_DIR / "configs" / "wan2.2" / "wan_ti2v_5b.yaml")
    return OmegaConf.to_container(config["vae_kwargs"])


def _load_official_vae(device, dtype):
    _add_official_to_path()
    _install_xfuser_stub()
    vae_path = WAN_BASE_DIR / "Wan2.2_VAE.pth"
    if not vae_path.exists():
        pytest.skip(f"Wan2.2 base VAE weights missing: {vae_path}")
    try:
        from models import AutoencoderKLWan3_8
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"Cannot import official DreamX VAE: {exc}")
    model = AutoencoderKLWan3_8.from_pretrained(str(vae_path), additional_kwargs=_vae_kwargs())
    return model.to(device=device, dtype=dtype).eval()


def _load_fastvideo_vae(device, dtype):
    vae_path = WAN_BASE_DIR / "Wan2.2_VAE.pth"
    if not vae_path.exists():
        pytest.skip(f"Wan2.2 raw VAE weights missing: {vae_path}")
    config = make_dreamx_world_5b_cam_vae_config()
    config.load_encoder = True
    config.load_decoder = True
    model = AutoencoderKLWan(config).to(device=device, dtype=dtype)
    raw_state = torch.load(str(vae_path), map_location="cpu", weights_only=True)
    mapped_state = {}
    for key, value in raw_state.items():
        mapped_key = _map_dreamx_raw_vae_key(key)
        if mapped_key is None:
            raise AssertionError(f"Unmapped DreamX raw VAE key: {key}")
        mapped_state[mapped_key] = value
    model.load_state_dict(mapped_state, strict=True)
    return model.eval()


def _normalize_fastvideo_vae_latent(latent: torch.Tensor) -> torch.Tensor:
    config = make_dreamx_world_5b_cam_vae_config()
    mean = torch.tensor(config.latents_mean, device=latent.device, dtype=latent.dtype).view(1, -1, 1, 1, 1)
    std = torch.tensor(config.latents_std, device=latent.device, dtype=latent.dtype).view(1, -1, 1, 1, 1)
    return (latent - mean) / std


def test_dreamx_world_vae_config_matches_wan22_shape():
    config = make_dreamx_world_5b_cam_vae_config()
    assert config.z_dim == 48
    assert config.in_channels == 12
    assert config.out_channels == 12
    assert config.base_dim == 160
    assert config.decoder_base_dim == 256
    assert config.scale_factor_temporal == 4
    assert config.scale_factor_spatial == 16
    assert config.patch_size == 2
    assert config.is_residual is True
    assert config.clip_output is False
    assert len(config.latents_mean) == 48
    assert len(config.latents_std) == 48


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for VAE parity.")
def test_dreamx_world_vae_encode_parity_scaffold():
    device = torch.device("cuda:0")
    dtype = torch.bfloat16
    official = _load_official_vae(device, dtype)
    fastvideo = _load_fastvideo_vae(device, dtype)
    torch.manual_seed(123)
    video = torch.randn(1, 3, 5, 64, 64, device=device, dtype=dtype).clamp(-1, 1)
    with torch.inference_mode():
        official_latent = official.encode(video).latent_dist.mean.float().cpu()
        fastvideo_latent = _normalize_fastvideo_vae_latent(fastvideo.encode(video).mean).float().cpu()
    assert official_latent.shape == fastvideo_latent.shape
    assert_close(fastvideo_latent, official_latent, atol=5e-2, rtol=5e-2)
