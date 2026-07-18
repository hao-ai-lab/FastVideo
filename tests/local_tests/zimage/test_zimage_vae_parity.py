# SPDX-License-Identifier: Apache-2.0
"""
Parity test for Z-Image VAE (AutoencoderKL decode path).

This compares FastVideo's AutoencoderKL wrapper against the Z-Image
reference AutoencoderKL implementation using the same local weights.

Usage:
    pytest tests/local_tests/zimage/test_zimage_vae_parity.py -v
"""

from __future__ import annotations

import importlib
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import load_file as safetensors_load_file
from torch.testing import assert_close

from fastvideo.configs.models.vaes.autoencoder_kl import AutoencoderKLVAEConfig
from fastvideo.models.loader.component_loader import VAELoader
from fastvideo.models.vaes.autoencoder_kl import AutoencoderKL as FastVideoAutoencoderKL


REPO_ROOT = Path(__file__).resolve().parents[3]
ZIMAGE_REPO = REPO_ROOT / "Z-Image"
ZIMAGE_SRC = REPO_ROOT / "Z-Image" / "src"
ZIMAGE_VAE_DIR = REPO_ROOT / "official_weights" / "Z-Image" / "vae"
ZIMAGE_VAE_CFG = ZIMAGE_VAE_DIR / "config.json"
ZIMAGE_VAE_WEIGHTS = ZIMAGE_VAE_DIR / "diffusion_pytorch_model.safetensors"
ZIMAGE_REFERENCE_REVISION = "26f23eda626ffadda020b04ff79488e1d72004cd"
ZIMAGE_VAE_SCALING_FACTOR = 0.3611
ZIMAGE_VAE_SHIFT_FACTOR = 0.1159
PARITY_SCOPE = "both"


def _require_pinned_reference_module(module_name: str, source_file: Path):
    if not ZIMAGE_REPO.exists():
        pytest.skip(f"Pinned Z-Image reference clone not found: {ZIMAGE_REPO}")
    if not source_file.is_file():
        pytest.fail(f"Z-Image reference clone is incomplete; missing {source_file}")

    try:
        result = subprocess.run(
            ["git", "-C", str(ZIMAGE_REPO), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        pytest.fail(f"Cannot verify Z-Image reference revision: {exc}")

    actual_revision = result.stdout.strip()
    assert actual_revision == ZIMAGE_REFERENCE_REVISION, (
        "Z-Image reference clone is not at the pinned revision: "
        f"expected {ZIMAGE_REFERENCE_REVISION}, got {actual_revision}"
    )

    if str(ZIMAGE_SRC) not in sys.path:
        sys.path.insert(0, str(ZIMAGE_SRC))
    try:
        module = importlib.import_module(module_name)
    except Exception as exc:
        pytest.fail(f"Cannot import pinned Z-Image module {module_name}: {exc}")

    module_file = Path(module.__file__ or "").resolve()
    assert module_file.is_relative_to(ZIMAGE_SRC.resolve()), (
        f"{module_name} resolved outside the pinned clone: {module_file}"
    )
    return module


@pytest.fixture(scope="module")
def reference_autoencoder_cls():
    module = _require_pinned_reference_module(
        "zimage.autoencoder",
        ZIMAGE_SRC / "zimage" / "autoencoder.py",
    )
    return module.AutoencoderKL


def _load_cfg() -> dict:
    if not ZIMAGE_VAE_CFG.exists():
        pytest.skip(f"Z-Image VAE config not found: {ZIMAGE_VAE_CFG}")
    with ZIMAGE_VAE_CFG.open("r", encoding="utf-8") as f:
        cfg = json.load(f)
    cfg.pop("_class_name", None)
    cfg.pop("_diffusers_version", None)
    cfg.pop("_name_or_path", None)
    return cfg


def _load_weights() -> dict[str, torch.Tensor]:
    if not ZIMAGE_VAE_WEIGHTS.exists():
        pytest.skip(f"Z-Image VAE weights not found: {ZIMAGE_VAE_WEIGHTS}")
    return safetensors_load_file(str(ZIMAGE_VAE_WEIGHTS), device="cpu")


def _build_reference(reference_autoencoder_cls, cfg: dict, weights: dict[str, torch.Tensor]) -> torch.nn.Module:
    ref = reference_autoencoder_cls(
        in_channels=cfg["in_channels"],
        out_channels=cfg["out_channels"],
        down_block_types=tuple(cfg["down_block_types"]),
        up_block_types=tuple(cfg["up_block_types"]),
        block_out_channels=tuple(cfg["block_out_channels"]),
        layers_per_block=cfg["layers_per_block"],
        latent_channels=cfg["latent_channels"],
        norm_num_groups=cfg["norm_num_groups"],
        scaling_factor=cfg["scaling_factor"],
        shift_factor=cfg["shift_factor"],
        use_quant_conv=cfg["use_quant_conv"],
        use_post_quant_conv=cfg["use_post_quant_conv"],
        mid_block_add_attention=cfg["mid_block_add_attention"],
    ).eval()
    ref.load_state_dict(weights, strict=True)
    return ref


def _build_fastvideo(cfg: dict, weights: dict[str, torch.Tensor]) -> torch.nn.Module:
    fv_cfg = AutoencoderKLVAEConfig()
    fv_cfg.update_model_arch(cfg)
    fv = FastVideoAutoencoderKL(fv_cfg).eval()
    fv.load_state_dict(weights, strict=True)
    return fv


def _load_fastvideo_production_vae(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        "fastvideo.models.loader.component_loader.get_local_torch_device",
        lambda: torch.device("cpu"),
    )
    pipeline_config = SimpleNamespace(
        vae_config=AutoencoderKLVAEConfig(),
        vae_precision="fp32",
    )
    fastvideo_args = SimpleNamespace(
        model_paths={},
        pipeline_config=pipeline_config,
        vae_cpu_offload=False,
    )
    vae = VAELoader().load(str(ZIMAGE_VAE_DIR), fastvideo_args)
    assert isinstance(vae, FastVideoAutoencoderKL)
    assert fastvideo_args.model_paths["vae"] == str(ZIMAGE_VAE_DIR)
    return vae


def test_zimage_vae_decode_parity(reference_autoencoder_cls):
    torch.manual_seed(7)

    cfg = _load_cfg()
    weights = _load_weights()

    ref = _build_reference(reference_autoencoder_cls, cfg, weights)
    fv = _build_fastvideo(cfg, weights)

    # Decode-only parity is the critical path for Z-Image pipeline integration.
    latents = torch.randn(1, cfg["latent_channels"], 8, 8, dtype=torch.float32)

    with torch.no_grad():
        ref_out = ref.decode(latents, return_dict=False)[0].detach().float()
        fv_out = fv.decode(latents, return_dict=False)[0].detach().float()

    assert ref_out.shape == fv_out.shape
    assert_close(ref_out, fv_out, atol=1e-4, rtol=1e-4)


def test_zimage_vae_production_loader_and_raw_decode_parity(
    reference_autoencoder_cls,
    monkeypatch: pytest.MonkeyPatch,
):
    torch.manual_seed(17)

    cfg = _load_cfg()
    weights = _load_weights()
    assert cfg["scaling_factor"] == ZIMAGE_VAE_SCALING_FACTOR
    assert cfg["shift_factor"] == ZIMAGE_VAE_SHIFT_FACTOR
    ref = _build_reference(reference_autoencoder_cls, cfg, weights)
    fv = _load_fastvideo_production_vae(monkeypatch)

    assert ref.config.scaling_factor == cfg["scaling_factor"]
    assert fv.config.scaling_factor == cfg["scaling_factor"]
    assert ref.config.shift_factor == cfg["shift_factor"]
    assert fv.config.shift_factor == cfg["shift_factor"]

    decode_latents = torch.randn(1, cfg["latent_channels"], 8, 8, dtype=torch.float32)

    with torch.no_grad():
        ref_out = ref.decode(decode_latents, return_dict=False)[0].detach().float()
        fv_out = fv.decode(decode_latents, return_dict=False)[0].detach().float()

    assert ref_out.shape == fv_out.shape
    assert_close(ref_out, fv_out, atol=1e-4, rtol=1e-4)
