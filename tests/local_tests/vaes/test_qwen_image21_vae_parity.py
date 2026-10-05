# SPDX-License-Identifier: Apache-2.0
"""Qwen-Image-2.1 VAE parity against a pinned Diffusers checkout.

Random-weight CPU tests cover the image-specific residual paths and tiling.
The production-loader gate requires one CUDA GPU and user-provided weights.
"""

from __future__ import annotations

import gc
import importlib
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from PIL import Image
from torch.testing import assert_close

REPO_ROOT = Path(__file__).resolve().parents[3]
REFERENCE_REVISION = "578c9b2c6636ab2424a0e56186268b83623656b2"
REFERENCE_ROOT = Path(os.environ.get("QWEN_IMAGE21_DIFFUSERS_DIR", REPO_ROOT / "official_reference" / "diffusers"))
MODEL_ROOT = Path(os.environ.get("QWEN_IMAGE21_MODEL_ROOT", REPO_ROOT / "official_weights" / "Qwen-Image-2.1"))
RUN_ENV = "QWEN_IMAGE21_RUN_VAE_PARITY"
PARITY_SCOPE = "both"


def _reference_class(required: bool = False):
    if not (REFERENCE_ROOT / "src").is_dir():
        message = f"set QWEN_IMAGE21_DIFFUSERS_DIR to Diffusers checkout at {REFERENCE_REVISION}"
        if required:
            pytest.fail(message, pytrace=False)
        pytest.skip(message)
    revision = subprocess.check_output(["git", "-C", str(REFERENCE_ROOT), "rev-parse", "HEAD"], text=True).strip()
    assert revision == REFERENCE_REVISION, f"reference revision mismatch: {revision}"
    sys.path.insert(0, str(REFERENCE_ROOT / "src"))
    try:
        module = importlib.import_module("diffusers.models.autoencoders.autoencoder_kl_qwenimage21")
    except ImportError as exc:
        if required:
            pytest.fail(f"cannot import Qwen-Image-2.1 reference: {exc}", pytrace=False)
        pytest.skip(f"reference dependencies unavailable: {exc}")
    expected_source = REFERENCE_ROOT / "src" / "diffusers" / "models" / "autoencoders"
    assert Path(module.__file__).resolve().parent == expected_source.resolve()
    return module.AutoencoderKLQwenImage21


def _tiny_config():
    from fastvideo.configs.models.vaes.qwen_image21 import QwenImage21VAEArchConfig, QwenImage21VAEConfig

    return QwenImage21VAEConfig(
        arch_config=QwenImage21VAEArchConfig(
            base_dim=4,
            decoder_base_dim=4,
            z_dim=4,
            dim_mult=(1, 2, 4, 8, 8),
            num_res_blocks=1,
            latents_mean=(0.0,) * 4,
            latents_std=(1.0,) * 4,
        ),
        use_tiling=False,
    )


@pytest.fixture
def _tiny_pair():
    official_cls = _reference_class()
    from fastvideo.models.vaes.qwen_image21 import AutoencoderKLQwenImage21

    config = _tiny_config()
    arch = config.arch_config
    official = official_cls(
        base_dim=arch.base_dim,
        decoder_base_dim=arch.decoder_base_dim,
        z_dim=arch.z_dim,
        dim_mult=list(arch.dim_mult),
        num_res_blocks=arch.num_res_blocks,
        temperal_downsample=list(arch.temperal_downsample),
        latents_mean=list(arch.latents_mean),
        latents_std=list(arch.latents_std),
    ).eval()
    native = AutoencoderKLQwenImage21(config).eval()
    official_state = official.state_dict()
    assert {key: value.shape for key, value in native.state_dict().items()} == {
        key: value.shape for key, value in official_state.items()
    }
    native.load_state_dict(official_state, strict=True)
    return official, native


@pytest.mark.parametrize("tiled", [False, True])
def test_qwen_image21_vae_random_cpu_parity(_tiny_pair, tiled):
    official, native = _tiny_pair
    if tiled:
        for model in (official, native):
            model.enable_tiling(
                tile_sample_min_height=32,
                tile_sample_min_width=32,
                tile_sample_stride_height=16,
                tile_sample_stride_width=16,
            )
    generator = torch.Generator().manual_seed(21)
    pixels = torch.rand((2, 4, 1, 48, 64), generator=generator).mul_(2).sub_(1)
    with torch.inference_mode():
        expected = official.encode(pixels).latent_dist
        actual = native.encode(pixels).latent_dist
        assert_close(actual.mean, expected.mean, atol=2e-6, rtol=2e-5)
        assert_close(actual.logvar, expected.logvar, atol=2e-6, rtol=2e-5)
        expected_pixels = official.decode(expected.mode()).sample
        actual_pixels = native.decode(expected.mode()).sample
    assert_close(actual_pixels, expected_pixels, atol=2e-6, rtol=2e-5)
    assert actual_pixels.shape == pixels.shape


def test_qwen_image21_vae_slicing_and_output_contract(_tiny_pair):
    official, native = _tiny_pair
    for model in (official, native):
        model.enable_slicing()
    pixels = torch.rand((2, 4, 1, 32, 48), generator=torch.Generator().manual_seed(22)).mul_(2).sub_(1)
    with torch.inference_mode():
        expected = official.encode(pixels, return_dict=False)[0].mode()
        actual = native.encode(pixels, return_dict=False)[0].mode()
        assert_close(actual, expected, atol=2e-6, rtol=2e-5)
        assert_close(
            native.decode(actual, return_dict=False)[0],
            official.decode(expected, return_dict=False)[0],
            atol=2e-6,
            rtol=2e-5,
        )


def test_qwen_image21_vae_production_loader_parity():
    if os.environ.get(RUN_ENV) != "1":
        pytest.skip(f"set {RUN_ENV}=1 on an allocated CUDA node")
    if not torch.cuda.is_available():
        pytest.fail("Qwen-Image-2.1 VAE production parity requires CUDA", pytrace=False)
    official_cls = _reference_class(required=True)
    component_path = MODEL_ROOT / "vae"
    if not (component_path / "config.json").is_file() or not list(component_path.glob("*.safetensors")):
        pytest.fail(f"local VAE checkpoint missing at {component_path}", pytrace=False)

    from fastvideo.configs.pipelines.qwen_image21 import QwenImage21PipelineConfig
    from fastvideo.models.loader.component_loader import VAELoader

    device = torch.device("cuda:0")
    args = SimpleNamespace(
        pipeline_config=QwenImage21PipelineConfig(),
        model_paths={},
        vae_cpu_offload=False,
    )
    args.pipeline_config.vae_precision = "fp32"
    args.pipeline_config.vae_config.use_tiling = False
    native = VAELoader().load(str(component_path), args)
    assert all(parameter.dtype == torch.float32 for parameter in native.parameters())
    generator = torch.Generator().manual_seed(23)
    pixels = torch.rand((1, 4, 1, 64, 80), generator=generator).mul_(2).sub_(1).to(device)
    with torch.inference_mode():
        posterior = native.encode(pixels).latent_dist
        actual_mean = posterior.mean.cpu()
        actual_logvar = posterior.logvar.cpu()
        actual_pixels = native.decode(posterior.mode()).sample.cpu()
    del native, posterior
    gc.collect()
    torch.cuda.empty_cache()

    official, info = official_cls.from_pretrained(
        component_path,
        local_files_only=True,
        torch_dtype=torch.float32,
        output_loading_info=True,
    )
    assert not {key: info.get(key) for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")
                if info.get(key)}
    official = official.to(device).eval()
    with torch.inference_mode():
        expected = official.encode(pixels).latent_dist
        assert_close(actual_mean, expected.mean.cpu(), atol=1e-5, rtol=1e-4)
        assert_close(actual_logvar, expected.logvar.cpu(), atol=1e-5, rtol=1e-4)
        assert_close(actual_pixels, official.decode(expected.mode()).sample.cpu(), atol=1e-5, rtol=1e-4)


@pytest.mark.parametrize("height,width,tiled", [(64, 80, False), (288, 320, True)])
def test_qwen_image21_vae_bf16_reference_layout_parity(height, width, tiled):
    """Independent preprocessing must preserve singleton strides for BF16 kernels."""
    if os.environ.get(RUN_ENV) != "1":
        pytest.skip(f"set {RUN_ENV}=1 on an allocated CUDA node")
    if not torch.cuda.is_available():
        pytest.fail("Qwen-Image-2.1 BF16 VAE production parity requires CUDA", pytrace=False)
    official_cls = _reference_class(required=True)
    component_path = MODEL_ROOT / "vae"
    if not (component_path / "config.json").is_file() or not list(component_path.glob("*.safetensors")):
        pytest.fail(f"local VAE checkpoint missing at {component_path}", pytrace=False)

    from diffusers.image_processor import VaeImageProcessor
    from diffusers.pipelines.qwenimage21.pipeline_qwenimage21 import QwenImage21Pipeline
    from fastvideo.configs.pipelines.qwen_image21 import QwenImage21PipelineConfig
    from fastvideo.models.loader.component_loader import VAELoader
    from fastvideo.pipelines.basic.qwen_image21.inputs import normalize_latents, pack_latents, reference_pixels

    rgb = torch.randint(0, 256, (height, width, 3), generator=torch.Generator().manual_seed(23), dtype=torch.uint8)
    image = Image.fromarray(rgb.numpy()).convert("RGBA")
    processor = VaeImageProcessor(vae_scale_factor=16, do_convert_rgb=False)
    native_pixels = reference_pixels(image)
    official_pixels = processor.preprocess(image, height=height, width=width).unsqueeze(2)
    assert_close(native_pixels, official_pixels, atol=0, rtol=0)
    # Size-one strides can select a different BF16 convolution kernel even
    # though both tensors contain identical values and are channels_last_3d.
    assert native_pixels.stride() == official_pixels.stride()
    device = torch.device("cuda:0")
    native_pixels = native_pixels.to(device=device, dtype=torch.bfloat16)
    official_pixels = official_pixels.to(device=device, dtype=torch.bfloat16)
    assert native_pixels.stride() == official_pixels.stride()

    args = SimpleNamespace(pipeline_config=QwenImage21PipelineConfig(), model_paths={}, vae_cpu_offload=False)
    args.pipeline_config.vae_config.use_tiling = tiled
    native = VAELoader().load(str(component_path), args)
    assert all(parameter.dtype == torch.bfloat16 for parameter in native.parameters())
    with torch.no_grad():
        posterior = native.encode(native_pixels).latent_dist
        actual_mean, actual_logvar = posterior.mean.cpu(), posterior.logvar.cpu()
        normalized = normalize_latents(posterior.mode(), native.config.latents_mean, native.config.latents_std)
        actual_packed = pack_latents(normalized).cpu()
    del native, posterior, normalized
    gc.collect()
    torch.cuda.empty_cache()

    official, info = official_cls.from_pretrained(
        component_path, local_files_only=True, torch_dtype=torch.bfloat16, output_loading_info=True)
    assert not {key: info.get(key) for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")
                if info.get(key)}
    official = official.to(device).eval()
    if tiled:
        official.enable_tiling()
    with torch.no_grad():
        expected = official.encode(official_pixels).latent_dist
        assert_close(actual_mean, expected.mean.cpu(), atol=1e-5, rtol=1e-4)
        assert_close(actual_logvar, expected.logvar.cpu(), atol=1e-5, rtol=1e-4)
        mean = torch.tensor(official.config.latents_mean).view(1, official.config.z_dim, 1, 1, 1)
        std = torch.tensor(official.config.latents_std).view(1, official.config.z_dim, 1, 1, 1)
        normalized = (expected.mode() - mean.to(device, torch.bfloat16)) / std.to(device, torch.bfloat16)
        expected_packed = QwenImage21Pipeline._pack_latents(
            normalized, 1, official.config.z_dim, height // 16, width // 16).cpu()
        assert_close(actual_packed, expected_packed, atol=1e-5, rtol=1e-4)
