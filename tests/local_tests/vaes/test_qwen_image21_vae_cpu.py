# SPDX-License-Identifier: Apache-2.0
"""Dependency-light CPU contracts for the native Qwen-Image-2.1 VAE.

Only logging and the CLI argparse action are replaced during imports. The
production architecture, config bases, and numerical code execute unchanged.
These tests exercise tiny random weights and are separate from weight parity.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import sys
from dataclasses import asdict
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch
from torch.testing import assert_close

REPO_ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture(scope="module")
def _native_modules():
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    with pytest.MonkeyPatch.context() as patch:
        for name in (
            "fastvideo", "fastvideo.configs", "fastvideo.configs.models", "fastvideo.configs.models.vaes",
            "fastvideo.models", "fastvideo.models.vaes",
        ):
            package = ModuleType(name)
            package.__path__ = []
            patch.setitem(sys.modules, name, package)
        logger = ModuleType("fastvideo.logger")
        logger.init_logger = logging.getLogger
        patch.setitem(sys.modules, logger.__name__, logger)
        utilities = ModuleType("fastvideo.utils")
        utilities.StoreBoolean = argparse.Action
        patch.setitem(sys.modules, utilities.__name__, utilities)

        def load(name, relative_path):
            spec = importlib.util.spec_from_file_location(name, REPO_ROOT / relative_path)
            module = importlib.util.module_from_spec(spec)
            patch.setitem(sys.modules, name, module)
            spec.loader.exec_module(module)
            return module

        load("fastvideo.configs.models.base", "fastvideo/configs/models/base.py")
        load("fastvideo.configs.models.vaes.base", "fastvideo/configs/models/vaes/base.py")
        config = load("fastvideo.configs.models.vaes.qwen_image21", "fastvideo/configs/models/vaes/qwen_image21.py")
        model = load("fastvideo.models.vaes.qwen_image21", "fastvideo/models/vaes/qwen_image21.py")
        yield SimpleNamespace(config=config, model=model)
    torch.set_num_threads(previous_threads)


def _tiny_config(modules, **overrides):
    arch = modules.config.QwenImage21VAEArchConfig(
        base_dim=4,
        decoder_base_dim=4,
        z_dim=4,
        num_res_blocks=1,
        latents_mean=(0.2, -1.5, 2.0, 0.9),
        latents_std=(2.0, 0.5, 1.25, 3.0),
    )
    return modules.config.QwenImage21VAEConfig(arch_config=arch, **overrides)


def _tiny_model(modules, **overrides):
    with torch.random.fork_rng():
        torch.manual_seed(21)
        return modules.model.AutoencoderKLQwenImage21(_tiny_config(modules, **overrides)).eval()


@pytest.mark.parametrize("tiled", [False, True])
def test_qwen_image21_vae_rgba_encode_decode_cpu(_native_modules, tiled):
    model = _tiny_model(_native_modules)
    if tiled:
        model.enable_tiling(
            tile_sample_min_height=32,
            tile_sample_min_width=32,
            tile_sample_stride_height=16,
            tile_sample_stride_width=16,
        )
    pixels = torch.rand((2, 4, 1, 48, 64), generator=torch.Generator().manual_seed(21)).mul_(2).sub_(1)
    with torch.inference_mode():
        posterior = model.encode(pixels).latent_dist
        latent = posterior.mode()
        decoded = model.decode(latent).sample
        reconstructed = model(pixels, return_dict=False)[0]
    assert latent.shape == (2, 4, 1, 3, 4)
    assert latent.data_ptr() == posterior.mean.data_ptr()
    assert decoded.shape == pixels.shape
    assert decoded.dtype == torch.float32
    assert torch.isfinite(latent).all() and torch.isfinite(decoded).all()
    assert decoded.min() >= -1 and decoded.max() <= 1
    assert_close(decoded, reconstructed, atol=0, rtol=0)


def test_qwen_image21_vae_slicing_and_posterior_cpu(_native_modules):
    model = _tiny_model(_native_modules)
    pixels = torch.rand((2, 4, 1, 32, 48), generator=torch.Generator().manual_seed(22)).mul_(2).sub_(1)
    with torch.inference_mode():
        expected = model.encode(pixels).latent_dist
        expected_pixels = model.decode(expected.mode()).sample
        model.enable_slicing()
        actual = model.encode(pixels, return_dict=False)[0]
        actual_pixels = model.decode(actual.mode(), return_dict=False)[0]
    assert_close(actual.mean, expected.mean, atol=2e-6, rtol=2e-5)
    assert_close(actual.logvar, expected.logvar, atol=2e-6, rtol=2e-5)
    assert_close(actual_pixels, expected_pixels, atol=2e-6, rtol=2e-5)
    first = actual.sample(generator=torch.Generator().manual_seed(23))
    second = actual.sample(generator=torch.Generator().manual_seed(23))
    assert_close(first, second, atol=0, rtol=0)
    assert not torch.equal(first, actual.mode())


def test_qwen_image21_vae_latent_normalization_cpu(_native_modules):
    model = _tiny_model(_native_modules)
    latents = torch.randn((2, 4, 1, 2, 3), generator=torch.Generator().manual_seed(24))
    actual = model.normalize_latents(latents)
    for channel in range(4):
        expected = (latents[:, channel] - model.config.latents_mean[channel]) / model.config.latents_std[channel]
        assert_close(actual[:, channel], expected, atol=0, rtol=0)
    assert_close(model.denormalize_latents(actual), latents, atol=5e-7, rtol=1e-6)
    production = _native_modules.config.QwenImage21VAEArchConfig()
    assert production.in_channels == production.out_channels == 4
    assert production.z_dim == 64 and len(production.latents_mean) == len(production.latents_std) == 64
    assert production.spatial_compression_ratio == 16 and production.temporal_compression_ratio == 1


@pytest.mark.parametrize("shape", [(1, 3, 1, 32, 32), (1, 4, 2, 32, 32), (1, 4, 1, 31, 32)])
def test_qwen_image21_vae_rejects_invalid_images_cpu(_native_modules, shape):
    model = _tiny_model(_native_modules)
    with pytest.raises(ValueError):
        model.encode(torch.zeros(shape))
    with pytest.raises(ValueError, match="must have shape"):
        model.decode(torch.zeros((1, 3, 1, 2, 2)))


def test_qwen_image21_vae_rejects_invalid_tiles_cpu(_native_modules):
    model = _tiny_model(_native_modules)
    with pytest.raises(ValueError, match="stride <= size"):
        model.enable_tiling(tile_sample_min_height=32, tile_sample_stride_height=48)
    with pytest.raises(ValueError, match="spatial-scale multiples"):
        model.enable_tiling(tile_sample_min_height=31, tile_sample_stride_height=16)
    with pytest.raises(ValueError, match="latent mean/std"):
        _native_modules.config.QwenImage21VAEArchConfig(z_dim=4)


def test_qwen_image21_vae_checkpoint_strictness_cpu(_native_modules, tmp_path, monkeypatch):
    model = _tiny_model(_native_modules)
    checkpoint = model.state_dict()
    config_json = asdict(model.config.arch_config)
    (tmp_path / "config.json").write_text(json.dumps(config_json))
    (tmp_path / "diffusion_pytorch_model.safetensors").touch()

    # Replace serialization IO so this test needs only torch and pytest; the
    # production from_pretrained parser and strict state loading still run.
    package = ModuleType("safetensors")
    package.__path__ = []
    serialization = ModuleType("safetensors.torch")
    serialization.load_file = lambda filename: checkpoint
    monkeypatch.setitem(sys.modules, "safetensors", package)
    monkeypatch.setitem(sys.modules, "safetensors.torch", serialization)
    loaded = model.from_pretrained(tmp_path, config=_tiny_config(_native_modules))
    for key, value in loaded.state_dict().items():
        assert_close(value, checkpoint[key], atol=0, rtol=0)
    checkpoint = {key: value for key, value in checkpoint.items() if key != "quant_conv.weight"}
    with pytest.raises(RuntimeError, match="Missing key"):
        model.from_pretrained(tmp_path, config=_tiny_config(_native_modules))
