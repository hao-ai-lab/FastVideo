# SPDX-License-Identifier: Apache-2.0
"""Qwen-Image-2.1 transformer reference and production-loader parity.

Set QWEN_IMAGE21_DIFFUSERS_DIR to Diffusers revision below for random-weight
CPU parity. Real-weight BF16 parity additionally needs an allocated CUDA GPU,
QWEN_IMAGE21_MODEL_ROOT, and QWEN_IMAGE21_RUN_DIT_PARITY=1. Explicitly requested
real-weight tests fail rather than skip when their prerequisites are missing.
Set QWEN_IMAGE21_DIT_PARITY_REPORT to save block and prediction diagnostics as JSON.
"""

from __future__ import annotations

import gc
import importlib
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch.testing import assert_close

ROOT = Path(__file__).resolve().parents[3]
REFERENCE_REVISION = "578c9b2c6636ab2424a0e56186268b83623656b2"
REFERENCE_ROOT = Path(os.environ.get("QWEN_IMAGE21_DIFFUSERS_DIR", ROOT / "official_reference/diffusers"))
MODEL_ROOT = Path(os.environ.get("QWEN_IMAGE21_MODEL_ROOT", ROOT / "official_weights/Qwen-Image-2.1"))
RUN_ENV = "QWEN_IMAGE21_RUN_DIT_PARITY"
PARITY_SCOPE = "both"


def _numerical_metrics(actual, expected):
    actual, expected = actual.detach().float().cpu(), expected.detach().float().cpu()
    difference = actual - expected
    reference_rms = expected.square().mean().sqrt()
    reference_abs_mean = expected.abs().mean()
    rmse = difference.square().mean().sqrt()
    mean_abs = difference.abs().mean()
    floor = torch.finfo(torch.float32).tiny
    return {
        "shape": list(actual.shape),
        "finite": bool(torch.isfinite(actual).all() and torch.isfinite(expected).all()),
        "max_abs": difference.abs().max().item(),
        "mean_abs": mean_abs.item(),
        "rmse": rmse.item(),
        "reference_rms": reference_rms.item(),
        "reference_abs_mean": reference_abs_mean.item(),
        "relative_l2": (rmse / reference_rms.clamp_min(floor)).item(),
        "relative_abs_mean": (mean_abs / reference_abs_mean.clamp_min(floor)).item(),
        "total_elements": actual.numel(),
    }


def _modality_summaries(actual, expected, masks):
    actual, expected = actual.detach().cpu(), expected.detach().cpu()
    summaries = {"global": _numerical_metrics(actual, expected)}
    for name, mask in masks.items():
        assert mask.numel() == actual.shape[1], f"{name} token mask does not match the joint sequence"
        if bool(mask.any()):
            summaries[name] = _numerical_metrics(actual[:, mask], expected[:, mask])
    return summaries


def _reference_module(required=False):
    if not (REFERENCE_ROOT / "src").is_dir():
        message = f"Set QWEN_IMAGE21_DIFFUSERS_DIR to Diffusers checkout at {REFERENCE_REVISION}"
        if required:
            pytest.fail(message, pytrace=False)
        pytest.skip(message)
    revision = subprocess.check_output(["git", "-C", str(REFERENCE_ROOT), "rev-parse", "HEAD"], text=True).strip()
    assert revision == REFERENCE_REVISION, f"Reference revision mismatch: {revision}"
    sys.path.insert(0, str(REFERENCE_ROOT / "src"))
    try:
        module = importlib.import_module("diffusers.models.transformers.transformer_qwenimage21")
    except ImportError as exc:
        if required:
            pytest.fail(f"Qwen-Image-2.1 reference dependencies unavailable: {exc}", pytrace=False)
        pytest.skip(f"Qwen-Image-2.1 reference dependencies unavailable: {exc}")
    expected = REFERENCE_ROOT / "src/diffusers/models/transformers"
    assert Path(module.__file__).resolve().parent == expected.resolve()
    return module


def _inputs(channels=4, text_dim=12, device="cpu", dtype=torch.float32, batch=2):
    generator = torch.Generator().manual_seed(21)
    image = torch.randn(batch, 12, channels, generator=generator).to(device=device, dtype=dtype)
    text = torch.randn(batch, 5, text_dim, generator=generator).to(device=device, dtype=dtype)
    return dict(hidden_states=image,
                encoder_hidden_states=text,
                timestep=torch.full((batch,), 0.65, device=device, dtype=dtype),
                img_shapes=[[(1, 2, 2), (1, 2, 2), (1, 2, 2)]] * batch,
                img_mask=torch.tensor([[False, True, True, False, False, True]] * batch, device=device),
                encoder_hidden_states_mask=torch.tensor([[True, True, True, True, False]] * batch, device=device))


@pytest.fixture
def tiny_pair():
    reference = _reference_module()
    from fastvideo.configs.models.dits.qwen_image21 import QwenImage21ArchConfig, QwenImage21Config
    from fastvideo.models.dits.qwen_image21 import QwenImage21Transformer2DModel

    kwargs = dict(in_channels=4, out_channels=4, num_layers=2, attention_head_dim=8,
                  num_attention_heads=2, context_in_dim=12, mlp_ratio=3, axes_dims_rope=(2, 2, 4))
    torch.manual_seed(20)
    official = reference.QwenImage21Transformer2DModel(**kwargs).eval()
    native = QwenImage21Transformer2DModel(QwenImage21Config(arch_config=QwenImage21ArchConfig(**kwargs)), {}).eval()
    official_state = official.state_dict()
    assert {key: value.shape for key, value in native.state_dict().items()} == {
        key: value.shape for key, value in official_state.items()
    }
    native.load_state_dict(official_state, strict=True)
    return reference, official, native


def test_qwen_image21_dit_random_reference_parity(tiny_pair):
    _, official, native = tiny_pair
    inputs = _inputs()
    with torch.inference_mode():
        expected = official(**inputs, return_dict=False)[0]
        actual = native(**inputs)
    assert_close(actual, expected, atol=2e-6, rtol=2e-5)


def test_qwen_image21_qk_norm_dtype_reference_parity():
    reference = _reference_module()
    from fastvideo.models.dits.qwen_image21 import QwenImage21RMSNorm

    for weight_dtype in (torch.float32, torch.bfloat16, torch.float16):
        for input_dtype in (torch.float32, torch.bfloat16, torch.float16):
            official = reference.RMSNorm(8, eps=1e-6).to(weight_dtype)
            native = QwenImage21RMSNorm(8, eps=1e-6).to(weight_dtype)
            weight = torch.randn(8, generator=torch.Generator().manual_seed(23)).to(weight_dtype)
            official.weight.data.copy_(weight)
            native.weight.data.copy_(weight)
            hidden = torch.randn(2, 7, 3, 8, generator=torch.Generator().manual_seed(24)).to(input_dtype)
            assert_close(native(hidden), official(hidden), atol=0, rtol=0)


@pytest.mark.parametrize("storage_device", [None, "cpu"])
def test_qwen_image21_dit_reference_cache_parity(tiny_pair, storage_device):
    reference, official, native = tiny_pair
    from fastvideo.models.dits.qwen_image21 import QwenImage21KVCache

    inputs = _inputs()
    official_cache = reference.QwenImage21KVCache(2)
    native_cache = QwenImage21KVCache(2, storage_device)
    with torch.inference_mode():
        expected = official(**inputs, kv_cache=official_cache, kv_cache_mode="extract", return_dict=False)[0]
        actual = native(**inputs, kv_cache=native_cache, kv_cache_mode="extract")
        assert_close(actual, expected, atol=2e-6, rtol=2e-5)
        for layer_idx in range(2):
            expected_k, expected_v = official_cache.get_layer(layer_idx).get()
            actual_k, actual_v = native_cache.get_layer(layer_idx).get()
            assert_close(actual_k, expected_k, atol=2e-6, rtol=2e-5)
            assert_close(actual_v, expected_v, atol=2e-6, rtol=2e-5)
        inputs["timestep"].fill_(0.3)
        inputs["hidden_states"][:, -4:] *= 0.8
        expected = official(**inputs, kv_cache=official_cache, kv_cache_mode="cached", return_dict=False)[0]
        actual = native(**inputs, kv_cache=native_cache, kv_cache_mode="cached")
    assert_close(actual, expected, atol=2e-6, rtol=2e-5)


def test_qwen_image21_dit_production_loader_parity():
    if os.environ.get(RUN_ENV) != "1":
        pytest.skip(f"Set {RUN_ENV}=1 on an allocated CUDA node")
    if not torch.cuda.is_available():
        pytest.fail("Qwen-Image-2.1 DiT production-loader parity requires CUDA", pytrace=False)
    reference = _reference_module(required=True)
    component = MODEL_ROOT / "transformer"
    if not (component / "config.json").is_file() or not list(component.glob("*.safetensors")):
        pytest.fail(f"Local transformer checkpoint missing at {component}", pytrace=False)

    from fastvideo.configs.pipelines.qwen_image21 import QwenImage21PipelineConfig
    from fastvideo.models.dits.qwen_image21 import QwenImage21KVCache
    from fastvideo.models.loader.component_loader import TransformerLoader

    args = SimpleNamespace(pipeline_config=QwenImage21PipelineConfig(), model_paths={},
                           override_transformer_cls_name=None, hsdp_replicate_dim=1, hsdp_shard_dim=1,
                           dit_cpu_offload=False, pin_cpu_memory=False, use_fsdp_inference=False,
                           training_mode=False, enable_torch_compile=False, torch_compile_kwargs={},
                           inference_torch_compile=False, VSA_tile_size=None, inference_mode=True,
                           dit_layerwise_offload=False)
    native = TransformerLoader().load(str(component), args)
    inputs = _inputs(channels=native.config.in_channels, text_dim=native.config.context_in_dim,
                     device="cuda:0", dtype=torch.bfloat16, batch=1)
    cached_inputs = dict(inputs)
    cached_inputs["timestep"] = inputs["timestep"] * 0.5
    actual_layers = []
    handles = [block.register_forward_hook(lambda module, args, output: actual_layers.append(output.float().cpu()))
               for block in native.transformer_blocks]
    cache = QwenImage21KVCache(len(native.transformer_blocks), "cpu")
    with torch.inference_mode():
        actual = native(**inputs, kv_cache=cache, kv_cache_mode="extract").float().cpu()
        for handle in handles:
            handle.remove()
        actual_cached = native(**cached_inputs, kv_cache=cache, kv_cache_mode="cached").float().cpu()
    del native, cache
    gc.collect()
    torch.cuda.empty_cache()

    official, info = reference.QwenImage21Transformer2DModel.from_pretrained(component, local_files_only=True,
                                                                            torch_dtype=torch.bfloat16,
                                                                            output_loading_info=True)
    assert not {key: info.get(key) for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")
                if info.get(key)}
    official = official.to("cuda:0").eval()
    expected_layers = []
    handles = [block.register_forward_hook(lambda module, args, output: expected_layers.append(output.float().cpu()))
               for block in official.transformer_blocks]
    official_cache = reference.QwenImage21KVCache(len(official.transformer_blocks))
    with torch.inference_mode():
        expected = official(**inputs, kv_cache=official_cache, kv_cache_mode="extract", return_dict=False)[0]
        for handle in handles:
            handle.remove()
        expected_cached = official(**cached_inputs, kv_cache=official_cache, kv_cache_mode="cached",
                                   return_dict=False)[0]

    # Image slots expand into four unpatched latent tokens; the input masks
    # independently identify text, reference images, target image and padding.
    slots = inputs["img_mask"][0].bool().cpu()
    repeats = torch.where(slots, 4, 1)
    image_tokens = slots.repeat_interleave(repeats)
    target_tokens = math.prod(inputs["img_shapes"][0][-1])
    prefix = torch.arange(image_tokens.numel()) < image_tokens.numel() - target_tokens
    prompt_valid = inputs["encoder_hidden_states_mask"][0].bool().cpu()
    slot_valid = torch.cat((prompt_valid, torch.ones(slots.numel() - prompt_valid.numel(), dtype=torch.bool)))
    text_valid = slot_valid.repeat_interleave(repeats)
    masks = {
        "text_prefix": ~image_tokens & text_valid & prefix,
        "image_tokens": image_tokens,
        "reference_images": image_tokens & prefix,
        "target_image": image_tokens & ~prefix,
        "padding": ~image_tokens & ~text_valid & prefix,
    }
    report = {
        "component": "qwen_image21_transformer",
        "checkpoint_revision": MODEL_ROOT.name,
        "reference_revision": REFERENCE_REVISION,
        "torch_version": torch.__version__,
        "atol": 3e-3,
        "rtol": 1e-2,
        "blocks": [],
    }
    for index, (actual_layer, expected_layer) in enumerate(zip(actual_layers, expected_layers, strict=True)):
        summary = {"block": index, **_modality_summaries(actual_layer, expected_layer, masks)}
        report["blocks"].append(summary)
        print(f"QWEN_IMAGE21_DIT_PARITY block_{index} {json.dumps(summary, sort_keys=True)}", flush=True)
    report["final_prediction"] = _modality_summaries(actual, expected, masks)
    report["cached_prediction"] = _modality_summaries(
        actual_cached, expected_cached, {"target_image": torch.ones(target_tokens, dtype=torch.bool)})
    for name in ("final_prediction", "cached_prediction"):
        print(f"QWEN_IMAGE21_DIT_PARITY {name} {json.dumps(report[name], sort_keys=True)}", flush=True)
    report_path = os.environ.get("QWEN_IMAGE21_DIT_PARITY_REPORT")
    if report_path:
        destination = Path(report_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

    for index, (actual_layer, expected_layer) in enumerate(zip(actual_layers, expected_layers, strict=True)):
        assert_close(actual_layer, expected_layer, atol=3e-3, rtol=1e-2, msg=f"Transformer block {index}")
    assert_close(actual, expected.float().cpu(), atol=3e-3, rtol=1e-2)
    assert_close(actual_cached, expected_cached.float().cpu(), atol=3e-3, rtol=1e-2)
