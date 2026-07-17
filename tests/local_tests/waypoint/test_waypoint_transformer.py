# SPDX-License-Identifier: Apache-2.0
"""Structural and numerical parity for Waypoint-1-Small."""

import importlib
import importlib.util
import json
import os
import re
import sys
import types

import pytest
import torch
from safetensors.torch import load_file

from fastvideo.attention.backends.sdpa import SDPAMetadata
from fastvideo.forward_context import set_forward_context

MODEL_ROOT = os.environ.get(
    "WAYPOINT_MODEL_PATH", "models/Waypoint-1-Small-Diffusers"
)
OFFICIAL_ROOT = os.environ.get(
    "WAYPOINT_OVERWORLD_PATH", "models/Waypoint-1-Small"
)
WEIGHTS = os.path.join(
    MODEL_ROOT, "transformer", "diffusion_pytorch_model.safetensors"
)
TRAJECTORY_FRAMES = 160


def _require_assets() -> None:
    if not os.path.isfile(WEIGHTS):
        pytest.skip(f"Waypoint weights not found at {WEIGHTS}")
    if not os.path.isfile(os.path.join(OFFICIAL_ROOT, "transformer", "model.py")):
        pytest.skip(f"Official Waypoint source not found at {OFFICIAL_ROOT}")


def _remap_weights(config) -> dict[str, torch.Tensor]:
    state = {}
    for key, value in load_file(WEIGHTS).items():
        for pattern, replacement in config.param_names_mapping.items():
            key = re.sub(pattern, replacement, key)
        state[key] = value
    return state


def _official_components():
    pytest.importorskip("tensordict")
    package = "waypoint_official_transformer_parity"
    root = types.ModuleType(package)
    root.__path__ = [OFFICIAL_ROOT]
    root.__package__ = package
    sys.modules[package] = root

    model_module = importlib.import_module(f"{package}.transformer.model")
    source = os.path.join(OFFICIAL_ROOT, "before_denoise.py")
    spec = importlib.util.spec_from_file_location(f"{package}.before_denoise", source)
    cache_module = importlib.util.module_from_spec(spec)
    cache_module.__package__ = package
    sys.modules[spec.name] = cache_module
    spec.loader.exec_module(cache_module)

    from torch.nn.attention.flex_attention import create_block_mask

    def make_block_mask(query_length, kv_length, written):
        def mask_mod(batch, head, query_index, kv_index):
            return written[kv_index]

        return create_block_mask(
            mask_mod,
            B=None,
            H=None,
            Q_LEN=query_length,
            KV_LEN=kv_length,
            device=written.device,
            _compile=False,
        )

    cache_module.make_block_mask = make_block_mask
    return model_module.WorldModel, cache_module.StaticKVCache


def test_official_checkpoint_loads_strictly():
    _require_assets()
    from fastvideo.configs.models.dits.waypoint_transformer import WaypointArchConfig
    from fastvideo.models.dits.waypoint_transformer import WaypointWorldModel

    config = WaypointArchConfig()
    model = WaypointWorldModel(config)
    state = _remap_weights(config)
    assert len(state) == 366
    model.load_state_dict(state, strict=True)


def test_meta_noise_frequency_is_materialized():
    from fastvideo.configs.models.dits.waypoint_transformer import WaypointArchConfig
    from fastvideo.models.dits.waypoint_transformer import WaypointWorldModel

    with torch.device("meta"):
        model = WaypointWorldModel(WaypointArchConfig())
    assert model.denoise_step_emb.freq.is_meta

    model.materialize_non_persistent_buffers(torch.device("cpu"))

    assert model.denoise_step_emb.freq.device.type == "cpu"
    assert model.denoise_step_emb.freq.dtype == torch.float32


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_single_frame_matches_official(distributed_setup):
    _require_assets()
    from fastvideo.configs.models.dits.waypoint_transformer import WaypointArchConfig
    from fastvideo.models.dits.waypoint_transformer import (
        WaypointKVCache,
        WaypointWorldModel,
    )

    device = torch.device("cuda")
    dtype = torch.float32
    config = WaypointArchConfig()
    raw_state = load_file(WEIGHTS)

    OfficialModel, OfficialCache = _official_components()
    with open(
        os.path.join(OFFICIAL_ROOT, "transformer", "config.json"),
        encoding="utf-8",
    ) as file:
        official_config = json.load(file)
    for key in ("_class_name", "_diffusers_version", "auto_map"):
        official_config.pop(key, None)

    official = OfficialModel(**official_config)
    official.load_state_dict(raw_state, strict=True)
    official = official.to(device=device, dtype=dtype).eval()
    native = WaypointWorldModel(config)
    native.load_state_dict(_remap_weights(config), strict=True)
    native = native.to(device=device, dtype=dtype).eval()
    native.denoise_step_emb.to(dtype=torch.float32)

    generator = torch.Generator(device=device).manual_seed(123)
    inputs = {
        "x": torch.randn(
            1, 1, 16, 32, 32, device=device, dtype=dtype, generator=generator
        ),
        "sigma": torch.tensor([[0.729332447052002]], device=device, dtype=dtype),
        "frame_timestamp": torch.zeros(1, 1, device=device, dtype=torch.long),
        "prompt_emb": torch.randn(
            1, 32, 2048, device=device, dtype=dtype, generator=generator
        ),
        "prompt_pad_mask": torch.zeros(1, 32, device=device, dtype=torch.bool),
        "mouse": torch.tensor([[[0.25, -0.5]]], device=device, dtype=dtype),
        "button": torch.zeros(1, 1, 256, device=device, dtype=dtype),
        "scroll": torch.ones(1, 1, 1, device=device, dtype=dtype),
    }
    inputs["button"][0, 0, 17] = 1

    official_cache = OfficialCache(official.config, 1, dtype).to(device)
    native_cache = WaypointKVCache(config, 1, dtype).to(device)
    metadata = SDPAMetadata(current_timestep=0, attn_mask=None)
    expected = inputs.pop("x")
    actual = expected.clone()
    sigmas = torch.tensor(config.scheduler_sigmas, device=device, dtype=dtype)
    with torch.no_grad(), set_forward_context(
        current_timestep=0,
        attn_metadata=metadata,
        forward_batch=None,
    ):
        for sigma, next_sigma in zip(sigmas[:-1], sigmas[1:], strict=True):
            inputs["sigma"] = sigma.expand(1, 1)
            expected_velocity = official(
                x=expected, **inputs, kv_cache=official_cache
            )
            actual_velocity = native(x=actual, **inputs, kv_cache=native_cache)
            expected = expected + (next_sigma - sigma) * expected_velocity
            actual = actual + (next_sigma - sigma) * actual_velocity

    absolute_error = (actual - expected).abs()
    cosine = torch.nn.functional.cosine_similarity(
        actual.flatten().float(), expected.flatten().float(), dim=0
    )
    print(
        f"mean_abs={absolute_error.mean().item():.8f} "
        f"max_abs={absolute_error.max().item():.8f} cosine={cosine.item():.8f}"
    )
    assert absolute_error.mean() < 0.002
    assert absolute_error.max() < 0.03
    assert cosine > 0.99999


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_160_frame_official_trajectory_parity(distributed_setup):
    _require_assets()
    from fastvideo.configs.models.dits.waypoint_transformer import WaypointArchConfig
    from fastvideo.models.dits.waypoint_transformer import (
        WaypointKVCache,
        WaypointWorldModel,
    )

    device = torch.device("cuda")
    dtype = torch.bfloat16
    config = WaypointArchConfig()
    raw_state = load_file(WEIGHTS)
    OfficialModel, OfficialCache = _official_components()
    with open(
        os.path.join(OFFICIAL_ROOT, "transformer", "config.json"),
        encoding="utf-8",
    ) as file:
        official_config = json.load(file)
    for key in ("_class_name", "_diffusers_version", "auto_map"):
        official_config.pop(key, None)

    official = OfficialModel(**official_config)
    official.load_state_dict(raw_state, strict=True)
    official = official.to(device=device, dtype=dtype).eval()
    native = WaypointWorldModel(config)
    native.load_state_dict(_remap_weights(config), strict=True)
    native = native.to(device=device, dtype=dtype).eval()
    native.denoise_step_emb.to(dtype=torch.float32)

    official_cache = OfficialCache(official.config, 1, dtype).to(device)
    native_cache = WaypointKVCache(config, 1, dtype).to(device)
    generator = torch.Generator(device=device).manual_seed(2026)
    prompt_emb = torch.randn(
        1,
        32,
        2048,
        device=device,
        dtype=dtype,
        generator=generator,
    )
    prompt_pad_mask = torch.zeros(1, 32, device=device, dtype=torch.bool)
    sigmas = torch.tensor(config.scheduler_sigmas, device=device, dtype=dtype)
    metadata = SDPAMetadata(current_timestep=0, attn_mask=None)
    frame_errors = []
    frame_cosines = []

    with torch.no_grad(), set_forward_context(
        current_timestep=0,
        attn_metadata=metadata,
        forward_batch=None,
    ):
        for frame_index in range(TRAJECTORY_FRAMES):
            noise = torch.randn(
                1,
                1,
                16,
                32,
                32,
                device=device,
                dtype=dtype,
                generator=generator,
            )
            expected = noise.clone()
            actual = noise.clone()
            frame_timestamp = torch.full(
                (1, 1), frame_index, device=device, dtype=torch.long
            )
            mouse = torch.tensor(
                [[[((frame_index % 9) - 4) / 8, ((frame_index % 7) - 3) / 6]]],
                device=device,
                dtype=dtype,
            )
            button = torch.zeros(1, 1, 256, device=device, dtype=dtype)
            button[0, 0, (17, 30, 31, 32)[frame_index % 4]] = 1
            scroll = torch.tensor(
                [[[(frame_index % 3) - 1]]], device=device, dtype=dtype
            )
            inputs = {
                "frame_timestamp": frame_timestamp,
                "prompt_emb": prompt_emb,
                "prompt_pad_mask": prompt_pad_mask,
                "mouse": mouse,
                "button": button,
                "scroll": scroll,
            }
            official_cache.set_frozen(True)
            native_cache.set_frozen(True)

            for sigma, next_sigma in zip(
                sigmas[:-1], sigmas[1:], strict=True
            ):
                step_sigma = sigma.expand(1, 1)
                expected_velocity = official(
                    x=expected,
                    sigma=step_sigma,
                    kv_cache=official_cache,
                    **inputs,
                )
                actual_velocity = native(
                    x=actual,
                    sigma=step_sigma,
                    kv_cache=native_cache,
                    **inputs,
                )
                expected = expected + (next_sigma - sigma) * expected_velocity
                actual = actual + (next_sigma - sigma) * actual_velocity

            difference = actual.float() - expected.float()
            frame_errors.append(difference.abs().mean())
            frame_cosines.append(
                torch.nn.functional.cosine_similarity(
                    actual.flatten().float(),
                    expected.flatten().float(),
                    dim=0,
                )
            )

            official_cache.set_frozen(False)
            native_cache.set_frozen(False)
            zero = torch.zeros(1, 1, device=device, dtype=dtype)
            # Keep both caches on the same official trajectory so errors are not fed back.
            official(
                x=expected,
                sigma=zero,
                kv_cache=official_cache,
                **inputs,
            )
            native(
                x=expected,
                sigma=zero,
                kv_cache=native_cache,
                **inputs,
            )

    errors = torch.stack(frame_errors)
    cosines = torch.stack(frame_cosines)
    worst_error_frame = int(errors.argmax().item())
    worst_cosine_frame = int(cosines.argmin().item())
    print(
        f"frames={TRAJECTORY_FRAMES} mean_abs={errors.mean().item():.8f} "
        f"worst_frame_abs={errors.max().item():.8f}@{worst_error_frame} "
        f"min_cosine={cosines.min().item():.8f}@{worst_cosine_frame}"
    )
    assert errors.mean() < 0.01
    assert errors.max() < 0.03
    assert cosines.min() > 0.999
