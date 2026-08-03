# SPDX-License-Identifier: Apache-2.0
"""Synthetic raw-checkpoint conversion for MiniMax-H3 native components."""

from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
from safetensors.torch import save_file


os.environ["FASTVIDEO_ATTENTION_BACKEND"] = "TORCH_SDPA"
os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
os.environ.setdefault("MASTER_PORT", "29617")
os.environ.setdefault("RANK", "0")
os.environ.setdefault("WORLD_SIZE", "1")
os.environ.setdefault("LOCAL_RANK", "0")

from scripts.checkpoint_conversion.convert_minimax_h3_to_diffusers import (  # noqa: E402
    MINIMAX_H3_FP32_SOURCE_PREFIXES,
    convert_transformer_key,
    convert_video_vae_key,
    get_transformer_key_plan,
    get_video_vae_key_plan,
    load_safetensors_directory,
    main as convert_main,
    reorder_interleaved_qkv,
    split_fused_qkv,
)
from tests.local_tests.minimax_h3._reference import assert_pinned_reference  # noqa: E402


assert_pinned_reference(
    "scripts/convert_minimax_h3_to_diffusers.py",
    "86f61f62934d1eccf2a76ec57b6d072e2c23767c6e49d96aa1cf24d40422b42c",
)

from fastvideo.configs.models.dits.minimax_h3 import (  # noqa: E402
    MiniMaxH3ArchConfig,
    MiniMaxH3Config,
)
from fastvideo.configs.models.vaes.minimax_h3_audio import (  # noqa: E402
    MiniMaxH3AudioVAEArchConfig,
    MiniMaxH3AudioVAEConfig,
)
from fastvideo.configs.models.vaes.minimax_h3_video import (  # noqa: E402
    MiniMaxH3VideoVAEArchConfig,
    MiniMaxH3VideoVAEConfig,
)
from fastvideo.distributed import (  # noqa: E402
    cleanup_dist_env_and_memory,
    maybe_init_distributed_environment_and_model_parallel,
)
from fastvideo.models.dits.minimax_h3 import (  # noqa: E402
    MiniMaxH3Transformer3DModel,
)
from fastvideo.models.vaes.minimax_h3_audio import MiniMaxH3AudioVAE  # noqa: E402
from fastvideo.models.vaes.minimax_h3_video import AutoencoderKLMiniMaxH3  # noqa: E402
from fastvideo.platforms import current_platform  # noqa: E402


TINY_TRANSFORMER_CONFIG = {
    "num_attention_heads": 2,
    "attention_head_dim": 16,
    "hidden_size": 24,
    "num_layers": 2,
    "num_refiner_layers": 2,
    "ffn_dim": 32,
    "in_channels": 4,
    "audio_in_channels": 6,
    "patch_size": (1, 2, 2),
    "text_dim": 8,
    "freq_dim": 8,
    "time_embed_hidden_dim": 24,
    "time_embed_dim": 16,
    "rope_freq_dim": 2,
    "rope_theta": 10000.0,
    "norm_eps": 1e-5,
    "qk_norm_eps": 1e-5,
    "final_norm_eps": 1e-5,
}

TINY_VIDEO_VAE_CONFIG = {
    "in_channels": 3,
    "out_channels": 3,
    "latent_channels": 4,
    "block_out_channels": (8, 16),
    "layers_per_block": 1,
    "spatial_downsample_factors": (2, 2),
    "temporal_downsample_factors": (2, 2),
    "norm_num_groups": 8,
    "norm_eps": 1e-6,
    "spatial_padding_mode": "reflect",
    "decoder_num_layers": 2,
    "decoder_num_attention_heads": 2,
    "decoder_attention_head_dim": 8,
    "decoder_num_register_tokens": 2,
    "decoder_ffn_mult": 2,
    "decoder_rope_theta": 100.0,
    "decoder_rope_dim_ratio": 0.75,
    "decoder_norm_eps": 1e-5,
    "clip_length": 17,
    "token_drop": 3,
    "latents_mean": (0.5, -0.25, 1.0, -1.5),
    "latents_std": (2.0, 0.5, 1.5, 4.0),
}

TINY_AUDIO_VAE_CONFIG = {
    "encoder_dim": 4,
    "encoder_rates": (2, 2),
    "latent_dim": 32,
    "latent_channels": 8,
    "num_attention_heads": 2,
    "decoder_dim": 16,
    "decoder_rates": (2, 2),
    "decoder_kernel_sizes": (4, 4),
    "resblock_kernel_sizes": (3, 7),
    "resblock_dilation_sizes": ((1, 3), (1, 3)),
    "sampling_rate": 32000,
    "latents_mean": [0.0] * 8,
    "latents_std": [1.0] * 8,
}


@pytest.fixture(scope="module", autouse=True)
def distributed_runtime():
    maybe_init_distributed_environment_and_model_parallel(1, 1)
    yield
    cleanup_dist_env_and_memory()


def _build_transformer() -> MiniMaxH3Transformer3DModel:
    arch = MiniMaxH3ArchConfig(**TINY_TRANSFORMER_CONFIG)
    config = MiniMaxH3Config(arch_config=arch)
    with patch.object(
            current_platform,
            "get_attn_backend_cls",
            return_value="fastvideo.attention.backends.sdpa.SDPABackend"):
        model = MiniMaxH3Transformer3DModel(
            config, dict(TINY_TRANSFORMER_CONFIG)).eval()
    return model.to(dtype=torch.bfloat16)


def _build_video_vae() -> AutoencoderKLMiniMaxH3:
    arch = MiniMaxH3VideoVAEArchConfig(**TINY_VIDEO_VAE_CONFIG)
    return AutoencoderKLMiniMaxH3(
        MiniMaxH3VideoVAEConfig(arch_config=arch)).eval()


def _build_audio_vae() -> MiniMaxH3AudioVAE:
    arch = MiniMaxH3AudioVAEArchConfig(**TINY_AUDIO_VAE_CONFIG)
    return MiniMaxH3AudioVAE(
        MiniMaxH3AudioVAEConfig(arch_config=arch)).eval()


def _clone_state(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    state: dict[str, torch.Tensor] = {}
    for index, (key, tensor) in enumerate(model.state_dict().items()):
        if tensor.is_floating_point():
            values = torch.arange(tensor.numel(), dtype=torch.float32)
            values = ((values + index) % 29 - 14) / 29
            state[key] = values.reshape(tensor.shape).to(tensor.dtype).contiguous()
        else:
            state[key] = tensor.detach().cpu().clone().contiguous()
    return state


def _interleave_qkv(query: torch.Tensor, key: torch.Tensor,
                    value: torch.Tensor, heads: int,
                    head_dim: int) -> torch.Tensor:
    """Inverse fixture transform: native Q/K/V -> raw per-head QKV rows."""
    trailing = query.shape[1:]
    grouped = torch.cat([
        tensor.reshape(heads, head_dim, *trailing)
        for tensor in (query, key, value)
    ],
                        dim=1)
    return grouped.reshape(heads * 3 * head_dim, *trailing).contiguous()


def _manual_deinterleave(raw: torch.Tensor, heads: int,
                         head_dim: int) -> tuple[torch.Tensor, ...]:
    grouped = raw.reshape(heads, 3 * head_dim, *raw.shape[1:])
    return tuple(
        torch.cat([
            grouped[head, part * head_dim:(part + 1) * head_dim]
            for head in range(heads)
        ],
                  dim=0).contiguous() for part in range(3))


def _raw_transformer_fixture(
        target: dict[str, torch.Tensor]
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    plan = get_transformer_key_plan(TINY_TRANSFORMER_CONFIG)
    planned_targets = {
        target_key
        for targets in plan.values() for target_key, _shape in targets
    }
    assert planned_targets == set(target)
    raw: dict[str, torch.Tensor] = {}
    expected = {key: tensor.clone() for key, tensor in target.items()}
    heads = TINY_TRANSFORMER_CONFIG["num_attention_heads"]
    head_dim = TINY_TRANSFORMER_CONFIG["attention_head_dim"]

    for source_key, targets in plan.items():
        if not targets:
            raw[source_key] = torch.arange(
                TINY_TRANSFORMER_CONFIG["rope_freq_dim"], dtype=torch.float32)
        elif len(targets) == 3:
            raw[source_key] = _interleave_qkv(
                *(target[name] for name, _shape in targets), heads, head_dim)
        else:
            target_key = targets[0][0]
            tensor = target[target_key]
            if source_key.endswith(".mlp.fc1.weight"):
                value, gate = tensor.chunk(2, dim=0)
                tensor = torch.cat([gate, value], dim=0)
            raw[source_key] = tensor.clone().contiguous()

    qkv_key = "blocks.0.attn.qkv_proj.weight"
    qkv = torch.arange(raw[qkv_key].numel(), dtype=torch.float32).remainder(97)
    raw[qkv_key] = qkv.reshape_as(raw[qkv_key]).to(torch.bfloat16)
    q_name, k_name, v_name = [name for name, _shape in plan[qkv_key]]
    expected[q_name], expected[k_name], expected[v_name] = _manual_deinterleave(
        raw[qkv_key], heads, head_dim)

    ffn_key = "blocks.0.mlp.fc1.weight"
    gate, value = raw[ffn_key].chunk(2, dim=0)
    raw[ffn_key] = torch.cat([torch.ones_like(gate),
                              torch.full_like(value, 2)],
                             dim=0)
    expected[plan[ffn_key][0][0]] = torch.cat(
        [torch.full_like(value, 2), torch.ones_like(gate)], dim=0)
    return raw, expected


def _raw_video_vae_fixture(
        target: dict[str, torch.Tensor]
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    plan = get_video_vae_key_plan(TINY_VIDEO_VAE_CONFIG)
    planned_targets = {
        target_key
        for targets in plan.values() for target_key in targets
    }
    assert planned_targets == set(target)
    raw: dict[str, torch.Tensor] = {}
    expected = {key: tensor.clone() for key, tensor in target.items()}
    heads = TINY_VIDEO_VAE_CONFIG["decoder_num_attention_heads"]
    head_dim = TINY_VIDEO_VAE_CONFIG["decoder_attention_head_dim"]

    for source_key, targets in plan.items():
        if not targets:
            raw[source_key] = torch.zeros(1, dtype=torch.float32)
        elif len(targets) == 3:
            raw[source_key] = _interleave_qkv(
                *(target[name] for name in targets), heads, head_dim)
        else:
            tensor = target[targets[0]]
            if ".ff.w1." in source_key:
                up, gate = tensor.chunk(2, dim=0)
                tensor = torch.cat([gate, up], dim=0)
            raw[source_key] = tensor.clone().contiguous()

    qkv_key = "decoder.transformer_blocks.0.attn.to_qkv.weight"
    qkv = torch.arange(raw[qkv_key].numel(), dtype=torch.float32).remainder(101)
    raw[qkv_key] = qkv.reshape_as(raw[qkv_key])
    q_name, k_name, v_name = plan[qkv_key]
    expected[q_name], expected[k_name], expected[v_name] = _manual_deinterleave(
        raw[qkv_key], heads, head_dim)

    ffn_key = "decoder.transformer_blocks.0.ff.w1.weight"
    gate, up = raw[ffn_key].chunk(2, dim=0)
    raw[ffn_key] = torch.cat(
        [torch.full_like(gate, 3), torch.full_like(up, 5)], dim=0)
    expected[plan[ffn_key][0]] = torch.cat(
        [torch.full_like(up, 5), torch.full_like(gate, 3)], dim=0)
    return raw, expected


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2))


def _assert_exact_state(actual: dict[str, torch.Tensor],
                        expected: dict[str, torch.Tensor]) -> None:
    assert actual.keys() == expected.keys()
    for key in expected:
        assert actual[key].shape == expected[key].shape, key
        assert actual[key].dtype == expected[key].dtype, key
        assert torch.equal(actual[key], expected[key]), key


def test_qkv_reorder_split_and_ffn_half_swap_are_exact() -> None:
    heads, head_dim, width = 2, 3, 4
    raw = torch.arange(heads * 3 * head_dim * width,
                       dtype=torch.float32).reshape(-1, width)
    expected = _manual_deinterleave(raw, heads, head_dim)
    reordered = reorder_interleaved_qkv(raw, heads, head_dim)
    assert all(
        torch.equal(actual, wanted)
        for actual, wanted in zip(split_fused_qkv(reordered, heads, head_dim),
                                  expected,
                                  strict=True))

    transformer_source = torch.cat(
        [torch.ones(2, 3), torch.full((2, 3), 2.0)], dim=0)
    [(key, transformed)] = convert_transformer_key(
        "blocks.0.mlp.fc1.weight", transformer_source,
        TINY_TRANSFORMER_CONFIG)
    assert key == "transformer_blocks.0.ff.net.0.proj.weight"
    assert torch.equal(
        transformed,
        torch.cat([torch.full((2, 3), 2.0), torch.ones(2, 3)], dim=0))

    [(key, transformed)] = convert_video_vae_key(
        "decoder.transformer_blocks.0.ff.w1.bias",
        torch.tensor([1.0, 1.0, 3.0, 3.0]), TINY_VIDEO_VAE_CONFIG)
    assert key == "decoder.transformer_blocks.0.ff.net.0.proj.bias"
    assert torch.equal(transformed, torch.tensor([3.0, 3.0, 1.0, 1.0]))


def test_synthetic_safetensors_cli_roundtrip_strict_loads_native_components(
        tmp_path: Path) -> None:
    torch.manual_seed(20260802)
    transformer = _build_transformer()
    video_vae = _build_video_vae()
    audio_vae = _build_audio_vae()
    raw_transformer, expected_transformer = _raw_transformer_fixture(
        _clone_state(transformer))
    raw_video_vae, expected_video_vae = _raw_video_vae_fixture(
        _clone_state(video_vae))
    expected_audio_vae = _clone_state(audio_vae)

    source = tmp_path / "raw"
    (source / "transformer").mkdir(parents=True)
    (source / "video_vae" / "source").mkdir(parents=True)
    (source / "audio_vae").mkdir(parents=True)
    save_file(raw_transformer,
              str(source / "transformer" / "model-00001.safetensors"))
    save_file(raw_video_vae,
              str(source / "video_vae" / "source" / "model.safetensors"))
    save_file(expected_audio_vae,
              str(source / "audio_vae" / "model.safetensors"))
    _write_json(
        source / "video_vae" / "config.json", {
            "source_path": "source",
            "source_safetensors_path": "model.safetensors",
            "latents_mean": list(TINY_VIDEO_VAE_CONFIG["latents_mean"]),
            "latents_std": list(TINY_VIDEO_VAE_CONFIG["latents_std"]),
        })
    _write_json(
        source / "model_index.json", {
            "_minimax_h3": {
                "sigma_shift_scales": {
                    "video": 12,
                    "audio": 3
                }
            }
        })

    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    transformer_config_path = config_dir / "transformer.json"
    video_config_path = config_dir / "video_vae.json"
    audio_config_path = config_dir / "audio_vae.json"
    _write_json(transformer_config_path, TINY_TRANSFORMER_CONFIG)
    _write_json(video_config_path, TINY_VIDEO_VAE_CONFIG)
    _write_json(audio_config_path, TINY_AUDIO_VAE_CONFIG)

    output = tmp_path / "converted"
    convert_main([
        "--checkpoint_path",
        str(source),
        "--output_path",
        str(output),
        "--transformer_config",
        str(transformer_config_path),
        "--video_vae_config",
        str(video_config_path),
        "--audio_vae_config",
        str(audio_config_path),
        "--max_shard_size",
        str(1 << 30),
    ])

    converted_transformer = load_safetensors_directory(output / "transformer")
    converted_video_vae = load_safetensors_directory(output / "vae")
    converted_audio_vae = load_safetensors_directory(output / "audio_vae")
    _assert_exact_state(converted_transformer, expected_transformer)
    _assert_exact_state(converted_video_vae, expected_video_vae)
    _assert_exact_state(converted_audio_vae, expected_audio_vae)

    transformer_plan = get_transformer_key_plan(TINY_TRANSFORMER_CONFIG)
    for source_key, targets in transformer_plan.items():
        expected_dtype = (torch.float32 if source_key.startswith(
            MINIMAX_H3_FP32_SOURCE_PREFIXES) else torch.bfloat16)
        for target_key, shape in targets:
            assert converted_transformer[target_key].dtype == expected_dtype
            assert list(converted_transformer[target_key].shape) == shape
    assert all(tensor.dtype == torch.float32
               for tensor in converted_video_vae.values())
    assert all(tensor.dtype == torch.float32
               for tensor in converted_audio_vae.values())

    assert not transformer.load_state_dict(converted_transformer,
                                           strict=True).missing_keys
    assert not video_vae.load_state_dict(converted_video_vae,
                                        strict=True).missing_keys
    assert not audio_vae.load_state_dict(converted_audio_vae,
                                        strict=True).missing_keys

    transformer_config = json.loads(
        (output / "transformer" / "config.json").read_text())
    video_config = json.loads((output / "vae" / "config.json").read_text())
    audio_config = json.loads(
        (output / "audio_vae" / "config.json").read_text())
    assert transformer_config["_class_name"] == "MiniMaxH3Transformer3DModel"
    assert video_config["_class_name"] == "AutoencoderKLMiniMaxH3"
    assert audio_config["_class_name"] == "AutoencoderKLMiniMaxH3Audio"
    assert json.loads((output / "scheduler" / "scheduler_config.json").read_text()) == {
        "_class_name": "MiniMaxH3Scheduler",
        "shift": 12.0,
    }
    assert json.loads((output / "audio_scheduler" /
                       "scheduler_config.json").read_text()) == {
                           "_class_name": "MiniMaxH3Scheduler",
                           "shift": 3.0,
                       }
