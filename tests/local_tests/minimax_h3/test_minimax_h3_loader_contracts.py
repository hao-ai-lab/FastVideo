# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.nn as nn
from safetensors.torch import save_file

from fastvideo.configs.models.dits.minimax_h3 import MiniMaxH3ArchConfig, MiniMaxH3Config
from fastvideo.configs.models.vaes.minimax_h3_audio import MiniMaxH3AudioVAEArchConfig, MiniMaxH3AudioVAEConfig
from fastvideo.configs.models.vaes.minimax_h3_video import MiniMaxH3VideoVAEArchConfig, MiniMaxH3VideoVAEConfig
from fastvideo.models.loader import component_loader
from fastvideo.models.loader.component_loader import (
    AudioDecoderLoader,
    ComponentLoader,
    SchedulerLoader,
    VAELoader,
    _validate_transformer_parameter_dtypes,
)
from fastvideo.models.loader.fsdp_load import maybe_load_fsdp_model
from fastvideo.models.registry import ModelRegistry
from fastvideo.models.dits.minimax_h3 import MiniMaxH3Transformer3DModel
from fastvideo.models.schedulers.scheduling_minimax_h3 import MiniMaxH3Scheduler
from fastvideo.models.vaes.minimax_h3_audio import MiniMaxH3AudioVAE
from fastvideo.models.vaes.minimax_h3_video import AutoencoderKLMiniMaxH3
from fastvideo.platforms import current_platform


def test_stage1_components_are_registry_discoverable() -> None:
    expected_modules = {
        "MiniMaxH3Transformer3DModel": "fastvideo.models.dits.minimax_h3",
        "AutoencoderKLMiniMaxH3": "fastvideo.models.vaes.minimax_h3_video",
        "AutoencoderKLMiniMaxH3Audio": "fastvideo.models.vaes.minimax_h3_audio",
        "MiniMaxH3Scheduler": "fastvideo.models.schedulers.scheduling_minimax_h3",
    }
    for architecture, module_name in expected_modules.items():
        model_cls, resolved_architecture = ModelRegistry.resolve_model_cls(architecture)
        assert resolved_architecture == architecture
        assert model_cls.__module__ == module_name


class _MixedDtypeModule(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fp32_projection = nn.Linear(2, 2).float()
        self.body = nn.Linear(2, 2).to(torch.bfloat16)

    @staticmethod
    def _get_parameter_dtype(name: str, default_dtype: torch.dtype) -> torch.dtype:
        return torch.float32 if name.startswith("fp32_projection") else default_dtype


def test_mixed_dtype_validation_checks_every_named_parameter() -> None:
    model = _MixedDtypeModule()
    _validate_transformer_parameter_dtypes(model, torch.bfloat16, torch.bfloat16)
    model.body.float()
    with pytest.raises(AssertionError, match="body.weight"):
        _validate_transformer_parameter_dtypes(model, torch.bfloat16, torch.bfloat16)


def test_fsdp_loader_materializes_h3_mixed_dtype_islands(tmp_path) -> None:
    architecture = {
        "num_attention_heads": 2,
        "attention_head_dim": 16,
        "hidden_size": 24,
        "num_layers": 1,
        "num_refiner_layers": 1,
        "ffn_dim": 32,
        "in_channels": 4,
        "audio_in_channels": 6,
        "patch_size": (1, 2, 2),
        "text_dim": 8,
        "freq_dim": 8,
        "time_embed_hidden_dim": 24,
        "time_embed_dim": 16,
        "rope_freq_dim": 2,
    }
    config = MiniMaxH3Config(arch_config=MiniMaxH3ArchConfig(**architecture))
    backend = "fastvideo.attention.backends.sdpa.SDPABackend"
    with patch.object(current_platform, "get_attn_backend_cls", return_value=backend):
        source = MiniMaxH3Transformer3DModel(config, dict(architecture)).eval().to(torch.bfloat16)
    checkpoint = tmp_path / "model.safetensors"
    save_file({name: tensor.contiguous() for name, tensor in source.state_dict().items()}, checkpoint)

    target_config = MiniMaxH3Config(arch_config=MiniMaxH3ArchConfig(**architecture))
    with patch.object(current_platform, "get_attn_backend_cls", return_value=backend):
        loaded = maybe_load_fsdp_model(
            model_cls=MiniMaxH3Transformer3DModel,
            init_params={"config": target_config, "hf_config": dict(architecture)},
            weight_dir_list=[str(checkpoint)],
            device=torch.device("cpu"),
            hsdp_replicate_dim=1,
            hsdp_shard_dim=1,
            default_dtype=torch.bfloat16,
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.float32,
            strict=True,
            training_mode=False,
            fsdp_inference=False,
            cpu_offload=False,
            pin_cpu_memory=False,
        )

    _validate_transformer_parameter_dtypes(loaded, torch.bfloat16, torch.bfloat16)
    assert loaded.proj_in.weight.dtype == torch.float32
    assert loaded.audio_proj_in.weight.dtype == torch.float32
    assert loaded.time_embedder.fc_in.weight.dtype == torch.float32
    assert loaded.proj_out.weight.dtype == torch.float32
    assert loaded.audio_proj_out.weight.dtype == torch.float32
    assert loaded.rope.inv_freq.dtype == torch.float32
    assert loaded.transformer_blocks[0].attn.to_q.weight.dtype == torch.bfloat16


def _write_scheduler_config(path: Path, shift: float) -> None:
    path.mkdir()
    (path / "scheduler_config.json").write_text(
        json.dumps({"_class_name": "MiniMaxH3Scheduler", "shift": shift}), encoding="utf-8"
    )


def test_scheduler_loader_keeps_video_and_audio_role_shifts_independent(tmp_path, monkeypatch) -> None:
    video_path = tmp_path / "scheduler"
    audio_path = tmp_path / "audio_scheduler"
    _write_scheduler_config(video_path, 12.0)
    _write_scheduler_config(audio_path, 3.0)
    monkeypatch.setattr(
        component_loader.ModelRegistry,
        "resolve_model_cls",
        lambda _: (MiniMaxH3Scheduler, None),
    )
    args = SimpleNamespace(pipeline_config=SimpleNamespace(flow_shift=9.0, audio_flow_shift=4.0))

    assert SchedulerLoader().load(str(video_path), args).shift == 12.0
    assert SchedulerLoader().load(str(audio_path), args).shift == 3.0
    assert isinstance(ComponentLoader.for_module_type("audio_scheduler", "diffusers"), SchedulerLoader)


def test_audio_vae_loader_keeps_full_fp32_checkpoint(tmp_path, monkeypatch) -> None:
    arch = MiniMaxH3AudioVAEArchConfig(
        encoder_dim=4,
        encoder_rates=(2, 2),
        latent_dim=32,
        latent_channels=8,
        num_attention_heads=2,
        decoder_dim=16,
        decoder_rates=(2, 2),
        decoder_kernel_sizes=(4, 4),
        resblock_kernel_sizes=(3, 7),
        resblock_dilation_sizes=((1, 3), (1, 3)),
        latents_mean=[0.0] * 8,
        latents_std=[1.0] * 8,
    )
    source = MiniMaxH3AudioVAE(MiniMaxH3AudioVAEConfig(arch_config=arch)).eval()
    config = {key: value for key, value in vars(arch).items() if not key.startswith("_")}
    config["_class_name"] = "AutoencoderKLMiniMaxH3Audio"
    (tmp_path / "config.json").write_text(json.dumps(config), encoding="utf-8")
    save_file(source.state_dict(), tmp_path / "diffusion_pytorch_model.safetensors")

    monkeypatch.setattr(component_loader, "get_local_torch_device", lambda: torch.device("cpu"))
    monkeypatch.setattr(
        component_loader.ModelRegistry,
        "resolve_model_cls",
        lambda _: (MiniMaxH3AudioVAE, None),
    )
    args = SimpleNamespace(pipeline_config=SimpleNamespace())
    loaded = AudioDecoderLoader().load(str(tmp_path), args)

    assert set(loaded.state_dict()) == set(source.state_dict())
    assert hasattr(loaded, "encoder") and hasattr(loaded, "decoder")
    assert all(parameter.dtype == torch.float32 for parameter in loaded.parameters())


def test_video_vae_loader_strict_loads_native_checkpoint(tmp_path, monkeypatch) -> None:
    arch = MiniMaxH3VideoVAEArchConfig(
        latent_channels=4,
        block_out_channels=(8, 16),
        layers_per_block=1,
        spatial_downsample_factors=(2, 2),
        temporal_downsample_factors=(2, 2),
        norm_num_groups=8,
        decoder_num_layers=1,
        decoder_num_attention_heads=2,
        decoder_attention_head_dim=8,
        decoder_num_register_tokens=2,
        decoder_ffn_mult=2,
        latents_mean=(0.0, ) * 4,
        latents_std=(1.0, ) * 4,
    )
    vae_config = MiniMaxH3VideoVAEConfig(arch_config=arch)
    source = AutoencoderKLMiniMaxH3(vae_config).eval()
    config = {key: value for key, value in vars(arch).items() if not key.startswith("_")}
    config["_class_name"] = "AutoencoderKLMiniMaxH3"
    (tmp_path / "config.json").write_text(json.dumps(config), encoding="utf-8")
    save_file(source.state_dict(), tmp_path / "diffusion_pytorch_model.safetensors")

    monkeypatch.setattr(component_loader, "get_local_torch_device", lambda: torch.device("cpu"))
    args = SimpleNamespace(
        model_paths={},
        vae_cpu_offload=False,
        pipeline_config=SimpleNamespace(vae_precision="fp32", vae_config=MiniMaxH3VideoVAEConfig()),
    )
    loaded = VAELoader().load(str(tmp_path), args)

    assert set(loaded.state_dict()) == set(source.state_dict())
    assert all(torch.equal(loaded.state_dict()[key], tensor) for key, tensor in source.state_dict().items())
