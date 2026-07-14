# SPDX-License-Identifier: Apache-2.0
"""Import/registry preflight and optional real Z-Image-Turbo smoke."""

from __future__ import annotations

import os
from contextlib import nullcontext
from pathlib import Path
from typing import Any, cast

import pytest
import torch


MODEL_DIR_ENV = os.getenv("ZIMAGE_MODEL_DIR")


def test_zimage_typed_surface_preflight() -> None:
    from fastvideo.api import GeneratorConfig, PipelineSelection
    from fastvideo.api.compat import generator_config_to_fastvideo_args
    import fastvideo.registry as registry
    from fastvideo.api.presets import get_preset, get_presets_for_family
    from fastvideo.configs.models.dits.zimage import ZImageDiTConfig
    from fastvideo.configs.models.encoders.qwen3 import Qwen3TextConfig
    from fastvideo.configs.models.vaes.autoencoder_kl import AutoencoderKLVAEConfig
    from fastvideo.configs.pipelines.zimage import ZImagePipelineConfig
    from fastvideo.fastvideo_args import WorkloadType
    from fastvideo.pipelines.basic.zimage.zimage_pipeline import EntryClass, ZImagePipeline

    assert ZImagePipeline.__name__ == "ZImagePipeline"
    assert EntryClass is ZImagePipeline
    assert ZImagePipeline.pipeline_config_cls is ZImagePipelineConfig
    assert ZImagePipeline._required_config_modules == [
        "scheduler",
        "text_encoder",
        "tokenizer",
        "transformer",
        "vae",
    ]

    default_preset, model_family = registry.get_preset_selection("Tongyi-MAI/Z-Image-Turbo")
    assert (default_preset, model_family) == ("zimage_turbo", "zimage")
    info = registry.get_model_info(
        "Tongyi-MAI/Z-Image-Turbo",
        workload_type=WorkloadType.T2I,
        override_pipeline_cls_name="ZImagePipeline",
    )
    assert info.pipeline_cls is ZImagePipeline
    assert info.pipeline_config_cls is ZImagePipelineConfig

    assert {preset.name for preset in get_presets_for_family("zimage")} == {"zimage_turbo"}
    preset = get_preset("zimage_turbo", "zimage")
    assert preset.defaults == {
        "height": 1024,
        "width": 1024,
        "num_frames": 1,
        "fps": 1,
        "seed": 42,
        "guidance_scale": 0.0,
        "num_inference_steps": 8,
        "negative_prompt": "",
        "max_sequence_length": 512,
        "cfg_normalization": False,
        "cfg_truncation": 1.0,
    }

    config = ZImagePipelineConfig()
    assert isinstance(config.dit_config, ZImageDiTConfig)
    assert isinstance(config.vae_config, AutoencoderKLVAEConfig)
    assert isinstance(config.text_encoder_configs[0], Qwen3TextConfig)
    assert config.text_encoder_configs[0].chat_template_enable_thinking is True
    assert config.text_encoder_configs[0].output_hidden_states is True
    assert config.text_encoder_configs[0].tokenizer_kwargs["max_length"] == 512
    assert config.scheduler_sigma_min == 0.0
    assert config.scheduler_use_reference_discrete_timesteps is True

    args = generator_config_to_fastvideo_args(
        GeneratorConfig(
            model_path="Tongyi-MAI/Z-Image-Turbo",
            pipeline=PipelineSelection(workload_type="t2i"),
        ))
    assert isinstance(args.pipeline_config, ZImagePipelineConfig)
    assert args.workload_type is WorkloadType.T2I


def test_model_download_honors_pinned_revision(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import fastvideo.utils as utils

    observed: dict[str, object] = {}

    def fake_snapshot_download(**kwargs):
        observed.update(kwargs)
        return str(tmp_path)

    monkeypatch.setattr(utils, "snapshot_download", fake_snapshot_download)
    monkeypatch.setattr(utils, "get_lock", lambda _key: nullcontext())
    resolved = utils.maybe_download_model(
        "Tongyi-MAI/Z-Image-Turbo",
        revision="f332072aa78be7aecdf3ee76d5c247082da564a6",
    )

    assert resolved == str(tmp_path)
    assert observed["repo_id"] == "Tongyi-MAI/Z-Image-Turbo"
    assert observed["revision"] == "f332072aa78be7aecdf3ee76d5c247082da564a6"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Z-Image load/generate smoke requires CUDA")
def test_zimage_pipeline_load_generate_smoke() -> None:
    if MODEL_DIR_ENV is None:
        pytest.skip("Set ZIMAGE_MODEL_DIR to activate the real load/generate smoke")
    model_dir = Path(MODEL_DIR_ENV)
    if not (model_dir / "model_index.json").is_file():
        pytest.skip(f"Z-Image model_index.json not found under {model_dir}")

    from fastvideo import VideoGenerator

    generator = VideoGenerator.from_pretrained(
        str(model_dir),
        num_gpus=1,
        tp_size=1,
        sp_size=1,
        use_fsdp_inference=False,
        dit_cpu_offload=False,
        dit_layerwise_offload=False,
        text_encoder_cpu_offload=False,
        vae_cpu_offload=False,
        pin_cpu_memory=False,
        output_type="latent",
    )
    try:
        result = generator.generate_video(
            prompt="a red panda reading a book",
            negative_prompt="",
            output_path="outputs/zimage/smoke",
            save_video=False,
            return_frames=True,
            height=64,
            width=64,
            num_frames=1,
            fps=1,
            num_inference_steps=1,
            guidance_scale=0.0,
            max_sequence_length=64,
            cfg_normalization=False,
            cfg_truncation=1.0,
            seed=42,
        )
    finally:
        generator.shutdown()

    result_dict = cast(dict[str, Any], result)
    samples = result_dict["samples"]
    assert torch.is_tensor(samples)
    assert samples.ndim in (4, 5)
    assert torch.isfinite(samples).all()
