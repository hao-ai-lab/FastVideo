# SPDX-License-Identifier: Apache-2.0
"""Import, registry, stage-contract, and optional real LingBot-Video smoke tests."""

from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch

from tests.local_tests.lingbot_video.hf_assets import FASTVIDEO_DENSE, download_components


def test_lingbot_video_pipeline_registry_and_preset(tmp_path: Path) -> None:
    """Verify exact class resolution, required modules, config, and official defaults."""
    from fastvideo.api.presets import get_preset
    from fastvideo.configs.models.encoders.lingbot_video import LingBotVideoQwen3VLConfig
    from fastvideo.configs.pipelines.lingbot_video import LingBotVideoT2VConfig, LingBotVideoTI2VConfig
    from fastvideo.fastvideo_args import WorkloadType
    from fastvideo.pipelines.basic.lingbot_video.lingbot_video_pipeline import (
        EntryClass,
        LingBotVideoImageToVideoPipeline,
        LingBotVideoPipeline,
    )
    from fastvideo.registry import get_model_info, get_preset_selection

    model_dir = tmp_path / "arbitrary-dense-layout"
    model_index = {
        "_class_name": "LingBotVideoDensePipeline",
        "_diffusers_version": "0.39.0",
        "scheduler": ["diffusers", "FlowUniPCMultistepScheduler"],
        "text_encoder": ["transformers", "LingBotVideoQwen3VLModel"],
        "tokenizer": ["transformers", "Qwen3VLProcessor"],
        "transformer": ["diffusers", "LingBotVideoTransformer3DModel"],
        "vae": ["diffusers", "AutoencoderKLWan"],
    }
    model_dir.mkdir()
    for component in LingBotVideoPipeline._required_config_modules:
        (model_dir / component).mkdir()
    (model_dir / "model_index.json").write_text(json.dumps(model_index), encoding="utf-8")
    assert model_index["_class_name"] == "LingBotVideoDensePipeline"
    assert EntryClass == [LingBotVideoPipeline, LingBotVideoImageToVideoPipeline]
    assert set(model_index) >= set(LingBotVideoPipeline._required_config_modules)
    preset_name, family = get_preset_selection(str(model_dir))
    assert (preset_name, family) == ("lingbot_video_dense_t2v", "lingbot_video")
    preset = get_preset(preset_name, family)
    expected_defaults = {
        "height": 480,
        "width": 832,
        "num_frames": 121,
        "fps": 24,
        "num_inference_steps": 40,
        "guidance_scale": 3.0,
        "batch_cfg": True,
        "seed": 42,
    }
    assert all(preset.defaults[key] == value for key, value in expected_defaults.items())
    info = get_model_info(str(model_dir), workload_type=WorkloadType.T2V)
    assert info.pipeline_cls is LingBotVideoPipeline
    assert info.pipeline_config_cls is LingBotVideoT2VConfig
    assert isinstance(LingBotVideoT2VConfig().text_encoder_configs[0], LingBotVideoQwen3VLConfig)
    i2v_info = get_model_info(str(model_dir), workload_type=WorkloadType.I2V)
    assert i2v_info.pipeline_cls is LingBotVideoImageToVideoPipeline
    assert i2v_info.pipeline_config_cls is LingBotVideoTI2VConfig


def test_lingbot_video_prompt_crop_contract() -> None:
    """Crop exactly 140 template tokens and trim right padding for batch one."""
    from fastvideo.configs.models.encoders import BaseEncoderOutput
    from fastvideo.configs.pipelines.lingbot_video import (
        PROMPT_CROP_START,
        postprocess_lingbot_video_text,
        preprocess_lingbot_video_prompt,
    )

    assert PROMPT_CROP_START == 140
    assert "<|im_start|>user\nfox<|im_end|>" in preprocess_lingbot_video_prompt("fox")
    hidden = torch.arange(146 * 2, dtype=torch.float32).reshape(1, 146, 2)
    mask = torch.cat((torch.ones(1, 144), torch.zeros(1, 2)), dim=1)
    embeds, cropped_mask = postprocess_lingbot_video_text(BaseEncoderOutput(hidden_states=(hidden, )), mask)
    assert embeds.shape == (1, 4, 2)
    assert cropped_mask.shape == (1, 4)
    assert torch.equal(embeds, hidden[:, 140:144])


def test_lingbot_video_tokenizer_loader_uses_multimodal_processor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Load the required Qwen3-VL processor even without processor_config.json."""
    from fastvideo.configs.pipelines.lingbot_video import LingBotVideoTI2VConfig
    from fastvideo.models.loader import component_loader

    expected = object()
    monkeypatch.setattr(
        component_loader.AutoProcessor,
        "from_pretrained",
        lambda *args, **kwargs: expected,
    )
    args = SimpleNamespace(
        pipeline_config=LingBotVideoTI2VConfig(),
        trust_remote_code=False,
    )
    assert component_loader.TokenizerLoader().load(str(tmp_path), args) is expected


def test_lingbot_video_latent_geometry_and_validation(monkeypatch: pytest.MonkeyPatch) -> None:
    """Validate 4n+1 frames, multiples of 16, and fp32 latent geometry."""
    from fastvideo.pipelines.basic.lingbot_video import stages
    from fastvideo.pipelines.pipeline_batch_info import ForwardBatch

    monkeypatch.setattr(stages, "get_local_torch_device", lambda: torch.device("cpu"))
    transformer = SimpleNamespace(num_channels_latents=16)
    stage = stages.LingBotVideoLatentPreparationStage(transformer)
    supplied = torch.zeros(1, 16, 3, 8, 10, dtype=torch.bfloat16)
    condition = torch.full((1, 16, 1, 8, 10), 0.25, dtype=torch.float32)
    batch = ForwardBatch(
        data_type="video",
        prompt="fox",
        num_frames=9,
        height=64,
        width=80,
        latents=supplied,
        image_latent=condition,
    )
    result = stage.forward(batch, cast(Any, SimpleNamespace()))
    assert result.raw_latent_shape == (1, 16, 3, 8, 10)
    assert result.latents is not None and result.latents.dtype == torch.float32
    assert torch.equal(result.latents[:, :, :1], condition)


def test_lingbot_video_image_latent_stage_accepts_native_vae_distribution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Consume FastVideo's direct posterior return instead of requiring a wrapper."""
    from fastvideo.pipelines.basic.lingbot_video import stages
    from fastvideo.pipelines.pipeline_batch_info import ForwardBatch

    class DirectDistribution:
        """Expose the native Wan VAE sample contract used by the stage."""

        def sample(self, generator: torch.Generator) -> torch.Tensor:
            del generator
            return torch.full((1, 16, 1, 2, 2), 0.5)

    class NativeVAE:
        """Return the posterior directly, as FastVideo's Wan VAE does."""

        config = SimpleNamespace(latents_mean=[0.0] * 16, latents_std=[1.0] * 16)

        def encode(self, _pixels: torch.Tensor) -> DirectDistribution:
            return DirectDistribution()

    monkeypatch.setattr(stages, "get_local_torch_device", lambda: torch.device("cpu"))
    batch = ForwardBatch(
        data_type="video",
        preprocessed_image=torch.zeros(1, 3, 1, 16, 16),
        generator=torch.Generator().manual_seed(42),
    )
    result = stages.LingBotVideoImageLatentPreparationStage(NativeVAE()).forward(
        batch,
        cast(Any, SimpleNamespace(vae_cpu_offload=False)),
    )
    assert result.image_latent is not None
    assert torch.equal(result.image_latent, torch.full((1, 16, 1, 2, 2), 0.5))


def test_lingbot_video_uses_released_vae_denormalization_arithmetic() -> None:
    """Use reciprocal-then-divide for LingBot while preserving the generic formula."""
    from fastvideo.configs.pipelines.lingbot_video import LingBotVideoT2VConfig
    from fastvideo.pipelines.stages.decoding import DecodingStage

    vae = SimpleNamespace(config=SimpleNamespace(latents_mean=(-0.7571, ), latents_std=(2.8184, )), )
    stage = DecodingStage(vae)
    latents = torch.tensor([[[[[-0.7278813123703003]]]]], dtype=torch.float32)
    mean = torch.tensor(vae.config.latents_mean).view(1, -1, 1, 1, 1)
    std = torch.tensor(vae.config.latents_std).view(1, -1, 1, 1, 1)
    released = latents / (1.0 / std) + mean
    generic = latents * std + mean
    assert not torch.equal(released, generic)

    lingbot_args = cast(Any, SimpleNamespace(pipeline_config=LingBotVideoT2VConfig()))
    generic_args = cast(Any, SimpleNamespace(pipeline_config=SimpleNamespace()))
    assert torch.equal(stage._denormalize_latents(latents, lingbot_args), released)
    assert torch.equal(stage._denormalize_latents(latents, generic_args), generic)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="real pipeline smoke requires CUDA")
def test_lingbot_video_pipeline_load_generate_smoke(tmp_path: Path) -> None:
    """Load every converted component and run one tiny latent denoising step."""
    if os.environ.get("LINGBOT_VIDEO_RUN_GPU_TESTS") != "1":
        pytest.skip("set LINGBOT_VIDEO_RUN_GPU_TESTS=1 on an allocated GPU")
    from fastvideo import VideoGenerator

    use_fsdp_inference = os.environ.get("LINGBOT_VIDEO_USE_FSDP") == "1"
    num_gpus = int(os.environ.get("LINGBOT_VIDEO_NUM_GPUS", "1"))
    sp_size = int(os.environ.get("LINGBOT_VIDEO_SP_SIZE", "1"))
    if torch.cuda.device_count() < num_gpus:
        pytest.skip(f"LingBot-Video pipeline smoke requires {num_gpus} CUDA devices")
    model_dir = download_components(
        FASTVIDEO_DENSE,
        "scheduler",
        "text_encoder",
        "tokenizer",
        "transformer",
        "vae",
    )
    generator = VideoGenerator.from_pretrained(
        str(model_dir),
        num_gpus=num_gpus,
        sp_size=sp_size,
        use_fsdp_inference=use_fsdp_inference,
        dit_cpu_offload=False,
        dit_layerwise_offload=False,
        vae_cpu_offload=True,
        text_encoder_cpu_offload=True,
        pin_cpu_memory=False,
        output_type="latent",
    )
    try:
        result = generator.generate_video(
            prompt="A red fox runs through fresh snow at sunrise.",
            output_path=str(tmp_path),
            save_video=False,
            return_frames=True,
            height=32,
            width=32,
            num_frames=1,
            num_inference_steps=1,
            guidance_scale=3.0,
            seed=42,
        )
    finally:
        generator.shutdown()
    samples = cast(dict[str, Any], result)["samples"]
    assert torch.is_tensor(samples)
    assert tuple(samples.shape) == (1, 16, 1, 4, 4)
    assert torch.isfinite(samples).all()
