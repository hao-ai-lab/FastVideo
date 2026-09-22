# SPDX-License-Identifier: Apache-2.0
"""Weight-free Wan-VACE registry and context-shape contracts."""

import json

import pytest
import torch

from fastvideo import registry
from fastvideo.api.presets import get_preset
from fastvideo.api.sampling_param import SamplingParam
from fastvideo.models.wan import pipeline_config
from fastvideo.pipelines.basic.wan.stages.vace_conditioning import WanVACEContextStage


@pytest.fixture(autouse=True)
def no_hub_access(monkeypatch):
    monkeypatch.setattr(registry, "maybe_download_model_index",
                        lambda *args, **kwargs: pytest.fail("Hub access not allowed in contract tests"))
    registry.get_model_info.cache_clear()
    yield
    registry.get_model_info.cache_clear()


def test_vace_1_3b_hf_id_resolves_config_and_preset(tmp_path):
    model_path = "Wan-AI/Wan2.1-VACE-1.3B-diffusers"
    info = registry._get_config_info(model_path)
    assert info.pipeline_config_cls is pipeline_config.WanVACE1_3B_Config
    assert info.default_preset == "wan_vace_1_3b"
    assert info.model_family == "wan"
    preset = get_preset("wan_vace_1_3b", "wan")
    sampling = SamplingParam.from_pretrained(model_path)
    for name, value in preset.defaults.items():
        assert getattr(sampling, name) == value

    manifest = {
        "_class_name": "WanVACEPipeline",
        "_diffusers_version": "0.34.0.dev0",
    }
    checkpoint_dir = tmp_path / "Wan2.1-VACE-1.3B-diffusers"
    checkpoint_dir.mkdir()
    (checkpoint_dir / "model_index.json").write_text(json.dumps(manifest))
    manifest_info = registry._get_config_info(str(checkpoint_dir))
    assert manifest_info.pipeline_config_cls is pipeline_config.WanVACE1_3B_Config


def test_vace_14b_hf_id_resolves_config_and_preset(tmp_path):
    model_path = "Wan-AI/Wan2.1-VACE-14B-diffusers"
    info = registry._get_config_info(model_path)
    assert info.pipeline_config_cls is pipeline_config.WanVACE14B_Config
    assert info.default_preset == "wan_vace_14b"
    assert info.model_family == "wan"
    preset = get_preset("wan_vace_14b", "wan")
    sampling = SamplingParam.from_pretrained(model_path)
    for name, value in preset.defaults.items():
        assert getattr(sampling, name) == value

    manifest = {
        "_class_name": "WanVACEPipeline",
        "_diffusers_version": "0.34.0.dev0",
    }
    (tmp_path / "model_index.json").write_text(json.dumps(manifest))
    manifest_info = registry._get_config_info(str(tmp_path / "Wan2.1-VACE-14B-diffusers"))
    assert manifest_info.pipeline_config_cls is pipeline_config.WanVACE14B_Config


def test_vace_manifest_routing_requires_param_size(tmp_path):
    manifest = {
        "_class_name": "WanVACEPipeline",
        "_diffusers_version": "0.34.0.dev0",
    }
    checkpoint_dir = tmp_path / "local-vace-checkpoint"
    checkpoint_dir.mkdir()
    (checkpoint_dir / "model_index.json").write_text(json.dumps(manifest))
    with pytest.raises(RuntimeError, match="No model info found"):
        registry._get_config_info(str(checkpoint_dir))


def test_vace_context_stage_builds_96_channel_control(monkeypatch):
    class TinyVAE:

        class Config:
            z_dim = 16

        config = Config()
        latents_mean = [0.0] * 16
        latents_std = [1.0] * 16

        def to(self, device):
            return self

        def encode(self, x):
            b, _, t, h, w = x.shape
            latent_t = (t - 1) // 4 + 1
            latent = torch.zeros(b, 16, latent_t, max(1, h // 8), max(1, w // 8))

            class _Dist:

                def mode(self_nonlocal):
                    return latent

            return _Dist()

    from fastvideo.fastvideo_args import FastVideoArgs
    from fastvideo.pipelines.pipeline_batch_info import ForwardBatch

    args = FastVideoArgs.from_kwargs(model_path="Wan-AI/Wan2.1-VACE-1.3B-diffusers")
    batch = ForwardBatch(
        data_type="video",
        height=32,
        width=32,
        num_frames=5,
        video_latent=torch.zeros(1, 3, 5, 32, 32),
        mask_video=torch.ones(1, 3, 5, 32, 32),
        vace_reference_images=[],
        conditioning_scale=1.0,
    )
    stage = WanVACEContextStage(TinyVAE())
    monkeypatch.setattr("fastvideo.pipelines.basic.wan.stages.vace_conditioning.get_local_torch_device",
                        lambda: torch.device("cpu"))
    stage.forward(batch, args)
    assert batch.vace_control_latents is not None
    assert batch.vace_control_latents.shape[1] == 96
