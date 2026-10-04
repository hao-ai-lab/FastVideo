# SPDX-License-Identifier: Apache-2.0
"""LTX-2 streaming-server config resolution tests.

``GPU_POOL_CONFIG`` is the typed ``GeneratorConfig`` form of the settings that the FastVideo-internal
``ui/ltx2-streaming/server/gpu_pool.py`` loads. The tests check that resolution puts each setting at the typed path
that the LTX-2 runtime reads.
"""
from __future__ import annotations

import pytest

from fastvideo.api.compat import normalize_generator_config
from fastvideo.api.inference_resolution import resolve_inference_config, torch_compile_kwargs
from fastvideo.tests.api.config_snapshot import isolated_environment

GPU_POOL_CONFIG = {
    "model_path": "FastVideo/LTX2-Distilled-Diffusers",
    "engine": {
        "num_gpus": 1,
        "use_fsdp_inference": False,
        "offload": {
            "dit": False,
            "dit_layerwise": False,
            "vae": False,
            "text_encoder": False,
            "pin_cpu_memory": True,
        },
        "compile": {
            "enabled": True,
            "text_encoder_enabled": True,
            "backend": "inductor",
            "fullgraph": True,
            "mode": "max-autotune-no-cudagraphs",
            "dynamic": False,
        },
    },
    "pipeline": {
        "components": {
            "config_root": "/models/ltx2-distilled/config",
            "upsampler_weights": "/models/ltx2-distilled/spatial_upsampler",
        },
        "vae_tiling": False,
        # An empty refine LoRA path keeps the refine LoRA disabled; None would load the checkpoint default.
        "ltx2": {
            "refine": {
                "lora_path": ""
            }
        },
        "preset_overrides": {
            "refine": {
                "enabled": True,
                "num_inference_steps": 2,
                "guidance_scale": 1.0,
                "add_noise": True,
            }
        },
    },
}


class TestGpuPoolResolution:
    """The typed gpu_pool config -> the resolved typed paths that the LTX-2 runtime reads."""

    @pytest.fixture
    def resolved(self):
        with isolated_environment():
            return resolve_inference_config(GPU_POOL_CONFIG)

    def test_empty_refine_lora_path_kept(self, resolved) -> None:
        assert resolved.pipeline.ltx2.refine.lora_path == ""

    def test_ltx2_refine_flags_copied_from_preset_overrides(self, resolved) -> None:
        refine = resolved.pipeline.ltx2.refine
        assert refine.enabled is True
        assert refine.add_noise is True
        assert refine.num_inference_steps == 2
        assert refine.guidance_scale == 1.0

    def test_refine_upsampler_path_kept(self, resolved) -> None:
        assert resolved.pipeline.components.upsampler_weights == ("/models/ltx2-distilled/spatial_upsampler")

    def test_config_root_kept(self, resolved) -> None:
        assert resolved.pipeline.components.config_root == "/models/ltx2-distilled/config"

    def test_torch_compile_kwargs_reassembled(self, resolved) -> None:
        assert torch_compile_kwargs(resolved) == {
            "backend": "inductor",
            "fullgraph": True,
            "mode": "max-autotune-no-cudagraphs",
            "dynamic": False,
        }

    def test_vae_tiling_kept(self, resolved) -> None:
        assert resolved.pipeline.vae_tiling is False

    def test_text_encoder_compile_kept(self, resolved) -> None:
        assert resolved.engine.compile.text_encoder_enabled is True


class TestRefinePresetOverridesCoverAllTypedFields:
    """Every field on LTX2Refine{Preset,Stage}Override must survive the
    copy from preset_overrides.refine into ``pipeline.ltx2.refine``.
    Guards against the hardcoded-key-tuple regression where
    image_crf / video_position_offset_sec silently dropped."""

    def test_all_fields_copied(self, monkeypatch) -> None:
        from fastvideo.api.schema import GeneratorConfig, PipelineSelection
        from fastvideo.pipelines.basic.ltx2.stage_overrides import (
            refine_preset_override_fields,
            refine_stage_override_fields,
        )

        # The model path is not a registered model, so skip the model definition.
        from fastvideo.api import inference_resolution
        monkeypatch.setattr(inference_resolution, "build_model_pipeline_config", lambda config: None)
        monkeypatch.setattr(inference_resolution, "pipeline_config_defaults_step", lambda config, defaults=None: lambda view: {})
        monkeypatch.setattr(inference_resolution, "materialize_pipeline_config", lambda resolved, pipeline_config: None)

        refine_payload = {
            # Preset-override fields.
            "enabled": True,
            "add_noise": False,
            # Stage-override fields.
            "num_inference_steps": 3,
            "guidance_scale": 1.5,
            "image_crf": 18,
            "video_position_offset_sec": 2.5,
        }
        all_fields = (refine_preset_override_fields() | refine_stage_override_fields())
        assert set(refine_payload) == all_fields, ("payload must cover every typed field to exercise the copy loop")

        config = GeneratorConfig(
            model_path="/models/ltx2",
            pipeline=PipelineSelection(preset_overrides={"refine": refine_payload}),
        )
        resolved = resolve_inference_config(config)

        for key, value in refine_payload.items():
            assert getattr(resolved.pipeline.ltx2.refine, key) == value


class TestCompileExtrasPreserved:
    """Additional torch.compile kwargs beyond the four typed fields
    round-trip through ``CompileConfig.extras``."""

    def test_extras_preserved(self) -> None:
        config = normalize_generator_config({
            "model_path": "FastVideo/LTX2-Distilled-Diffusers",
            "engine": {
                "compile": {
                    "enabled": True,
                    "backend": "inductor",
                    "extras": {
                        "options": {
                            "triton.cudagraphs": False
                        },
                        "disable": False,
                    },
                }
            },
        })
        assert config.engine.compile.backend == "inductor"
        assert config.engine.compile.extras == {
            "options": {
                "triton.cudagraphs": False
            },
            "disable": False,
        }

        with isolated_environment():
            resolved = resolve_inference_config(config)
        assert torch_compile_kwargs(resolved) == {
            "backend": "inductor",
            "options": {
                "triton.cudagraphs": False
            },
            "disable": False,
        }
