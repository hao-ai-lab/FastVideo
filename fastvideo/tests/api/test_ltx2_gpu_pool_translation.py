# SPDX-License-Identifier: Apache-2.0
"""LTX-2 streaming-server config flattening tests.

``GPU_POOL_CONFIG`` is the typed ``GeneratorConfig`` form of the settings that the FastVideo-internal
``ui/ltx2-streaming/server/gpu_pool.py`` loads. The tests check that ``generator_config_to_fastvideo_args`` turns it
into the flat ``FastVideoArgs`` keywords that the LTX-2 runtime reads.
"""
from __future__ import annotations

import pytest

from fastvideo.api.compat import (
    from_pretrained_kwargs_to_config,
    generator_config_to_fastvideo_args,
    normalize_generator_config,
)

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


class TestGpuPoolFlattening:
    """The typed gpu_pool config -> the FastVideoArgs keywords that the LTX-2 runtime reads."""

    @pytest.fixture
    def args_kwargs(self, monkeypatch):
        from fastvideo import fastvideo_args as fva

        captured: dict[str, object] = {}

        def _capture(**kw):
            captured.update(kw)
            return _Captured(**kw)

        class _Captured:

            def __init__(self, **kw):
                self.kwargs = kw

        monkeypatch.setattr(fva.FastVideoArgs, "from_kwargs", _capture)

        generator_config_to_fastvideo_args(normalize_generator_config(GPU_POOL_CONFIG))
        return captured

    def test_empty_refine_lora_path_reemitted(self, args_kwargs) -> None:
        assert args_kwargs["ltx2_refine_lora_path"] == ""

    def test_ltx2_refine_flags_reemitted(self, args_kwargs) -> None:
        assert args_kwargs["ltx2_refine_enabled"] is True
        assert args_kwargs["ltx2_refine_add_noise"] is True
        assert args_kwargs["ltx2_refine_num_inference_steps"] == 2
        assert args_kwargs["ltx2_refine_guidance_scale"] == 1.0

    def test_refine_upsampler_path_reemitted(self, args_kwargs) -> None:
        assert args_kwargs["ltx2_refine_upsampler_path"] == ("/models/ltx2-distilled/spatial_upsampler")

    def test_config_model_path_reemitted(self, args_kwargs) -> None:
        assert args_kwargs["config_model_path"] == "/models/ltx2-distilled/config"

    def test_torch_compile_kwargs_reassembled(self, args_kwargs) -> None:
        assert args_kwargs["torch_compile_kwargs"] == {
            "backend": "inductor",
            "fullgraph": True,
            "mode": "max-autotune-no-cudagraphs",
            "dynamic": False,
        }

    def test_vae_tiling_reemitted_with_legacy_name(self, args_kwargs) -> None:
        assert args_kwargs["ltx2_vae_tiling"] is False

    def test_text_encoder_compile_reemitted(self, args_kwargs) -> None:
        # Present in the captured kwargs dict even though
        # ``FastVideoArgs.from_kwargs`` will filter it out — realtime
        # runtime upstream (PR 7.6) reads it off this dict.
        assert args_kwargs["enable_torch_compile_text_encoder"] is True

    def test_no_stray_refine_dict(self, args_kwargs) -> None:
        """preset_overrides.refine must flatten to ltx2_refine_* kwargs
        rather than landing as a nested ``refine`` kwarg that
        FastVideoArgs doesn't understand."""
        assert "refine" not in args_kwargs


class TestRefineFlattenCoversAllTypedFields:
    """Every field on LTX2Refine{Preset,Stage}Override must survive the
    round-trip through preset_overrides.refine back to ltx2_refine_*
    kwargs. Guards against the hardcoded-key-tuple regression where
    image_crf / video_position_offset_sec silently dropped."""

    def test_all_fields_reemitted(self, monkeypatch) -> None:
        from fastvideo import fastvideo_args as fva
        from fastvideo.api.compat import (
            generator_config_to_fastvideo_args, )
        from fastvideo.api.schema import GeneratorConfig, PipelineSelection
        from fastvideo.pipelines.basic.ltx2.stage_overrides import (
            refine_preset_override_fields,
            refine_stage_override_fields,
        )

        captured: dict[str, object] = {}

        class _Captured:

            def __init__(self, **kw):
                self.kwargs = kw

        def _capture(**kw):
            captured.update(kw)
            return _Captured(**kw)

        monkeypatch.setattr(fva.FastVideoArgs, "from_kwargs", _capture)
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
        assert set(refine_payload) == all_fields, ("payload must cover every typed field to exercise the flatten loop")

        config = GeneratorConfig(
            model_path="/models/ltx2",
            pipeline=PipelineSelection(preset_overrides={"refine": refine_payload}),
        )
        generator_config_to_fastvideo_args(config)

        for key, value in refine_payload.items():
            assert captured[f"ltx2_refine_{key}"] == value


class TestCompileExtrasPreserved:
    """Additional torch.compile kwargs beyond the four typed fields
    round-trip through ``CompileConfig.extras``."""

    def test_extras_preserved(self, monkeypatch) -> None:
        from fastvideo import fastvideo_args as fva

        captured: dict[str, object] = {}

        def _capture(**kw):
            captured.update(kw)

            class _Captured:

                def __init__(self, **kw):
                    self.kwargs = kw

            return _Captured(**kw)

        monkeypatch.setattr(fva.FastVideoArgs, "from_kwargs", _capture)

        kwargs = {
            "enable_torch_compile": True,
            "torch_compile_kwargs": {
                "backend": "inductor",
                "options": {
                    "triton.cudagraphs": False
                },
                "disable": False,
            },
        }
        config = from_pretrained_kwargs_to_config("FastVideo/LTX2-Distilled-Diffusers", kwargs)
        assert config.engine.compile.backend == "inductor"
        assert config.engine.compile.extras == {
            "options": {
                "triton.cudagraphs": False
            },
            "disable": False,
        }

        generator_config_to_fastvideo_args(config)
        assert captured["torch_compile_kwargs"] == {
            "backend": "inductor",
            "options": {
                "triton.cudagraphs": False
            },
            "disable": False,
        }
