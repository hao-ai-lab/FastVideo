# SPDX-License-Identifier: Apache-2.0
import json

import yaml

from fastvideo.api import (
    config_to_dict,
    ContinuationState,
    GenerationRequest,
    GeneratorConfig,
    GpuPoolConfig,
    load_run_config,
    load_serve_config,
    parse_config,
    PlannedStage,
    PromptEnhancerConfig,
    PromptSafetyConfig,
    RunConfig,
    ServeConfig,
    StreamingConfig,
    WarmupConfig,
)


def test_parse_config_builds_nested_typed_config() -> None:
    raw = {
        "generator": {
            "model_path": "/models/ltx2",
            "pipeline": {
                "workload_type": "t2v",
                "preset": "ltx2_two_stage",
            },
        },
        "request": {
            "prompt": ["a fox", "a wolf"],
            "sampling": {
                "num_frames": 121,
                "height": 1024,
                "width": 1536,
                "guidance_scale": 1.5,
            },
            "state": {
                "kind": "ltx2_continuation",
                "payload": {"segment_index": 1},
            },
            "plan": {
                "stages": [
                    {
                        "name": "base",
                        "kind": "sample",
                    }
                ]
            },
        },
    }

    config = parse_config(RunConfig, raw)

    assert config.generator.pipeline.preset == "ltx2_two_stage"
    assert config.request.prompt == ["a fox", "a wolf"]
    assert config.request.state == ContinuationState(
        kind="ltx2_continuation",
        payload={"segment_index": 1},
    )
    assert config.request.plan is not None
    assert config.request.plan.stages == [PlannedStage(name="base", kind="sample")]


def test_parse_config_accepts_existing_typed_instance() -> None:
    typed = RunConfig(
        generator=GeneratorConfig(model_path="/models/base"),
        request=GenerationRequest(prompt="hello"),
    )

    assert parse_config(RunConfig, typed) is typed


def test_load_run_config_supports_yaml_roundtrip(tmp_path) -> None:
    raw = {
        "generator": {"model_path": "/models/wan"},
        "request": {
            "prompt": "hello",
            "sampling": {"num_frames": 16},
        },
    }
    path = tmp_path / "run.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    loaded = load_run_config(path)

    assert config_to_dict(loaded) == {
        "generator": {
            "model_path": "/models/wan",
            "revision": None,
            "trust_remote_code": False,
            "engine": {
                "num_gpus": 1,
                "execution_backend": "mp",
                "parallelism": {
                    "tp_size": -1,
                    "sp_size": -1,
                    "hsdp_replicate_dim": 1,
                    "hsdp_shard_dim": -1,
                    "dist_timeout": None,
                },
                "offload": {
                    "dit": True,
                    "dit_layerwise": True,
                    "text_encoder": True,
                    "image_encoder": True,
                    "vae": True,
                    "pin_cpu_memory": True,
                },
                "compile": {
                    "enabled": False,
                    "backend": None,
                    "fullgraph": None,
                    "mode": None,
                    "dynamic": None,
                    "extras": {},
                    "text_encoder_enabled": None,
                    "vae_enabled": None,
                    "audio_vae_enabled": None,
                    "dit_kwargs": {},
                    "text_encoder_kwargs": {},
                    "vae_kwargs": {},
                    "audio_vae_kwargs": {},
                },
                "enable_stage_verification": True,
                "use_fsdp_inference": False,
                "disable_autocast": False,
                "quantization": None,
            },
            "pipeline": {
                "workload_type": None,
                "preset": None,
                "preset_version": None,
                "components": {
                    "config_root": None,
                    "pipeline_config_path": None,
                    "text_encoder_weights": None,
                    "transformer_weights": None,
                    "transformer_2_weights": None,
                    "vae_weights": None,
                    "upsampler_weights": None,
                    "lora_path": None,
                    "override_pipeline_cls_name": None,
                    "override_transformer_cls_name": None,
                },
                "vae_tiling": None,
                "preset_overrides": {},
                "experimental": {},
            },
        },
        "request": {
            "prompt": "hello",
            "negative_prompt": None,
            "inputs": {
                "prompt_path": None,
                "image_path": None,
                "video_path": None,
                "pil_image": None,
                "pose": None,
                "mouse_cond": None,
                "keyboard_cond": None,
                "grid_sizes": None,
                "c2ws_plucker_emb": None,
                "refine_from": None,
                "stage1_video": None,
            },
            "sampling": {
                "num_videos_per_prompt": 1,
                "seed": 1024,
                "num_frames": 16,
                "height": 720,
                "width": 1280,
                "height_sr": 1072,
                "width_sr": 1920,
                "fps": 24,
                "num_inference_steps": 50,
                "num_inference_steps_sr": 50,
                "guidance_scale": 1.0,
                "guidance_scale_2": None,
                "guidance_rescale": 0.0,
                "true_cfg_scale": None,
                "boundary_ratio": None,
                "sigmas": None,
            },
            "runtime": {
                "enable_teacache": False,
                "return_trajectory_latents": False,
                "return_trajectory_decoded": False,
            },
            "output": {
                "output_path": "outputs/",
                "output_video_name": None,
                "save_video": True,
                "return_frames": True,
                "return_state": False,
            },
            "stage_overrides": {},
            "state": None,
            "plan": None,
            "extensions": {},
        },
    }


def test_load_serve_config_supports_json_roundtrip(tmp_path) -> None:
    raw = {
        "generator": {"model_path": "/models/server"},
        "server": {"port": 9000},
        "default_request": {"prompt": "serve default"},
    }
    path = tmp_path / "serve.json"
    path.write_text(json.dumps(raw), encoding="utf-8")

    loaded = load_serve_config(path)

    assert isinstance(loaded, ServeConfig)
    assert loaded.server.port == 9000
    assert loaded.default_request.prompt == "serve default"


def test_serve_config_streaming_defaults_to_none() -> None:
    raw = {"generator": {"model_path": "/models/server"}}
    loaded = parse_config(ServeConfig, raw)
    assert loaded.streaming is None


def test_serve_config_parses_streaming_block() -> None:
    raw = {
        "generator": {"model_path": "/models/server"},
        "streaming": {
            "session_timeout_seconds": 120,
            "generation_segment_cap": 4,
            "stream_mode": "legacy_jpeg",
            "warmup": {
                "enabled": False,
                "prompt": "warmup prompt",
                "timeout_seconds": 600,
            },
            "pool": {
                "num_workers": 2,
                "enable_audio_reencode": False,
                "conditioning_num_frames": 5,
                "conditioning_end_offset": 1,
            },
            "prompt": {
                "provider": "groq",
                "model": "llama-3-70b",
                "timeout_ms": 10000,
                "system_prompt_dir": "/opt/prompts",
            },
            "safety": {
                "enabled": True,
                "classifier_path": "/opt/safety.pt",
            },
        },
    }

    loaded = parse_config(ServeConfig, raw)

    assert isinstance(loaded.streaming, StreamingConfig)
    assert loaded.streaming.session_timeout_seconds == 120
    assert loaded.streaming.generation_segment_cap == 4
    assert loaded.streaming.stream_mode == "legacy_jpeg"
    assert loaded.streaming.warmup == WarmupConfig(
        enabled=False, prompt="warmup prompt", timeout_seconds=600)
    assert loaded.streaming.pool == GpuPoolConfig(
        num_workers=2,
        enable_audio_reencode=False,
        conditioning_num_frames=5,
        conditioning_end_offset=1,
    )
    assert loaded.streaming.prompt == PromptEnhancerConfig(
        provider="groq",
        model="llama-3-70b",
        timeout_ms=10000,
        system_prompt_dir="/opt/prompts",
    )
    assert loaded.streaming.safety == PromptSafetyConfig(
        enabled=True, classifier_path="/opt/safety.pt")


def test_serve_config_streaming_round_trip_through_config_to_dict() -> None:
    raw = {
        "generator": {"model_path": "/models/server"},
        "streaming": {"session_timeout_seconds": 600},
    }
    loaded = parse_config(ServeConfig, raw)
    dumped = config_to_dict(loaded)
    assert dumped["streaming"]["session_timeout_seconds"] == 600
    assert dumped["streaming"]["warmup"]["enabled"] is True
    assert dumped["streaming"]["prompt"]["enabled"] is False
    assert dumped["streaming"]["prompt"]["provider"] == "cerebras"
    assert dumped["streaming"]["safety"]["enabled"] is False


def test_load_serve_config_with_streaming_from_yaml(tmp_path) -> None:
    raw = {
        "generator": {"model_path": "/models/server"},
        "streaming": {
            "stream_mode": "av_fmp4",
            "pool": {"num_workers": 4},
        },
    }
    path = tmp_path / "serve.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    loaded = load_serve_config(path)

    assert loaded.streaming is not None
    assert loaded.streaming.stream_mode == "av_fmp4"
    assert loaded.streaming.pool.num_workers == 4
