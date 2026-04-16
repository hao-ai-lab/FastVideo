# SPDX-License-Identifier: Apache-2.0
import json

import yaml

from fastvideo.api import (
    config_to_dict,
    ContinuationState,
    GenerationRequest,
    GeneratorConfig,
    load_run_config,
    load_serve_config,
    parse_config,
    PlannedStage,
    RunConfig,
    ServeConfig,
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
                "compile": {"enabled": False, "kwargs": {}},
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
