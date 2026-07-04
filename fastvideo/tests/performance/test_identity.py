# SPDX-License-Identifier: Apache-2.0

from copy import deepcopy

import pytest
import torch

from fastvideo.tests.performance import identity as identity_module
from fastvideo.tests.performance.identity import (
    build_recipe_from_benchmark_config,
    canonical_json,
    environment_fingerprint,
    environment_metadata,
    hardware_profile,
    hardware_profile_id,
    recipe_fingerprint,
    resolved_revision_from_model_path,
    software_profile,
    software_profile_id,
)


def _benchmark_config():
    return {
        "benchmark_id": "wan-t2v-1.3b-2gpu",
        "workload_id": "wan-t2v",
        "variant_id": "1.3b-sp2",
        "benchmark_version": 2,
        "model": {
            "model_path": "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
            "model_short_name": "Wan2.1-T2V-1.3B",
        },
        "init_kwargs": {
            "num_gpus": 2,
            "sp_size": 2,
            "tp_size": 1,
            "vae_sp": True,
            "vae_tiling": True,
            "text_encoder_precisions": ["fp32"],
        },
        "generation_kwargs": {
            "height": 480,
            "width": 832,
            "num_frames": 45,
            "num_inference_steps": 4,
            "guidance_scale": 3,
            "embedded_cfg_scale": 6,
            "seed": 1024,
            "fps": 24,
            "neg_prompt": "low quality",
        },
        "test_prompts": ["A cinematic video."],
    }


def _fingerprint(cfg, attention_backend="FLASH_ATTN"):
    recipe = build_recipe_from_benchmark_config(cfg, attention_backend=attention_backend)
    return recipe_fingerprint(recipe)


def test_canonical_json_is_stable_for_mapping_order():
    left = {"b": [2, 1], "a": {"z": "last", "m": "middle"}}
    right = {"a": {"m": "middle", "z": "last"}, "b": [2, 1]}

    assert canonical_json(left) == canonical_json(right)


def test_canonical_json_handles_sets_deterministically():
    assert canonical_json({"values": {"b", "a"}}) == '{"values":["a","b"]}'
    assert canonical_json({"values": frozenset(("b", "a"))}) == '{"values":["a","b"]}'


def test_same_config_produces_same_recipe_fingerprint():
    cfg = _benchmark_config()

    assert _fingerprint(cfg) == _fingerprint(deepcopy(cfg))


def test_recipe_includes_first_class_benchmark_identity():
    recipe = build_recipe_from_benchmark_config(_benchmark_config())

    assert recipe["benchmark"] == {
        "benchmark_id": "wan-t2v-1.3b-2gpu",
        "workload_id": "wan-t2v",
        "variant_id": "1.3b-sp2",
        "benchmark_version": 2,
    }


def test_recipe_requires_first_class_benchmark_identity():
    cfg = _benchmark_config()
    del cfg["workload_id"]

    with pytest.raises(ValueError, match="workload_id"):
        build_recipe_from_benchmark_config(cfg)


def test_semantically_equivalent_sequence_forms_match():
    cfg = _benchmark_config()
    equivalent = deepcopy(cfg)
    equivalent["init_kwargs"]["text_encoder_precisions"] = tuple(
        equivalent["init_kwargs"]["text_encoder_precisions"])

    assert _fingerprint(cfg) == _fingerprint(equivalent)


def test_output_path_does_not_change_recipe_fingerprint():
    cfg = _benchmark_config()
    with_output_path = deepcopy(cfg)
    with_output_path["generation_kwargs"]["output_path"] = "/tmp/generated"

    assert _fingerprint(cfg) == _fingerprint(with_output_path)


def test_measured_prompt_override_ignores_unused_configured_prompts():
    cfg = _benchmark_config()
    changed = deepcopy(cfg)
    changed["test_prompts"] = [
        "Unused prompt that should not affect this measured workload.",
        cfg["test_prompts"][0],
    ]

    recipe = build_recipe_from_benchmark_config(cfg, measured_prompts=[cfg["test_prompts"][0]])
    changed_recipe = build_recipe_from_benchmark_config(changed, measured_prompts=[cfg["test_prompts"][0]])

    assert recipe_fingerprint(recipe) == recipe_fingerprint(changed_recipe)


def test_num_inference_steps_changes_recipe_fingerprint():
    cfg = _benchmark_config()
    changed = deepcopy(cfg)
    changed["generation_kwargs"]["num_inference_steps"] = 8

    assert _fingerprint(cfg) != _fingerprint(changed)


def test_benchmark_version_changes_recipe_fingerprint():
    cfg = _benchmark_config()
    changed = deepcopy(cfg)
    changed["benchmark_version"] = 3

    assert _fingerprint(cfg) != _fingerprint(changed)


def test_attention_backend_changes_recipe_fingerprint():
    cfg = _benchmark_config()

    assert _fingerprint(cfg, attention_backend="FLASH_ATTN") != _fingerprint(
        cfg, attention_backend="TORCH_SDPA")


def test_precision_changes_recipe_fingerprint():
    cfg = _benchmark_config()
    changed = deepcopy(cfg)
    changed["init_kwargs"]["text_encoder_precisions"] = ["bf16"]

    assert _fingerprint(cfg) != _fingerprint(changed)


def test_dimensions_and_seed_change_recipe_fingerprint():
    cfg = _benchmark_config()
    different_dimensions = deepcopy(cfg)
    different_dimensions["generation_kwargs"]["width"] = 1024
    different_seed = deepcopy(cfg)
    different_seed["generation_kwargs"]["seed"] = 2048

    assert _fingerprint(cfg) != _fingerprint(different_dimensions)
    assert _fingerprint(cfg) != _fingerprint(different_seed)


def test_distributed_layout_changes_recipe_fingerprint():
    cfg = _benchmark_config()
    changed = deepcopy(cfg)
    changed["init_kwargs"]["sp_size"] = 1
    changed["init_kwargs"]["tp_size"] = 2

    assert _fingerprint(cfg) != _fingerprint(changed)


def test_software_profile_uses_exact_attention_kernel_package_versions_for_id():
    profile_a = software_profile(
        python_version="3.12.4",
        torch_version="2.12.0+cu130",
        cuda_version="13.0",
        package_versions={
            "triton": "3.4.1",
            "fastvideo_kernel": "0.3.2",
            "flash_attn_4": "4.0.0.dev0",
            "flashinfer": "0.2.11",
            "nvidia_cutlass_dsl": "4.5.0",
        },
    )
    profile_b = software_profile(
        python_version="3.12.5",
        torch_version="2.12.1+cu130",
        cuda_version="13.0",
        package_versions={
            "triton": "3.4.9",
            "fastvideo_kernel": "0.3.7",
            "flash_attn_4": "4.0.0.dev1",
            "flashinfer": "0.2.12",
            "nvidia_cutlass_dsl": "4.5.1",
        },
    )

    assert profile_a == {
        "python": "3.12",
        "pytorch": "2.12",
        "cuda": "13.0",
        "packages": {
            "fastvideo_kernel": "0.3.2",
            "flash_attn_4": "4.0.0.dev0",
            "flashinfer": "0.2.11",
            "nvidia_cutlass_dsl": "4.5.0",
            "triton": "3.4.1",
        },
    }
    assert profile_b["python"] == "3.12"
    assert profile_b["pytorch"] == "2.12"
    assert profile_b["cuda"] == "13.0"
    assert software_profile_id(profile_a) != software_profile_id(profile_b)


def test_installed_package_versions_checks_attention_kernel_distributions(monkeypatch):
    versions = {
        "fastvideo-kernel": "0.3.2",
        "flash-attn": "2.8.1",
        "flash-attn-4": "4.0.0.dev0",
        "flash-attention-fp4": "0.1.0",
        "flashinfer-python": "0.2.11",
        "nvidia-cutlass-dsl": "4.5.0",
    }

    def fake_version(name):
        if name not in versions:
            raise identity_module.importlib.metadata.PackageNotFoundError(name)
        return versions[name]

    monkeypatch.setattr(identity_module.importlib.metadata, "version", fake_version)

    installed = identity_module._installed_package_versions()

    assert installed["fastvideo_kernel"] == "0.3.2"
    assert installed["flash_attn"] == "2.8.1"
    assert installed["flash_attn_4"] == "4.0.0.dev0"
    assert installed["flash_attention_fp4"] == "0.1.0"
    assert installed["flashinfer"] == "0.2.11"
    assert installed["nvidia_cutlass_dsl"] == "4.5.0"


def test_hardware_profile_id_uses_normalized_gpu_cohort():
    profile = hardware_profile(
        num_gpus=2,
        gpu_devices=[
            {
                "name": "NVIDIA L40S",
                "memory_bytes": 48 * 1024**3,
                "compute_capability": "8.9",
            },
            {
                "name": "NVIDIA L40S",
                "memory_gb": 48,
                "compute_capability": "8.9",
            },
        ],
        interconnect="none_or_partial",
    )

    assert profile == {
        "device_type": "cuda",
        "gpu_count": 2,
        "gpus": [
            {
                "name": "NVIDIA L40S",
                "memory_gb": 48,
                "compute_capability": "8.9",
            },
            {
                "name": "NVIDIA L40S",
                "memory_gb": 48,
                "compute_capability": "8.9",
            },
        ],
        "interconnect": "none_or_partial",
    }
    assert hardware_profile_id(profile).startswith("hw-")


def test_hardware_profile_pads_missing_requested_cuda_devices(monkeypatch):
    class Props:
        name = "NVIDIA L40S"
        total_memory = 48 * 1024**3

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda device_id: Props())
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device_id: (8, 9))

    profile = hardware_profile(num_gpus=2, interconnect="unknown")

    assert profile["gpu_count"] == 2
    assert profile["gpus"] == [
        {
            "name": "NVIDIA L40S",
            "memory_gb": 48,
            "compute_capability": "8.9",
        },
        {
            "name": "unknown",
            "memory_gb": None,
            "compute_capability": None,
        },
    ]


def test_local_path_detection_checks_both_path_separators(monkeypatch):
    monkeypatch.setattr(identity_module.os, "sep", "\\")

    assert identity_module._looks_like_local_path("models/local/checkpoint") is True
    assert identity_module._looks_like_local_path(r"C:\models\checkpoint") is True
    assert identity_module._looks_like_local_path("Wan-AI/Wan2.1-T2V-1.3B-Diffusers") is False


def test_environment_metadata_is_separate_audit_fingerprint():
    cfg = _benchmark_config()
    recipe_hash = _fingerprint(cfg)
    audit_a = environment_metadata(
        env={"IMAGE_VERSION": "py3.12-cuda13.0.0"},
        package_versions={"triton": "3.4.1"},
        hardware={"device_type": "cuda", "gpu_count": 2},
        software={"python": "3.12", "pytorch": "2.12", "cuda": "13.0"},
    )
    audit_b = environment_metadata(
        env={"IMAGE_VERSION": "py3.12-cuda13.0.1"},
        package_versions={"triton": "3.4.1"},
        hardware={"device_type": "cuda", "gpu_count": 2},
        software={"python": "3.12", "pytorch": "2.12", "cuda": "13.0"},
    )

    assert recipe_hash == _fingerprint(cfg)
    assert environment_fingerprint(audit_a) != environment_fingerprint(audit_b)


def test_resolved_revision_from_hf_snapshot_path():
    revision = "a" * 40

    assert resolved_revision_from_model_path(
        f"/root/.cache/huggingface/hub/models--org--repo/snapshots/{revision}") == revision
    assert resolved_revision_from_model_path("/models/local-repo") is None


def test_runtime_identity_from_generator_summarizes_worker_records():
    from fastvideo.tests.performance.test_inference_performance import _runtime_identity_from_generator

    revision = "b" * 40

    class FakeExecutor:
        def collective_rpc(self, _method):
            return [
                {
                    "resolved_attention_backends": ["FLASH_ATTN"],
                    "resolved_model_path": f"/root/.cache/huggingface/hub/models--org--repo/snapshots/{revision}",
                },
                {
                    "resolved_attention_backends": ["FLASH_ATTN"],
                    "resolved_model_path": f"/root/.cache/huggingface/hub/models--org--repo/snapshots/{revision}",
                },
            ]

    class FakeGenerator:
        executor = FakeExecutor()

    assert _runtime_identity_from_generator(FakeGenerator()) == {
        "resolved_attention_backend": "FLASH_ATTN",
        "resolved_model_revision": revision,
    }


def test_collect_worker_identity_handles_torch_module_pipeline():
    from fastvideo.tests.performance.test_inference_performance import _collect_worker_identity

    class Backend:
        name = "FLASH_ATTN"

    class Leaf(torch.nn.Module):
        backend = Backend()

    class Pipeline(torch.nn.Module):
        model_path = "/models/local"

        def __init__(self):
            super().__init__()
            self.leaf = Leaf()

    class Worker:
        pipeline = Pipeline()

    assert _collect_worker_identity(Worker()) == {
        "resolved_attention_backends": ["FLASH_ATTN"],
        "resolved_model_path": "/models/local",
    }
