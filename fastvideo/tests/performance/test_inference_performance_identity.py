# SPDX-License-Identifier: Apache-2.0

from fastvideo.tests.performance import compare_baseline
from fastvideo.tests.performance import test_inference_performance as perf


def _benchmark_config():
    return {
        "benchmark_id": "wan-t2v-1.3b-2gpu",
        "config_schema_version": perf.V2_CONFIG_SCHEMA_VERSION,
        "workload_id": "wan-t2v-1.3b",
        "variant_id": "canonical",
        "benchmark_version": 1,
        "model": {
            "model_path": "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
            "model_short_name": "Wan2.1-T2V-1.3B",
        },
        "init_kwargs": {
            "num_gpus": 2,
            "flow_shift": 7.0,
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
        "run_config": {
            "num_warmup_runs": 2,
            "num_measurement_runs": 5,
            "required_gpus": 2,
        },
    }


def test_performance_producer_emits_v2_identity_from_raw_result_shape(monkeypatch):
    monkeypatch.setenv("FASTVIDEO_ATTENTION_BACKEND", "FLASH_ATTN")
    cfg = _benchmark_config()
    init_kwargs = dict(cfg["init_kwargs"])
    generation_kwargs = dict(cfg["generation_kwargs"])
    generation_kwargs["output_path"] = "/tmp/generated"
    prompt = "A cinematic video."

    identity_fields = perf._build_v2_identity_fields(
        cfg,
        init_kwargs,
        generation_kwargs,
        prompt,
        "NVIDIA L40S PCIe",
    )
    raw_result = {
        "benchmark_id": cfg["benchmark_id"],
        "model_short_name": cfg["model"]["model_short_name"],
        "device": "NVIDIA L40S PCIe",
        "num_gpus": init_kwargs["num_gpus"],
        "num_warmup_runs": 2,
        "num_measurement_runs": 5,
        "avg_generation_time_s": 10.0,
        "individual_times_s": [10.0],
        "throughput_fps": 4.5,
        "max_peak_memory_mb": 10000.0,
        "individual_peak_memories_mb": [10000.0],
        "thresholds": {},
        "commit": "a" * 40,
        "pr_number": "123",
        "timestamp": "2026-06-16T00:00:00+00:00",
        "text_encoder_time_s": 1.0,
        "dit_time_s": 2.0,
        "vae_decode_time_s": 3.0,
        **identity_fields,
    }

    record = compare_baseline.normalize_performance_result(raw_result)

    assert record["workload_id"] == "wan-t2v-1.3b"
    assert record["variant_id"] == "canonical"
    assert record["benchmark_version"] == 1
    assert record["hardware_profile_id"].startswith("hw-")
    assert record["software_profile_id"].startswith("sw-")
    assert record["software_profile_id"] == perf._profile_id(
        "sw",
        record["software_comparison_profile"],
    )
    assert len(record["recipe_fingerprint"]) == 64
    assert "output_path" not in record["recipe"]["generation_kwargs"]


def test_software_profile_tracks_exact_runtime_attention_kernel_and_container_versions(monkeypatch):
    package_versions = {
        "triton": "3.2.1",
        "flash-attn": "2.8.1",
        "sageattention": "2.1.1",
        "xformers": "0.0.31",
        "fastvideo-kernel": "0.3.2",
        "flashinfer-python": "0.2.3",
    }

    def fake_version(package_name):
        if package_name not in package_versions:
            raise perf.importlib_metadata.PackageNotFoundError(package_name)
        return package_versions[package_name]

    monkeypatch.setattr(perf.importlib_metadata, "version", fake_version)
    monkeypatch.setattr(perf.platform, "python_version", lambda: "3.12.11")
    monkeypatch.setattr(perf.torch, "__version__", "2.7.1+cu128")
    monkeypatch.setattr(perf.torch.version, "cuda", "12.8.1", raising=False)
    monkeypatch.setenv("FASTVIDEO_ATTENTION_BACKEND", "SAGE_ATTN")
    monkeypatch.setenv("FASTVIDEO_FA4", "1")
    monkeypatch.setenv("FASTVIDEO_PERFORMANCE_PROFILE_VERSION", "perf-profile-v2")
    monkeypatch.setenv("IMAGE_VERSION", "py3.12-cuda13.0")
    monkeypatch.setenv("FASTVIDEO_CONTAINER_IMAGE_REF", "ghcr.io/hao-ai-lab/fastvideo/fastvideo-dev@sha256:abc")

    profile = perf._software_profile()
    comparison_profile = perf._software_comparison_profile(profile)

    assert profile["profile_schema_version"] == perf.SOFTWARE_PROFILE_SCHEMA_VERSION
    assert profile["python"] == "3.12.11"
    assert profile["pytorch"] == "2.7.1+cu128"
    assert profile["cuda"] == "12.8.1"
    assert profile["attention_backend"] == "SAGE_ATTN"
    assert profile["flash_attention_4_enabled"] is True
    assert profile["performance_profile_version"] == "perf-profile-v2"
    assert profile["container_image_version"] == "py3.12-cuda13.0"
    assert profile["container_image_ref"] == "ghcr.io/hao-ai-lab/fastvideo/fastvideo-dev@sha256:abc"
    assert profile["packages"] == {
        "triton": "3.2.1",
        "flash-attn": "2.8.1",
        "sageattention": "2.1.1",
        "xformers": "0.0.31",
        "fastvideo-kernel": "0.3.2",
        "flashinfer-python": "0.2.3",
    }
    assert comparison_profile == {
        "comparison_profile_schema_version": perf.SOFTWARE_COMPARISON_PROFILE_SCHEMA_VERSION,
        "python": "3.12",
        "pytorch": "2.7",
        "cuda": "12.8",
        "attention_backend": "SAGE_ATTN",
        "flash_attention_4_enabled": True,
        "packages": {
            "triton": "3.2",
            "flash-attn": "2.8",
            "sageattention": "2.1",
            "xformers": "0.0",
            "fastvideo-kernel": "0.3",
            "flashinfer-python": "0.2",
        },
        "performance_profile_version": "perf-profile-v2",
        "container_image_version": "py3.12-cuda13.0",
    }

    patch_changed_audit_profile = {
        **profile,
        "packages": {
            **profile["packages"],
            "triton": "3.2.2",
        },
        "container_image_ref": "ghcr.io/hao-ai-lab/fastvideo/fastvideo-dev@sha256:def",
    }
    assert perf._profile_id("sw", comparison_profile) == perf._profile_id(
        "sw",
        perf._software_comparison_profile(patch_changed_audit_profile),
    )
    assert perf._profile_id("sw", comparison_profile) != perf._profile_id(
        "sw",
        {
            **comparison_profile,
            "flash_attention_4_enabled": False,
        },
    )
    assert perf._profile_id("sw", comparison_profile) != perf._profile_id(
        "sw",
        {
            **comparison_profile,
            "performance_profile_version": "perf-profile-v3",
        },
    )
