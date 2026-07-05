# SPDX-License-Identifier: Apache-2.0

from fastvideo.tests.performance import test_inference_performance as perf_test


def _benchmark_config():
    return {
        "benchmark_id": "wan-t2v-1.3b-2gpu",
        "workload_id": "wan-t2v",
        "variant_id": "1.3b-sp2",
        "benchmark_version": 2,
        "quality_metadata": {
            "quality_status": "canonical",
        },
        "model": {
            "model_path": "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
            "model_short_name": "Wan2.1-T2V-1.3B",
        },
        "generation_kwargs": {
            "num_frames": 45,
        },
    }


def _identity_fields():
    return {
        "workload_id": "wan-t2v",
        "variant_id": "1.3b-sp2",
        "benchmark_version": 2,
        "recipe": {
            "recipe_schema_version": 1,
            "benchmark": {
                "benchmark_id": "wan-t2v-1.3b-2gpu",
                "workload_id": "wan-t2v",
                "variant_id": "1.3b-sp2",
                "benchmark_version": 2,
            },
        },
        "recipe_fingerprint": "recipe-1",
        "hardware_profile": {
            "device_type": "cuda",
            "gpu_count": 2,
        },
        "hardware_profile_id": "hw-l40s",
        "software_profile": {
            "python": "3.12",
            "pytorch": "2.12",
            "cuda": "13.0",
        },
        "software_profile_id": "sw-cu130",
        "environment_metadata": {
            "env": {
                "IMAGE_VERSION": "py3.12-cuda13.0.0",
            },
        },
        "environment_fingerprint": "env-ci",
    }


def test_build_result_record_emits_v2_wan_shape(monkeypatch):
    monkeypatch.setenv("PERF_RUN_SOURCE", "scheduled_main")
    monkeypatch.setenv("BUILDKITE_COMMIT", "a" * 40)
    monkeypatch.setenv("BUILDKITE_PULL_REQUEST", "false")
    monkeypatch.setenv("BUILDKITE_BRANCH", "main")
    monkeypatch.setenv("TEST_SCOPE", "full")
    monkeypatch.setenv("BUILDKITE_BUILD_URL", "https://buildkite.example/build")
    monkeypatch.setenv("BUILDKITE_BUILD_ID", "build-1")
    monkeypatch.setenv("BUILDKITE_JOB_ID", "job-1")
    monkeypatch.setattr(perf_test, "_build_identity_fields", lambda *_args: _identity_fields())

    record = perf_test._build_result_record(
        cfg=_benchmark_config(),
        model_info={"model_short_name": "Wan2.1-T2V-1.3B"},
        init_kwargs={"num_gpus": 2},
        gen_kwargs={"num_frames": 45},
        num_warmup=2,
        num_measure=3,
        thresholds={
            "max_generation_time_s": 34.0,
            "max_peak_memory_mb": 11000.0,
        },
        times=[30.0, 31.0, 32.0],
        peak_memories=[10000.0, 10100.0, 10050.0],
        all_component_times=[
            {
                "text_encoder_time_s": 1.0,
                "dit_time_s": 8.0,
                "vae_decode_time_s": 3.0,
            },
            {
                "text_encoder_time_s": 1.2,
                "dit_time_s": 8.2,
                "vae_decode_time_s": 3.2,
            },
            {
                "text_encoder_time_s": None,
                "dit_time_s": 8.4,
                "vae_decode_time_s": 3.4,
            },
        ],
        prompt="A cinematic video.",
        runtime_identity={
            "resolved_attention_backend": "FLASH_ATTN",
        },
        device_name="NVIDIA L40S",
        timestamp="2026-07-05T00:00:00+00:00",
    )

    assert record["result_schema_version"] == perf_test.RESULT_SCHEMA_VERSION
    assert record["benchmark_id"] == "wan-t2v-1.3b-2gpu"
    assert record["workload_id"] == "wan-t2v"
    assert record["variant_id"] == "1.3b-sp2"
    assert record["benchmark_version"] == 2
    assert record["recipe_fingerprint"] == "recipe-1"
    assert record["hardware_profile_id"] == "hw-l40s"
    assert record["software_profile_id"] == "sw-cu130"
    assert record["environment_fingerprint"] == "env-ci"
    assert record["environment_metadata"]["env"]["IMAGE_VERSION"] == "py3.12-cuda13.0.0"
    assert record["quality_metadata"] == {"quality_status": "canonical"}
    assert record["run_source"] == "scheduled_main"
    assert record["branch"] == "main"
    assert record["test_scope"] == "full"
    assert record["build_url"] == "https://buildkite.example/build"
    assert record["build_id"] == "build-1"
    assert record["job_id"] == "job-1"
    assert record["pr_number"] == ""
    assert record["commit"] == "a" * 40
    assert record["avg_generation_time_s"] == 31.0
    assert record["throughput_fps"] == 1.452
    assert record["max_peak_memory_mb"] == 10100.0
    assert record["text_encoder_time_s"] == 1.1
    assert record["dit_time_s"] == 8.2
    assert record["vae_decode_time_s"] == 3.2
