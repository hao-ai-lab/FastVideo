# SPDX-License-Identifier: Apache-2.0

import copy

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
    cfg["generation_kwargs"]["output_path"] = "/tmp/generated"
    prompt = "A cinematic video."

    identity_fields = perf._build_identity_fields(
        cfg,
        init_kwargs,
        prompt,
        {
            "resolved_attention_backend": "FLASH_ATTN",
            "resolved_model_revision": None,
        },
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
    assert record["software_profile_id"] == perf.software_profile_id(record["software_profile"])
    assert len(record["recipe_fingerprint"]) == 64
    assert "output_path" not in record["recipe"]["generation_kwargs"]


def test_display_benchmark_id_rename_preserves_producer_fingerprint_and_cohort(monkeypatch):
    monkeypatch.setenv("FASTVIDEO_ATTENTION_BACKEND", "FLASH_ATTN")
    original = _benchmark_config()
    renamed = copy.deepcopy(original)
    renamed["benchmark_id"] = "wan-t2v-renamed-display-id"

    def build_identity(cfg):
        return perf._build_identity_fields(
            cfg,
            dict(cfg["init_kwargs"]),
            "A cinematic video.",
            {
                "resolved_attention_backend": "FLASH_ATTN",
                "resolved_model_revision": None,
            },
        )

    original_identity = build_identity(original)
    renamed_identity = build_identity(renamed)

    assert original_identity["recipe"]["benchmark"]["benchmark_id"] == original["benchmark_id"]
    assert renamed_identity["recipe"]["benchmark"]["benchmark_id"] == renamed["benchmark_id"]
    assert original_identity["recipe_fingerprint"] == renamed_identity["recipe_fingerprint"]
    assert compare_baseline._comparison_identity_filters(
        original_identity) == compare_baseline._comparison_identity_filters(renamed_identity)


def test_v2_identity_tolerates_null_run_config(monkeypatch):
    monkeypatch.setenv("FASTVIDEO_ATTENTION_BACKEND", "FLASH_ATTN")
    cfg = _benchmark_config()
    cfg["run_config"] = None
    init_kwargs = dict(cfg["init_kwargs"])

    identity_fields = perf._build_identity_fields(
        cfg,
        init_kwargs,
        "A cinematic video.",
        {
            "resolved_attention_backend": "FLASH_ATTN",
            "resolved_model_revision": None,
        },
    )

    assert identity_fields["hardware_profile"]["gpu_count"] == 2


def test_producer_tracks_runtime_software_identity_and_container_audit(monkeypatch):
    base_profile = {
        "python": "3.12",
        "pytorch": "2.7",
        "cuda": "12.8",
        "packages": {
            "fastvideo_kernel": "0.3.2",
            "triton": "3.2.1",
        },
    }
    monkeypatch.setattr(perf, "software_profile", lambda: dict(base_profile))
    monkeypatch.setenv("FASTVIDEO_ATTENTION_BACKEND", "SAGE_ATTN")
    monkeypatch.setenv("FASTVIDEO_FA4", "1")
    monkeypatch.setenv("FASTVIDEO_PERFORMANCE_PROFILE_VERSION", "perf-profile-v2")
    monkeypatch.setenv("IMAGE_VERSION", "py3.12-cuda13.0")
    monkeypatch.setenv(
        "FASTVIDEO_CONTAINER_IMAGE_REF",
        "ghcr.io/hao-ai-lab/fastvideo/fastvideo-dev@sha256:abc",
    )

    cfg = _benchmark_config()
    identity_fields = perf._build_identity_fields(
        cfg,
        dict(cfg["init_kwargs"]),
        "A cinematic video.",
        {
            "resolved_attention_backend": "SAGE_ATTN",
            "resolved_model_revision": None,
        },
    )

    expected_profile = {
        **base_profile,
        "attention_backend": "SAGE_ATTN",
        "flash_attention_4_enabled": True,
        "performance_profile_version": "perf-profile-v2",
        "container_image_version": "py3.12-cuda13.0",
    }
    assert identity_fields["software_profile"] == expected_profile
    assert identity_fields["software_profile_id"] == perf.software_profile_id(expected_profile)
    assert identity_fields["environment_metadata"]["env"]["FASTVIDEO_CONTAINER_IMAGE_REF"].endswith(
        "sha256:abc"
    )

    first_profile_id = identity_fields["software_profile_id"]
    monkeypatch.setenv(
        "FASTVIDEO_CONTAINER_IMAGE_REF",
        "ghcr.io/hao-ai-lab/fastvideo/fastvideo-dev@sha256:def",
    )
    changed_audit = perf._build_identity_fields(
        cfg,
        dict(cfg["init_kwargs"]),
        "A cinematic video.",
        {
            "resolved_attention_backend": "SAGE_ATTN",
            "resolved_model_revision": None,
        },
    )
    assert changed_audit["software_profile_id"] == first_profile_id
    assert changed_audit["environment_metadata"]["env"]["FASTVIDEO_CONTAINER_IMAGE_REF"].endswith(
        "sha256:def"
    )

    monkeypatch.setenv("FASTVIDEO_FA4", "0")
    changed_runtime = perf._build_identity_fields(
        cfg,
        dict(cfg["init_kwargs"]),
        "A cinematic video.",
        {
            "resolved_attention_backend": "SAGE_ATTN",
            "resolved_model_revision": None,
        },
    )
    assert changed_runtime["software_profile_id"] != first_profile_id
