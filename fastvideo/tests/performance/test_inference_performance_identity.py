# SPDX-License-Identifier: Apache-2.0

from fastvideo.tests.performance import compare_baseline
from fastvideo.tests.performance.test_inference_performance import _build_v2_identity_fields


def _benchmark_config():
    return {
        "benchmark_id": "wan-t2v-1.3b-2gpu",
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

    identity_fields = _build_v2_identity_fields(
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
    assert len(record["recipe_fingerprint"]) == 64
    assert "output_path" not in record["recipe"]["generation_kwargs"]
