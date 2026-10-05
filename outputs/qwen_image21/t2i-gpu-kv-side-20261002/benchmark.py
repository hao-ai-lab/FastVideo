"""Isolated T2I benchmark: preloaded models, GPU KV cache, synchronized stages."""

import json
import os
import subprocess
import threading
import time
from pathlib import Path


def install_observer(worker):
    import torch
    from fastvideo.models.dits.qwen_image21 import QwenImage21KVLayerCache

    assert worker.fastvideo_args.pipeline_config.kv_cache_device == "cuda"
    worker._t2i_cache_stats = {}
    original_store = QwenImage21KVLayerCache.store
    original_get = QwenImage21KVLayerCache.get
    original_clear = QwenImage21KVLayerCache.clear

    def store(layer, k, v):
        original_store(layer, k, v)
        assert layer.k.device.type == layer.v.device.type == "cuda"
        stats = worker._t2i_cache_stats
        size = sum(t.numel() * t.element_size() for t in (layer.k, layer.v))
        stats["current_bytes"] += size - getattr(layer, "_benchmark_bytes", 0)
        layer._benchmark_bytes = size
        stats["peak_bytes"] = max(stats["peak_bytes"], stats["current_bytes"])
        stats["store_calls"] += 1
        stats["layer_shape"] = list(layer.k.shape)
        stats["dtype"] = str(layer.k.dtype)
        stats["device"] = str(layer.k.device)

    def get(layer, device=None, dtype=None):
        result = original_get(layer, device, dtype)
        worker._t2i_cache_stats["get_calls"] += 1
        assert result[0].data_ptr() == layer.k.data_ptr()
        assert result[1].data_ptr() == layer.v.data_ptr()
        return result

    def clear(layer):
        worker._t2i_cache_stats["current_bytes"] -= getattr(layer, "_benchmark_bytes", 0)
        layer._benchmark_bytes = 0
        original_clear(layer)

    QwenImage21KVLayerCache.store = store
    QwenImage21KVLayerCache.get = get
    QwenImage21KVLayerCache.clear = clear
    return {"kv_cache_device": worker.fastvideo_args.pipeline_config.kv_cache_device}


def reset_measurement(worker):
    import torch

    torch.cuda.synchronize()
    worker._t2i_cache_stats.clear()
    worker._t2i_cache_stats.update(current_bytes=0, peak_bytes=0, store_calls=0, get_calls=0)
    torch.cuda.reset_peak_memory_stats()


def snapshot(worker):
    import torch

    torch.cuda.synchronize()
    args = worker.fastvideo_args
    return {
        "cache": dict(worker._t2i_cache_stats),
        "gpu_peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
        "gpu_peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
        "dit_cpu_offload": args.dit_cpu_offload,
        "dit_layerwise_offload": args.dit_layerwise_offload,
        "text_encoder_cpu_offload": args.text_encoder_cpu_offload,
        "vae_cpu_offload": args.vae_cpu_offload,
        "lazy_module_load": args.lazy_module_load,
        "kv_cache_device": args.pipeline_config.kv_cache_device,
    }


def main():
    os.environ["FASTVIDEO_STAGE_LOGGING"] = "1"
    import cloudpickle
    import numpy as np
    import torch
    from fastvideo import VideoGenerator
    from fastvideo.api import (
        EngineConfig, GenerationRequest, GeneratorConfig, InputConfig, OffloadConfig,
        OutputConfig, ParallelismConfig, PipelineSelection, SamplingConfig,
    )

    out = Path(__file__).resolve().parent
    baseline = out.parent / "parity4-additional/test_qwen_image21_modes_1024_p0"
    spec = json.loads((baseline / "spec.json").read_text())
    noise = torch.load(spec["latents"], map_location="cpu", weights_only=True)
    expected = torch.load(baseline / "native.pt", map_location="cpu", weights_only=True)["pixels"]
    report = {
        "workload": "t2i", "height": 1024, "width": 1024, "steps": 40,
        "seed": spec["seed"], "prompt": spec["prompt"], "model_path": spec["model"],
        "kv_cache_device": "cuda", "warmup_steps": 4, "measurement_runs": 2,
        "observational_cache_checks": True, "stage_logging": True, "runs": [],
    }
    stop = threading.Event()
    memory_samples = []

    def monitor():
        while not stop.is_set():
            value = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                capture_output=True, text=True,
            )
            if value.returncode == 0:
                memory_samples.append({"time": time.perf_counter(), "used_mib": float(value.stdout.strip())})
            stop.wait(0.3)

    thread = threading.Thread(target=monitor, daemon=True)
    thread.start()
    generator = None
    try:
        started = time.perf_counter()
        generator = VideoGenerator.from_config(GeneratorConfig(
            model_path=spec["model"],
            engine=EngineConfig(
                num_gpus=1, use_fsdp_inference=False,
                parallelism=ParallelismConfig(tp_size=1, sp_size=1),
                offload=OffloadConfig(dit=False, dit_layerwise=False, text_encoder=True,
                                      vae=True, lazy_module_load=False),
            ),
            pipeline=PipelineSelection(workload_type="t2i", experimental={"kv_cache_device": "cuda"}),
        ))
        report["initialization_seconds"] = time.perf_counter() - started
        report["observer_installation"] = generator.executor.collective_rpc(cloudpickle.dumps(install_observer))[0]

        def run(steps, label, save):
            generator.executor.collective_rpc(cloudpickle.dumps(reset_measurement))
            request = GenerationRequest(
                prompt=spec["prompt"],
                inputs=InputConfig(latents=noise.clone()),
                sampling=SamplingConfig(
                    height=1024, width=1024, num_frames=1, fps=1, num_inference_steps=steps,
                    guidance_scale=1, true_cfg_scale=1, seed=spec["seed"], use_kv_cache=True,
                    reference_resolution=spec["reference_resolution"],
                    sigmas=np.linspace(1.0, 1.0 / steps, steps).tolist(),
                ),
                output=OutputConfig(output_path=str(out / f"{label}.png"), return_frames=True, save_video=save),
            )
            started = time.perf_counter()
            result = generator.generate(request)
            ended = time.perf_counter()
            stages = {
                key: item["execution_time"]
                for key, item in result.logging_info.stages.items() if "execution_time" in item
            }
            core = {key: stages[key] for key in ("text_encoding", "denoising", "decoding")}
            measurement = {
                "label": label, "steps": steps, "request_wall_seconds": ended - started,
                "generation_time_seconds": result.generation_time,
                "api_e2e_seconds": result.extra.get("e2e_latency"),
                "stage_seconds": stages,
                "encoder_dit_vae_seconds": sum(core.values()),
                "runtime": generator.executor.collective_rpc(cloudpickle.dumps(snapshot))[0],
                "sampled_peak_gpu_used_mib": max(
                    (sample["used_mib"] for sample in memory_samples if started <= sample["time"] <= ended),
                    default=None,
                ),
            }
            print("T2I_RUN " + json.dumps(measurement), flush=True)
            return result.samples.detach().float().cpu(), measurement

        pixels, report["warmup"] = run(4, "warmup", False)
        delta = (pixels - expected).abs()
        report["warmup_output_vs_validated_upstream_parity"] = {
            "exact": torch.equal(pixels, expected), "max_abs": delta.max().item(),
            "mean_abs": delta.mean().item(),
        }
        assert report["warmup_output_vs_validated_upstream_parity"]["exact"]
        previous = None
        for index in range(2):
            pixels, measurement = run(40, f"run-{index + 1}", True)
            report["runs"].append(measurement)
            if previous is not None:
                report["repeated_40_step_outputs_exact"] = torch.equal(pixels, previous)
                assert report["repeated_40_step_outputs_exact"]
            previous = pixels
            (out / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
        report["average_encoder_dit_vae_seconds"] = sum(r["encoder_dit_vae_seconds"] for r in report["runs"]) / 2
        report["average_request_wall_seconds"] = sum(r["request_wall_seconds"] for r in report["runs"]) / 2
        report["status"] = "success"
    except BaseException as exc:
        report["status"] = "failed"
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        if generator is not None:
            generator.shutdown()
        stop.set()
        thread.join(timeout=2)
        (out / "gpu-memory.json").write_text(json.dumps(memory_samples, indent=2) + "\n")
        (out / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
        print("T2I_GPU_KV_BENCHMARK " + json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
