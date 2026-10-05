"""Warm 40-step Qwen T2I timing and pixel comparisons for selected backends.

All runs share explicit CPU BF16 noise and sigmas. Images are not saved.
Use --backend sdpa/flash; FA4 additionally requires FASTVIDEO_FA4=1.
Compiled variants use the existing public CompileConfig, with warmup excluded.
"""

import argparse
import json
import os
from pathlib import Path
import time


def backend_receipt(worker):
    import torch
    from fastvideo.attention.layer import LocalAttention

    transformer = worker.pipeline.get_module("transformer")
    resolved = transformer.config._resolved_attention_backend
    return {
        "resolved": resolved.name if resolved is not None else None,
        "dense_layers": {name: module.backend.name
                         for name, module in transformer.named_modules()
                         if isinstance(module, LocalAttention)},
        "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["sdpa", "flash", "sage"], required=True)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--ready-file", type=Path)
    args = parser.parse_args()
    os.environ["FASTVIDEO_STAGE_LOGGING"] = "1"
    os.environ["FASTVIDEO_ATTENTION_BACKEND"] = {
        "sdpa": "TORCH_SDPA", "flash": "FLASH_ATTN", "sage": "SAGE_ATTN"}[args.backend]
    import cloudpickle
    import numpy as np
    import torch
    from fastvideo import VideoGenerator
    from fastvideo.api import (CompileConfig, EngineConfig, GenerationRequest, GeneratorConfig,
                              InputConfig, OffloadConfig, OutputConfig, ParallelismConfig,
                              PipelineSelection, SamplingConfig)

    out = Path(__file__).resolve().parent
    label = args.backend + ("-compiled" if args.compile else "")
    model = (out.parent / "restore-20261005/model-path.txt").read_text().strip()
    report = {"backend": args.backend, "compiled": args.compile, "torch": torch.__version__,
              "resolution": [1024, 1024], "steps": 40, "seed": 42, "runs": []}
    noise = torch.randn(1, 4096, 64, generator=torch.Generator(device="cpu").manual_seed(42),
                        dtype=torch.bfloat16)
    generator = None
    try:
        started = time.perf_counter()
        generator = VideoGenerator.from_config(GeneratorConfig(
            model_path=model,
            engine=EngineConfig(num_gpus=1, use_fsdp_inference=False,
                parallelism=ParallelismConfig(tp_size=1, sp_size=1),
                offload=OffloadConfig(dit=False, dit_layerwise=False, text_encoder=True,
                                      vae=True, lazy_module_load=False),
                compile=CompileConfig(enabled=args.compile, fullgraph=False, dynamic=False)),
            pipeline=PipelineSelection(workload_type="t2i", experimental={"kv_cache_device": "cuda"}),
        ))
        report["initialization_seconds"] = time.perf_counter() - started
        if args.ready_file is not None:
            print("Waiting for background build completion before measured requests", flush=True)
            deadline = time.perf_counter() + 1800
            while not args.ready_file.is_file():
                if time.perf_counter() >= deadline:
                    raise RuntimeError("Timed out waiting for build-completion receipt")
                time.sleep(.2)

        def run(steps, name):
            request = GenerationRequest(
                prompt="A red ceramic teapot on a white studio background, product photography.",
                inputs=InputConfig(latents=noise.clone()),
                sampling=SamplingConfig(height=1024, width=1024, num_frames=1, fps=1,
                    num_inference_steps=steps, guidance_scale=1, true_cfg_scale=1,
                    seed=42, reference_resolution=1024, use_kv_cache=True,
                    sigmas=np.linspace(1., 1. / steps, steps).tolist()),
                output=OutputConfig(return_frames=True, save_video=False),
            )
            started = time.perf_counter()
            result = generator.generate(request)
            wall = time.perf_counter() - started
            stages = {key: value["execution_time"] for key, value in result.logging_info.stages.items()
                      if "execution_time" in value}
            metrics = {"name": name, "steps": steps, "wall_seconds": wall,
                "stage_seconds": stages,
                "core_seconds": sum(stages[key] for key in ("text_encoding", "denoising", "decoding"))}
            pixels = result.samples.detach().float().cpu()
            assert torch.isfinite(pixels).all()
            print("MEASUREMENT " + json.dumps(metrics), flush=True)
            return pixels, metrics

        _, report["warmup"] = run(4, "warmup")
        pixels, first = run(40, "run-1")
        report["runs"].append(first)
        again, second = run(40, "run-2")
        report["runs"].append(second)
        report["repeat_exact"] = torch.equal(pixels, again)
        report["average_core_seconds"] = sum(r["core_seconds"] for r in report["runs"]) / 2
        report["average_denoising_seconds"] = sum(r["stage_seconds"]["denoising"] for r in report["runs"]) / 2
        report["receipt"] = generator.executor.collective_rpc(cloudpickle.dumps(backend_receipt))[0]
        baseline_path = out / "baseline-pixels.pt"
        if label == "sdpa":
            torch.save(pixels, baseline_path)
        elif baseline_path.is_file():
            baseline = torch.load(baseline_path, weights_only=True, map_location="cpu")
            delta = (pixels - baseline).abs()
            report["pixel_difference_vs_sdpa"] = {"exact": torch.equal(pixels, baseline),
                "mean_abs": delta.mean().item(), "max_abs": delta.max().item(),
                "rmse": delta.square().mean().sqrt().item(),
                "fraction_outside_original_pipeline_tolerance":
                    (delta > .04 + .02 * baseline.abs()).float().mean().item()}
            torch.testing.assert_close(pixels, baseline, atol=.04, rtol=.02)
        report["status"] = "PASS"
    except BaseException as exc:
        report["status"] = "FAIL"
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        if generator is not None:
            generator.shutdown()
        (out / f"pipeline-{label}.json").write_text(json.dumps(report, indent=2) + "\n")
        print("PIPELINE_BENCHMARK " + json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
