"""Warm up Qwen DiT compilation, time one request, then disable compilation."""

import copy
import json
import os
import time
from pathlib import Path

os.environ["FASTVIDEO_STAGE_LOGGING"] = "1"
os.environ["FASTVIDEO_ATTENTION_BACKEND"] = "TORCH_SDPA"
os.environ["FASTVIDEO_INFERENCE_TORCH_COMPILE"] = "0"
os.environ["FASTVIDEO_FA4"] = "0"
os.environ["TORCHINDUCTOR_COMPILE_THREADS"] = "4"
os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(Path(__file__).resolve().parent / "inductor-cache")

import cloudpickle
import torch
import yaml
from PIL import Image

from fastvideo import VideoGenerator
from fastvideo.api import OutputConfig, load_run_config


def compilation_snapshot(worker):
    from torch._dynamo.utils import counters

    model = worker.pipeline.modules["transformer"]
    blocks = model.transformer_blocks
    return {
        "enabled": worker.fastvideo_args.enable_torch_compile,
        "inference_torch_compile": worker.fastvideo_args.inference_torch_compile,
        "kwargs": worker.fastvideo_args.torch_compile_kwargs,
        "wrapped_blocks": sum(hasattr(block.forward, "_torchdynamo_orig_callable") for block in blocks),
        "total_blocks": len(blocks),
        "dynamo_stats": dict(counters["stats"]),
        "dynamo_frames": dict(counters["frames"]),
        "graph_breaks": dict(counters["graph_break"]),
    }


def main():
    root = Path(__file__).resolve().parents[3]
    out = Path(__file__).resolve().parent
    config_path = root / "examples/inference/basic/qwen_image21_t2i.yaml"
    model_path = (root / "outputs/qwen_image21/restore-20261005/model-path.txt").read_text().strip()
    overrides = {
        "generator.model_path": model_path,
        "generator.engine.compile.enabled": True,
        "generator.engine.compile.backend": "inductor",
        "generator.engine.compile.fullgraph": False,
        "generator.engine.use_fsdp_inference": False,
        "generator.engine.offload.dit": False,
        "generator.engine.offload.dit_layerwise": False,
        "generator.engine.offload.lazy_module_load": False,
        "generator.engine.offload.pin_cpu_memory": True,
        "generator.pipeline.experimental.kv_cache_device": "cuda",
        "request.prompt": "A red ceramic teapot on a white studio background, product photography.",
        "request.sampling.seed": 42,
        "request.sampling.num_inference_steps": 40,
        "request.sampling.height": 1024,
        "request.sampling.width": 1024,
    }
    run = load_run_config(config_path, overrides=overrides)
    report = {
        "torch": torch.__version__,
        "height": 1024,
        "width": 1024,
        "steps": 40,
        "seed": 42,
        "backend": "inductor",
        "fullgraph": False,
        "attention_backend": "TORCH_SDPA",
        "dit_offload": False,
        "dit_layerwise_offload": False,
        "pin_cpu_memory": True,
        "kv_cache_device": "cuda",
        "warmup_requests": 1,
        "measured_requests": 1,
        "config_path": str(config_path),
    }
    generator = None
    try:
        started = time.perf_counter()
        generator = VideoGenerator.from_config(run.generator)
        report["initialization_seconds"] = time.perf_counter() - started
        report["before_warmup"] = generator.executor.collective_rpc(
            cloudpickle.dumps(compilation_snapshot)
        )[0]
        assert report["before_warmup"]["enabled"]

        def generate(label, save):
            request = copy.deepcopy(run.request)
            request.output = OutputConfig(
                output_path=str(out / f"{label}.png"), save_video=save, return_frames=True
            )
            started = time.perf_counter()
            result = generator.generate(request)
            wall_seconds = time.perf_counter() - started
            stages = {
                key: value["execution_time"]
                for key, value in result.logging_info.stages.items()
                if "execution_time" in value
            }
            measurement = {
                "wall_seconds": wall_seconds,
                "generation_seconds": result.generation_time,
                "stage_seconds": stages,
                "encoder_dit_decoder_seconds": sum(
                    stages[key] for key in ("text_encoding", "denoising", "decoding")
                ),
                "compile": generator.executor.collective_rpc(cloudpickle.dumps(compilation_snapshot))[0],
            }
            pixels = result.samples.detach().float().cpu()
            assert pixels.shape == (1, 4, 1, 1024, 1024), pixels.shape
            assert torch.isfinite(pixels).all().item()
            assert pixels[:, :3].std().item() > 0
            print(label.upper() + " " + json.dumps(measurement), flush=True)
            return pixels, measurement

        warmup_pixels, report["warmup"] = generate("warmup", False)
        assert report["warmup"]["compile"]["wrapped_blocks"] == 32
        assert report["warmup"]["compile"]["dynamo_stats"].get("unique_graphs", 0) > 0
        (out / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
        measured_pixels, report["measured"] = generate("compiled", True)
        delta = (measured_pixels - warmup_pixels).abs()
        report["warmup_vs_measured"] = {
            "exact": torch.equal(measured_pixels, warmup_pixels),
            "max_abs": delta.max().item(),
            "mean_abs": delta.mean().item(),
        }
        with Image.open(out / "compiled.png") as image:
            assert image.size == (1024, 1024) and image.mode == "RGBA"
        report["status"] = "PASS"
    except BaseException as exc:
        report["status"] = "FAIL"
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        run.generator.engine.compile.enabled = False
        # Only change the compile switch if a concurrent edit enabled it.
        # The benchmark uses typed overrides, keeping the public default off.
        text = config_path.read_text()
        if "    compile:\n      enabled: true\n" in text:
            config_path.write_text(text.replace(
                "    compile:\n      enabled: true\n", "    compile:\n      enabled: false\n", 1
            ))
        report["compile_enabled_after_run"] = yaml.safe_load(config_path.read_text())[
            "generator"
        ]["engine"]["compile"]["enabled"]
        if generator is not None:
            generator.shutdown()
        (out / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
        print("COMPILE_BENCHMARK " + json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
