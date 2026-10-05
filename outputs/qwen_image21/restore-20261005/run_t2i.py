"""Run one full Qwen-Image-2.1 T2I request after restoring the environment."""

import json
import os
import time
from pathlib import Path

os.environ["FASTVIDEO_STAGE_LOGGING"] = "1"

import torch
from PIL import Image

from fastvideo import VideoGenerator
from fastvideo.api import (
    EngineConfig,
    GenerationRequest,
    GeneratorConfig,
    OffloadConfig,
    OutputConfig,
    ParallelismConfig,
    PipelineSelection,
    SamplingConfig,
)


def main():
    out = Path(__file__).resolve().parent
    model_path = (out / "model-path.txt").read_text().strip()
    image_path = out / "t2i.png"
    report = {
        "model": "Qwen/Qwen-Image-2.1",
        "revision": "d26bb61231c349cf6b7896fa83353113880e1ba3",
        "torch": torch.__version__,
        "height": 1024,
        "width": 1024,
        "steps": 40,
        "seed": 42,
        "prompt": "A red ceramic teapot on a white studio background, product photography.",
        "dit_cpu_offload": False,
        "dit_layerwise_offload": False,
        "kv_cache_device": "cuda",
        "lazy_module_load": False,
    }
    generator = None
    try:
        started = time.perf_counter()
        generator = VideoGenerator.from_config(
            GeneratorConfig(
                model_path=model_path,
                engine=EngineConfig(
                    num_gpus=1,
                    use_fsdp_inference=False,
                    parallelism=ParallelismConfig(tp_size=1, sp_size=1),
                    offload=OffloadConfig(
                        dit=False,
                        dit_layerwise=False,
                        text_encoder=True,
                        vae=True,
                        lazy_module_load=False,
                    ),
                ),
                pipeline=PipelineSelection(
                    workload_type="t2i", experimental={"kv_cache_device": "cuda"}
                ),
            )
        )
        report["initialization_seconds"] = time.perf_counter() - started
        started = time.perf_counter()
        result = generator.generate(
            GenerationRequest(
                prompt=report["prompt"],
                sampling=SamplingConfig(
                    height=1024,
                    width=1024,
                    num_frames=1,
                    fps=1,
                    num_inference_steps=40,
                    guidance_scale=1.0,
                    true_cfg_scale=1.0,
                    seed=42,
                    reference_resolution=1024,
                    use_kv_cache=True,
                ),
                output=OutputConfig(
                    output_path=str(image_path), save_video=True, return_frames=True
                ),
            )
        )
        report["request_wall_seconds"] = time.perf_counter() - started
        report["generation_seconds"] = result.generation_time
        stages = {
            key: value["execution_time"]
            for key, value in result.logging_info.stages.items()
            if "execution_time" in value
        }
        report["stage_seconds"] = stages
        report["encoder_dit_decoder_seconds"] = sum(
            stages[key] for key in ("text_encoding", "denoising", "decoding")
        )
        samples = result.samples.detach().float().cpu()
        assert samples.shape == (1, 4, 1, 1024, 1024), samples.shape
        assert torch.isfinite(samples).all().item()
        assert samples.min().item() >= 0 and samples.max().item() <= 1
        assert samples[:, :3].std().item() > 0
        report["output_shape"] = list(samples.shape)
        report["finite_pixels"] = True
        report["rgb_std"] = samples[:, :3].std().item()
        with Image.open(image_path) as image:
            assert image.size == (1024, 1024), image.size
            assert image.mode == "RGBA", image.mode
            report["image_size"] = list(image.size)
            report["image_mode"] = image.mode
        report["image_path"] = str(image_path)
        report["status"] = "PASS"
    except BaseException as exc:
        report["status"] = "FAIL"
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        if generator is not None:
            generator.shutdown()
        (out / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
        print("T2I_RESTORE_RESULT " + json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
