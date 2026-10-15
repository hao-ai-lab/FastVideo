# SPDX-License-Identifier: Apache-2.0
"""Profile local Wan 2.1 1.3B component loading and one tiny verification request.

One GPU, eager execution, cached local checkpoint. The 9-frame, 128x128,
4-step request verifies execution, not visual quality or production latency.
"""
import argparse
import json
from pathlib import Path
import time


def main():
    started = time.perf_counter()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True, help="Local Wan 2.1 1.3B Diffusers checkpoint root")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--tensor-output", type=Path)
    parser.add_argument("--synchronize", action="store_true")
    parser.add_argument("--fsdp", action="store_true", help="Exercise the single-rank FSDP inference loader")
    args = parser.parse_args()
    mark = time.perf_counter()
    import torch
    torch_seconds = time.perf_counter() - mark
    mark = time.perf_counter()
    from torch.distributed.fsdp import FSDPModule
    from fastvideo import VideoGenerator
    from fastvideo.models.loader.startup import collect_startup_timings, startup_phase
    fastvideo_seconds = time.perf_counter() - mark
    if not torch.cuda.is_available():
        raise RuntimeError("One CUDA GPU is required")
    report = {"imports": {"torch_seconds": torch_seconds, "fastvideo_and_helpers_seconds": fastvideo_seconds},
              "model_path": args.model_path, "torch": torch.__version__, "fsdp": args.fsdp,
              "cuda_initialized_before_pipeline": torch.cuda.is_initialized(),
              "workload": {"frames": 9, "height": 128, "width": 128, "steps": 4, "seed": 1024}}
    def synchronize_if_initialized():
        if torch.cuda.is_initialized():
            torch.cuda.synchronize()
    generator = None
    try:
        with collect_startup_timings(synchronize=synchronize_if_initialized if args.synchronize else None) as timings:
            with startup_phase("pipeline.create"):
                generator = VideoGenerator.from_pretrained(
                    args.model_path, num_gpus=1, distributed_executor_backend="uni", sp_size=1, tp_size=1,
                    use_fsdp_inference=args.fsdp, dit_cpu_offload=False, dit_layerwise_offload=False,
                    vae_cpu_offload=False,
                    text_encoder_cpu_offload=False, inference_torch_compile=False,
                    vae_tiling=False, vae_sp=False,
                )
                torch.cuda.synchronize()
            report["ready_seconds"] = time.perf_counter() - started
            effective = generator.fastvideo_args
            report["effective_config"] = {
                name: getattr(effective, name) for name in (
                    "use_fsdp_inference", "dit_layerwise_offload", "dit_cpu_offload",
                    "text_encoder_cpu_offload", "vae_cpu_offload", "inference_torch_compile",
                )
            }
            assert effective.use_fsdp_inference == args.fsdp, "Requested FSDP mode was overridden"
            assert not any(value for name, value in report["effective_config"].items()
                           if name != "use_fsdp_inference"), "Offload or compilation unexpectedly enabled"
            # This benchmark deliberately uses the in-process executor.
            transformer = generator.executor.driver_worker.worker.pipeline.modules["transformer"]
            report["transformer_is_fsdp_module"] = isinstance(transformer, FSDPModule)
            assert report["transformer_is_fsdp_module"] == args.fsdp, "Live model does not match FSDP request"
            with startup_phase("first_request"):
                result = generator.generate_video(
                    "A red cube on a wooden table.", num_frames=9, height=128, width=128,
                    num_inference_steps=4, seed=1024, save_video=False, return_frames=True,
                )
                torch.cuda.synchronize()
            samples = result["samples"]
            assert isinstance(samples, torch.Tensor)
            assert samples.shape == (1, 3, 9, 128, 128) and torch.isfinite(samples).all()
            report["verified_seconds"] = time.perf_counter() - started
            report["output_shape"] = list(samples.shape)
            report["device"] = torch.cuda.get_device_name(0)
            report["status"] = "ok"
            if args.tensor_output:
                args.tensor_output.parent.mkdir(parents=True, exist_ok=True)
                torch.save(samples.detach().cpu(), args.tensor_output)
    except Exception as error:
        report["status"] = "error"
        report["error"] = repr(error)
        raise
    finally:
        if "timings" in locals():
            report["startup"] = timings.to_dict()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        if generator is not None:
            generator.executor.shutdown()


if __name__ == "__main__":
    main()
