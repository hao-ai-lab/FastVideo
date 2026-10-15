# SPDX-License-Identifier: Apache-2.0
"""Profile a local Wan 2.1 1.3B VAE load and first decode on one CUDA GPU.

Run by filename so import timing begins before importing torch/FastVideo.
The checkpoint must already be downloaded; this does not measure Hub downloads.
"""
import argparse
import json
from pathlib import Path
import time


def main():
    started = time.perf_counter()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True, help="Local Wan 2.1 1.3B Diffusers VAE component directory")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tensor-output", type=Path, help="Optional decoded tensor for independent parity checks")
    parser.add_argument("--synchronize", action="store_true", help="Synchronize CUDA at phase boundaries")
    args = parser.parse_args()
    mark = time.perf_counter()
    import torch
    torch_import = time.perf_counter() - mark
    mark = time.perf_counter()
    import fastvideo.envs as envs
    from fastvideo.utils import get_open_port
    from fastvideo.configs.pipelines import PipelineConfig
    from fastvideo.distributed import cleanup_dist_env_and_memory, maybe_init_distributed_environment_and_model_parallel
    from fastvideo.fastvideo_args import FastVideoArgs
    from fastvideo.models.loader.component_loader import PipelineComponentLoader
    from fastvideo.models.loader.startup import collect_startup_timings, startup_phase
    from fastvideo.models.wan.vae_config import WanVAEConfig
    fastvideo_import = time.perf_counter() - mark
    if not torch.cuda.is_available():
        raise RuntimeError("This profile requires one CUDA GPU")
    mark = time.perf_counter()
    runtime = FastVideoArgs(model_path=args.model_path,
                            pipeline_config=PipelineConfig(vae_config=WanVAEConfig(), vae_precision="fp32"))
    runtime.vae_cpu_offload = False
    configuration_seconds = time.perf_counter() - mark
    result = {"imports": {"torch_seconds": torch_import, "fastvideo_and_helpers_seconds": fastvideo_import},
              "model_path": args.model_path, "torch": torch.__version__,
              "configuration_seconds": configuration_seconds,
              "cuda_initialized_before_device_setup": torch.cuda.is_initialized()}
    distributed_ready = False
    def synchronize_if_initialized():
        # Do not move CUDA context initialization out of the device_setup phase.
        if torch.cuda.is_initialized():
            torch.cuda.synchronize()
    try:
        with collect_startup_timings(synchronize=synchronize_if_initialized if args.synchronize else None) as timings:
            with startup_phase("device_setup"):
                envs.setdefault_external("MASTER_ADDR", "127.0.0.1")
                envs.setdefault_external("MASTER_PORT", str(get_open_port()))
                maybe_init_distributed_environment_and_model_parallel(1, 1)
                distributed_ready = True
            vae = PipelineComponentLoader.load_module("vae", args.model_path, "diffusers", runtime)
            torch.cuda.synchronize()
            result["ready_seconds"] = time.perf_counter() - started
            latents = torch.randn((1, 16, 2, 12, 12), generator=torch.Generator().manual_seed(1024)).cuda()
            with startup_phase("first_forward"), torch.no_grad():
                output = vae.decode(latents)
                torch.cuda.synchronize()
            assert output.shape == (1, 3, 5, 96, 96) and torch.isfinite(output).all()
            result["verified_seconds"] = time.perf_counter() - started
            result["output_shape"] = list(output.shape)
            result["device"] = torch.cuda.get_device_name(0)
            if args.tensor_output:
                args.tensor_output.parent.mkdir(parents=True, exist_ok=True)
                torch.save(output.cpu(), args.tensor_output)
        result["status"] = "ok"
    except Exception as error:
        result["status"] = "error"
        result["error"] = repr(error)
        raise
    finally:
        if "timings" in locals():
            result["startup"] = timings.to_dict()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        if distributed_ready:
            cleanup_dist_env_and_memory()


if __name__ == "__main__":
    main()
