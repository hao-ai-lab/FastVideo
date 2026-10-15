# SPDX-License-Identifier: Apache-2.0
"""Retain Wan DiT/T5/VAE weights; verify fresh normal-loader inference workers.

Use a local Wan 2.1 T2V 1.3B Diffusers checkpoint on one GPU. Start a donor,
then run consumers in separate processes with the same private cache directory.
The tiny request checks execution/parity, not video quality or throughput.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import signal
import threading
import time

import torch

from fastvideo import VideoGenerator
from fastvideo.models.loader.inference_weight_cache import COMPONENTS
from fastvideo.models.loader.weight_cache import serve_weight_cache


def fingerprints(modules):
    result = {}
    for component in COMPONENTS:
        digest = hashlib.sha256()
        model = modules[component]
        for name, tensor in [*model.named_parameters(remove_duplicate=False),
                             *model.named_buffers(remove_duplicate=False)]:
            digest.update(name.encode())
            digest.update(str((tuple(tensor.shape), tensor.dtype, tuple(tensor.stride()))).encode())
            digest.update(tensor.detach().contiguous().reshape(-1).view(torch.uint8).cpu().numpy().tobytes())
        result[component] = digest.hexdigest()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("role", choices=("donor", "consumer", "disk"))
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--checkpoint-id", required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tensor-output", type=Path)
    parser.add_argument("--executor", choices=("uni", "mp"), default="uni")
    args = parser.parse_args()
    report = {"role": args.role, "torch": torch.__version__, "executor": args.executor}
    generator = None
    stop = threading.Event()
    threads = []
    failures = []
    started = time.perf_counter()
    try:
        # Import/runtime allocations are separate from retained weight storage.
        torch.cuda.init()
        allocated = torch.cuda.memory_allocated()
        generator = VideoGenerator.from_pretrained(
            args.model_path, num_gpus=1, distributed_executor_backend=args.executor,
            tp_size=1, sp_size=1, hsdp_replicate_dim=1, hsdp_shard_dim=1,
            dit_cpu_offload=False, dit_layerwise_offload=False, vae_cpu_offload=False,
            text_encoder_cpu_offload=False, image_encoder_cpu_offload=False,
            lazy_module_load=False, inference_torch_compile=False,
            vae_tiling=False, vae_sp=False,
            weight_cache_dir=str(args.cache_dir) if args.role == "consumer" else None,
            weight_cache_checkpoint=args.checkpoint_id,
        )
        torch.cuda.synchronize()
        report["ready_seconds"] = time.perf_counter() - started
        report["pipeline_allocation_delta_bytes"] = torch.cuda.memory_allocated() - allocated
        # FastVideo resolves both uni and single-GPU mp to UniprocExecutor.
        # Record the effective executor instead of assuming the requested name
        # implies a spawned worker.
        report["effective_executor"] = type(generator.executor).__name__
        modules = None
        if report["effective_executor"] == "UniprocExecutor":
            pipeline = generator.executor.driver_worker.worker.pipeline
            modules = pipeline.modules
            report["keys"] = {name: asdict(modules[name]._fastvideo_weight_cache_key) for name in COMPONENTS}
            report["attachments"] = {name: modules[name]._fastvideo_weight_cache_witness for name in COMPONENTS}
            report["fingerprints_before"] = fingerprints(modules)
        if args.role == "donor":
            if modules is None:
                raise ValueError("The donor must use the uni executor")
            args.cache_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
            for sig in (signal.SIGINT, signal.SIGTERM):
                signal.signal(sig, lambda *_: stop.set())
            def serve(component):
                try:
                    torch.cuda.set_device(0)
                    serve_weight_cache(modules[component], modules[component]._fastvideo_weight_cache_key,
                                       args.cache_dir / f"{component}.sock", stop=stop)
                except BaseException as error:
                    failures.append(error)
                    stop.set()
            for component in COMPONENTS:
                thread = threading.Thread(target=serve, args=(component,))
                thread.start()
                threads.append(thread)
            deadline = time.monotonic() + 30
            while not all((args.cache_dir / f"{name}.sock").exists() for name in COMPONENTS):
                if stop.wait(.1) or time.monotonic() > deadline:
                    raise RuntimeError(f"Donor failed to become ready: {failures}")
            report["donor_ready_seconds"] = time.perf_counter() - started
            print("DiT, T5 and VAE donor sockets ready", flush=True)
            while not stop.wait(.5):
                pass
            if failures:
                raise failures[0]
        else:
            outputs = []
            for _ in range(2):
                result = generator.generate_video(
                    "A red cube on a wooden table.", num_frames=9, height=128, width=128,
                    num_inference_steps=4, seed=1024, save_video=False, return_frames=True,
                )
                samples = result["samples"].detach().cpu()
                assert samples.shape == (1, 3, 9, 128, 128) and torch.isfinite(samples).all()
                outputs.append(samples)
            assert torch.equal(outputs[0], outputs[1]), "Repeated seeded requests differ"
            report["output_shape"] = list(outputs[0].shape)
            if args.tensor_output:
                torch.save(outputs[0], args.tensor_output)
        # Close servers before the final fingerprint so no consumer can still
        # execute on these tensors during the donor's final verification.
        stop.set()
        for thread in threads:
            thread.join()
        if modules is not None:
            report["fingerprints_after"] = fingerprints(modules)
            assert report["fingerprints_before"] == report["fingerprints_after"]
        report["status"] = "ok"
    except BaseException as error:
        report["status"] = "error"
        report["error"] = repr(error)
        raise
    finally:
        stop.set()
        for thread in threads:
            thread.join()
        if generator is not None:
            generator.executor.shutdown()
        report["total_seconds"] = time.perf_counter() - started
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
