# SPDX-License-Identifier: Apache-2.0
"""Persistent native Wan VAE weights with independently restartable IPC consumers.

Run donor in one process; run consumer repeatedly in fresh processes on the same
GPU. The donor must stay alive. The example uses a local, immutable Wan 2.1 1.3B
Diffusers VAE component and FP32. It shares weights, not mutable VAE caches.
"""
import argparse
import hashlib
import json
from pathlib import Path
import signal
import threading
import time

import torch

from fastvideo.configs.pipelines import PipelineConfig
from fastvideo.distributed import cleanup_dist_env_and_memory, maybe_init_distributed_environment_and_model_parallel
import fastvideo.envs as envs
from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.models.loader.component_loader import VAELoader
from fastvideo.models.loader.utils import set_default_torch_dtype
from fastvideo.models.loader.weight_cache import WeightCacheClient, WeightCacheKey, serve_weight_cache
from fastvideo.models.wan.vae import AutoencoderKLWan
from fastvideo.models.wan.vae_config import WanVAEConfig
from fastvideo.utils import get_open_port


def fingerprint(model):
    digest = hashlib.sha256()
    for name, tensor in [*model.named_parameters(remove_duplicate=False), *model.named_buffers(remove_duplicate=False)]:
        digest.update(name.encode())
        digest.update(str((tensor.dtype, tuple(tensor.shape), tuple(tensor.stride()), tensor.storage_offset())).encode())
        digest.update(tensor.detach().contiguous().reshape(-1).view(torch.uint8).cpu().numpy().tobytes())
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("role", choices=["donor", "consumer", "disk"])
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--checkpoint-id", required=True, help="Immutable checkpoint revision or content digest")
    parser.add_argument("--socket", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tensor-output", type=Path)
    args = parser.parse_args()
    if args.role != "disk" and args.socket is None:
        parser.error("--socket is required for donor/consumer")
    config_bytes = (args.model_path / "config.json").read_bytes()
    architecture = json.loads(config_bytes)
    if architecture.get("_class_name") != "AutoencoderKLWan":
        raise ValueError("This adapter supports native Wan VAE only")
    key = WeightCacheKey(args.checkpoint_id, "vae", hashlib.sha256(config_bytes).hexdigest(),
                         "torch.float32", "native-wan-fp32-v1")
    envs.setdefault_external("MASTER_ADDR", "127.0.0.1")
    envs.setdefault_external("MASTER_PORT", str(get_open_port()))
    maybe_init_distributed_environment_and_model_parallel(1, 1)
    device = torch.device("cuda", torch.cuda.current_device())
    report = {"role": args.role, "checkpoint": args.checkpoint_id, "torch": torch.__version__}
    def load_disk():
        runtime = FastVideoArgs(model_path=str(args.model_path), vae_cpu_offload=False,
                                pipeline_config=PipelineConfig(vae_config=WanVAEConfig(), vae_precision="fp32"))
        return VAELoader().load(str(args.model_path), runtime).eval()
    native_config = WanVAEConfig()
    native_config.update_model_arch(architecture)
    def construct():
        return VAELoader().build_model(AutoencoderKLWan, native_config, torch.device("meta"))
    def verify(model):
        report["weight_sha256_before"] = fingerprint(model)
        x = torch.randn((1, 16, 2, 12, 12), generator=torch.Generator().manual_seed(1024)).to(device)
        with torch.no_grad():
            first = model.decode(x)
            second = model.decode(x)
        torch.cuda.synchronize(device)
        assert first.shape == (1, 3, 5, 96, 96) and torch.isfinite(first).all()
        assert torch.equal(first, second), "Request-local VAE caches were not reset"
        report["weight_sha256_after"] = fingerprint(model)
        assert report["weight_sha256_before"] == report["weight_sha256_after"]
        if args.tensor_output:
            args.tensor_output.parent.mkdir(parents=True, exist_ok=True)
            torch.save(first.cpu(), args.tensor_output)
    try:
        if args.role == "donor":
            model = load_disk()
            report["weight_sha256_before"] = fingerprint(model)
            stop = threading.Event()
            for sig in (signal.SIGINT, signal.SIGTERM):
                signal.signal(sig, lambda *_: stop.set())
            print("Serving donor weights", flush=True)
            serve_weight_cache(model, key, args.socket, stop=stop)
            report["weight_sha256_after"] = fingerprint(model)
            assert report["weight_sha256_before"] == report["weight_sha256_after"]
        else:
            start = time.perf_counter()
            allocated = torch.cuda.memory_allocated(device)
            if args.role == "consumer":
                with set_default_torch_dtype(torch.float32):
                    with WeightCacheClient(args.socket, key, device).load_model(construct) as attachment:
                        report["load_seconds"] = time.perf_counter() - start
                        report["allocated_weight_delta_bytes"] = torch.cuda.memory_allocated(device) - allocated
                        report["donor_pid"] = attachment.donor_pid
                        report["ipc_handles"] = attachment.handles
                        report["weight_source"] = "cuda_ipc"
                        verify(attachment.model)
            else:
                model = load_disk()
                torch.cuda.synchronize(device)
                report["load_seconds"] = time.perf_counter() - start
                report["allocated_weight_delta_bytes"] = torch.cuda.memory_allocated(device) - allocated
                report["weight_source"] = "disk"
                verify(model)
        report["status"] = "ok"
    except Exception as error:
        report.update(status="error", error=repr(error))
        raise
    finally:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        cleanup_dist_env_and_memory()


if __name__ == "__main__":
    main()
