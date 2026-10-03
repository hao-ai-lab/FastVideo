"""Pruned FastH3 8-step benchmark on a single RTX 4090.

usage: python bench_pod.py <name> <model_dir> <quant: nvfp4|fp8> [--timed N] [--offload-buffers] [--no-layerwise]
Stage logs (FASTVIDEO_STAGE_LOGGING=1) carry per-stage time and memory peaks; results.json lands in outputs/<name>/.
"""
import argparse
import json
import os
import pathlib
import shlex
import statistics
import subprocess
import sys
import threading
import time


class HostMemoryPeak:
    """Sample pod-wide cgroup usage; anon excludes cached checkpoint file pages."""

    def __init__(self):
        self.stop = threading.Event()
        self.peak_bytes = 0
        self.peak_anon_bytes = 0
        self.peak_gpu_bytes = None
        self._gpu_used = None
        self._nvml_shutdown = None
        self.thread = threading.Thread(target=self._sample, daemon=True)
        try:
            import pynvml
            pynvml.nvmlInit()
            self._nvml_shutdown = pynvml.nvmlShutdown
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            self._gpu_used = lambda: pynvml.nvmlDeviceGetMemoryInfo(handle).used
        except Exception as exc:
            print(f"GPU memory sampler unavailable: {exc}", flush=True)

    def _sample(self):
        while not self.stop.is_set():
            try:
                root = pathlib.Path("/sys/fs/cgroup")
                self.peak_bytes = max(self.peak_bytes, int((root / "memory.current").read_text()))
                stats = dict(line.split() for line in (root / "memory.stat").read_text().splitlines())
                self.peak_anon_bytes = max(self.peak_anon_bytes, int(stats["anon"]))
                if self._gpu_used is not None:
                    self.peak_gpu_bytes = max(self.peak_gpu_bytes or 0, self._gpu_used())
            except (OSError, KeyError, ValueError):
                return
            self.stop.wait(0.1)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *_args):
        self.stop.set()
        self.thread.join()
        if self._nvml_shutdown is not None:
            self._nvml_shutdown()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("name"); ap.add_argument("model"); ap.add_argument("quant", choices=("nvfp4", "fp8", "bf16"))
    ap.add_argument("--timed", type=int, default=2)
    ap.add_argument("--offload-buffers", action="store_true")
    ap.add_argument("--no-layerwise", action="store_true")
    ap.add_argument("--resident-encoder", action="store_true")
    ap.add_argument("--tile-batch", default=None)
    ap.add_argument("--frames", type=int, default=243)
    ap.add_argument("--height", type=int, default=768)
    ap.add_argument("--width", type=int, default=1344)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--prompt-file", type=pathlib.Path, default=pathlib.Path(__file__).with_name("prompts_1k.json"))
    ap.add_argument("--output-root", type=pathlib.Path, default=pathlib.Path("/workspace/outputs"))
    ap.add_argument("--no-vae-compile", action="store_true")
    ap.add_argument("--adaln-cache", action="store_true")
    ap.add_argument("--profile", action="store_true")
    ap.add_argument("--sparsity", type=float, default=0.8)
    ap.add_argument("--decode", default="h3-vae")
    ap.add_argument("--lazy", action="store_true", help="lazy_module_load: reload released modules per request")
    ap.add_argument("--prompts", default=None, help="comma-separated prompt ids (default: both)")
    a = ap.parse_args()
    if a.timed < 2 or a.warmup < 1:
        ap.error("Use at least one warmup and two timed runs")
    if not pathlib.Path(a.model, "fastvideo_inference.json").is_file():
        ap.error("The model directory must contain fastvideo_inference.json for the 8-step DMD contract")

    os.environ.setdefault("FASTVIDEO_STAGE_LOGGING", "1")
    # NVFP4 is retained for other GPUs; the RTX 4090 DiT uses FP8.
    if a.quant == "nvfp4":
        os.environ.setdefault("FASTVIDEO_H3_VSA_FP4", "1")
        os.environ.setdefault("FASTVIDEO_NVFP4_MM_BACKEND", "cutlass")
    os.environ.setdefault("FASTVIDEO_MINIMAX_H3_FUSIONS", "all")
    os.environ.setdefault("FASTVIDEO_H3_VAE_TILE_BATCH", "28")
    os.environ.setdefault("FASTVIDEO_VSA_TRITON", "1")
    os.environ.setdefault("FASTVIDEO_VSA_SM100A", "0")
    os.environ.setdefault("FASTVIDEO_FA4", "0")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    ap_table = os.environ.get("FASTVIDEO_H3_ADALN_TABLE")
    if a.profile:
        os.environ["FASTVIDEO_H3_SP_PROFILE"] = "1"
    if a.adaln_cache and not ap_table:
        os.environ["FASTVIDEO_H3_ADALN_CACHE"] = "1"
        # Export the exact modulation tables (and inputs) for table-only loads and low-rank experiments.
        os.environ.setdefault("FASTVIDEO_H3_ADALN_DUMP", f"/workspace/adaln_tables_{a.name}.pt")
    if a.offload_buffers:
        os.environ["FASTVIDEO_LAYERWISE_OFFLOAD_BUFFERS"] = "1"
    if a.tile_batch:
        os.environ["FASTVIDEO_H3_VAE_TILE_BATCH"] = a.tile_batch
    import torch
    from fastvideo import VideoGenerator

    texts = json.loads(a.prompt_file.read_text())
    layerwise = not a.no_layerwise
    engine = {"num_gpus": 1, "use_fsdp_inference": False,
              "parallelism": {"tp_size": 1, "sp_size": 1},
              "offload": {"dit": False, "dit_layerwise": layerwise, "text_encoder": not a.resident_encoder,
                          "vae": layerwise, "pin_cpu_memory": True, "lazy_module_load": a.lazy},
              "compile": {"enabled": False, "vae_enabled": not a.no_vae_compile}}
    if a.quant == "nvfp4":
        engine["quantization"] = {"transformer_quant": "NVFP4", "layer_profile": "h3_dit_vsa"}
    elif a.quant == "fp8":
        engine["quantization"] = {"transformer_quant": "FP8"}
    experimental = {"attention_backend": "VIDEO_SPARSE_ATTN_H3", "VSA_sparsity": a.sparsity, "VSA_tile_size": 64,
                    "h3_sequential_load": not a.resident_encoder, "inference_torch_compile": False,
                    "vae_parallel_decode": False, "video_decode_backend": a.decode}
    config = {"model_path": a.model, "engine": engine, "pipeline": {"experimental": experimental}}
    out_dir = a.output_root / a.name
    out_dir.mkdir(parents=True, exist_ok=True)
    model_root = pathlib.Path(a.model)
    revision_file = model_root / ".cache/huggingface/download/fastvideo_inference.json.metadata"
    model_revision = revision_file.read_text().splitlines()[0] if revision_file.is_file() else None
    hardware = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=name,memory.total,driver_version,pci.bus_id", "--format=csv,noheader"], text=True
    ).strip()
    sampling = {"seed": 20260929, "height": a.height, "width": a.width, "num_frames": a.frames, "fps": 24,
                "num_inference_steps": 9, "guidance_scale": 1.0, "batch_cfg": False}
    results = {"name": a.name, "quant": a.quant, "command": shlex.join([sys.executable, "-P", *sys.argv]),
               "env": {k: v for k, v in os.environ.items() if k.startswith(("FASTVIDEO_", "PYTORCH_"))
                       or k in ("CUDA_VISIBLE_DEVICES", "MAX_JOBS")},
               "torch": torch.__version__, "cuda": torch.version.cuda,
               "hardware": hardware, "model_revision": model_revision,
               "model_contract": json.loads((model_root / "fastvideo_inference.json").read_text()),
               "source_commit": os.environ.get("FASTVIDEO_SOURCE_COMMIT"),
               "gpu": torch.cuda.get_device_name(0), "config": config, "sampling": sampling, "runs": []}
    (out_dir / "results.json").write_text(json.dumps(results, indent=2))
    t0 = time.perf_counter()
    generator = VideoGenerator.from_config(config)
    results["load_s"] = round(time.perf_counter() - t0, 1)
    ids = a.prompts.split(",") if a.prompts else list(texts)
    order = [ids[i % len(ids)] for i in range(a.warmup + a.timed)]
    try:
        for i, pid in enumerate(order):
            request = {"prompt": texts[pid], "negative_prompt": "",
                       "sampling": sampling,
                       "output": {"output_path": str(out_dir / f"{i:02d}_{pid}.mp4"), "save_video": True,
                                  "return_frames": False}}
            t = time.perf_counter()
            with HostMemoryPeak() as host_peak:
                generator.generate(request)
            wall = round(time.perf_counter() - t, 2)
            results["runs"].append({"prompt": pid, "warmup": i < a.warmup, "wall_s": wall,
                                    "clip": request["output"]["output_path"],
                                    "peak_gpu_used_gib": (round(host_peak.peak_gpu_bytes / 2**30, 3)
                                                          if host_peak.peak_gpu_bytes is not None else None),
                                    "peak_host_cgroup_gib": round(host_peak.peak_bytes / 2**30, 3),
                                    "peak_host_anon_gib": round(host_peak.peak_anon_bytes / 2**30, 3)})
            timed = [run["wall_s"] for run in results["runs"] if not run["warmup"]]
            if timed:
                results["median_e2e_s"] = statistics.median(timed)
            print("RUN", json.dumps(results["runs"][-1]), flush=True)
            (out_dir / "results.json").write_text(json.dumps(results, indent=1))
    finally:
        generator.shutdown()
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
