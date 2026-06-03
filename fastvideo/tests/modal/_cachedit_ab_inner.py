"""
Inner-script half of the cache-dit A/B harness.

Runs one denoising pass (caching off or on) over the harness's prompt + seed
set, records per-prompt walls, and writes a JSON results blob to the path
given on argv. Invoked by ``cachedit_ab.py`` via ``/opt/venv/bin/python`` so
FastVideo runs inside the image's venv (Modal's main process Python lacks
torch).

cache-dit step caching is LOSSY: the patched output is not bit-identical, so
the SSIM column is a quality measurement (target >= ~0.95), not a 1.0 gate.
"""
import argparse
import json
import os
import sys
import time


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("run_pass", "compute_ssim"), default="run_pass")
    parser.add_argument("--config-json")
    parser.add_argument("--results-json")
    parser.add_argument("--baseline-results-json")
    parser.add_argument("--patched-results-json")
    parser.add_argument("--ssim-output-json")
    args = parser.parse_args()

    if args.mode == "compute_ssim":
        return _compute_ssim_main(args)

    if not args.config_json or not args.results_json:
        parser.error("--config-json and --results-json are required for mode=run_pass")
    with open(args.config_json) as f:
        cfg = json.load(f)

    model_id = cfg["model_id"]
    num_gpus = cfg["num_gpus"]
    use_cachedit = cfg["use_cachedit"]
    fn = cfg.get("cachedit_fn_compute_blocks", 8)
    bn = cfg.get("cachedit_bn_compute_blocks", 0)
    threshold = cfg.get("cachedit_residual_threshold", 0.08)
    warmup = cfg.get("cachedit_max_warmup_steps", 8)
    taylorseer = cfg.get("cachedit_taylorseer", False)
    taylorseer_order = cfg.get("cachedit_taylorseer_order", 1)
    enable_compile = cfg.get("enable_compile", False)
    height, width = cfg["height"], cfg["width"]
    num_frames = cfg["num_frames"]
    num_inference_steps = cfg["num_inference_steps"]
    output_dir = cfg["output_dir"]
    prompts = cfg["prompts"]
    seed_base = cfg["seed_base"]

    from fastvideo.attention.backends.flash_attn import fa_version
    print(f"[inner] resolved flash-attn version: FA{fa_version}", flush=True)

    from fastvideo import VideoGenerator

    # cache-dit skips blocks, incompatible with layerwise/CPU offload (the
    # offload prefetch chain assumes every block runs each step). Disable it on
    # BOTH passes so the A/B isolates the cache effect. Wan 1.3B fits a single
    # L40S (48GB) without offload.
    generator = VideoGenerator.from_pretrained(
        model_id,
        num_gpus=num_gpus,
        use_cachedit=use_cachedit,
        cachedit_fn_compute_blocks=fn,
        cachedit_bn_compute_blocks=bn,
        cachedit_residual_threshold=threshold,
        cachedit_max_warmup_steps=warmup,
        cachedit_taylorseer=taylorseer,
        cachedit_taylorseer_order=taylorseer_order,
        dit_layerwise_offload=False,
        dit_cpu_offload=False,
        enable_torch_compile=enable_compile,
    )
    resolved = getattr(generator.fastvideo_args, "use_cachedit", None)
    if resolved is not use_cachedit:
        raise RuntimeError(f"use_cachedit did not propagate: requested {use_cachedit}, resolved {resolved}")
    if use_cachedit:
        print(f"[inner] cache-dit ON: Fn={fn} Bn={bn} threshold={threshold} warmup={warmup} "
              f"taylorseer={taylorseer}(order={taylorseer_order})", flush=True)
    else:
        print("[inner] caching OFF (baseline)", flush=True)

    os.makedirs(output_dir, exist_ok=True)
    records = []
    for i, prompt in enumerate(prompts):
        prompt_out = os.path.join(output_dir, f"prompt_{i:02d}")
        t0 = time.perf_counter()
        generator.generate_video(
            prompt,
            output_path=prompt_out,
            save_video=True,
            height=height,
            width=width,
            num_frames=num_frames,
            num_inference_steps=num_inference_steps,
            seed=seed_base + i,
        )
        wall = time.perf_counter() - t0
        # generate_video appends _1, _2, ... rather than overwriting; pick the
        # mp4 we just wrote by mtime (newest).
        mp4s = sorted([os.path.join(prompt_out, f) for f in os.listdir(prompt_out) if f.endswith(".mp4")],
                      key=os.path.getmtime)
        if not mp4s:
            raise RuntimeError(f"no .mp4 produced for prompt {i} at {prompt_out}")
        records.append({"i": i, "wall_s": wall, "mp4": mp4s[-1]})
        print(f"[inner] [{'cachedit' if use_cachedit else 'baseline'}] prompt {i}: {wall:.3f}s -> {mp4s[-1]}",
              flush=True)

    with open(args.results_json, "w") as f:
        json.dump(records, f)
    print(f"[inner] wrote {len(records)} records to {args.results_json}", flush=True)
    return 0


def _compute_ssim_main(args) -> int:
    """Pairwise SSIM between baseline and patched output mp4s. Avoids importing
    fastvideo (its top-level import pulls triton, which needs a CUDA driver);
    runs on CPU with pytorch_msssim + torchvision/av directly."""
    if not args.ssim_output_json or not (args.baseline_results_json and args.patched_results_json):
        raise SystemExit("compute_ssim needs --baseline-results-json, --patched-results-json, --ssim-output-json")

    import torch
    from pytorch_msssim import ssim as pm_ssim

    def _read_video_frames(path):
        try:
            from torchvision.io import read_video
            frames, _, _ = read_video(path, pts_unit="sec", output_format="TCHW")
            if frames.shape[0] > 0:
                return frames
        except Exception:
            # torchvision's backend (FFmpeg/PyAV) can raise more than
            # ImportError/AttributeError; fall back to the PyAV path below.
            pass
        import av
        container = av.open(path)
        frames = []
        for frame in container.decode(video=0):
            frames.append(torch.from_numpy(frame.to_ndarray(format="rgb24")).permute(2, 0, 1))
        container.close()
        if not frames:
            raise RuntimeError(f"No video frames decoded from {path}")
        return torch.stack(frames)

    def _ssim(p1, p2):
        f1, f2 = _read_video_frames(p1), _read_video_frames(p2)
        n = min(f1.shape[0], f2.shape[0])
        if n == 0:
            raise RuntimeError(f"no decodable frames to compare: {p1} ({f1.shape[0]}) vs {p2} ({f2.shape[0]})")
        f1 = (f1[:n].float() / 255.0).contiguous()
        f2 = (f2[:n].float() / 255.0).contiguous()
        return [pm_ssim(f1[i:i + 1], f2[i:i + 1], data_range=1.0).item() for i in range(n)]

    with open(args.baseline_results_json) as f:
        baseline = json.load(f)
    with open(args.patched_results_json) as f:
        patched = json.load(f)

    rows = []
    for b, p in zip(baseline, patched, strict=True):
        if b["i"] != p["i"]:
            raise ValueError(f"baseline/patched prompt index mismatch: {b['i']} != {p['i']}")
        vals = _ssim(b["mp4"], p["mp4"])
        rows.append({
            "i": b["i"],
            "baseline_wall_s": b["wall_s"],
            "patched_wall_s": p["wall_s"],
            "ssim_mean": float(sum(vals) / len(vals)),
            "ssim_worst": float(min(vals)),
        })
        print(f"[inner]   prompt {b['i']} SSIM mean={rows[-1]['ssim_mean']:.6f} worst={rows[-1]['ssim_worst']:.6f}",
              flush=True)
    with open(args.ssim_output_json, "w") as f:
        json.dump(rows, f)
    return 0


if __name__ == "__main__":
    sys.exit(main())
