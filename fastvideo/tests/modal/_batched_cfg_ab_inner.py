"""
Inner-script half of the W3b batched-CFG A/B harness.

Runs one denoising pass (sequential or batched) over the harness's
prompt + seed set, records per-prompt walls, and writes a JSON results
blob to the path given on argv.

Invoked by ``batched_cfg_ab.py`` via ``/opt/venv/bin/python`` so that
FastVideo runs inside the image's venv (Modal's main function process
runs in Modal's own Python, where FastVideo is not installed).
"""
import argparse
import json
import os
import sys
import time


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("run_pass", "compute_ssim"), default="run_pass",
                        help="run_pass: execute one A/B pass. compute_ssim: read baseline+patched "
                        "results JSONs, write a pairwise SSIM JSON.")
    parser.add_argument("--config-json", help="run_pass: path to JSON file with run config")
    parser.add_argument("--results-json", help="run_pass: path to write JSON results")
    parser.add_argument("--baseline-results-json", help="compute_ssim: path to baseline records JSON (mode A)")
    parser.add_argument("--patched-results-json", help="compute_ssim: path to patched records JSON (mode A)")
    parser.add_argument("--baseline-dir", help="compute_ssim: scan dir for newest mp4 per prompt_NN (mode B)")
    parser.add_argument("--patched-dir", help="compute_ssim: scan dir for newest mp4 per prompt_NN (mode B)")
    parser.add_argument("--ssim-output-json", help="compute_ssim: path to write SSIM rows JSON")
    args = parser.parse_args()

    if args.mode == "compute_ssim":
        return _compute_ssim_main(args)

    if not args.config_json or not args.results_json:
        parser.error("--config-json and --results-json are required for mode=run_pass")
    with open(args.config_json) as f:
        cfg = json.load(f)

    model_id: str = cfg["model_id"]
    num_gpus: int = cfg["num_gpus"]
    use_batched_cfg: bool = cfg["use_batched_cfg"]
    enable_compile: bool = cfg.get("enable_compile", False)
    height: int = cfg["height"]
    width: int = cfg["width"]
    num_frames: int = cfg["num_frames"]
    num_inference_steps: int = cfg["num_inference_steps"]
    output_dir: str = cfg["output_dir"]
    prompts: list[str] = cfg["prompts"]
    seed_base: int = cfg["seed_base"]

    # Log resolved flash-attn version (Hopper -> FA3 expected per Will
    # Slack 2026-05-28; L40S -> FA2 baked into the image).
    from fastvideo.attention.backends.flash_attn import fa_version
    print(f"[inner] resolved flash-attn version: FA{fa_version}", flush=True)

    from fastvideo import VideoGenerator

    generator = VideoGenerator.from_pretrained(
        model_id,
        num_gpus=num_gpus,
        use_batched_cfg=use_batched_cfg,
        enable_torch_compile=enable_compile,
    )
    if enable_compile:
        print("[inner] enable_torch_compile=True — first prompt will include compile warmup", flush=True)
    resolved_flag = getattr(generator.fastvideo_args, "use_batched_cfg", None)
    if resolved_flag is not use_batched_cfg:
        raise RuntimeError(f"use_batched_cfg did not propagate: requested {use_batched_cfg}, "
                           f"resolved {resolved_flag}")
    print(f"[inner] use_batched_cfg resolved to {resolved_flag}", flush=True)

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
        # FastVideo's generate_video doesn't overwrite — it appends
        # _1, _2, ... to avoid collisions. After multiple harness runs
        # the same prompt_NN directory accumulates mp4s from each run;
        # we want the one we just wrote. Pick by mtime (newest).
        mp4s = sorted(
            [os.path.join(prompt_out, f) for f in os.listdir(prompt_out) if f.endswith(".mp4")],
            key=os.path.getmtime,
        )
        if not mp4s:
            raise RuntimeError(f"no .mp4 produced for prompt {i} at {prompt_out}")
        records.append({"i": i, "wall_s": wall, "mp4": mp4s[-1]})
        label = "batched" if use_batched_cfg else "sequential"
        print(f"[inner] [{label}] prompt {i}: {wall:.3f}s -> {mp4s[0]}", flush=True)

    with open(args.results_json, "w") as f:
        json.dump(records, f)
    print(f"[inner] wrote {len(records)} records to {args.results_json}", flush=True)
    return 0


def _compute_ssim_main(args) -> int:
    """Pairwise SSIM between baseline and patched output mp4s.

    Two input modes:
      A. --baseline-results-json + --patched-results-json: read the
         JSON records the per-pass runs wrote; each carries the mp4
         path the in-run picker selected.
      B. --baseline-dir + --patched-dir: scan each directory's
         prompt_NN/ subdirs and pick the NEWEST mp4 per prompt by
         mtime. Use this to recover after a run where stale mp4s
         from a previous run accumulated in the same dir.

    Avoids importing ``fastvideo`` entirely — its top-level package
    transitively imports triton (via vmoba), which needs a CUDA driver
    to initialise. The recovery function runs on CPU, so we inline the
    SSIM logic here using only torch / torchvision / av directly.
    """
    if not args.ssim_output_json:
        raise SystemExit("--ssim-output-json is required for mode=compute_ssim")
    mode_a = args.baseline_results_json and args.patched_results_json
    mode_b = args.baseline_dir and args.patched_dir
    if not (mode_a or mode_b):
        raise SystemExit("Provide either --baseline-results-json + --patched-results-json (mode A) "
                         "OR --baseline-dir + --patched-dir (mode B).")

    # Match fastvideo/tests/utils.py: pytorch_msssim's ssim (single-scale)
    # so our numbers are directly comparable to anyone re-running the
    # canonical helper. The reference helper defaults to MS-SSIM but the
    # SSIM=1.0 gate is the more conservative bar — both should be 1.0
    # on identical output, so use ssim (not ms_ssim) here as it's the
    # stricter single-scale comparison.
    import torch
    from pytorch_msssim import ssim as pm_ssim

    def _read_video_frames(path: str) -> torch.Tensor:
        try:
            from torchvision.io import read_video
            frames, _, _ = read_video(path, pts_unit="sec", output_format="TCHW")
            return frames
        except (ImportError, AttributeError):
            pass
        import av
        container = av.open(path)
        frames = []
        for frame in container.decode(video=0):
            arr = frame.to_ndarray(format="rgb24")
            frames.append(torch.from_numpy(arr).permute(2, 0, 1))
        container.close()
        if not frames:
            raise RuntimeError(f"No video frames decoded from {path}")
        return torch.stack(frames)

    def _ssim(video1_path: str, video2_path: str) -> list[float]:
        f1 = _read_video_frames(video1_path)
        f2 = _read_video_frames(video2_path)
        n = min(f1.shape[0], f2.shape[0])
        f1 = (f1[:n].float() / 255.0).contiguous()
        f2 = (f2[:n].float() / 255.0).contiguous()
        vals = []
        for i in range(n):
            v = pm_ssim(f1[i:i + 1], f2[i:i + 1], data_range=1.0).item()
            vals.append(v)
        return vals

    def _scan_dir_for_newest_mp4s(root: str) -> list[dict]:
        """For each prompt_NN/ subdir, pick the newest .mp4 by mtime."""
        rows: list[dict] = []
        for entry in sorted(os.listdir(root)):
            sub = os.path.join(root, entry)
            if not (os.path.isdir(sub) and entry.startswith("prompt_")):
                continue
            mp4s = sorted(
                [os.path.join(sub, f) for f in os.listdir(sub) if f.endswith(".mp4")],
                key=os.path.getmtime,
            )
            if not mp4s:
                continue
            idx = int(entry.split("_", 1)[1])
            rows.append({"i": idx, "wall_s": float("nan"), "mp4": mp4s[-1]})
        rows.sort(key=lambda r: r["i"])
        return rows

    if mode_b:
        baseline = _scan_dir_for_newest_mp4s(args.baseline_dir)
        patched = _scan_dir_for_newest_mp4s(args.patched_dir)
        print(f"[inner] scanned {args.baseline_dir} -> {len(baseline)} prompts", flush=True)
        print(f"[inner] scanned {args.patched_dir} -> {len(patched)} prompts", flush=True)
    else:
        with open(args.baseline_results_json) as f:
            baseline = json.load(f)
        with open(args.patched_results_json) as f:
            patched = json.load(f)

    rows = []
    for b, p in zip(baseline, patched, strict=True):
        assert b["i"] == p["i"]
        print(f"[inner] computing SSIM for prompt {b['i']}: {b['mp4']} vs {p['mp4']}", flush=True)
        ssim_vals = _ssim(b["mp4"], p["mp4"])
        ssim_mean = float(sum(ssim_vals) / len(ssim_vals))
        ssim_worst = float(min(ssim_vals))
        rows.append({
            "i": b["i"],
            "baseline_wall_s": b["wall_s"],
            "patched_wall_s": p["wall_s"],
            "ssim_mean": ssim_mean,
            "ssim_worst": ssim_worst,
        })
        print(f"[inner]   prompt {b['i']} SSIM mean={ssim_mean:.6f} worst={ssim_worst:.6f}", flush=True)
    with open(args.ssim_output_json, "w") as f:
        json.dump(rows, f)
    print(f"[inner] wrote {len(rows)} SSIM rows to {args.ssim_output_json}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
