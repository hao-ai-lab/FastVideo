# SPDX-License-Identifier: Apache-2.0
"""Single-GPU data-parallel worker for large-scale Wan2.2-T2V-A14B synthetic gen.

One VideoGenerator per GPU (num_gpus=1, no FSDP/SP/TP). Each worker owns a stride
slice of the prompt list: prompts[worker_id :: num_workers]. Idempotent/resumable:
a video whose final mp4 exists is skipped, so a requeue after a cordon just continues.

First-frame saturation fix baked in: generate (num_frames + drop) frames, drop the
first `drop` decoded frames, keep `num_frames`. (See research_log/08-first-frame-saturation.md.)

Per-worker manifest shard (manifest_shards/worker_<id>.jsonl) avoids 48 procs racing on
one file; merge_manifests.py compiles them into the FastVideo videos2caption.json.
"""
from __future__ import annotations
import argparse, json, os, time, traceback
from pathlib import Path
import numpy as np

MODEL = "Wan-AI/Wan2.2-T2V-A14B-Diffusers"

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--prompts", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--worker-id", type=int, required=True)
    p.add_argument("--num-workers", type=int, required=True)
    p.add_argument("--max-videos", type=int, default=None, help="global cap on #prompts consumed (across all workers)")
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--num-frames", type=int, default=121, help="frames to KEEP")
    p.add_argument("--drop", type=int, default=8, help="leading frames dropped (gen = keep+drop, must be 4k+1)")
    p.add_argument("--fps", type=int, default=16)
    p.add_argument("--steps", type=int, default=40)
    p.add_argument("--guidance-scale", type=float, default=4.0)
    p.add_argument("--guidance-scale-2", type=float, default=3.0)
    p.add_argument("--seed-base", type=int, default=1024, help="per-video seed = seed_base + global_prompt_idx")
    p.add_argument("--shuffle-seed", type=int, default=1234, help="deterministic prompt shuffle (same across workers)")
    return p.parse_args()

def load_prompts(path: Path):
    lines = [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    return lines

def main():
    a = parse_args()
    gen_frames = a.num_frames + a.drop
    assert (gen_frames - 1) % 4 == 0, f"gen frames {gen_frames} must be 4k+1"

    videos_dir = a.output_dir / "videos"
    meta_dir = a.output_dir / "meta"
    shard_dir = a.output_dir / "manifest_shards"
    for d in (videos_dir, meta_dir, shard_dir):
        d.mkdir(parents=True, exist_ok=True)
    shard_manifest = shard_dir / f"worker_{a.worker_id:04d}.jsonl"
    fail_log = a.output_dir / f"failures_worker_{a.worker_id:04d}.log"

    prompts = load_prompts(a.prompts)
    # Deterministic global shuffle so prompt order is stable across restarts and workers.
    order = np.random.RandomState(a.shuffle_seed).permutation(len(prompts))
    if a.max_videos is not None:
        order = order[:a.max_videos]
    # This worker's stride slice.
    my_positions = list(range(a.worker_id, len(order), a.num_workers))
    my_global_idx = [int(order[pos]) for pos in my_positions]
    print(f"[w{a.worker_id}/{a.num_workers}] assigned {len(my_global_idx)} prompts "
          f"(gen {gen_frames}->keep {a.num_frames}, {a.width}x{a.height}@{a.fps}fps)", flush=True)

    # already-done set from existing mp4s
    def final_path(idx): return videos_dir / f"vid_{idx:06d}.mp4"

    import imageio.v2 as imageio
    from fastvideo import VideoGenerator
    t0 = time.time()
    g = VideoGenerator.from_pretrained(
        MODEL, num_gpus=1, use_fsdp_inference=False,
        dit_cpu_offload=False, vae_cpu_offload=False,
        text_encoder_cpu_offload=True, pin_cpu_memory=True,
    )
    print(f"[w{a.worker_id}] model ready in {time.time()-t0:.1f}s", flush=True)

    n_ok = 0
    for gi in my_global_idx:
        fp = final_path(gi)
        if fp.exists():
            continue
        prompt = prompts[gi]
        tmp = videos_dir / f".tmp_w{a.worker_id}_{gi:06d}.mp4"
        t = time.time()
        try:
            res = g.generate_video(
                prompt, save_video=False, return_frames=True,
                height=a.height, width=a.width, num_frames=gen_frames, fps=a.fps,
                seed=a.seed_base + gi, num_inference_steps=a.steps,
                guidance_scale=a.guidance_scale, guidance_scale_2=a.guidance_scale_2,
            )
            if isinstance(res, list):
                res = res[0]
            frames = np.asarray(res["frames"])[a.drop:a.drop + a.num_frames]
            if frames.shape[0] != a.num_frames:
                raise RuntimeError(f"got {frames.shape[0]} frames after drop, want {a.num_frames}")
            imageio.mimsave(tmp, list(frames), fps=a.fps, format="mp4")
            os.replace(tmp, fp)  # atomic publish
        except Exception as e:  # keep the worker alive
            with fail_log.open("a") as f:
                f.write(json.dumps({"idx": gi, "err": repr(e)[:500]}) + "\n")
            print(f"[w{a.worker_id}] FAIL idx={gi}: {e!r}", flush=True)
            if tmp.exists(): tmp.unlink()
            continue
        dt = time.time() - t
        rec = {"idx": gi, "path": fp.name, "cap": [prompt], "fps": float(a.fps),
               "num_frames": a.num_frames, "resolution": {"width": a.width, "height": a.height},
               "gen_seconds": round(dt, 1)}
        with shard_manifest.open("a") as f:
            f.write(json.dumps(rec) + "\n")
        (meta_dir / f"vid_{gi:06d}.json").write_text(json.dumps(rec))
        n_ok += 1
        if n_ok % 10 == 0:
            print(f"[w{a.worker_id}] {n_ok} done, last {dt:.0f}s idx={gi}", flush=True)
    print(f"[w{a.worker_id}] DONE_WORKER made {n_ok} new videos", flush=True)

if __name__ == "__main__":
    main()
