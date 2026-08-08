# SPDX-License-Identifier: Apache-2.0
"""Single-GPU I2V synthetic gen: one SHARED seed image + a fixed caption list -> videos.

Reproduces the wantrack_synth_toy setup (all clips start from the same seed frame; each caption
varies only the motion) but at 720p/24fps. Uses Wan2.1-I2V-14B-720P: for every caption it
generates a clip conditioned on --seed-image at --height x --width.

Unlike the T2V worker there is NO first-frame drop: I2V's frame 0 IS the (clean) seed, so we
keep all num_frames. Output layout matches gen_synth_worker.py (videos/, meta/, manifest_shards/)
so merge_synth_manifests.py + the tracks/preprocess stages work unchanged.

Idempotent/resumable (skips finished mp4s). Parallelize across GPUs with --worker-id/--num-workers.

  CUDA_VISIBLE_DEVICES=0 python data_pipeline/gen_synth_i2v_worker.py \
    --seed-image data_pipeline/synth_toy_720p/synthetic_seed.png \
    --captions   data_pipeline/synth_toy_720p/captions.txt \
    --output-dir /home/hal-kevin/data/motion-stream-synth
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

MODEL_DEFAULT = "/home/hal-kevin/models/Wan2.1-I2V-14B-720P-Diffusers"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--seed-image", type=Path, required=True, help="shared first-frame conditioning image")
    p.add_argument("--captions", type=Path, required=True, help="one caption per line")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--model", default=MODEL_DEFAULT)
    p.add_argument("--worker-id", type=int, default=0)
    p.add_argument("--num-workers", type=int, default=1)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--num-frames", type=int, default=121, help="kept 1:1 (I2V, no drop); must be 4k+1")
    p.add_argument("--fps", type=int, default=24)
    p.add_argument("--steps", type=int, default=40)
    p.add_argument("--guidance-scale", type=float, default=5.0)
    p.add_argument("--seed-base", type=int, default=1024, help="per-clip seed = seed_base + caption_idx")
    return p.parse_args()


def main():
    a = parse_args()
    assert (a.num_frames - 1) % 4 == 0, f"num_frames {a.num_frames} must be 4k+1"
    assert a.seed_image.exists(), f"seed image not found: {a.seed_image}"

    videos_dir = a.output_dir / "videos"
    meta_dir = a.output_dir / "meta"
    shard_dir = a.output_dir / "manifest_shards"
    for d in (videos_dir, meta_dir, shard_dir):
        d.mkdir(parents=True, exist_ok=True)
    shard_manifest = shard_dir / f"worker_{a.worker_id:04d}.jsonl"
    fail_log = a.output_dir / f"failures_worker_{a.worker_id:04d}.log"

    captions = [ln.strip() for ln in a.captions.read_text().splitlines() if ln.strip()]
    # Fixed order (caption line -> vid index); this worker owns a stride slice.
    my_idx = list(range(a.worker_id, len(captions), a.num_workers))
    print(f"[w{a.worker_id}/{a.num_workers}] {len(my_idx)} caption(s) "
          f"seed={a.seed_image.name} {a.width}x{a.height}@{a.fps}fps x{a.num_frames}f", flush=True)

    import imageio.v2 as imageio
    from fastvideo import VideoGenerator
    t0 = time.time()
    g = VideoGenerator.from_pretrained(
        a.model, num_gpus=1, use_fsdp_inference=False,
        dit_cpu_offload=False, vae_cpu_offload=False,
        text_encoder_cpu_offload=True, pin_cpu_memory=True,
    )
    print(f"[w{a.worker_id}] model ready in {time.time()-t0:.1f}s", flush=True)

    seed_path = str(a.seed_image.resolve())
    n_ok = 0
    for gi in my_idx:
        fp = videos_dir / f"vid_{gi:06d}.mp4"
        if fp.exists():
            continue
        prompt = captions[gi]
        tmp = videos_dir / f".tmp_w{a.worker_id}_{gi:06d}.mp4"
        t = time.time()
        try:
            res = g.generate_video(
                prompt, image_path=seed_path, save_video=False, return_frames=True,
                height=a.height, width=a.width, num_frames=a.num_frames, fps=a.fps,
                seed=a.seed_base + gi, num_inference_steps=a.steps,
                guidance_scale=a.guidance_scale,
            )
            if isinstance(res, list):
                res = res[0]
            frames = np.asarray(res["frames"])          # I2V: keep all frames, no drop
            if frames.shape[0] != a.num_frames:
                raise RuntimeError(f"got {frames.shape[0]} frames, want {a.num_frames}")
            imageio.mimsave(tmp, list(frames), fps=a.fps, format="mp4")
            os.replace(tmp, fp)
        except Exception as e:                          # keep the worker alive
            with fail_log.open("a") as f:
                f.write(json.dumps({"idx": gi, "err": repr(e)[:500]}) + "\n")
            print(f"[w{a.worker_id}] FAIL idx={gi}: {e!r}", flush=True)
            if tmp.exists():
                tmp.unlink()
            continue
        dt = time.time() - t
        rec = {"idx": gi, "path": fp.name, "cap": [prompt], "fps": float(a.fps),
               "num_frames": a.num_frames, "duration": a.num_frames / float(a.fps),
               "resolution": {"width": a.width, "height": a.height},
               "gen_seconds": round(dt, 1)}
        with shard_manifest.open("a") as f:
            f.write(json.dumps(rec) + "\n")
        (meta_dir / f"vid_{gi:06d}.json").write_text(json.dumps(rec))
        n_ok += 1
        print(f"[w{a.worker_id}] {n_ok}/{len(my_idx)} done, {dt:.0f}s idx={gi}", flush=True)
    print(f"[w{a.worker_id}] DONE_WORKER made {n_ok} new videos", flush=True)


if __name__ == "__main__":
    main()
