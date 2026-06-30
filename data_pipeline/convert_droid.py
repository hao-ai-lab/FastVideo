# SPDX-License-Identifier: Apache-2.0
"""Stage 0b (DROID): build a track-conditioning training set from DROID.

Goal (per the project decision): a set where the **first frame is similar** across
clips but the **motion is diverse**, with a single generic prompt, so the model is
forced to read the point tracks to reduce loss (content cannot determine the output).

DROID has no scene grouping in its LeRobot metadata, so "similar first frame" is
obtained by *clustering frame-0*: we download a candidate pool, embed each first
frame (downscaled RGB), and pick the ``--num-clips`` episodes whose first frames are
mutually most similar (the tightest dense cluster -- the centroid that minimises the
radius to its k-th nearest neighbour, then its k nearest).

Output mirrors ``generate_videos.py`` exactly so the existing downstream
(``extract_tracks.py`` -> ``v1_preprocess.py --preprocess_task i2v_track``) ingests it
unchanged:

    <out>/videos/vid_000000.mp4 ...        libx264, ``--num-frames`` frames @ ``--fps``
    <out>/videos2caption.json              FastVideo manifest (one generic caption)
    <out>/merge.txt                        "<videos_dir>,<json_path>"

Source: HuggingFace LeRobot mirror ``IPEC-COMMUNITY/droid_lerobot`` (v2.0): per-episode
av1 mp4 at ``videos/chunk-{c:03d}/{video_key}/episode_{i:06d}.mp4`` (av1 -> decoded
with PyAV; re-encoded to libx264 so downstream decord reads it). 180x320, 15 fps.

Run on a compute node (network + a little CPU; no GPU)::

    srun --jobid=<job> --overlap --ntasks=1 .venv/bin/python data_pipeline/convert_droid.py \
        --out-dir /mnt/weka/home/hao.zhang/shao/data/motion_pipeline/droid_track_200 \
        --num-clips 200 --candidate-scan 1500
"""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

REPO = "IPEC-COMMUNITY/droid_lerobot"
CHUNK_SIZE = 1000  # meta/info.json chunks_size
DEFAULT_VIDEO_KEY = "observation.images.exterior_image_1_left"
DEFAULT_PROMPT = "a robot arm manipulating objects on a tabletop"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--num-clips", type=int, default=200, help="How many episodes to keep (the cluster size).")
    p.add_argument("--candidate-scan",
                   type=int,
                   default=1500,
                   help="How many episodes to download+embed before clustering. Larger => tighter cluster.")
    p.add_argument("--video-key", type=str, default=DEFAULT_VIDEO_KEY)
    p.add_argument("--num-frames", type=int, default=121)
    p.add_argument("--fps", type=int, default=24, help="Written fps == train_fps so preprocess does NOT resample.")
    p.add_argument("--frame-start", type=int, default=0, help="First source-frame index of the kept window.")
    p.add_argument("--prompt", type=str, default=DEFAULT_PROMPT)
    p.add_argument("--embed-size", type=int, default=64, help="Frame-0 is resized to NxN for the similarity embedding.")
    p.add_argument("--no-cluster", action="store_true", help="Skip clustering; keep the first --num-clips candidates.")
    p.add_argument("--download-workers", type=int, default=16)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def list_candidate_episodes(num_frames: int, scan: int) -> list[int]:
    """Episode indices (length >= num_frames), first ``scan`` in index order."""
    from huggingface_hub import hf_hub_download
    ep_path = hf_hub_download(REPO, "meta/episodes.jsonl", repo_type="dataset")
    out: list[int] = []
    with open(ep_path) as f:
        for line in f:
            d = json.loads(line)
            if int(d.get("length", 0)) >= num_frames:
                out.append(int(d["episode_index"]))
                if len(out) >= scan:
                    break
    return out


def episode_video_path(ep_idx: int, video_key: str) -> str:
    chunk = ep_idx // CHUNK_SIZE
    return f"videos/chunk-{chunk:03d}/{video_key}/episode_{ep_idx:06d}.mp4"


def download_one(ep_idx: int, video_key: str) -> tuple[int, str | None]:
    from huggingface_hub import hf_hub_download
    try:
        p = hf_hub_download(REPO, episode_video_path(ep_idx, video_key), repo_type="dataset")
        return ep_idx, p
    except Exception:  # noqa: BLE001
        return ep_idx, None


def decode_frames(path: str, start: int, count: int) -> np.ndarray | None:
    """Decode ``count`` frames starting at ``start`` (RGB uint8 [T,H,W,3]) via PyAV (handles av1)."""
    import av
    try:
        container = av.open(path)
        frames = []
        for idx, frame in enumerate(container.decode(video=0)):
            if idx >= start:
                frames.append(frame.to_ndarray(format="rgb24"))
                if len(frames) >= count:
                    break
        container.close()
    except Exception:  # noqa: BLE001
        return None
    if len(frames) < count:
        return None
    return np.stack(frames, axis=0)


def embed_frame0(path: str, size: int) -> np.ndarray | None:
    from PIL import Image
    fr = decode_frames(path, 0, 1)
    if fr is None:
        return None
    img = Image.fromarray(fr[0]).resize((size, size), Image.BILINEAR)
    v = np.asarray(img, dtype=np.float32).reshape(-1)
    n = np.linalg.norm(v) + 1e-8
    return v / n


def pick_tightest_cluster(embs: np.ndarray, k: int) -> np.ndarray:
    """Return indices of the k mutually-most-similar embeddings.

    Centroid = the point minimising the distance to its k-th nearest neighbour
    (densest region); then take that point's k nearest neighbours.
    """
    # cosine distance (embs are L2-normalised) -> 1 - sim
    sim = embs @ embs.T
    dist = 1.0 - sim
    np.fill_diagonal(dist, 0.0)
    part = np.partition(dist, kth=min(k, dist.shape[0] - 1), axis=1)
    kth = part[:, min(k, dist.shape[0] - 1)]  # radius to k-th NN per row
    centroid = int(np.argmin(kth))
    order = np.argsort(dist[centroid])[:k]
    return order


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir
    videos_dir = out_dir / "videos"
    videos_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "videos2caption.json"
    merge_path = out_dir / "merge.txt"

    print(f"[droid] listing candidates (length>={args.num_frames}, scan={args.candidate_scan})...", flush=True)
    candidates = list_candidate_episodes(args.num_frames, args.candidate_scan)
    print(f"[droid] {len(candidates)} candidate episodes", flush=True)

    # Download (cached) + embed frame-0 in parallel.
    print(f"[droid] downloading + embedding frame-0 ({args.download_workers} workers)...", flush=True)
    ep_to_path: dict[int, str] = {}
    with ThreadPoolExecutor(max_workers=args.download_workers) as ex:
        futs = {ex.submit(download_one, ep, args.video_key): ep for ep in candidates}
        for i, fut in enumerate(as_completed(futs), 1):
            ep, path = fut.result()
            if path is not None:
                ep_to_path[ep] = path
            if i % 100 == 0:
                print(f"[droid]   downloaded {i}/{len(candidates)}", flush=True)
    print(f"[droid] downloaded {len(ep_to_path)} mp4s; embedding...", flush=True)

    eps_ok: list[int] = []
    embs: list[np.ndarray] = []
    for ep in candidates:
        if ep not in ep_to_path:
            continue
        e = embed_frame0(ep_to_path[ep], args.embed_size)
        if e is not None:
            eps_ok.append(ep)
            embs.append(e)
    embs_arr = np.stack(embs, axis=0)
    print(f"[droid] embedded {len(eps_ok)} frame-0s", flush=True)

    if args.no_cluster or len(eps_ok) <= args.num_clips:
        sel_local = list(range(min(args.num_clips, len(eps_ok))))
        print(f"[droid] no clustering -> first {len(sel_local)} candidates", flush=True)
    else:
        sel_local = pick_tightest_cluster(embs_arr, args.num_clips).tolist()
        # report cluster tightness
        sub = embs_arr[sel_local]
        c = sub.mean(0, keepdims=True)
        c /= np.linalg.norm(c) + 1e-8
        mean_sim = float((sub @ c.T).mean())
        print(f"[droid] tightest-cluster of {len(sel_local)} clips; mean cos-sim to centroid={mean_sim:.3f}",
              flush=True)

    selected_eps = [eps_ok[i] for i in sel_local]

    # Decode + write libx264 mp4s + manifest.
    import imageio.v2 as imageio
    records = []
    seq = 0
    for ep in selected_eps:
        frames = decode_frames(ep_to_path[ep], args.frame_start, args.num_frames)
        if frames is None:
            print(f"[droid]   skip ep{ep}: decode<{args.num_frames} frames", flush=True)
            continue
        H, W = int(frames.shape[1]), int(frames.shape[2])
        name = f"vid_{seq:06d}.mp4"
        imageio.mimsave(str(videos_dir / name),
                        list(frames),
                        fps=args.fps,
                        codec="libx264",
                        macro_block_size=1,
                        output_params=["-pix_fmt", "yuv420p"])
        records.append({
            "idx": seq,
            "path": name,
            "cap": [args.prompt],
            "fps": float(args.fps),
            "duration": float(args.num_frames) / float(args.fps),
            "num_frames": int(args.num_frames),
            "resolution": {
                "width": W,
                "height": H
            },
            "source_episode": int(ep),
        })
        seq += 1
        if seq % 25 == 0:
            print(f"[droid]   wrote {seq}/{len(selected_eps)}", flush=True)

    json_path.write_text(json.dumps(records, indent=2))
    merge_path.write_text(f"{videos_dir.resolve()},{json_path.resolve()}\n")
    print(f"[droid] DONE: wrote {len(records)} clips -> {videos_dir}", flush=True)
    print(f"[droid] manifest -> {json_path}", flush=True)
    print(f"[droid] next: extract_tracks.py --data-dir {out_dir} ; then v1_preprocess i2v_track", flush=True)


if __name__ == "__main__":
    main()
