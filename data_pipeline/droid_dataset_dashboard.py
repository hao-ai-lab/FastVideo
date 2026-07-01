# SPDX-License-Identifier: Apache-2.0
"""Inspect a track-conditioning dataset: are the clips diverse, and do the
CoTracker tracks actually capture the motion?

Two concerns this surfaces:
  1. Near-duplicate clips (the frame-0 cluster can over-select the same scene).
  2. Weak control signal: CoTracker uses a frame-0 grid, so anything that ENTERS
     the frame later (or that the tracker loses) is never queried -> its motion is
     invisible to the tracks. A clip can look dynamic yet have near-static tracks.

``render`` (CPU; run on a node) writes, per clip:
  - <id>__overlay.mp4   the video with the CoTracker tracks drawn (dots + tails),
  - <id>__thumb.png     first frame,
and a ``stats.json`` with per-clip motion-coverage + a near-duplicate grouping.

``serve`` (no GPU) is a gradio gallery: clips sorted by motion (least first, so
dead/duplicate clips float to the top), each with its overlay + stats.

    # 1) render artifacts (node)
    srun --jobid=<job> --overlap --ntasks=1 .venv/bin/python data_pipeline/droid_dataset_dashboard.py render \
        --data-dir /mnt/weka/home/hao.zhang/shao/data/motion_pipeline/droid_track_200 --out <dash_dir>
    # 2) serve (login node ok)
    .venv/bin/python data_pipeline/droid_dataset_dashboard.py serve --dash-dir <dash_dir> --share
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

MOVE_THRESH_PX = 8.0  # a point "moves" if it travels > this from frame 0 (input px)
VIS_THRESH = 0.5


def _load_frames(path: str) -> np.ndarray:
    try:
        from decord import VideoReader, cpu
        vr = VideoReader(path, ctx=cpu(0))
        return vr.get_batch(list(range(len(vr)))).asnumpy()
    except Exception:  # noqa: BLE001
        import imageio.v2 as imageio
        rd = imageio.get_reader(path, format="ffmpeg")
        fr = np.stack([np.asarray(f) for f in rd], axis=0)
        rd.close()
        return fr


def _clip_stats(tracks: np.ndarray, vis: np.ndarray) -> dict[str, Any]:
    """Motion coverage from tracks [T,N,2] (px) + vis [T,N]."""
    T, N, _ = tracks.shape
    v = vis > VIS_THRESH
    # displacement of each point from its frame-0 position, only where visible
    disp = np.sqrt(((tracks - tracks[0:1])**2).sum(-1))  # [T,N]
    max_disp = np.where(v, disp, 0.0).max(0)  # [N] per-point peak travel
    moving = max_disp > MOVE_THRESH_PX
    return {
        "n_points": int(N),
        "frac_visible": float(v.mean()),
        "frac_moving": float(moving.mean()),  # fraction of grid points that ever move
        "n_moving": int(moving.sum()),
        "mean_motion_px": float(max_disp.mean()),
        "p95_motion_px": float(np.percentile(max_disp, 95)),
        "max_motion_px": float(max_disp.max()),
    }


def cmd_render(args: argparse.Namespace) -> None:
    from fastvideo.train.callbacks.track_validation import (_draw_overlay, _grid_colors, _subsample)
    data_dir = Path(args.data_dir)
    manifest = json.loads((data_dir / "videos2caption.json").read_text())
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    import imageio.v2 as imageio
    from PIL import Image

    embs: list[np.ndarray] = []
    entries: list[dict[str, Any]] = []
    for rec in manifest:
        cid = Path(rec["path"]).stem
        vpath = str(data_dir / "videos" / rec["path"])
        ppath = rec.get("points_path")
        frames = _load_frames(vpath)
        d = np.load(ppath)
        tracks = d["tracks"].astype(np.float32)[:frames.shape[0]]  # [T,N,2] px
        vis = d["visibility"].astype(np.float32)[:frames.shape[0]]
        H, W = int(frames.shape[1]), int(frames.shape[2])

        stats = _clip_stats(tracks, vis)

        # overlay (subsampled grid for clarity)
        grid = int(round(tracks.shape[1]**0.5))
        tn = tracks / np.array([W, H], np.float32)  # normalize for _subsample
        tr_s, vs_s = _subsample(tn, vis, grid, args.stride)
        colors = _grid_colors(grid, args.stride)[:tr_s.shape[1]]
        trpx = tr_s.copy()
        trpx[..., 0] *= W
        trpx[..., 1] *= H
        ov = _draw_overlay(frames, trpx, vs_s, colors, args.tail, 2, VIS_THRESH)
        imageio.mimsave(str(out / f"{cid}__overlay.mp4"), ov, fps=args.fps, macro_block_size=1)
        Image.fromarray(frames[0]).save(str(out / f"{cid}__thumb.png"))

        # frame-0 embedding for near-duplicate grouping (32x32 gray, L2-norm)
        g = np.asarray(Image.fromarray(frames[0]).convert("L").resize((32, 32)), np.float32).reshape(-1)
        embs.append(g / (np.linalg.norm(g) + 1e-8))

        entries.append({"id": cid, "episode": rec.get("source_episode"), "caption": rec["cap"][0], **stats})
        print(
            f"[dash] {cid} move%={stats['frac_moving']*100:4.1f} mean={stats['mean_motion_px']:5.1f}px "
            f"max={stats['max_motion_px']:5.1f}px",
            flush=True)

    # near-duplicate grouping: union-find on cosine-sim >= threshold
    E = np.stack(embs, 0)
    sim = E @ E.T
    n = len(entries)
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(n):
        for j in range(i + 1, n):
            if sim[i, j] >= args.dup_thresh:
                parent[find(i)] = find(j)
    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    group_id = {}
    for gi, (_, members) in enumerate(sorted(groups.items(), key=lambda kv: -len(kv[1]))):
        for m in members:
            group_id[m] = gi
    for i, e in enumerate(entries):
        e["dup_group"] = int(group_id[i])

    n_groups = len(groups)
    n_dead = sum(1 for e in entries if e["frac_moving"] < 0.02)
    summary = {
        "n_clips": n,
        "n_dup_groups": n_groups,
        "largest_dup_group": max(len(v) for v in groups.values()),
        "n_low_motion_clips(<2%)": n_dead,
        "median_frac_moving": float(np.median([e["frac_moving"] for e in entries])),
        "median_mean_motion_px": float(np.median([e["mean_motion_px"] for e in entries])),
        "dup_thresh": args.dup_thresh,
        "move_thresh_px": MOVE_THRESH_PX,
    }
    (out / "stats.json").write_text(json.dumps({"summary": summary, "clips": entries}, indent=2))
    print(f"[dash] SUMMARY: {json.dumps(summary, indent=2)}", flush=True)
    print(f"[dash] wrote dashboard artifacts -> {out}", flush=True)


def cmd_serve(args: argparse.Namespace) -> None:
    import gradio as gr
    dash = Path(args.dash_dir)
    data = json.loads((dash / "stats.json").read_text())
    summary, clips = data["summary"], data["clips"]
    # sort: least motion first (dead/duplicate clips surface), then by dup group
    order = sorted(range(len(clips)), key=lambda i: (clips[i]["frac_moving"], clips[i]["dup_group"]))

    def label(i: int) -> str:
        c = clips[i]
        return (f"#{i} ep{c['episode']} | move {c['frac_moving']*100:.0f}% | "
                f"mean {c['mean_motion_px']:.0f}px | dupgrp {c['dup_group']}")

    gallery_items = [(str(dash / f"{clips[i]['id']}__thumb.png"), label(i)) for i in order]

    def show(evt):  # gr.SelectData; unannotated so gradio's get_type_hints doesn't eval it
        i = order[evt.index]
        c = clips[i]
        return (str(dash / f"{c['id']}__overlay.mp4"), json.dumps(c, indent=2))

    head = (f"### DROID dataset inspection — {summary['n_clips']} clips\n"
            f"- near-duplicate groups: **{summary['n_dup_groups']}** "
            f"(largest group **{summary['largest_dup_group']}** clips)\n"
            f"- low-motion clips (<2% points move): **{summary['n_low_motion_clips(<2%)']}**\n"
            f"- median fraction of points moving: **{summary['median_frac_moving']*100:.1f}%**, "
            f"median mean-motion **{summary['median_mean_motion_px']:.1f}px**\n\n"
            f"Sorted least-motion first. Overlay = CoTracker tracks (dots + tails); "
            f"watch for moving objects with NO dots on them (off-frame entries / lost tracks).")

    with gr.Blocks(title="DROID dataset dashboard") as demo:
        gr.Markdown(head)
        with gr.Row():
            gal = gr.Gallery(value=gallery_items, columns=6, height=560, label="clips (least motion first)")
            with gr.Column():
                vid = gr.Video(label="track overlay")
                meta = gr.Code(label="clip stats", language="json")
        gal.select(show, None, [vid, meta])
    demo.queue().launch(server_name=args.host, server_port=args.port, share=args.share)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("render")
    r.add_argument("--data-dir", required=True, help="dataset root (videos/ + videos2caption.json + tracks/)")
    r.add_argument("--out", required=True)
    r.add_argument("--stride", type=int, default=3, help="subsample the NxN grid for overlay clarity")
    r.add_argument("--tail", type=int, default=12)
    r.add_argument("--fps", type=int, default=12)
    r.add_argument("--dup-thresh", type=float, default=0.985, help="cosine-sim on frame-0 to call clips duplicates")
    r.set_defaults(func=cmd_render)
    s = sub.add_parser("serve")
    s.add_argument("--dash-dir", required=True)
    s.add_argument("--host", default="0.0.0.0")
    s.add_argument("--port", type=int, default=7872)
    s.add_argument("--share", action="store_true")
    s.set_defaults(func=cmd_serve)
    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
