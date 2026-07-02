# SPDX-License-Identifier: Apache-2.0
"""Visualize the segmentation pipeline: FastSAM masks + chosen per-object points +
CoTracker tracks overlaid on the source video. `render` (GPU) writes artifacts;
`serve` (no GPU) is a gradio gallery with --share.

    srun ... env CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$PWD .venv/bin/python data_pipeline/segment_viz.py render \
        --data-dir <dataset root> --out <viz dir> --limit 12
    .venv/bin/python data_pipeline/segment_viz.py serve --viz-dir <viz dir> --share
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))


def _colors(n: int) -> np.ndarray:
    import colorsys
    return np.array([[int(255 * c) for c in colorsys.hsv_to_rgb((i * 0.61803) % 1.0, 0.65, 1.0)]
                     for i in range(max(1, n))], np.uint8)


def _read_frames(path: str) -> np.ndarray:
    try:
        from decord import VideoReader, cpu
        vr = VideoReader(path, ctx=cpu(0))
        return vr.get_batch(list(range(len(vr)))).asnumpy()
    except Exception:  # noqa: BLE001
        import av
        c = av.open(path)
        return np.stack([f.to_ndarray(format="rgb24") for f in c.decode(video=0)])


def cmd_render(args: argparse.Namespace) -> None:
    from PIL import Image, ImageDraw
    from ultralytics import FastSAM
    from fastvideo.train.callbacks.track_validation import _draw_overlay
    import imageio.v2 as imageio
    model = FastSAM(args.model)
    data = Path(args.data_dir)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    items = json.loads((data / "videos2caption.json").read_text())[:args.limit]
    entries = []
    for k, it in enumerate(items, 1):
        vpath = str(data / "videos" / it["path"])
        npz = it.get("points_path") or str(data / "tracks" / f"{Path(it['path']).stem}.npz")
        frames = _read_frames(vpath)
        H, W = frames.shape[1], frames.shape[2]
        d = np.load(npz)
        tracks = d["tracks"].astype(np.float32)[:frames.shape[0]]  # [T,N,2] px
        vis = d["visibility"].astype(np.float32)[:frames.shape[0]]
        oid = d["object_ids"].astype(np.int64) if "object_ids" in d else np.full(tracks.shape[1], -1)

        # (1) SAM masks + chosen points on frame 0
        res = model(frames[0],
                    device=args.device,
                    retina_masks=True,
                    imgsz=args.imgsz,
                    conf=args.conf,
                    iou=args.iou,
                    verbose=False)
        img = Image.fromarray(frames[0]).convert("RGB")
        objs = sorted(int(o) for o in np.unique(oid) if int(o) >= 0)
        cols = _colors(len(objs) + 1)
        if res and res[0].masks is not None:
            masks = res[0].masks.data.cpu().numpy().astype(bool)
            base = np.asarray(img).astype(np.float32)
            for mi, m in enumerate(masks):
                if m.shape != (H, W):
                    import cv2
                    m = cv2.resize(m.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST).astype(bool)
                base[m] = 0.55 * base[m] + 0.45 * cols[mi % len(cols)][None].astype(np.float32)
            img = Image.fromarray(base.clip(0, 255).astype(np.uint8))
        draw = ImageDraw.Draw(img)
        # chosen point per object = highest-motion track in that object
        disp = np.sqrt(((tracks - tracks[0:1])**2).sum(-1)).max(0)  # [N]
        chosen = []
        for oi, o in enumerate(objs):
            idx = np.where(oid == o)[0]
            pick = idx[np.argmax(disp[idx])]
            x, y = float(tracks[0, pick, 0]), float(tracks[0, pick, 1])
            chosen.append(pick)
            draw.ellipse([x - 5, y - 5, x + 5, y + 5], fill=(255, 255, 255), outline=(0, 0, 0))
        img.save(str(out / f"{Path(it['path']).stem}_sam.png"))

        # (2) CoTracker tracks over the video, colored by object (background gray, objects colored)
        pcols = np.tile(np.array([[110, 110, 110]], np.uint8), (tracks.shape[1], 1))
        for oi, o in enumerate(objs):
            pcols[oid == o] = cols[oi % len(cols)]
        st = max(1, tracks.shape[1] // 900)  # subsample for clarity
        sel = np.arange(0, tracks.shape[1], st)
        trpx = tracks[:, sel].copy()
        ov = _draw_overlay(frames, trpx, vis[:, sel], pcols[sel], 12, 2, 0.5)
        imageio.mimsave(str(out / f"{Path(it['path']).stem}_tracks.mp4"),
                        ov,
                        fps=int(it.get("fps", 24)),
                        macro_block_size=1)
        entries.append({
            "id": Path(it["path"]).stem,
            "caption": it["cap"][0][:120],
            "n_objects": len(objs),
            "n_labeled": int((oid >= 0).sum())
        })
        print(f"[viz] [{k}/{len(items)}] {it['path']}: {len(objs)} objects", flush=True)
    (out / "manifest.json").write_text(json.dumps(entries, indent=2))
    print(f"[viz] rendered {len(entries)} clips -> {out}", flush=True)


def cmd_serve(args: argparse.Namespace) -> None:
    import gradio as gr
    viz = Path(args.viz_dir)
    entries = json.loads((viz / "manifest.json").read_text())
    gal = [(str(viz / f"{e['id']}_sam.png"), f"{e['id']} | {e['n_objects']} objs") for e in entries]

    def show(evt):  # gr.SelectData
        e = entries[evt.index]
        return (str(viz / f"{e['id']}_sam.png"), str(viz / f"{e['id']}_tracks.mp4"), json.dumps(e, indent=2))

    with gr.Blocks(title="Segmentation + tracks viz") as demo:
        gr.Markdown("### FastSAM segments + chosen per-object points + CoTracker tracks\n"
                    "Left gallery: frame-0 with SAM masks (color) and the chosen point per object (white dot). "
                    "Click a clip: SAM frame + the video with CoTracker tracks (background gray, objects colored).")
        with gr.Row():
            g = gr.Gallery(value=gal, columns=4, height=520, label="clips")
            with gr.Column():
                samimg = gr.Image(label="frame-0: SAM masks + chosen points")
                trkvid = gr.Video(label="CoTracker tracks (colored by object)")
                meta = gr.Code(label="info", language="json")
        g.select(show, None, [samimg, trkvid, meta])
    demo.queue().launch(server_name=args.host, server_port=args.port, share=args.share)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("render")
    r.add_argument("--data-dir", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--model", default="FastSAM-s.pt")
    r.add_argument("--device", default="cuda")
    r.add_argument("--imgsz", type=int, default=1024)
    r.add_argument("--conf", type=float, default=0.4)
    r.add_argument("--iou", type=float, default=0.9)
    r.add_argument("--limit", type=int, default=12)
    r.set_defaults(func=cmd_render)
    s = sub.add_parser("serve")
    s.add_argument("--viz-dir", required=True)
    s.add_argument("--host", default="0.0.0.0")
    s.add_argument("--port", type=int, default=7880)
    s.add_argument("--share", action="store_true")
    s.set_defaults(func=cmd_serve)
    a = p.parse_args()
    a.func(a)


if __name__ == "__main__":
    main()
