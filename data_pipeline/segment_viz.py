# SPDX-License-Identifier: Apache-2.0
"""Visualize / sweep the segmentation pipeline: FastSAM masks + chosen per-object
points + CoTracker tracks overlaid on the source video.

``render`` (GPU) sweeps a set of FastSAM configs over the clips and writes one
SAM-panel PNG + one track-overlay mp4 per (config, clip). ``serve`` (no GPU) is a
gradio gallery with config/clip filters so you can scroll and compare any combo.

    srun ... env CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$PWD .venv/bin/python data_pipeline/segment_viz.py render \
        --data-dir <dataset root> --out <viz dir> --limit 8
    .venv/bin/python data_pipeline/segment_viz.py serve --viz-dir <viz dir> --share

Configs default to DEFAULT_SWEEP (below); override with --configs '<json>' or
--configs-json <file>. Each config: {name, conf, iou, imgsz, min_area_frac, max_masks}.
``min_area_frac`` drops masks smaller than that fraction of the frame; ``max_masks``
keeps only the N largest — both fight FastSAM over-segmentation.
"""
# NOTE: intentionally no ``from __future__ import annotations`` — gradio needs the
# real ``gr.SelectData`` annotation object on the click handler to inject the event.
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

# name, FastSAM conf/iou/imgsz, then two over-segmentation knobs:
#   min_area_frac: drop masks whose area < this fraction of the frame (0 = keep all)
#   max_masks:     keep only the N largest masks (0 = keep all)
DEFAULT_SWEEP = [
    {
        "name": "baseline",
        "conf": 0.4,
        "iou": 0.9,
        "imgsz": 1024,
        "min_area_frac": 0.0,
        "max_masks": 0
    },
    {
        "name": "conf0.6",
        "conf": 0.6,
        "iou": 0.9,
        "imgsz": 1024,
        "min_area_frac": 0.0,
        "max_masks": 0
    },
    {
        "name": "conf0.75",
        "conf": 0.75,
        "iou": 0.9,
        "imgsz": 1024,
        "min_area_frac": 0.0,
        "max_masks": 0
    },
    {
        "name": "iou0.6",
        "conf": 0.4,
        "iou": 0.6,
        "imgsz": 1024,
        "min_area_frac": 0.0,
        "max_masks": 0
    },
    {
        "name": "areafloor",
        "conf": 0.4,
        "iou": 0.9,
        "imgsz": 1024,
        "min_area_frac": 0.006,
        "max_masks": 0
    },
    {
        "name": "clean",
        "conf": 0.6,
        "iou": 0.7,
        "imgsz": 1024,
        "min_area_frac": 0.004,
        "max_masks": 25
    },
    {
        "name": "img1536",
        "conf": 0.5,
        "iou": 0.8,
        "imgsz": 1536,
        "min_area_frac": 0.003,
        "max_masks": 0
    },
]


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


def _load_configs(args: argparse.Namespace) -> list[dict]:
    if args.configs:
        raw = json.loads(args.configs)
    elif args.configs_json:
        raw = json.loads(Path(args.configs_json).read_text())
    else:
        raw = DEFAULT_SWEEP
    cfgs = []
    for i, c in enumerate(raw):
        cfgs.append({
            "name": str(c.get("name", f"cfg{i}")),
            "conf": float(c.get("conf", 0.4)),
            "iou": float(c.get("iou", 0.9)),
            "imgsz": int(c.get("imgsz", 1024)),
            "min_area_frac": float(c.get("min_area_frac", 0.0)),
            "max_masks": int(c.get("max_masks", 0)),
        })
    return cfgs


def _extract_masks(res, H: int, W: int) -> np.ndarray:
    if not res or res[0].masks is None:
        return np.zeros((0, H, W), bool)
    masks = res[0].masks.data.cpu().numpy().astype(bool)  # [M,h,w]
    if masks.shape[0] and masks.shape[1:] != (H, W):
        import cv2
        masks = np.stack(
            [cv2.resize(m.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST).astype(bool) for m in masks])
    return masks


def filter_masks(masks: np.ndarray, min_area_frac: float, max_masks: int) -> np.ndarray:
    """Drop tiny masks (< min_area_frac of frame) and keep only the N largest."""
    if masks.shape[0] == 0:
        return masks
    H, W = masks.shape[1], masks.shape[2]
    areas = masks.reshape(masks.shape[0], -1).sum(1).astype(np.float64)
    if min_area_frac > 0:
        keep = (areas / float(H * W)) >= min_area_frac
        masks, areas = masks[keep], areas[keep]
    if max_masks and masks.shape[0] > max_masks:
        masks = masks[np.argsort(-areas)[:max_masks]]
    return masks


def cmd_render(args: argparse.Namespace) -> None:
    from PIL import Image, ImageDraw
    from ultralytics import FastSAM
    from fastvideo.train.callbacks.track_validation import _draw_overlay
    from segment_tracks import object_ids_for_points
    import imageio.v2 as imageio

    configs = _load_configs(args)
    model = FastSAM(args.model)
    data = Path(args.data_dir)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    items = json.loads((data / "videos2caption.json").read_text())[:args.limit]
    print(f"[viz] {len(items)} clips x {len(configs)} configs = {len(items) * len(configs)} panels", flush=True)

    entries: list[dict] = []
    clip_ids: list[str] = []
    for k, it in enumerate(items, 1):
        stem = Path(it["path"]).stem
        clip_ids.append(stem)
        vpath = str(data / "videos" / it["path"])
        npz = it.get("points_path") or str(data / "tracks" / f"{stem}.npz")
        frames = _read_frames(vpath)
        H, W = frames.shape[1], frames.shape[2]
        d = np.load(npz)
        tracks = d["tracks"].astype(np.float32)[:frames.shape[0]]  # [T,N,2] px
        vis = d["visibility"].astype(np.float32)[:frames.shape[0]]  # [T,N]
        disp = np.sqrt(((tracks - tracks[0:1])**2).sum(-1)).max(0)  # [N] max displacement

        for cfg in configs:
            res = model(frames[0],
                        device=args.device,
                        retina_masks=True,
                        imgsz=cfg["imgsz"],
                        conf=cfg["conf"],
                        iou=cfg["iou"],
                        verbose=False)
            masks = filter_masks(_extract_masks(res, H, W), cfg["min_area_frac"], cfg["max_masks"])
            oid = object_ids_for_points(masks, tracks[0], H, W)  # [N] mask idx per point (-1 none)
            objs = sorted(int(o) for o in np.unique(oid) if int(o) >= 0)

            # (1) SAM panel: colored masks + chosen point (highest-motion track) per object
            mcols = _colors(len(masks) + 1)
            base = frames[0].astype(np.float32)
            for mi, m in enumerate(masks):
                base[m] = 0.55 * base[m] + 0.45 * mcols[mi % len(mcols)][None].astype(np.float32)
            img = Image.fromarray(base.clip(0, 255).astype(np.uint8))
            draw = ImageDraw.Draw(img)
            for o in objs:
                idx = np.where(oid == o)[0]
                pick = idx[np.argmax(disp[idx])]
                x, y = float(tracks[0, pick, 0]), float(tracks[0, pick, 1])
                draw.ellipse([x - 5, y - 5, x + 5, y + 5], fill=(255, 255, 255), outline=(0, 0, 0))
            sam_name = f"{cfg['name']}__{stem}_sam.png"
            img.save(str(out / sam_name))

            # (2) CoTracker tracks over the video, colored by object (background gray)
            trk_name = ""
            if not args.no_tracks:
                ocols = _colors(len(objs) + 1)
                pcols = np.tile(np.array([[110, 110, 110]], np.uint8), (tracks.shape[1], 1))
                for oi, o in enumerate(objs):
                    pcols[oid == o] = ocols[oi % len(ocols)]
                # grid-aware subsample: a flat row-major stride staggers columns and
                # looks like half the grid; stride rows AND cols equally instead so the
                # true (e.g. 50x50) grid stays visible and aligned.
                N = tracks.shape[1]
                G = int(round(N**0.5))
                if G * G == N:
                    k = max(1, G // 50)  # keep full grid up to 50x50
                    sel = np.arange(N).reshape(G, G)[::k, ::k].reshape(-1)
                else:
                    st = max(1, N // 1500)
                    sel = np.arange(0, N, st)
                ov = _draw_overlay(frames, tracks[:, sel].copy(), vis[:, sel], pcols[sel], 12, 2, 0.5)
                trk_name = f"{cfg['name']}__{stem}_tracks.mp4"
                imageio.mimsave(str(out / trk_name), ov, fps=int(it.get("fps", 24)), macro_block_size=1)

            entries.append({
                "config":
                cfg["name"],
                "clip":
                stem,
                "caption": (it["cap"][0] if isinstance(it.get("cap"), list) else str(it.get("cap", "")))[:120],
                "n_masks":
                int(len(masks)),
                "n_objects":
                len(objs),
                "n_labeled":
                int((oid >= 0).sum()),
                "n_points":
                int(oid.shape[0]),
                "sam":
                sam_name,
                "tracks":
                trk_name,
            })
            print(
                f"[viz] [{k}/{len(items)}] {stem} [{cfg['name']}]: "
                f"{len(masks)} masks, {len(objs)} objs, {(oid >= 0).sum()}/{oid.shape[0]} pts labeled",
                flush=True)

    manifest = {"configs": configs, "clips": clip_ids, "entries": entries}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"[viz] rendered {len(entries)} panels -> {out}", flush=True)


def cmd_serve(args: argparse.Namespace) -> None:
    import gradio as gr
    viz = Path(args.viz_dir).resolve()
    manifest = json.loads((viz / "manifest.json").read_text())
    entries = manifest["entries"]
    cfg_names = [c["name"] for c in manifest["configs"]]
    clip_ids = manifest["clips"]
    cfg_params = {c["name"]: c for c in manifest["configs"]}
    # closure state: the currently-filtered entry list that the gallery reflects
    current = {"entries": list(entries)}

    def _label(e: dict) -> str:
        return f'{e["clip"]} | {e["config"]} | {e["n_masks"]}m/{e["n_objects"]}o'

    def gallery_for(cfg_sel: str, clip_sel: str):
        es = entries
        if cfg_sel and cfg_sel != "(all)":
            es = [e for e in es if e["config"] == cfg_sel]
        if clip_sel and clip_sel != "(all)":
            es = [e for e in es if e["clip"] == clip_sel]
        current["entries"] = es
        return [(str(viz / e["sam"]), _label(e)) for e in es]

    def show(evt: gr.SelectData):
        e = current["entries"][evt.index]
        tv = str(viz / e["tracks"]) if e.get("tracks") else None
        info = {**e, "config_params": cfg_params.get(e["config"], {})}
        return str(viz / e["sam"]), tv, json.dumps(info, indent=2)

    with gr.Blocks(title="FastSAM config sweep") as demo:
        gr.Markdown("### FastSAM config sweep — masks + chosen per-object points + CoTracker tracks\n"
                    "Filter by **config** and/or **clip**, scroll the gallery, click a tile to inspect. "
                    "Tile label: `clip | config | <#masks>m/<#objects-with-points>o`. "
                    "Fewer, cleaner masks = less over-segmentation.")
        with gr.Row():
            cfg_dd = gr.Dropdown(["(all)"] + cfg_names, value="(all)", label="config")
            clip_dd = gr.Dropdown(["(all)"] + clip_ids, value="(all)", label="clip")
        with gr.Row():
            g = gr.Gallery(value=gallery_for("(all)", "(all)"), columns=4, height=620, label="panels (click to view)")
            with gr.Column():
                samimg = gr.Image(label="frame-0: SAM masks + chosen points")
                trkvid = gr.Video(label="CoTracker tracks (colored by object)")
                meta = gr.Code(label="info", language="json")
        cfg_dd.change(gallery_for, [cfg_dd, clip_dd], g)
        clip_dd.change(gallery_for, [cfg_dd, clip_dd], g)
        g.select(show, None, [samimg, trkvid, meta])
    # allowed_paths: without this gradio blocks serving the PNGs/mp4s that live
    # outside its app root -> the three side panels error out on click.
    demo.queue().launch(server_name=args.host, server_port=args.port, share=args.share, allowed_paths=[str(viz)])


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("render")
    r.add_argument("--data-dir", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--model", default="FastSAM-s.pt")
    r.add_argument("--device", default="cuda")
    r.add_argument("--limit", type=int, default=8)
    r.add_argument("--configs", default=None, help="inline JSON list of config dicts (overrides sweep)")
    r.add_argument("--configs-json", default=None, help="path to a JSON list of config dicts")
    r.add_argument("--no-tracks", action="store_true", help="skip the (slow) track-overlay mp4s")
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
