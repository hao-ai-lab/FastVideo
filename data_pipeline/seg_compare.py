# SPDX-License-Identifier: Apache-2.0
"""Compare segmentation-model *variants* (not just FastSAM configs) for frame-0
object segmentation + sparse track sampling, on real openvid clips.

``render`` (GPU) runs each model in no-prompt "everything" mode on frame-0 of each
clip, labels the CoTracker grid points by object (smallest containing mask wins),
runs the 1-per-object + extras sparse sampler, and writes per (model, clip):
  - <..>_masks.png    frame-0 with colored SAM masks
  - <..>_sparse.png   frame-0 with kept sparse tracks (colored by object) vs dropped (gray)
  - <..>_tracks.mp4   kept sparse tracks over the whole clip
plus a per-entry .npz (oid/tw/vis0/xy0) so the dashboard can re-sample live.

``serve`` (CPU, login node) is a gradio dashboard: pick a clip, see every model's
masks / sparse-sampling side by side, tweak num_sampled live, click to inspect.

aarch64-safe: PyAV decode, headless cv2, no decord.

    # render on a GPU node (piggyback job 365)
    srun --overlap --jobid=365 --ntasks=1 --chdir=/mnt/lustre/vlm-s4duan/FastVideo \
      bash -lc 'source .venv/bin/activate && export CUDA_VISIBLE_DEVICES=1 \
        YOLO_CONFIG_DIR=/mnt/lustre/vlm-s4duan/.ultralytics \
        TORCH_HOME=/mnt/lustre/vlm-s4duan/.torch HF_HOME=/mnt/lustre/vlm-s4duan/.hf \
        PYTHONPATH=$PWD:$PWD/data_pipeline MPLCONFIGDIR=/mnt/lustre/vlm-s4duan/.mpl && \
        python -u data_pipeline/seg_compare.py render --limit 8'

    # serve on the login node, then ssh -L 7862:localhost:7862 <login>
    .venv/bin/python data_pipeline/seg_compare.py serve
"""
# NOTE: no ``from __future__ import annotations`` — gradio needs the real
# gr.SelectData annotation on the click handler.
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from segment_tracks import object_ids_for_points, lowrank_track_weights  # noqa: E402
from sparse_sampling_dashboard import sparse_sample  # noqa: E402
from track_informativeness import _draw_overlay  # noqa: E402

DATA_DEFAULT = "/mnt/lustre/vlm-s4duan/openvid_1m"
WEIGHTS_DIR = "/mnt/lustre/vlm-s4duan/models/seg"
OUT_DEFAULT = "/mnt/lustre/vlm-s4duan/seg_compare_out"

# The model zoo. loader in {FastSAM, SAM}. All run no-prompt everything-mode.
# min_area_frac drops mask specks (< frac of frame); nms_contain drops a mask if
# > that fraction of it is already covered by a larger kept mask (collapses
# part-level fragments into their object). max_masks caps to N largest (0=all).
DEFAULT_MODELS = [
    {"name": "fastsam-s", "loader": "FastSAM", "weight": "FastSAM-s.pt",
     "imgsz": 1024, "conf": 0.4, "iou": 0.9, "min_area_frac": 0.0015, "nms_contain": 0.0, "max_masks": 0},
    {"name": "fastsam-x", "loader": "FastSAM", "weight": "FastSAM-x.pt",
     "imgsz": 1024, "conf": 0.4, "iou": 0.9, "min_area_frac": 0.0015, "nms_contain": 0.0, "max_masks": 0},
    {"name": "mobilesam", "loader": "SAM", "weight": "mobile_sam.pt",
     "imgsz": 1024, "conf": 0.4, "iou": 0.9, "min_area_frac": 0.0015, "nms_contain": 0.0, "max_masks": 0},
    {"name": "sam-b", "loader": "SAM", "weight": "sam_b.pt",
     "imgsz": 1024, "conf": 0.4, "iou": 0.9, "min_area_frac": 0.0015, "nms_contain": 0.0, "max_masks": 0},
    {"name": "sam2-b", "loader": "SAM", "weight": "sam2_b.pt",
     "imgsz": 1024, "conf": 0.4, "iou": 0.9, "min_area_frac": 0.0015, "nms_contain": 0.0, "max_masks": 0},
    {"name": "sam2.1-b+", "loader": "SAM", "weight": "sam2.1_b.pt",
     "imgsz": 1024, "conf": 0.4, "iou": 0.9, "min_area_frac": 0.0015, "nms_contain": 0.0, "max_masks": 0},
    {"name": "sam2.1-l", "loader": "SAM", "weight": "sam2.1_l.pt",
     "imgsz": 1024, "conf": 0.4, "iou": 0.9, "min_area_frac": 0.0015, "nms_contain": 0.0, "max_masks": 0},
]

# sparse-sampler defaults baked into the rendered PNG/mp4 (dashboard can re-sample the PNG live)
SPARSE_DEFAULT = {"num_sampled": 20, "mode": "weighted", "seed": 0}


def read_frames(path: str) -> np.ndarray:
    """Full clip -> [T,H,W,3] uint8 via PyAV (aarch64-safe)."""
    import av
    c = av.open(path)
    frames = [f.to_ndarray(format="rgb24") for f in c.decode(video=0)]
    c.close()
    return np.stack(frames)


def colors(n: int) -> np.ndarray:
    import colorsys
    return np.array([[int(255 * v) for v in colorsys.hsv_to_rgb((i * 0.61803) % 1.0, 0.65, 1.0)]
                     for i in range(max(1, n))], np.uint8)


def extract_masks(res, H: int, W: int) -> np.ndarray:
    if not res or res[0].masks is None:
        return np.zeros((0, H, W), bool)
    m = res[0].masks.data.cpu().numpy().astype(bool)  # [M,h,w]
    if m.shape[0] and m.shape[1:] != (H, W):  # nearest resize w/o cv2
        import torch
        t = torch.from_numpy(m.astype(np.uint8))[None].float()
        t = torch.nn.functional.interpolate(t, size=(H, W), mode="nearest")[0]
        m = t.numpy().astype(bool)
    return m


def filter_masks(masks: np.ndarray, min_area_frac: float, nms_contain: float, max_masks: int) -> np.ndarray:
    if masks.shape[0] == 0:
        return masks
    H, W = masks.shape[1], masks.shape[2]
    areas = masks.reshape(masks.shape[0], -1).sum(1).astype(np.float64)
    if min_area_frac > 0:
        keep = (areas / float(H * W)) >= min_area_frac
        masks, areas = masks[keep], areas[keep]
    if masks.shape[0] and nms_contain > 0:  # drop fragments mostly inside a larger kept mask
        order = np.argsort(-areas)  # large -> small
        kept_idx, covered = [], np.zeros((H, W), bool)
        for i in order:
            m = masks[i]
            a = areas[i]
            if a > 0 and (m & covered).sum() / a > nms_contain:
                continue
            kept_idx.append(i)
            covered |= m
        masks, areas = masks[kept_idx], areas[kept_idx]
    if max_masks and masks.shape[0] > max_masks:
        masks = masks[np.argsort(-areas)[:max_masks]]
    return masks


def mask_panel(frame0: np.ndarray, masks: np.ndarray, oid: np.ndarray, xy0: np.ndarray) -> "Image.Image":
    from PIL import Image, ImageDraw
    mcols = colors(len(masks) + 1)
    base = frame0.astype(np.float32)
    for mi, m in enumerate(masks):
        base[m] = 0.55 * base[m] + 0.45 * mcols[mi % len(mcols)][None].astype(np.float32)
    img = Image.fromarray(base.clip(0, 255).astype(np.uint8))
    draw = ImageDraw.Draw(img)
    for o in sorted({int(v) for v in np.unique(oid) if int(v) >= 0}):  # 1 dot per object (its centroid pt)
        idx = np.where(oid == o)[0]
        cx, cy = float(xy0[idx, 0].mean()), float(xy0[idx, 1].mean())
        draw.ellipse([cx - 4, cy - 4, cx + 4, cy + 4], fill=(255, 255, 255), outline=(0, 0, 0))
    return img


def sparse_panel(frame0: np.ndarray, xy0: np.ndarray, oid: np.ndarray, keep: np.ndarray, title: str,
                 out_path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cmap = plt.get_cmap("tab20")
    fig, ax = plt.subplots(figsize=(9, 5.2))
    ax.imshow(frame0)
    x, y = xy0[:, 0], xy0[:, 1]
    ax.scatter(x[~keep], y[~keep], c="0.5", s=3, alpha=0.30)
    for o in np.unique(oid[keep]):
        m = keep & (oid == o)
        col = "white" if int(o) < 0 else cmap(int(o) % 20)
        ax.scatter(x[m], y[m], c=[col], s=80, edgecolors="black", linewidths=1.1)
    ax.set_title(title, fontsize=10)
    ax.axis("off")
    fig.savefig(out_path, dpi=90, bbox_inches="tight")
    plt.close(fig)


def cmd_render(args: argparse.Namespace) -> None:
    import torch
    import imageio.v2 as imageio
    from ultralytics import FastSAM, SAM

    data = Path(args.data_dir)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    clips_dir = data / ("clips" if (data / "clips").exists() else "videos")
    models_cfg = json.loads(Path(args.models_json).read_text()) if args.models_json else DEFAULT_MODELS
    if args.models:  # subset by name
        want = set(args.models.split(","))
        models_cfg = [m for m in models_cfg if m["name"] in want]
    loaders = {"FastSAM": FastSAM, "SAM": SAM}

    # load every model once (reused across all clips)
    print(f"[seg] loading {len(models_cfg)} models ...", flush=True)
    loaded = {}
    for mc in models_cfg:
        wp = Path(WEIGHTS_DIR) / mc["weight"]
        loaded[mc["name"]] = loaders[mc["loader"]](str(wp) if wp.exists() else mc["weight"])

    items = json.loads((data / "videos2caption.json").read_text())[:args.limit]
    print(f"[seg] {len(items)} clips x {len(models_cfg)} models", flush=True)
    entries, clip_ids = [], []
    for k, it in enumerate(items, 1):
        stem = Path(it["path"]).stem
        clip_ids.append(stem)
        vpath = str(clips_dir / it["path"])
        npz = it.get("points_path") or str(data / "tracks" / f"{stem}.npz")
        frames = read_frames(vpath)
        H, W = frames.shape[1], frames.shape[2]
        d = np.load(npz)
        tracks = d["tracks"].astype(np.float32)[:frames.shape[0]]  # [T,N,2] px
        vis = d["visibility"].astype(np.float32)[:frames.shape[0]]  # [T,N]
        xy0, vis0 = tracks[0], vis[0]
        tw = lowrank_track_weights(tracks)  # [N] informativeness in [0,1]
        # frame-0 png (shared, for dashboard live re-sampling background)
        from PIL import Image
        f0png = f"{stem}__frame0.png"
        Image.fromarray(frames[0]).save(str(out / f0png))

        for mc in models_cfg:
            t0 = time.time()
            res = loaded[mc["name"]](frames[0], device=args.device, retina_masks=True,
                                     imgsz=mc["imgsz"], conf=mc["conf"], iou=mc["iou"], verbose=False)
            masks = filter_masks(extract_masks(res, H, W), mc["min_area_frac"], mc["nms_contain"], mc["max_masks"])
            oid = object_ids_for_points(masks, xy0, H, W)
            n_obj = int(len({int(v) for v in np.unique(oid) if int(v) >= 0}))
            keep = sparse_sample(oid, tw, vis0, SPARSE_DEFAULT["num_sampled"], SPARSE_DEFAULT["mode"],
                                 SPARSE_DEFAULT["seed"])
            dt = time.time() - t0
            pfx = f"{mc['name']}__{stem}"
            mask_panel(frames[0], masks, oid, xy0).save(str(out / f"{pfx}_masks.png"))
            sparse_panel(frames[0], xy0, oid, keep,
                         f"{stem} | {mc['name']}: {int(keep.sum())} kept ({n_obj} objs +"
                         f"{SPARSE_DEFAULT['num_sampled']} extra)", str(out / f"{pfx}_sparse.png"))
            trk_name = ""
            if not args.no_video:
                import matplotlib.pyplot as plt
                cmap = plt.get_cmap("tab20")
                kidx = np.where(keep)[0]
                cols = np.zeros((kidx.size, 3), np.uint8)
                for o in np.unique(oid[kidx]):
                    mm = oid[kidx] == o
                    rgba = (1., 1., 1., 1.) if int(o) < 0 else cmap(int(o) % 20)
                    cols[mm] = (np.array(rgba[:3]) * 255).astype(np.uint8)
                ov = _draw_overlay(frames, tracks[:, kidx].copy(), vis[:, kidx], cols, 14, 3, 0.5)
                trk_name = f"{pfx}_tracks.mp4"
                imageio.mimsave(str(out / trk_name), ov, fps=int(it.get("fps", 24)), macro_block_size=1)
            # per-entry npz so the dashboard can re-sample live (no GPU)
            np.savez(str(out / f"{pfx}.npz"), oid=oid.astype(np.int64), tw=tw, vis0=vis0, xy0=xy0)
            entries.append({
                "model": mc["name"], "clip": stem,
                "caption": (it["cap"][0] if isinstance(it.get("cap"), list) else str(it.get("cap", "")))[:140],
                "n_masks": int(masks.shape[0]), "n_objects": n_obj,
                "n_labeled": int((oid >= 0).sum()), "n_points": int(oid.shape[0]),
                "n_kept": int(keep.sum()), "sec": round(dt, 2),
                "masks_png": f"{pfx}_masks.png", "sparse_png": f"{pfx}_sparse.png",
                "tracks_mp4": trk_name, "frame0_png": f0png, "npz": f"{pfx}.npz",
            })
            print(f"[seg] [{k}/{len(items)}] {stem} [{mc['name']}]: {masks.shape[0]} masks, "
                  f"{n_obj} objs, {(oid >= 0).sum()}/{oid.shape[0]} pts labeled, {dt:.1f}s", flush=True)
        del frames
        torch.cuda.empty_cache()

    manifest = {"models": models_cfg, "clips": clip_ids, "sparse_default": SPARSE_DEFAULT, "entries": entries}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"[seg] wrote {len(entries)} entries -> {out}/manifest.json", flush=True)


def cmd_serve(args: argparse.Namespace) -> None:
    import gradio as gr
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    viz = Path(args.viz_dir).resolve()
    manifest = json.loads((viz / "manifest.json").read_text())
    entries = manifest["entries"]
    model_names = [m["name"] for m in manifest["models"]]
    clip_ids = manifest["clips"]
    by_clip = {c: [e for e in entries if e["clip"] == c] for c in clip_ids}
    caption_of = {c: (by_clip[c][0]["caption"] if by_clip.get(c) else "") for c in clip_ids}
    npz_cache = {}

    def _npz(e):
        p = e["npz"]
        if p not in npz_cache:
            npz_cache[p] = {k: v for k, v in np.load(str(viz / p)).items()}
        return npz_cache[p]

    def masks_gallery(clip):
        es = sorted(by_clip.get(clip, []), key=lambda e: model_names.index(e["model"]))
        return [(str(viz / e["masks_png"]), f'{e["model"]}  •  {e["n_masks"]}m / {e["n_objects"]}o') for e in es]

    def resample_png(e, num_sampled, mode, seed):
        z = _npz(e)
        oid, tw, vis0, xy0 = z["oid"], z["tw"], z["vis0"], z["xy0"]
        keep = sparse_sample(oid, tw, vis0, int(num_sampled), mode, int(seed))
        frame0 = plt.imread(str(viz / e["frame0_png"]))
        cmap = plt.get_cmap("tab20")
        fig, ax = plt.subplots(figsize=(9, 5.2))
        ax.imshow(frame0)
        x, y = xy0[:, 0], xy0[:, 1]
        ax.scatter(x[~keep], y[~keep], c="0.5", s=3, alpha=0.30)
        for o in np.unique(oid[keep]):
            m = keep & (oid == o)
            col = "white" if int(o) < 0 else cmap(int(o) % 20)
            ax.scatter(x[m], y[m], c=[col], s=80, edgecolors="black", linewidths=1.1)
        n_obj = int(len({int(v) for v in np.unique(oid) if int(v) >= 0}))
        ax.set_title(f'{e["clip"]} | {e["model"]}: {int(keep.sum())} kept ({n_obj} objs +{num_sampled}, {mode})',
                     fontsize=10)
        ax.axis("off")
        import numpy as _np
        fig.canvas.draw()
        buf = _np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
        plt.close(fig)
        return buf

    def sparse_gallery(clip, num_sampled, mode, seed):
        es = sorted(by_clip.get(clip, []), key=lambda e: model_names.index(e["model"]))
        return [(resample_png(e, num_sampled, mode, seed), f'{e["model"]}  •  {e["n_kept"]}→') for e in es]

    with gr.Blocks(title="Segmentation model comparison") as demo:
        gr.Markdown(
            "## Segmentation model comparison — frame-0 masks + sparse track sampling\n"
            "Pick a **clip**; every model is shown side by side. **Masks** tab = raw everything-mode "
            "segmentation (tile label `<#masks>m / <#objects-with-points>o`). **Sparse sampling** tab = "
            "1 track per object + N extras (tweak live). Fewer, cleaner object masks = better for our "
            "1-per-object recipe. Click a tile to inspect it big + its track-overlay video.")
        clip_dd = gr.Dropdown(clip_ids, value=clip_ids[0], label=f"clip (1 of {len(clip_ids)})")
        clip_cap = gr.Markdown(f"*{caption_of[clip_ids[0]]}*")
        with gr.Tab("Masks (everything-mode)"):
            mg = gr.Gallery(value=masks_gallery(clip_ids[0]), columns=3, height=680,
                            label="SAM masks per model (click a tile)")
        with gr.Tab("Sparse sampling"):
            with gr.Row():
                num_sl = gr.Slider(0, 150, value=manifest["sparse_default"]["num_sampled"], step=1,
                                   label="num_sampled (extras beyond 1-per-object)")
                mode_rd = gr.Radio(["weighted", "random"], value=manifest["sparse_default"]["mode"], label="extras mode")
                seed_sl = gr.Slider(0, 50, value=manifest["sparse_default"]["seed"], step=1, label="seed")
            sg = gr.Gallery(value=sparse_gallery(clip_ids[0], manifest["sparse_default"]["num_sampled"],
                                                 manifest["sparse_default"]["mode"],
                                                 manifest["sparse_default"]["seed"]),
                            columns=3, height=640, label="sparse-sampled tracks per model (click a tile)")
        gr.Markdown("### Inspect")
        with gr.Row():
            big = gr.Image(label="selected panel", height=460)
            vid = gr.Video(label="kept-tracks overlay over the clip", height=460)
        meta = gr.Code(label="metrics", language="json")

        def on_clip(clip, num_sampled, mode, seed):
            return masks_gallery(clip), sparse_gallery(clip, num_sampled, mode, seed), f"*{caption_of.get(clip, '')}*"

        def on_sparse_ctrl(clip, num_sampled, mode, seed):
            return sparse_gallery(clip, num_sampled, mode, seed)

        def pick(clip, evt: gr.SelectData):
            es = sorted(by_clip.get(clip, []), key=lambda e: model_names.index(e["model"]))
            e = es[evt.index]
            v = str(viz / e["tracks_mp4"]) if e.get("tracks_mp4") else None
            return str(viz / e["masks_png"]), v, json.dumps(e, indent=2)

        clip_dd.change(on_clip, [clip_dd, num_sl, mode_rd, seed_sl], [mg, sg, clip_cap])
        for c in (num_sl, mode_rd, seed_sl):
            c.change(on_sparse_ctrl, [clip_dd, num_sl, mode_rd, seed_sl], sg)
        mg.select(pick, clip_dd, [big, vid, meta])
        sg.select(pick, clip_dd, [big, vid, meta])

    demo.queue().launch(server_name=args.host, server_port=args.port, share=args.share,
                        allowed_paths=[str(viz)])


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("render")
    r.add_argument("--data-dir", default=DATA_DEFAULT)
    r.add_argument("--out", default=OUT_DEFAULT)
    r.add_argument("--limit", type=int, default=8)
    r.add_argument("--device", default="cuda")
    r.add_argument("--models", default=None, help="comma-separated subset of model names")
    r.add_argument("--models-json", default=None, help="path to a JSON list of model config dicts")
    r.add_argument("--no-video", action="store_true", help="skip the track-overlay mp4s (faster)")
    r.set_defaults(func=cmd_render)
    s = sub.add_parser("serve")
    s.add_argument("--viz-dir", default=OUT_DEFAULT)
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=7862)
    s.add_argument("--share", action="store_true")
    s.set_defaults(func=cmd_serve)
    a = p.parse_args()
    a.func(a)


if __name__ == "__main__":
    main()
