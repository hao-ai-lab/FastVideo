# SPDX-License-Identifier: Apache-2.0
"""Interactive dashboard for the SPARSE ~1-per-object + few-background sampler.

Per Yongqi's advice, our track budget for the sparse recipe is dramatically smaller
than MotionStream's 1000-2500: keep 1 track per SAM object, then add ``num_sampled``
extra points drawn from the rest (either uniformly at random, or weighted by the
precomputed low-rank informativeness). The dashboard lets us eyeball whether the
resulting subset actually covers the "action" of each clip before we commit to
launching training runs on it.

CPU only, gradio share (no GPU needed).

    .venv/bin/python data_pipeline/sparse_sampling_dashboard.py \
        --data-dir /mnt/weka/home/hao.zhang/shao/data/motion_pipeline/synthetic_toy --share
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from track_informativeness import _draw_overlay, _read_frames  # noqa: E402


def sparse_sample(oid: np.ndarray, tw: np.ndarray | None, vis0: np.ndarray, num_sampled: int, mode: str,
                  seed: int) -> np.ndarray:
    """One point per SAM object (weighted-by-`tw` inside each object) + ``num_sampled`` extras
    drawn from the rest via ``mode`` in {"random", "weighted"}. Returns kept bool [N]."""
    rng = np.random.default_rng(int(seed))
    N = oid.shape[0]
    valid = vis0 > 0.5
    keep = np.zeros(N, bool)
    # (a) 1 per object
    for o in np.unique(oid):
        if int(o) < 0:
            continue
        idx = np.where((oid == o) & valid)[0]
        if idx.size == 0:
            continue
        if tw is not None and tw[idx].sum() > 0:
            p = tw[idx] / tw[idx].sum()
            pick = int(rng.choice(idx, p=p))
        else:
            pick = int(rng.choice(idx))
        keep[pick] = True
    # (b) num_sampled extras from the remaining valid points
    pool = np.where((~keep) & valid)[0]
    k = int(min(max(num_sampled, 0), pool.size))
    if k > 0:
        if mode == "weighted" and tw is not None and tw[pool].sum() > 0:
            p = tw[pool] / tw[pool].sum()
            picks = rng.choice(pool, size=k, replace=False, p=p)
        else:
            picks = rng.choice(pool, size=k, replace=False)
        keep[picks] = True
    return keep


def _load(data_dir: Path) -> list[dict]:
    man = json.loads((data_dir / "videos2caption.json").read_text())
    labels_dir = data_dir / "sam_labels"
    items = []
    for it in man:
        stem = Path(it["path"]).stem
        vpath = str(data_dir / "videos" / it["path"])
        npzp = it.get("points_path") or str(data_dir / "tracks" / f"{stem}.npz")
        d = np.load(npzp)
        lp = labels_dir / f"{stem}.npy"
        labels = np.load(lp) if lp.exists() else None  # [H,W] int16, -1 = background
        items.append({
            "stem": stem,
            "caption": it["cap"][0] if isinstance(it.get("cap"), list) else "",
            "fps": int(it.get("fps", 24)),
            "vpath": vpath,
            "tracks": d["tracks"].astype(np.float32),
            "vis": d["visibility"].astype(np.float32),
            "oid": d["object_ids"].astype(np.int64) if "object_ids" in d else None,
            "tw": d["track_weights"].astype(np.float32) if "track_weights" in d else None,
            "labels": labels,
        })
    return items


def build_ui(items: list[dict], out_dir: Path):
    import gradio as gr
    import imageio.v2 as imageio
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def _render_frame(clip_idx: int, num_sampled: int, mode: str, seed: int, mask_alpha: float = 0.45):
        it = items[int(clip_idx)]
        frames = _read_frames(it["vpath"])
        tracks = it["tracks"][:frames.shape[0]]
        vis = it["vis"][:frames.shape[0]]
        oid = it["oid"]
        tw = it["tw"]
        labels = it.get("labels")
        vis0 = vis[0]
        if oid is None:
            return None, "No object_ids in npz — run segment_tracks first."
        keep = sparse_sample(oid, tw, vis0, int(num_sampled), mode, int(seed))
        n_kept = int(keep.sum())
        n_obj = int(len(np.unique(oid[oid >= 0])))
        # Blend SAM segmentation on top of frame 0: color each segment by the same tab20
        # index we use for the tracked points (mask id == object id).
        base = frames[0].astype(np.float32)
        if labels is not None:
            cmap = plt.get_cmap("tab20")
            H, W = base.shape[:2]
            overlay = np.zeros_like(base)
            uniq = np.unique(labels)
            for u in uniq:
                if int(u) < 0:
                    continue
                col = np.array(cmap(int(u) % 20)[:3]) * 255
                m = labels == u
                overlay[m] = col
            mask = (labels >= 0)[..., None]
            base = np.where(mask, (1 - mask_alpha) * base + mask_alpha * overlay, base)
        base = base.clip(0, 255).astype(np.uint8)
        fig, ax = plt.subplots(1, 1, figsize=(9, 5.5))
        ax.imshow(base)
        x, y = tracks[0, :, 0], tracks[0, :, 1]
        ax.scatter(x[~keep], y[~keep], c="0.5", s=3, alpha=0.35)  # dropped
        # kept: color per unique object id (SAME palette as the mask overlay above)
        cmap = plt.get_cmap("tab20")
        for o in np.unique(oid[keep]):
            mask_pts = keep & (oid == o)
            col = "white" if int(o) < 0 else cmap(int(o) % 20)
            ax.scatter(x[mask_pts], y[mask_pts], c=[col], s=90, edgecolors="black", linewidths=1.2)
        ax.set_title(
            f"{it['stem']}: {n_kept} tracks kept  ({n_obj} SAM objects, +{num_sampled} background, "
            f"{mode}, seed {seed})",
            fontsize=10)
        ax.axis("off")
        img_path = str(out_dir / f"preview_{it['stem']}.png")
        fig.savefig(img_path, dpi=90, bbox_inches="tight")
        plt.close(fig)
        info = json.dumps(
            {
                "caption": it["caption"],
                "n_kept": n_kept,
                "n_objects_present": n_obj,
                "kept_from_objects": int(len(np.unique(oid[keep][oid[keep] >= 0]))),
                "kept_from_background": int((keep & (oid < 0)).sum()),
            },
            indent=2)
        return img_path, info

    def _render_video(clip_idx: int, num_sampled: int, mode: str, seed: int):
        it = items[int(clip_idx)]
        frames = _read_frames(it["vpath"])
        tracks = it["tracks"][:frames.shape[0]]
        vis = it["vis"][:frames.shape[0]]
        oid = it["oid"]
        tw = it["tw"]
        keep = sparse_sample(oid, tw, vis[0], int(num_sampled), mode, int(seed))
        kidx = np.where(keep)[0]
        if kidx.size == 0:
            return None
        cmap = plt.get_cmap("tab20")
        cols = np.zeros((kidx.size, 3), dtype=np.uint8)
        for o in np.unique(oid[kidx]):
            m = (oid[kidx] == o)
            rgba = (1.0, 1.0, 1.0, 1.0) if int(o) < 0 else cmap(int(o) % 20)
            cols[m] = (np.array(rgba[:3]) * 255).astype(np.uint8)
        ov = _draw_overlay(frames, tracks[:, kidx].copy(), vis[:, kidx], cols, 14, 3, 0.5)
        out = str(out_dir / f"preview_{it['stem']}_overlay.mp4")
        imageio.mimsave(out, ov, fps=int(it.get("fps", 24)), macro_block_size=1)
        return out

    with gr.Blocks(title="Sparse track sampler dashboard") as demo:
        gr.Markdown("### Sparse-sampling recipe: 1 track per SAM object + `num_sampled` extras\n"
                    "Frame 0 shows all 2500 grid tracks in gray, and the kept ones colored by object. "
                    "Set `num_sampled` = 0 for pure 1-per-object; raise it to add background context. "
                    "Toggle `weighted` to bias the extras toward high-lowrank-informativeness points. "
                    "The overlay video shows only the kept tracks over the whole clip.")
        with gr.Row():
            with gr.Column(scale=1):
                clip_dd = gr.Dropdown([f"{i:02d} — {items[i]['stem']}" for i in range(len(items))],
                                      value=f"00 — {items[0]['stem']}",
                                      label="Clip",
                                      type="index")
                num_slider = gr.Slider(0, 200, value=20, step=1, label="num_sampled (extras)")
                mode_radio = gr.Radio(["random", "weighted"], value="weighted", label="extras sampling mode")
                seed_slider = gr.Slider(0, 100, value=0, step=1, label="seed")
                info_box = gr.Code(label="info", language="json")
                vid_btn = gr.Button("Render overlay video", variant="primary")
            with gr.Column(scale=2):
                frame_img = gr.Image(label="frame 0 with kept tracks", height=460)
                overlay_vid = gr.Video(label="kept-tracks overlay video", height=460)

        for src in [clip_dd, num_slider, mode_radio, seed_slider]:
            src.change(_render_frame, [clip_dd, num_slider, mode_radio, seed_slider], [frame_img, info_box])
        vid_btn.click(_render_video, [clip_dd, num_slider, mode_radio, seed_slider], overlay_vid)
        demo.load(_render_frame, [clip_dd, num_slider, mode_radio, seed_slider], [frame_img, info_box])
    return demo


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", required=True)
    p.add_argument("--out-dir",
                   default="/mnt/weka/home/hao.zhang/shao/data/motion_pipeline/sparse_sampling_dashboard_out")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=7894)
    p.add_argument("--share", action="store_true")
    args = p.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    items = _load(Path(args.data_dir))
    demo = build_ui(items, out_dir)
    demo.queue().launch(server_name=args.host, server_port=args.port, share=args.share, allowed_paths=[str(out_dir)])


if __name__ == "__main__":
    main()
