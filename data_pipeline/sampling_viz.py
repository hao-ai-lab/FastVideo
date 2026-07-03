# SPDX-License-Identifier: Apache-2.0
"""Visualize the training-time track sampler on a preprocessed dataset.

For each clip it shows, straight from the stored ``tracks.npz`` (tracks + visibility +
``object_ids`` + ``track_weights``):
  (1) the low-rank informativeness HEATMAP over the 50x50 grid on frame 0,
  (2) the points the SAMPLER KEEPS (exact port of WanTrack ``_augment_tracks``:
      >=1 pt per SAM object  +  low-rank-weighted draw (1-uniform_frac)  +  uniform draw),
  (3) an overlay video of ONLY the kept tracks over the source video, colored by low-rank
      weight (blue = generic, red = informative), so you can see if good traces are sampled.

CPU only (no GPU / no model needed).

    .venv/bin/python data_pipeline/sampling_viz.py render --data-dir <dataset root> --out <dir> --limit 8
    .venv/bin/python data_pipeline/sampling_viz.py serve --viz-dir <dir> --share
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from track_informativeness import _draw_overlay, _read_frames, _norm  # noqa: E402


def sample_kept(tracks: np.ndarray,
                vis: np.ndarray,
                oid: np.ndarray | None,
                tw: np.ndarray | None,
                k: int,
                uniform_frac: float = 0.3,
                diversity: float = 1.0,
                seed: int = 0) -> np.ndarray:
    """Exact numpy port of WanTrack._augment_tracks sampling -> kept bool [N]."""
    rng = np.random.default_rng(seed)
    N = tracks.shape[1]
    valid = (vis[0] > 0.5).astype(np.float64)  # only frame-0-visible are valid queries
    if tw is not None and tw.size == N:
        w = (0.05 + diversity * tw.astype(np.float64)) * valid
    else:
        disp = np.sqrt(((tracks - tracks[0:1])**2).sum(-1))
        motion = (disp * (vis > 0.5)).max(0)
        w = (1.0 + diversity * (motion / (motion.mean() + 1e-6))) * valid
    if w.sum() <= 0:
        w = valid.copy()
    if w.sum() <= 0:
        w = np.ones(N)
    keep = np.zeros(N, bool)
    # (a) object coverage: one weighted pick per present segment
    if oid is not None:
        for o in np.unique(oid):
            if int(o) < 0:
                continue
            idx = np.where(oid == o)[0]
            ww = w[idx]
            ww = np.ones_like(ww) if ww.sum() <= 0 else ww
            keep[rng.choice(idx, p=ww / ww.sum())] = True
    # (b) low-rank-weighted draw for (1 - uniform_frac) of the remaining budget
    rem = max(0, k - int(keep.sum()))
    n_weighted = rem - int(round(rem * uniform_frac))
    pool = np.where((~keep) & (w > 0))[0]
    nw = min(n_weighted, pool.size)
    if nw > 0:
        pw = w[pool] / w[pool].sum()
        keep[rng.choice(pool, size=nw, replace=False, p=pw)] = True
    # (c) uniform draw to fill up to k (also backfills weighted shortfall)
    pool2 = np.where((~keep) & (valid > 0))[0]
    nu = min(max(0, k - int(keep.sum())), pool2.size)
    if nu > 0:
        keep[rng.choice(pool2, size=nu, replace=False)] = True
    return keep


def cmd_render(args: argparse.Namespace) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import imageio.v2 as imageio
    turbo = matplotlib.colormaps["turbo"]

    data = Path(args.data_dir)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    items = json.loads((data / "videos2caption.json").read_text())[:args.limit]
    entries = []
    for i, it in enumerate(items, 1):
        stem = Path(it["path"]).stem
        vpath = str(data / "videos" / it["path"])
        npz = it.get("points_path") or str(data / "tracks" / f"{stem}.npz")
        frames = _read_frames(vpath)
        d = np.load(npz)
        tracks = d["tracks"].astype(np.float32)[:frames.shape[0]]
        vis = d["visibility"].astype(np.float32)[:frames.shape[0]]
        oid = d["object_ids"].astype(np.int64) if "object_ids" in d else None
        tw = d["track_weights"].astype(np.float32) if "track_weights" in d else None
        N = tracks.shape[1]
        k = int(args.k)
        keep = sample_kept(tracks, vis, oid, tw, k, args.uniform_frac, args.diversity, args.seed)
        twn = _norm(tw) if tw is not None else np.zeros(N)
        x, y = tracks[0, :, 0], tracks[0, :, 1]

        # ---- figure: (1) lowrank heatmap  (2) kept vs dropped
        fig, axes = plt.subplots(1, 2, figsize=(17, 5.6))
        axes[0].imshow(frames[0])
        sc = axes[0].scatter(x, y, c=twn, cmap="turbo", s=12, vmin=0, vmax=1)
        axes[0].set_title(f"low-rank informativeness heatmap (all {N} pts)", fontsize=11)
        axes[0].axis("off")
        fig.colorbar(sc, ax=axes[0], fraction=0.03)
        axes[1].imshow(frames[0])
        axes[1].scatter(x[~keep], y[~keep], c="0.5", s=3, alpha=0.5)  # dropped
        axes[1].scatter(x[keep],
                        y[keep],
                        c=twn[keep],
                        cmap="turbo",
                        s=16,
                        vmin=0,
                        vmax=1,
                        edgecolors="white",
                        linewidths=0.3)  # kept, colored by weight
        n_obj = len(np.unique(oid[oid >= 0])) if oid is not None else 0
        n_cov = len(np.unique(oid[keep][oid[keep] >= 0])) if oid is not None else 0
        axes[1].set_title(f"kept by sampler: {int(keep.sum())}/{N}  (objects covered {n_cov}/{n_obj})", fontsize=11)
        axes[1].axis("off")
        fig.suptitle(f"{stem}  |  {(it['cap'][0] if isinstance(it.get('cap'), list) else '')[:80]}", fontsize=10)
        fig.tight_layout()
        heat = f"{stem}_sampling.png"
        fig.savefig(str(out / heat), dpi=80, bbox_inches="tight")
        plt.close(fig)

        # ---- overlay video: ONLY kept tracks, colored by low-rank weight
        kidx = np.where(keep)[0]
        # subsample for legibility if many kept
        if kidx.size > 700:
            kidx = kidx[np.linspace(0, kidx.size - 1, 700).round().astype(int)]
        pcols = (np.array([turbo(v)[:3] for v in twn[kidx]]) * 255).astype(np.uint8)
        ov = _draw_overlay(frames, tracks[:, kidx].copy(), vis[:, kidx], pcols, 14, 2, 0.5)
        vid = f"{stem}_kept.mp4"
        imageio.mimsave(str(out / vid), ov, fps=int(it.get("fps", 24)), macro_block_size=1)

        stats = {
            "id": stem,
            "n_points": int(N),
            "k_sampled": k,
            "kept": int(keep.sum()),
            "objects_covered": f"{n_cov}/{n_obj}",
            "mean_weight_kept": round(float(twn[keep].mean()), 3) if keep.any() else 0.0,
            "mean_weight_dropped": round(float(twn[~keep].mean()), 3) if (~keep).any() else 0.0,
            "frac_kept_informative(>0.5)": round(float((twn[keep] > 0.5).mean()), 3) if keep.any() else 0.0,
        }
        entries.append({"id": stem, "heat": heat, "video": vid, "stats": stats})
        print(
            f"[samp] [{i}/{len(items)}] {stem}: kept {int(keep.sum())}/{N}, cov {n_cov}/{n_obj}, "
            f"w_kept={stats['mean_weight_kept']} vs w_drop={stats['mean_weight_dropped']}",
            flush=True)

    (out / "manifest.json").write_text(json.dumps(entries, indent=2))
    print(f"[samp] rendered {len(entries)} clips -> {out}", flush=True)


def cmd_serve(args: argparse.Namespace) -> None:
    import gradio as gr
    viz = Path(args.viz_dir).resolve()
    entries = json.loads((viz / "manifest.json").read_text())
    gal = [(str(viz / e["heat"]), e["id"]) for e in entries]

    def show(evt: gr.SelectData):
        e = entries[evt.index]
        return str(viz / e["heat"]), str(viz / e["video"]), json.dumps(e["stats"], indent=2)

    with gr.Blocks(title="Track sampling viz") as demo:
        gr.Markdown("### Training-time track sampler: low-rank heatmap + kept points + kept-track overlay\n"
                    "Left panel: low-rank informativeness heatmap (bright = informative). Right panel: the points "
                    "the sampler **keeps** (colored by weight, gray = dropped) — should cover every object and "
                    "favor bright/informative points while keeping some uniform. Click a clip -> the overlay video "
                    "of ONLY the kept tracks (blue = generic, red = informative). Good sampling = kept traces follow "
                    "the moving hands/objects. `stats.mean_weight_kept` should exceed `mean_weight_dropped`.")
        with gr.Row():
            g = gr.Gallery(value=gal, columns=2, height=620, label="clips (click)")
            with gr.Column():
                heat = gr.Image(label="heatmap (left) + kept points (right)")
                vid = gr.Video(label="kept tracks over video (colored by low-rank weight)")
                meta = gr.Code(label="stats", language="json")
        g.select(show, None, [heat, vid, meta])
    demo.queue().launch(server_name=args.host, server_port=args.port, share=args.share, allowed_paths=[str(viz)])


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("render")
    r.add_argument("--data-dir", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--limit", type=int, default=8)
    r.add_argument("--k", type=int, default=1500, help="tracks to keep (training samples U[1000,2500])")
    r.add_argument("--uniform-frac", type=float, default=0.3)
    r.add_argument("--diversity", type=float, default=1.0)
    r.add_argument("--seed", type=int, default=0)
    r.set_defaults(func=cmd_render)
    s = sub.add_parser("serve")
    s.add_argument("--viz-dir", required=True)
    s.add_argument("--host", default="0.0.0.0")
    s.add_argument("--port", type=int, default=7889)
    s.add_argument("--share", action="store_true")
    s.set_defaults(func=cmd_serve)
    a = p.parse_args()
    a.func(a)


if __name__ == "__main__":
    main()
