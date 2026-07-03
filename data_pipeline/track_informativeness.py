# SPDX-License-Identifier: Apache-2.0
"""Diagnostic bake-off: which point tracks are *informative* vs *generic*?

A point that moves a lot is not necessarily important. If the camera pans, every
background point moves, but that motion is shared by the crowd, so any one background
point is redundant. We compare several ways to score / select useful points, from
simple statistics to clustering and learned embeddings.

Per-point scalar scores (heatmaps, bright = informative):
  abs_disp          max_t ||p_t - p_0||                       naive raw motion
  affine_residual   residual after a robust per-frame global   removes camera pan/zoom/rot
                    affine x0 -> p_t
  lowrank_residual  residual after subtracting mean trajectory  unique vs ALL points, but
                    + top-k shared SVD modes (ONE subspace)     assumes a single subspace
  motion_outlier    1 - HDBSCAN membership prob on trajectory   density outlier in motion space
                    embeddings                                  (#1 motion clustering)
  subspace_resid    per-cluster low-rank residual after         union-of-subspaces; the
                    Spectral motion segmentation                per-object lowrank (#2 SSC-lite)
  dino_uniqueness   1 - membership prob clustering DINOv2       semantic ⊕ motion (#3)
                    appearance ⊕ trajectory embeddings

Grouping / subset panels:
  motion_clusters   HDBSCAN labels on trajectory embeddings     motion groups (#1)
  subspace_clusters Spectral motion segmentation labels         object subspaces (#2)
  dpp / fps / greedy  diverse K-subset selection                what sampling wants (#4)

``render`` (CPU; DINOv2 on CPU) writes per clip a big multi-panel PNG + an overlay mp4
colored by low-rank informativeness + stats. ``serve`` is a gradio gallery (--share).

    .venv/bin/python data_pipeline/track_informativeness.py render --data-dir <root> --out <dir> --limit 8
    .venv/bin/python data_pipeline/track_informativeness.py serve --viz-dir <dir> --share
"""
import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

K_SELECT = 64  # size of the illustrative "useful subset" for the selection methods


# ---------------------------------------------------------------------------- overlay / io
def _draw_overlay(frames: Any,
                  tracks: Any,
                  vis: Any,
                  colors: Any,
                  tail: int = 12,
                  radius: int = 2,
                  vis_thresh: float = 0.5) -> list:
    """Self-contained track overlay (dots + short tails); no fastvideo/triton dependency."""
    from PIL import Image, ImageDraw
    T, N, _ = tracks.shape
    out = []
    for t in range(T):
        img = Image.fromarray(np.ascontiguousarray(frames[t]))
        dr = ImageDraw.Draw(img)
        t0 = max(0, t - tail)
        for i in range(N):
            col = (int(colors[i][0]), int(colors[i][1]), int(colors[i][2]))
            if t - t0 >= 1:
                dr.line([(float(p[0]), float(p[1])) for p in tracks[t0:t + 1, i]], fill=col, width=1)
            if vis[t, i] > vis_thresh:
                x, y = float(tracks[t, i, 0]), float(tracks[t, i, 1])
                dr.ellipse([x - radius, y - radius, x + radius, y + radius], fill=col)
        out.append(np.asarray(img))
    return out


def _read_frames(path: str) -> np.ndarray:
    try:
        from decord import VideoReader, cpu
        vr = VideoReader(path, ctx=cpu(0))
        return vr.get_batch(list(range(len(vr)))).asnumpy()
    except Exception:  # noqa: BLE001
        import av
        c = av.open(path)
        return np.stack([f.to_ndarray(format="rgb24") for f in c.decode(video=0)])


def _norm(x: np.ndarray, pct: float = 97.0) -> np.ndarray:
    lo = float(x.min())
    hi = float(np.percentile(x, pct))
    return np.zeros_like(x) if hi <= lo else np.clip((x - lo) / (hi - lo), 0.0, 1.0)


# ---------------------------------------------------------------------------- statistical scores
def abs_disp(tracks: np.ndarray) -> np.ndarray:
    return np.sqrt(((tracks - tracks[0:1])**2).sum(-1)).max(0)


def affine_residual(tracks: np.ndarray, vis: np.ndarray, iters: int = 3) -> np.ndarray:
    T, N, _ = tracks.shape
    X0 = np.concatenate([tracks[0], np.ones((N, 1), np.float32)], 1)  # [N,3]
    resid = np.zeros((T, N), np.float32)
    for t in range(1, T):
        pt = tracks[t]
        w = (vis[t] > 0.5).astype(np.float64)
        if w.sum() < 6:
            w = np.ones(N)
        for _ in range(iters):
            W = X0.T * w
            A = np.linalg.lstsq((W @ X0), (W @ pt), rcond=None)[0]
            r = np.sqrt(((X0 @ A - pt)**2).sum(-1)) + 1e-6
            delta = 1.5 * np.median(r[w > 0]) + 1e-3
            w = np.where(r <= delta, 1.0, delta / r) * (vis[t] > 0.5)
        resid[t] = r
    return np.median(resid[1:], 0)


def global_affine_motion(tracks: np.ndarray, vis: np.ndarray) -> float:
    """Camera-fixedness proxy: median per-frame magnitude of the fitted GLOBAL affine
    displacement (in px), normalized by frame diagonal. ~0 => fixed camera; large => the
    whole scene translates/zooms (moving/egocentric camera). Robust (Huber) fit."""
    T, N, _ = tracks.shape
    X0 = np.concatenate([tracks[0], np.ones((N, 1), np.float32)], 1)
    mags = []
    for t in range(1, T):
        pt = tracks[t]
        w = (vis[t] > 0.5).astype(np.float64)
        if w.sum() < 6:
            w = np.ones(N)
        for _ in range(2):
            Wm = X0.T * w
            A = np.linalg.lstsq((Wm @ X0), (Wm @ pt), rcond=None)[0]
            r = np.sqrt(((X0 @ A - pt)**2).sum(-1)) + 1e-6
            delta = 1.5 * np.median(r[w > 0]) + 1e-3
            w = np.where(r <= delta, 1.0, delta / r) * (vis[t] > 0.5)
        mags.append(float(np.median(np.sqrt(((X0 @ A - tracks[0])**2).sum(-1)))))
    diag = float(np.hypot(tracks[..., 0].max(), tracks[..., 1].max()) + 1e-6)
    return round(float(np.median(mags)) / diag, 4)


def displacement_matrix(tracks: np.ndarray) -> np.ndarray:
    """[N, 2T] per-point displacement relative to frame 0."""
    T, N, _ = tracks.shape
    return (tracks - tracks[0:1]).transpose(1, 0, 2).reshape(N, 2 * T)


def lowrank_residual(tracks: np.ndarray, rank: int = 3) -> np.ndarray:
    D = displacement_matrix(tracks)
    Dc = D - D.mean(0, keepdims=True)
    U, S, Vt = np.linalg.svd(Dc, full_matrices=False)
    r = min(rank, S.shape[0])
    return np.sqrt(((Dc - (U[:, :r] * S[:r]) @ Vt[:r])**2).sum(-1))


# ---------------------------------------------------------------------------- embeddings + clustering
def traj_embedding(tracks: np.ndarray, n_pca: int = 16) -> np.ndarray:
    from sklearn.decomposition import PCA
    D = displacement_matrix(tracks)
    D = (D - D.mean(0)) / (D.std(0) + 1e-6)
    d = int(min(n_pca, min(D.shape) - 1))
    return PCA(n_components=d, random_state=0).fit_transform(D).astype(np.float32)


def motion_cluster(emb: np.ndarray):
    """#1: HDBSCAN on trajectory embeddings -> labels, outlierness score (1 - membership prob)."""
    from sklearn.cluster import HDBSCAN
    mcs = max(10, emb.shape[0] // 100)
    cl = HDBSCAN(min_cluster_size=mcs, min_samples=5).fit(emb)
    return cl.labels_, (1.0 - cl.probabilities_).astype(np.float32)


def subspace_cluster(tracks: np.ndarray, emb: np.ndarray, k: int):
    """#2: Spectral motion segmentation into k groups; per-cluster low-rank residual."""
    from sklearn.cluster import SpectralClustering
    k = int(np.clip(k, 2, 8))
    lab = SpectralClustering(n_clusters=k,
                             affinity="nearest_neighbors",
                             n_neighbors=15,
                             assign_labels="cluster_qr",
                             random_state=0).fit_predict(emb)
    D = displacement_matrix(tracks)
    resid = np.zeros(lab.shape[0], np.float32)
    for c in np.unique(lab):
        idx = np.where(lab == c)[0]
        Z = D[idx] - D[idx].mean(0, keepdims=True)
        r = min(4, min(Z.shape) - 1)
        if r > 0:
            U, S, Vt = np.linalg.svd(Z, full_matrices=False)
            resid[idx] = np.sqrt(((Z - (U[:, :r] * S[:r]) @ Vt[:r])**2).sum(-1))
    return lab, resid


def dino_point_features(frame0: np.ndarray, xy: np.ndarray, model, device: str) -> np.ndarray:
    """#3: sample DINOv2 patch tokens at each frame-0 point (bilinear). Returns [N,C] L2-normed."""
    import torch
    import torch.nn.functional as F
    H, W = frame0.shape[0], frame0.shape[1]
    gw, gh = max(1, round(W / 14)), max(1, round(H / 14))
    rw, rh = gw * 14, gh * 14
    img = torch.from_numpy(frame0).float().permute(2, 0, 1)[None] / 255.0
    img = F.interpolate(img, size=(rh, rw), mode="bilinear", align_corners=False)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    img = ((img - mean) / std).to(device)
    with torch.no_grad():
        tok = model.forward_features(img)["x_norm_patchtokens"][0]  # [gh*gw, C]
    C = tok.shape[-1]
    grid = tok.reshape(gh, gw, C).permute(2, 0, 1)[None]  # [1,C,gh,gw]
    # normalized sample coords in [-1,1] from pixel xy
    gx = (xy[:, 0] / W) * 2 - 1
    gy = (xy[:, 1] / H) * 2 - 1
    samp = torch.from_numpy(np.stack([gx, gy], -1)).float().view(1, -1, 1, 2).to(device)
    feat = F.grid_sample(grid, samp, mode="bilinear", align_corners=False)[0, :, :, 0].T  # [N,C]
    feat = feat.cpu().numpy()
    return (feat / (np.linalg.norm(feat, axis=1, keepdims=True) + 1e-6)).astype(np.float32)


def dino_motion_cluster(dino_feat: np.ndarray, traj_emb: np.ndarray):
    """Cluster fused (semantic ⊕ motion) embedding -> labels, uniqueness (1 - membership prob)."""
    from sklearn.cluster import HDBSCAN
    from sklearn.decomposition import PCA
    dz = PCA(n_components=int(min(16, min(dino_feat.shape) - 1)), random_state=0).fit_transform(dino_feat)
    dz = (dz - dz.mean(0)) / (dz.std(0) + 1e-6)
    tz = (traj_emb - traj_emb.mean(0)) / (traj_emb.std(0) + 1e-6)
    fused = np.concatenate([dz, tz], 1)
    cl = HDBSCAN(min_cluster_size=max(10, fused.shape[0] // 100), min_samples=5).fit(fused)
    return cl.labels_, (1.0 - cl.probabilities_).astype(np.float32)


# ---------------------------------------------------------------------------- subset selection (#4)
def select_fps(emb: np.ndarray, K: int, seed: int) -> np.ndarray:
    sel = [int(seed)]
    d = np.linalg.norm(emb - emb[seed], axis=1)
    for _ in range(K - 1):
        i = int(np.argmax(d))
        sel.append(i)
        d = np.minimum(d, np.linalg.norm(emb - emb[i], axis=1))
    return np.array(sorted(set(sel)))


def select_dpp(quality: np.ndarray, emb: np.ndarray, K: int) -> np.ndarray:
    """Greedy MAP for a DPP with L = diag(q) S diag(q), S a Gaussian similarity kernel."""
    from scipy.spatial.distance import cdist
    q = _norm(quality) + 0.05
    d2 = cdist(emb, emb, "sqeuclidean")
    S = np.exp(-d2 / (np.median(d2) + 1e-6))
    L = (q[:, None] * q[None, :]) * S
    N = L.shape[0]
    cis = np.zeros((K, N))
    di2s = np.diag(L).copy()
    j = int(np.argmax(di2s))
    sel = [j]
    for it in range(1, K):
        k = it - 1
        ei = (L[j] - cis[:k, j] @ cis[:k]) / np.sqrt(max(di2s[j], 1e-12))
        cis[k] = ei
        di2s = di2s - ei**2
        di2s[sel] = -np.inf
        j = int(np.argmax(di2s))
        if di2s[j] <= 1e-10:
            break
        sel.append(j)
    return np.array(sorted(sel))


def select_greedy_recon(xy0: np.ndarray, D: np.ndarray, K: int) -> np.ndarray:
    """Add the point whose displacement is least predictable by RBF-interpolating the
    already-selected points over frame-0 positions -> targets motion discontinuities."""
    from scipy.spatial.distance import cdist
    sel = [int(np.argmax(np.linalg.norm(D - D.mean(0), axis=1)))]
    for _ in range(K - 1):
        S = np.array(sel)
        d2 = cdist(xy0, xy0[S], "sqeuclidean")
        W = np.exp(-d2 / (np.median(d2) + 1e-6))
        W /= (W.sum(1, keepdims=True) + 1e-9)
        err = np.linalg.norm(D - W @ D[S], axis=1)
        err[S] = -1
        sel.append(int(np.argmax(err)))
    return np.array(sorted(sel))


# ---------------------------------------------------------------------------- render
def _spearman(a, b) -> float:
    from scipy.stats import spearmanr
    return round(float(spearmanr(a, b).correlation), 3)


def cmd_render(args: argparse.Namespace) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import imageio.v2 as imageio

    dino_model, dino_dev = None, "cpu"
    if not args.no_dino:
        try:
            import torch
            dino_dev = "cuda" if torch.cuda.is_available() else "cpu"
            dino_model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14").to(dino_dev).eval()
            print(f"[info] DINOv2 loaded on {dino_dev}", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"[info] DINOv2 unavailable ({type(e).__name__}: {e}); skipping semantic panels", flush=True)

    data = Path(args.data_dir)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    items = json.loads((data / "videos2caption.json").read_text())[:args.limit]
    turbo = matplotlib.colormaps["turbo"]
    entries = []
    for k, it in enumerate(items, 1):
        stem = Path(it["path"]).stem
        vpath = str(data / "videos" / it["path"])
        npz = it.get("points_path") or str(data / "tracks" / f"{stem}.npz")
        frames = _read_frames(vpath)
        d = np.load(npz)
        tracks = d["tracks"].astype(np.float32)[:frames.shape[0]]
        vis = d["visibility"].astype(np.float32)[:frames.shape[0]]
        N = tracks.shape[1]
        x, y = tracks[0, :, 0], tracks[0, :, 1]
        D = displacement_matrix(tracks)
        emb = traj_embedding(tracks)

        # scalar scores
        scores = {
            "abs_disp": abs_disp(tracks),
            "affine_residual": affine_residual(tracks, vis),
            "lowrank_residual": lowrank_residual(tracks),
        }
        m_lab, m_out = motion_cluster(emb)
        scores["motion_outlier"] = m_out
        k_est = len(np.unique(m_lab[m_lab >= 0])) or 4
        s_lab, s_res = subspace_cluster(tracks, emb, k_est)
        scores["subspace_resid"] = s_res
        d_lab = None
        if dino_model is not None:
            try:
                dfeat = dino_point_features(frames[0], tracks[0], dino_model, dino_dev)
                d_lab, d_uniq = dino_motion_cluster(dfeat, emb)
                scores["dino_uniqueness"] = d_uniq
            except Exception as e:  # noqa: BLE001
                print(f"[info]   dino features failed for {stem}: {e}", flush=True)

        # subset selections (#4)
        seed = int(np.argmax(scores["lowrank_residual"]))
        subsets = {
            "dpp": select_dpp(scores["lowrank_residual"], emb, K_SELECT),
            "fps": select_fps(emb, K_SELECT, seed),
            "greedy_recon": select_greedy_recon(tracks[0], D, K_SELECT),
        }

        # ---- figure: row1 scalar scores, row2 cluster maps + subsets
        label_panels = [("motion_clusters", m_lab), ("subspace_clusters", s_lab)]
        if d_lab is not None:
            label_panels.append(("dino_clusters", d_lab))
        panels = ([("score", n, s)
                   for n, s in scores.items()] + [("labels", n, lab)
                                                  for n, lab in label_panels] + [("subset", n, idx)
                                                                                 for n, idx in subsets.items()])
        ncol = 5
        nrow = int(np.ceil(len(panels) / ncol))
        fig, axes = plt.subplots(nrow, ncol, figsize=(4.4 * ncol, 3.1 * nrow))
        axes = np.array(axes).reshape(-1)
        for ax in axes:
            ax.axis("off")
        for ax, (kind, name, val) in zip(axes, panels, strict=False):
            ax.imshow(frames[0])
            if kind == "score":
                ax.scatter(x, y, c=_norm(val), cmap="turbo", s=10, vmin=0, vmax=1)
                ax.set_title(f"{name}", fontsize=10)
            elif kind == "labels":
                noise = val < 0
                ax.scatter(x[noise], y[noise], c="0.5", s=6)
                ax.scatter(x[~noise], y[~noise], c=val[~noise], cmap="tab20", s=10)
                ax.set_title(f"{name} ({len(np.unique(val[val >= 0]))} groups)", fontsize=10)
            else:  # subset
                ax.scatter(x, y, c="0.4", s=3)
                ax.scatter(x[val], y[val], c="red", s=26, edgecolors="white", linewidths=0.4)
                ax.set_title(f"{name} (K={len(val)})", fontsize=10)
        fig.suptitle(f"{stem}  |  {(it['cap'][0] if isinstance(it.get('cap'), list) else '')[:90]}", fontsize=11)
        fig.tight_layout()
        heat = f"{stem}_scores.png"
        fig.savefig(str(out / heat), dpi=80, bbox_inches="tight")
        plt.close(fig)

        # ---- overlay colored by lowrank informativeness
        lr = _norm(scores["lowrank_residual"])
        pcols = (np.array([turbo(v)[:3] for v in lr]) * 255).astype(np.uint8)
        G = int(round(N**0.5))
        sel = (np.arange(N).reshape(G, G)[::max(1, G // 25), ::max(1, G // 25)].reshape(-1) if G *
               G == N else np.arange(0, N, max(1, N // 625)))
        ov = _draw_overlay(frames, tracks[:, sel].copy(), vis[:, sel], pcols[sel], 14, 2, 0.5)
        vid = f"{stem}_lowrank.mp4"
        imageio.mimsave(str(out / vid), ov, fps=int(it.get("fps", 24)), macro_block_size=1)

        # ---- stats
        names = list(scores.keys())
        rho = {f"{a}|{b}": _spearman(scores[a], scores[b]) for i, a in enumerate(names) for b in names[i + 1:]}
        lr_raw = scores["lowrank_residual"]
        lo_info = lr_raw <= np.percentile(lr_raw, 50)

        def _cov_generic(idx: Any, m_lab: Any = m_lab, lo_info: Any = lo_info) -> dict:
            grp = m_lab[idx]
            ngrp = len(np.unique(grp[grp >= 0]))
            tot = max(len(np.unique(m_lab[m_lab >= 0])), 1)
            return {"coverage_of_motion_groups": f"{ngrp}/{tot}", "frac_generic": round(float(lo_info[idx].mean()), 3)}

        stats = {
            "id": stem,
            "n_points": int(N),
            "camera_motion": global_affine_motion(tracks, vis),  # ~0 = fixed cam, high = moving
            "n_motion_groups": int(len(np.unique(m_lab[m_lab >= 0]))),
            "noise_frac": round(float((m_lab < 0).mean()), 3),
            "spearman": rho,
            "subset_quality": {
                name: _cov_generic(idx)
                for name, idx in subsets.items()
            },
        }
        entries.append({"id": stem, "heat": heat, "video": vid, "stats": stats})
        print(
            f"[info] [{k}/{len(items)}] {stem}: cam_motion={stats['camera_motion']}, "
            f"{stats['n_motion_groups']} motion groups, noise={stats['noise_frac']}, "
            f"rho(abs|lowrank)={rho.get('abs_disp|lowrank_residual')}",
            flush=True)

    (out / "manifest.json").write_text(json.dumps(entries, indent=2))
    print(f"[info] rendered {len(entries)} clips -> {out}", flush=True)


def cmd_serve(args: argparse.Namespace) -> None:
    import gradio as gr
    viz = Path(args.viz_dir).resolve()
    entries = json.loads((viz / "manifest.json").read_text())
    gal = [(str(viz / e["heat"]), e["id"]) for e in entries]

    def show(evt: gr.SelectData):
        e = entries[evt.index]
        return str(viz / e["heat"]), str(viz / e["video"]), json.dumps(e["stats"], indent=2)

    with gr.Blocks(title="Track informativeness bake-off") as demo:
        gr.Markdown(
            "### Which point tracks are useful? Statistics vs clustering vs embeddings vs subset selection\n"
            "**Row 1 (scores, bright = informative):** `abs_disp` raw motion · `affine_residual` after "
            "removing camera affine · `lowrank_residual` unique vs all (one subspace) · `motion_outlier` "
            "HDBSCAN density outlier · `subspace_resid` per-object lowrank · `dino_uniqueness` semantic⊕motion. "
            "**Row 2:** `motion_clusters` / `subspace_clusters` / `dino_clusters` (colored groups, gray = noise) "
            "and the diverse K-subsets `dpp` / `fps` / `greedy_recon` (red = chosen). "
            "Click a clip -> panels + overlay colored by low-rank informativeness. `stats.subset_quality` shows "
            "how well each subset covers the motion groups and how many chosen points are generic.")
        with gr.Row():
            g = gr.Gallery(value=gal, columns=2, height=640, label="clips (click)")
            with gr.Column():
                heat = gr.Image(label="method panels")
                vid = gr.Video(label="overlay colored by low-rank informativeness")
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
    r.add_argument("--no-dino", action="store_true", help="skip the DINOv2 semantic panels")
    r.set_defaults(func=cmd_render)
    s = sub.add_parser("serve")
    s.add_argument("--viz-dir", required=True)
    s.add_argument("--host", default="0.0.0.0")
    s.add_argument("--port", type=int, default=7883)
    s.add_argument("--share", action="store_true")
    s.set_defaults(func=cmd_serve)
    a = p.parse_args()
    a.func(a)


if __name__ == "__main__":
    main()
