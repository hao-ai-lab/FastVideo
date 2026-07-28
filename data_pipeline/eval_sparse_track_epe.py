#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Evaluate super-sparse track adherence via CoTracker EPE.

For each val clip, we build the exact adversarial case the user described:
  * Pick ONE active foreground trace from the GT parquet — the highest-motion
    foreground track (object_ids >= 0, visible in frame 0).
  * Pick N static background anchors — random background tracks (object_ids == -1)
    with position frozen at frame-0 (visibility=1 across all frames).
  * Generate video conditioned on (active + anchors) — this is what training
    was supposed to teach the model to handle.
  * Also generate an ablation with track_points=None (unconditional motion).
  * Run CoTracker on both generated videos, initialised at frame 0 with the
    conditioning points. Compare CoTracker-extracted tracks against the GT
    trace to compute End-Point-Error in pixel space.

Metrics logged per clip and aggregated to wandb:
  epe_active_full   EPE(GT_active, CoTracker(gen_full))
  epe_active_notrack EPE(GT_active, CoTracker(gen_no_track))
  epe_bg            EPE(static_anchor, CoTracker(gen_full)) — should be small
  DELTA = epe_active_notrack - epe_active_full — positive means training helped.

Usage:
  srun --overlap --jobid=529 --ntasks=1 -w hpc-rack-2-13 --chdir=$PWD bash -lc '
    source .venv/bin/activate
    export HOME=/mnt/lustre/vlm-s4duan HF_HOME=/mnt/lustre/vlm-s4duan/.hf \
      TORCH_HOME=/mnt/lustre/vlm-s4duan/.torch MPLCONFIGDIR=/mnt/lustre/vlm-s4duan/.mpl \
      TRITON_CACHE_DIR=/tmp/triton_eval TOKENIZERS_PARALLELISM=false NCCL_CUMEM_ENABLE=0 \
      PYTHONPATH=$PWD TRACKWAN_TRACK_BIAS=1 CUDA_VISIBLE_DEVICES=2 \
      WANDB_API_KEY=<key> WANDB_MODE=online
    python data_pipeline/eval_sparse_track_epe.py \
      --model-dir /mnt/lustre/vlm-s4duan/exports/merged_bias_ckpt4800 \
      --yaml examples/train/scenario/worldmodel/finetune_wantrack_openvid_sparse_1p3b_merged_bias.yaml \
      --data-path /mnt/lustre/vlm-s4duan/openvid_1m/combined_parquet_dataset \
      --num-clips 5 --steps 30 --w-motion 1.5 \
      --wandb-run-name eval_mb4800_sparse
  '
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

# make trackwan_infer importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))


# ==========================================================================
# CoTracker
# ==========================================================================
def load_cotracker(device: str) -> torch.nn.Module:
    """Load CoTracker v3 offline. Uses the warm hub cache under $TORCH_HOME/hub."""
    hub_dir = Path(torch.hub.get_dir())
    local = hub_dir / "facebookresearch_co-tracker_main"
    if local.exists():
        model = torch.hub.load(str(local), "cotracker3_offline", source="local", trust_repo=True)
    else:
        model = torch.hub.load("facebookresearch/co-tracker", "cotracker3_offline", trust_repo=True)
    return model.to(device).eval()


@torch.no_grad()
def cotracker_query(cotracker: torch.nn.Module, video: np.ndarray, queries_xy: np.ndarray,
                    device: str) -> tuple[np.ndarray, np.ndarray]:
    """Track `queries_xy` (N, 2 pixel coords) forward through `video` [T,H,W,3] uint8.

    Returns tracks [T,N,2] (pixel coords) and visibility [T,N] (bool).
    """
    T, H, W, C = video.shape
    v = torch.from_numpy(video).permute(0, 3, 1, 2).float()[None].to(device)  # [1,T,3,H,W]
    N = queries_xy.shape[0]
    q = np.zeros((1, N, 3), dtype=np.float32)
    q[0, :, 0] = 0  # start-frame index
    q[0, :, 1] = queries_xy[:, 0]  # x
    q[0, :, 2] = queries_xy[:, 1]  # y
    tracks, visibility = cotracker(v, queries=torch.from_numpy(q).to(device))
    return tracks[0].cpu().numpy(), visibility[0].cpu().numpy()


# ==========================================================================
# Sparse-conditioning sampler
# ==========================================================================
def pick_sparse_conditioning(row: dict, *, num_active: int, num_bg: int,
                             seed: int) -> dict:
    """From a preprocessed parquet row, pick the sparse case: `num_active` moving fg
    traces (highest-displacement fg tracks visible in frame 0) + `num_bg` static bg
    anchors (random bg tracks, position frozen at frame-0).

    Everything is normalised to [0,1] to match training. Pixel-space is only used at
    the metric-computation step.
    """
    tp = row["track_points"]  # [T,N,2] normalised
    tv = row["track_visibility"]  # [T,N]
    oid = row["object_ids"]  # [N] int, -1 for bg
    T, N, _ = tp.shape
    rng = np.random.RandomState(seed)

    fg = np.where((oid >= 0) & (tv[0] > 0.5))[0]
    bg = np.where((oid == -1) & (tv[0] > 0.5))[0]
    if fg.size == 0:
        raise RuntimeError("no visible foreground tracks in frame 0")
    # displacement of each fg over the full clip (only over visible frames)
    disp = np.linalg.norm(tp[-1, fg] - tp[0, fg], axis=-1)
    active_idx = fg[np.argsort(-disp)[:num_active]]  # top-`num_active` moving fg
    n_bg = min(num_bg, bg.size)
    bg_idx = rng.choice(bg, size=n_bg, replace=False) if n_bg > 0 else np.zeros((0, ), dtype=int)

    active_gt = tp[:, active_idx]  # [T, num_active, 2] — GT motion
    active_vis = tv[:, active_idx]
    bg_static_pos = np.broadcast_to(tp[0:1, bg_idx], (T, n_bg, 2)).copy()  # position frozen at f0
    bg_vis = np.ones((T, n_bg), dtype=np.float32)

    cond_tracks = np.concatenate([active_gt, bg_static_pos], axis=1)  # [T, num_active+n_bg, 2]
    cond_vis = np.concatenate([active_vis, bg_vis], axis=1)
    return {
        "active_idx": active_idx,
        "active_gt": active_gt,  # [T,num_active,2] normalised
        "active_vis": active_vis,
        "bg_static_pos": bg_static_pos,  # [T,n_bg,2] normalised (const in t)
        "cond_tracks": cond_tracks,  # [T, num_active+n_bg, 2] normalised
        "cond_vis": cond_vis,
        "n_active": len(active_idx),
        "n_bg": n_bg,
    }


# ==========================================================================
# Metrics
# ==========================================================================
def epe_norm_to_px(pred_norm: np.ndarray, gt_norm: np.ndarray, vis: np.ndarray,
                   res_h: int, res_w: int) -> float:
    """L2 distance between predicted/GT normalised tracks, evaluated in pixel space,
    averaged only over visible frames × visible points."""
    diff = (pred_norm - gt_norm) * np.array([res_w, res_h])[None, None, :]
    l2 = np.linalg.norm(diff, axis=-1)  # [T, N]
    mask = vis > 0.5
    if not mask.any():
        return float("nan")
    return float(l2[mask].mean())


def epe_px(pred_px: np.ndarray, gt_px: np.ndarray, vis: np.ndarray) -> float:
    """L2 in pixel space — both inputs already in pixels."""
    l2 = np.linalg.norm(pred_px - gt_px, axis=-1)
    mask = vis > 0.5
    if not mask.any():
        return float("nan")
    return float(l2[mask].mean())


# ==========================================================================
# Trace overlay for wandb visualisation
# ==========================================================================
def overlay_tracks(video: np.ndarray, tracks_norm: np.ndarray, vis: np.ndarray,
                   colors: list[tuple[int, int, int]], radius: int = 3,
                   tail: int = 12) -> np.ndarray:
    """Draw normalised tracks on a video [T,H,W,3] uint8.  Small tail + current dot."""
    from PIL import Image, ImageDraw
    T, H, W, C = video.shape
    N = tracks_norm.shape[1]
    out = []
    for t in range(T):
        img = Image.fromarray(np.ascontiguousarray(video[t]))
        dr = ImageDraw.Draw(img)
        t0 = max(0, t - tail)
        for i in range(N):
            col = colors[i % len(colors)]
            # tail
            if t - t0 >= 1:
                pts = [(float(tracks_norm[k, i, 0] * W), float(tracks_norm[k, i, 1] * H))
                       for k in range(t0, t + 1)]
                dr.line(pts, fill=col, width=1)
            if vis[t, i] > 0.5:
                x = float(tracks_norm[t, i, 0] * W)
                y = float(tracks_norm[t, i, 1] * H)
                dr.ellipse([x - radius, y - radius, x + radius, y + radius], fill=col)
        out.append(np.asarray(img))
    return np.stack(out)


# ==========================================================================
# Parquet loader (small subset — avoid loading the whole 259k row set)
# ==========================================================================
def load_first_n_from_parquet(data_path: str, n: int, text_len: int) -> list:
    import pyarrow.parquet as pq
    from fastvideo.dataset.dataloader.schema import pyarrow_schema_i2v_track
    from fastvideo.dataset.utils import collate_rows_from_parquet_schema
    files = sorted(glob.glob(os.path.join(data_path, "**", "*.parquet"), recursive=True))
    if not files:
        raise FileNotFoundError(data_path)
    rows: list = []
    for f in files:
        rows.extend(pq.read_table(f).to_pylist())
        if len(rows) >= n:
            break
    sel = rows[:n]
    batch = collate_rows_from_parquet_schema(sel, pyarrow_schema_i2v_track,
                                             text_padding_length=int(text_len), cfg_rate=0.0)
    infos = batch.get("info_list") or [{} for _ in sel]
    out = []
    for i in range(len(sel)):
        out.append({
            "text_embedding": batch["text_embedding"][i:i + 1].clone(),
            "text_attention_mask": batch["text_attention_mask"][i:i + 1].clone(),
            "first_frame_latent": batch["first_frame_latent"][i:i + 1].clone(),
            "clip_feature": batch["clip_feature"][i:i + 1].clone(),
            "vae_latent": batch["vae_latent"][i:i + 1].clone(),
            "track_points": batch["track_points"][i].numpy(),  # [T,N,2] normalised
            "track_visibility": batch["track_visibility"][i].numpy(),  # [T,N]
            "object_ids": batch["object_ids"][i].numpy(),  # [N]
            "caption": str(infos[i].get("caption", "") if i < len(infos) else ""),
        })
    return out


# ==========================================================================
# Main
# ==========================================================================
def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model-dir", required=True)
    p.add_argument("--yaml", required=True)
    p.add_argument("--data-path", required=True)
    p.add_argument("--num-clips", type=int, default=5)
    p.add_argument("--num-active", type=int, default=1)
    p.add_argument("--num-bg", type=int, default=20)
    p.add_argument("--steps", type=int, default=30)
    p.add_argument("--w-motion", type=float, default=1.5)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--fps", type=int, default=24)
    p.add_argument("--wandb-project", default="wantrack-bidir")
    p.add_argument("--wandb-run-name", required=True)
    p.add_argument("--out-dir", default="/mnt/lustre/vlm-s4duan/eval_sparse_epe")
    args = p.parse_args()

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    import wandb
    import imageio.v2 as imageio
    import trackwan_infer as twi

    print(f"[eval] loading model {args.model_dir}", flush=True)
    model, tc = twi.load_trackwan(args.model_dir, args.yaml)
    text_len = int(tc.data.text_padding_length) if hasattr(tc.data, "text_padding_length") else 256

    print(f"[eval] loading {args.num_clips} parquet clips", flush=True)
    samples = load_first_n_from_parquet(args.data_path, args.num_clips, text_len)

    print("[eval] loading CoTracker", flush=True)
    device = model.device
    cotracker = load_cotracker(device.type + ":" + str(device.index) if device.index is not None else device.type)

    res_h = int(tc.data.num_height)
    res_w = int(tc.data.num_width)
    palette = [(255, 60, 60), (60, 255, 60), (60, 90, 255), (255, 200, 60), (200, 60, 255), (60, 255, 220)]

    run = wandb.init(project=args.wandb_project, name=args.wandb_run_name,
                     config={"model_dir": args.model_dir, "num_clips": args.num_clips,
                             "num_active": args.num_active, "num_bg": args.num_bg,
                             "steps": args.steps, "w_motion": args.w_motion, "res": [res_h, res_w]})

    per_clip: list = []

    for idx, s in enumerate(samples):
        try:
            sc = pick_sparse_conditioning(s, num_active=args.num_active, num_bg=args.num_bg,
                                          seed=args.seed + idx)
        except RuntimeError as exc:
            print(f"[eval] clip {idx}: {exc}; skipping", flush=True)
            continue

        # ── Generate two videos: with sparse tracks (motion CFG) + without any tracks ──
        tp_t = torch.from_numpy(sc["cond_tracks"])[None].float()  # [1,T,N,2]
        tv_t = torch.from_numpy(sc["cond_vis"])[None].float()  # [1,T,N]

        seed = args.seed + idx
        common = dict(first_frame_latent=s["first_frame_latent"],
                      text_embedding=s["text_embedding"],
                      text_attention_mask=s["text_attention_mask"],
                      clip_feature=s["clip_feature"],
                      num_steps=args.steps, seed=seed)

        print(f"[eval] clip {idx}: generating full ({sc['n_active']} active + {sc['n_bg']} anchors)", flush=True)
        t0 = time.time()
        lat_full = twi.generate(model, track_points=tp_t, track_visibility=tv_t,
                                guidance_scale=args.w_motion, **common)
        gen_full = twi.decode_to_pixels(model, lat_full)  # [T,H,W,3] uint8
        t_full = time.time() - t0
        print(f"[eval] clip {idx}: full took {t_full:.1f}s", flush=True)

        print(f"[eval] clip {idx}: generating no-track", flush=True)
        t0 = time.time()
        lat_no = twi.generate(model, track_points=None, track_visibility=None,
                              guidance_scale=1.0, **common)
        gen_no = twi.decode_to_pixels(model, lat_no)
        t_no = time.time() - t0
        print(f"[eval] clip {idx}: no-track took {t_no:.1f}s", flush=True)

        T, H, W, _ = gen_full.shape

        # ── CoTracker on both generations, queried at frame-0 with same points ──
        query_norm = sc["cond_tracks"][0]  # [N_total, 2] in [0,1]
        queries_px = query_norm * np.array([W, H])[None]  # pixel space
        print(f"[eval] clip {idx}: CoTracker on full", flush=True)
        tk_full_px, tk_full_vis = cotracker_query(cotracker, gen_full, queries_px, device=device.type)
        print(f"[eval] clip {idx}: CoTracker on no-track", flush=True)
        tk_no_px, tk_no_vis = cotracker_query(cotracker, gen_no, queries_px, device=device.type)

        # GT in pixel space, at the SAME resolution as the generation
        gt_full_px = sc["cond_tracks"] * np.array([W, H])[None, None, :]  # [T,N_total,2]
        active_slice = slice(0, sc["n_active"])
        bg_slice = slice(sc["n_active"], sc["n_active"] + sc["n_bg"])

        # Active EPE: CoTracker(gen) vs the GT trace we specified.
        # Filter by GT visibility (active_vis) AND CoTracker's own visibility.
        active_vis_gt = sc["active_vis"]  # [T,n_active]
        both_vis_full = (active_vis_gt > 0.5) & (tk_full_vis[:, active_slice] > 0.5)
        both_vis_no = (active_vis_gt > 0.5) & (tk_no_vis[:, active_slice] > 0.5)

        epe_active_full = epe_px(tk_full_px[:, active_slice],
                                 gt_full_px[:, active_slice],
                                 both_vis_full.astype(np.float32))
        epe_active_no = epe_px(tk_no_px[:, active_slice],
                               gt_full_px[:, active_slice],
                               both_vis_no.astype(np.float32))

        # BG EPE (only for full): CoTracker(gen_full) vs static anchor position
        bg_static_px = sc["bg_static_pos"] * np.array([W, H])[None, None, :]
        bg_vis_mask = (tk_full_vis[:, bg_slice] > 0.5).astype(np.float32)
        epe_bg = epe_px(tk_full_px[:, bg_slice], bg_static_px, bg_vis_mask) if sc["n_bg"] > 0 else float("nan")

        delta = epe_active_no - epe_active_full  # positive => tracks helped

        # ── Save + log videos ──
        overlay_full = overlay_tracks(gen_full, sc["cond_tracks"], sc["cond_vis"], palette)
        overlay_no = overlay_tracks(gen_no, sc["cond_tracks"], sc["cond_vis"], palette)

        # Also show the CoTracker-extracted tracks over the full generation — this is
        # what the model *actually* produced motion-wise.
        cotr_tracks_norm = tk_full_px / np.array([W, H])[None, None, :]
        overlay_cotr = overlay_tracks(gen_full, cotr_tracks_norm, tk_full_vis.astype(np.float32), palette)

        clip_dir = Path(args.out_dir) / f"clip_{idx:03d}"
        clip_dir.mkdir(parents=True, exist_ok=True)
        p_full = clip_dir / "gen_full_overlay.mp4"
        p_no = clip_dir / "gen_notrack_overlay.mp4"
        p_cotr = clip_dir / "gen_full_cotracker.mp4"
        imageio.mimsave(p_full, overlay_full, fps=args.fps, macro_block_size=1)
        imageio.mimsave(p_no, overlay_no, fps=args.fps, macro_block_size=1)
        imageio.mimsave(p_cotr, overlay_cotr, fps=args.fps, macro_block_size=1)

        print(f"[eval] clip {idx}: EPE_active_full={epe_active_full:.2f}  "
              f"EPE_active_notrack={epe_active_no:.2f}  DELTA={delta:.2f}  "
              f"EPE_bg={epe_bg:.2f}", flush=True)

        # per-clip logs
        entry = {
            "clip": idx,
            "epe_active_full_px": epe_active_full,
            "epe_active_notrack_px": epe_active_no,
            "delta_px": delta,
            "epe_bg_px": epe_bg,
            "n_active": sc["n_active"],
            "n_bg": sc["n_bg"],
            "sec_gen_full": t_full,
            "sec_gen_notrack": t_no,
            "caption": s["caption"][:120],
        }
        per_clip.append(entry)
        wandb.log({
            f"clip_{idx:03d}/gen_full": wandb.Video(str(p_full), fps=args.fps, format="mp4"),
            f"clip_{idx:03d}/gen_notrack": wandb.Video(str(p_no), fps=args.fps, format="mp4"),
            f"clip_{idx:03d}/gen_full_cotracker": wandb.Video(str(p_cotr), fps=args.fps, format="mp4"),
            "clip": idx,
            "epe_active_full_px": epe_active_full,
            "epe_active_notrack_px": epe_active_no,
            "delta_px": delta,
            "epe_bg_px": epe_bg,
        })

    # ── Aggregate + summary ──
    if per_clip:
        arr = lambda k: np.array([r[k] for r in per_clip if not np.isnan(r[k])])
        summary = {
            "n_clips": len(per_clip),
            "mean_epe_active_full_px": float(np.mean(arr("epe_active_full_px"))),
            "median_epe_active_full_px": float(np.median(arr("epe_active_full_px"))),
            "mean_epe_active_notrack_px": float(np.mean(arr("epe_active_notrack_px"))),
            "median_epe_active_notrack_px": float(np.median(arr("epe_active_notrack_px"))),
            "mean_delta_px": float(np.mean(arr("delta_px"))),
            "median_delta_px": float(np.median(arr("delta_px"))),
            "mean_epe_bg_px": float(np.mean(arr("epe_bg_px"))),
        }
        (Path(args.out_dir) / f"{args.wandb_run_name}_summary.json").write_text(json.dumps({
            "summary": summary, "per_clip": per_clip}, indent=2))
        wandb.log({f"summary/{k}": v for k, v in summary.items()})
        wandb.summary.update(summary)
        print("=" * 70)
        print(f"[eval] summary ({len(per_clip)} clips):")
        for k, v in summary.items():
            print(f"  {k}: {v:.3f}" if isinstance(v, float) else f"  {k}: {v}")
        print("=" * 70)

    wandb.finish()


if __name__ == "__main__":
    main()
