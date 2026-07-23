#!/usr/bin/env python3
"""2-track validation ablation.

For each val sample, keep exactly 2 tracks (one per top-motion object) and zero
the rest. This is the extreme sparse-inference case: what the user actually gets
when they draw 2 traces at inference. Compares checkpoints side-by-side.

Uses paper CFG (wt=3.0, wm=1.5) unless overridden. Runs `_sample`-style denoise
matching track_validation.py exactly.

Usage:
  python data_pipeline/two_track_val.py \
    --model-dir /mnt/lustre/vlm-s4duan/exports/merged_bias_ckpt4800 \
    --yaml examples/train/scenario/worldmodel/finetune_wantrack_synth_stage2_paperLR.yaml \
    --val-parquet /mnt/lustre/vlm-s4duan/val_examples_mixed/combined_parquet_dataset \
    --wandb-run-name twotrack_mb4800
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
import time
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import pyarrow.parquet as pq
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
import trackwan_infer as twi  # noqa: E402
from fastvideo.forward_context import set_forward_context  # noqa: E402
from fastvideo.models.schedulers.scheduling_flow_match_euler_discrete import (  # noqa: E402
    FlowMatchEulerDiscreteScheduler,
)


def load_val_samples(parquet_dir: str, text_len: int) -> list:
    from fastvideo.dataset.dataloader.schema import pyarrow_schema_i2v_track
    from fastvideo.dataset.utils import collate_rows_from_parquet_schema
    fs = sorted(glob.glob(os.path.join(parquet_dir, "**", "*.parquet"), recursive=True))
    if not fs:
        raise FileNotFoundError(parquet_dir)
    rows = []
    for f in fs:
        rows.extend(pq.read_table(f).to_pylist())
    batch = collate_rows_from_parquet_schema(rows, pyarrow_schema_i2v_track,
                                             text_padding_length=int(text_len), cfg_rate=0.0)
    infos = batch.get("info_list") or [{} for _ in rows]
    return [{
        "id": str(infos[i].get("id", f"clip{i:03d}")) if i < len(infos) else f"clip{i:03d}",
        "file_name": rows[i].get("file_name", ""),
        "text_embedding": batch["text_embedding"][i:i + 1].clone(),
        "text_attention_mask": batch["text_attention_mask"][i:i + 1].clone(),
        "first_frame_latent": batch["first_frame_latent"][i:i + 1].clone(),
        "clip_feature": batch["clip_feature"][i:i + 1].clone(),
        "vae_latent": batch["vae_latent"][i:i + 1].clone(),
        "track_points": batch["track_points"][i:i + 1].clone(),  # [1,T,N,2] normalised
        "track_visibility": batch["track_visibility"][i:i + 1].clone(),
        "object_ids": batch["object_ids"][i:i + 1].clone() if "object_ids" in batch else None,
        "caption": str(infos[i].get("caption", "") if i < len(infos) else ""),
    } for i in range(len(rows))]


def pick_two_tracks(sample: dict, seed: int, num_bg: int = 0,
                    num_fg: int = 2) -> tuple[torch.Tensor, torch.Tensor, list]:
    """Return (tp_2, tv_2, info) — ``num_fg`` foreground tracks + ``num_bg`` bg anchor tracks.

    Foreground picks require the track to STAY IN FRAME [0,1] for all 121 frames — no
    point in showing traces that fly off the canvas.
    """
    tp = sample["track_points"].float()  # [1,T,N,2]
    tv = sample["track_visibility"].float()  # [1,T,N]
    oid = sample["object_ids"]  # [1,N]
    if oid is None:
        raise ValueError("no object_ids in val sample")
    oid = oid[0].to(torch.int64)  # [N]

    tp0 = tp[0]  # [T, N, 2]
    N = tp0.shape[1]
    motion_per_track = torch.linalg.norm(tp0[-1] - tp0[0], dim=-1)  # [N]

    # Which tracks stay fully within [0,1] on both x and y across ALL frames?
    in_frame = ((tp0 >= 0.0) & (tp0 <= 1.0)).all(dim=-1).all(dim=0)  # [N]

    # Rank object_ids by max in-frame within-object track motion.
    fg_oids = torch.unique(oid[oid >= 0])
    obj_best = []  # (oid, best_track_idx, best_motion)
    for o in fg_oids.tolist():
        idxs = ((oid == o) & in_frame & (tv[0, 0] > 0.5)).nonzero(as_tuple=True)[0]
        if idxs.numel() == 0:
            continue
        m = motion_per_track[idxs]
        best = idxs[int(torch.argmax(m).item())]
        obj_best.append((o, int(best.item()), float(m.max().item())))
    obj_best.sort(key=lambda x: -x[2])
    picked_tracks = [idx for _, idx, _ in obj_best[:num_fg]]
    info = [{"oid": int(o), "motion_max": mm, "track_idx": pk}
            for (o, pk, mm) in obj_best[:num_fg]]
    rng = np.random.RandomState(seed)
    vis0 = tv[0, 0]  # [N] visibility at frame 0

    # Add num_bg background anchor tracks — near-static (bottom of motion), visible at frame 0.
    bg_picked = []
    if num_bg > 0:
        bg_pool = ((oid == -1) & (vis0 > 0.5)).nonzero(as_tuple=True)[0]
        if bg_pool.numel() > 0:
            # Sort bg tracks by motion (ascending), take lowest-motion ones and pick uniformly
            bg_motion = motion_per_track[bg_pool]
            n_pool = int(min(bg_pool.numel(), max(num_bg * 20, num_bg)))
            top_idxs = torch.argsort(bg_motion)[:n_pool]
            candidate = bg_pool[top_idxs]
            if candidate.numel() >= num_bg:
                sel = rng.choice(candidate.numel(), num_bg, replace=False)
                bg_picked = [int(candidate[k].item()) for k in sel]
            else:
                bg_picked = [int(k.item()) for k in candidate]
            for k in bg_picked:
                info.append({"oid": -1, "motion_max": float(motion_per_track[k].item()), "track_idx": k})

    # Build zeroed visibility: only picked tracks visible.
    tv_new = torch.zeros_like(tv)
    for k in picked_tracks + bg_picked:
        tv_new[0, :, k] = tv[0, :, k]

    return tp, tv_new, info


@torch.no_grad()
def sample_denoise(model, sample: dict, tp: torch.Tensor, tv: torch.Tensor,
                   *, w_text: float, w_motion: float, num_steps: int, seed: int) -> np.ndarray:
    """Reproduce track_validation._sample denoise EXACTLY."""
    device = model.device
    dtype = torch.bfloat16

    ff = sample["first_frame_latent"].to(device, dtype)
    txt = sample["text_embedding"].to(device, dtype)
    mask = sample["text_attention_mask"].to(device, dtype)
    clip = sample["clip_feature"].to(device, dtype)
    tp_d = tp.to(device, dtype)
    tv_d = tv.to(device, dtype)
    cond20 = model._build_i2v_cond_concat(ff)

    _, _, T, H, W = ff.shape
    gen = torch.Generator(device="cpu").manual_seed(int(seed))
    latents = torch.randn((1, 16, T, H, W), generator=gen, dtype=torch.float32).to(device)
    sched = FlowMatchEulerDiscreteScheduler(shift=float(model.timestep_shift))
    sched.set_timesteps(int(num_steps), device=device)

    cfg_on = (w_text != 1.0) or (w_motion != 1.0)
    txt_null = torch.zeros_like(txt) if cfg_on else None

    def _fwd(text_e, tp_e, tv_e, mi, ts_):
        with torch.autocast(device.type, dtype=dtype), \
             set_forward_context(current_timestep=ts_, attn_metadata=None):
            return model.transformer(hidden_states=mi, encoder_hidden_states=text_e,
                                     encoder_attention_mask=mask, timestep=ts_,
                                     encoder_hidden_states_image=clip,
                                     track_points=tp_e, track_visibility=tv_e,
                                     return_dict=False)

    for tt in sched.timesteps:
        mi = torch.cat([latents.to(dtype), cond20], dim=1)
        ts_ = tt.reshape(1).to(device, dtype)
        v_full = _fwd(txt, tp_d, tv_d, mi, ts_)
        if not cfg_on:
            v = v_full
        elif w_text != 1.0 and w_motion == 1.0:
            v_no_text = _fwd(txt_null, tp_d, tv_d, mi, ts_)
            v = v_no_text + w_text * (v_full - v_no_text)
        elif w_text == 1.0 and w_motion != 1.0:
            v_no_motion = _fwd(txt, None, None, mi, ts_)
            v = v_no_motion + w_motion * (v_full - v_no_motion)
        else:
            v_no_text = _fwd(txt_null, tp_d, tv_d, mi, ts_)
            v_no_motion = _fwd(txt, None, None, mi, ts_)
            alpha = w_text / max(w_text + w_motion, 1e-6)
            v_base = alpha * v_no_text + (1.0 - alpha) * v_no_motion
            v = v_base + w_text * (v_full - v_no_text) + w_motion * (v_full - v_no_motion)
        latents = sched.step(v.float(), tt, latents.float(), return_dict=False)[0]

    return twi.decode_to_pixels(model, latents)


def draw_track_overlay(video: np.ndarray, tp: torch.Tensor, tv: torch.Tensor,
                       info: list | None = None) -> np.ndarray:
    """Draw kept tracks on the video. Foreground = red/blue, background anchors = gray."""
    from PIL import Image, ImageDraw
    T, H, W, _ = video.shape
    tp_np = tp[0].cpu().numpy()  # [T, N, 2]
    tv_np = tv[0].cpu().numpy()  # [T, N]
    N = tp_np.shape[1]
    kept = [k for k in range(N) if tv_np[:, k].max() > 0.5]
    fg_colors = [(255, 60, 60), (60, 200, 255)]
    bg_color = (180, 180, 180)
    # figure out which are fg vs bg by looking at info if given
    is_bg_by_idx = {}
    if info is not None:
        for entry in info:
            is_bg_by_idx[entry["track_idx"]] = (entry["oid"] == -1)
    out = []
    fg_seen = 0
    for t in range(T):
        img = Image.fromarray(np.ascontiguousarray(video[t]))
        dr = ImageDraw.Draw(img)
        fg_seen = 0
        for k in kept:
            if tv_np[t, k] <= 0.5:
                continue
            x = float(tp_np[t, k, 0] * W)
            y = float(tp_np[t, k, 1] * H)
            if is_bg_by_idx.get(k, False):
                col = bg_color
                r = 3
            else:
                col = fg_colors[fg_seen % len(fg_colors)]
                fg_seen += 1
                r = 5
            dr.ellipse([x - r, y - r, x + r, y + r], fill=col, outline=(255, 255, 255))
        out.append(np.asarray(img))
    return np.stack(out)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model-dir", required=True)
    p.add_argument("--yaml", required=True)
    p.add_argument("--val-parquet", required=True)
    p.add_argument("--wandb-project", default="wantrack-bidir")
    p.add_argument("--wandb-run-name", required=True)
    p.add_argument("--out-dir", default="/mnt/lustre/vlm-s4duan/two_track_val")
    p.add_argument("--w-text", type=float, default=3.0)
    p.add_argument("--w-motion", type=float, default=1.5)
    p.add_argument("--steps", type=int, default=30)
    p.add_argument("--seed", type=int, default=1000)
    p.add_argument("--fps", type=int, default=24)
    p.add_argument("--num-fg-tracks", type=int, default=2,
                   help="Number of foreground (moving object) tracks to keep.")
    p.add_argument("--num-bg-tracks", type=int, default=0,
                   help="Extra static bg anchor tracks in addition to the fg ones.")
    p.add_argument("--swap-text-from", type=int, default=None,
                   help="For every sample, replace text_embedding+mask with THIS sample idx's text.")
    p.add_argument("--swap-clip-from", type=int, default=None,
                   help="For every sample, replace clip_feature with THIS sample idx's clip.")
    p.add_argument("--swap-ff-from", type=int, default=None,
                   help="For every sample, replace first_frame_latent with THIS sample idx's ff.")
    args = p.parse_args()

    out_dir = Path(args.out_dir) / args.wandb_run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    import wandb

    print(f"[2trk] loading model {args.model_dir}", flush=True)
    model, tc = twi.load_trackwan(args.model_dir, args.yaml)
    text_len = int(getattr(tc.data, "text_padding_length", 256))
    print(f"[2trk] text_len={text_len}", flush=True)

    print(f"[2trk] loading val samples", flush=True)
    samples = load_val_samples(args.val_parquet, text_len)
    print(f"[2trk] {len(samples)} val samples", flush=True)

    run = wandb.init(project=args.wandb_project, name=args.wandb_run_name,
                     config={"model_dir": args.model_dir, "w_text": args.w_text,
                             "w_motion": args.w_motion, "steps": args.steps})

    for i, s in enumerate(samples):
        sid = s["id"] or f"clip{i:03d}"
        print(f"\n[2trk] === sample {i}: {sid} ({s['file_name'][:40]}) ===", flush=True)
        # Optional cross-sample field swaps (isolate which condition drives quality)
        if args.swap_text_from is not None:
            src = samples[args.swap_text_from]
            s = {**s, "text_embedding": src["text_embedding"].clone(),
                       "text_attention_mask": src["text_attention_mask"].clone()}
            print(f"[2trk]   SWAPPED text from sample {args.swap_text_from}", flush=True)
        if args.swap_clip_from is not None:
            src = samples[args.swap_clip_from]
            s = {**s, "clip_feature": src["clip_feature"].clone()}
            print(f"[2trk]   SWAPPED clip from sample {args.swap_clip_from}", flush=True)
        if args.swap_ff_from is not None:
            src = samples[args.swap_ff_from]
            s = {**s, "first_frame_latent": src["first_frame_latent"].clone()}
            print(f"[2trk]   SWAPPED first_frame_latent from sample {args.swap_ff_from}", flush=True)
        tp2, tv2, info = pick_two_tracks(s, seed=args.seed + i,
                                         num_fg=args.num_fg_tracks, num_bg=args.num_bg_tracks)
        print(f"[2trk]   picked: {info}", flush=True)

        t0 = time.time()
        frames = sample_denoise(model, s, tp2, tv2, w_text=args.w_text, w_motion=args.w_motion,
                                num_steps=args.steps, seed=args.seed + i)
        dt = time.time() - t0
        print(f"[2trk]   denoise {dt:.1f}s -> frames {frames.shape}", flush=True)

        overlay = draw_track_overlay(frames, tp2, tv2, info)
        fn_raw = out_dir / f"{sid}_gen.mp4"
        fn_ov = out_dir / f"{sid}_gen_overlay.mp4"
        imageio.mimsave(str(fn_raw), frames, fps=args.fps, macro_block_size=1)
        imageio.mimsave(str(fn_ov), overlay, fps=args.fps, macro_block_size=1)

        # GT for reference
        try:
            ref = twi.decode_reference(model, s["vae_latent"].to(model.device, torch.bfloat16))
            ref_ov = draw_track_overlay(ref, tp2, tv2, info)
            fn_ref = out_dir / f"{sid}_gt_overlay.mp4"
            imageio.mimsave(str(fn_ref), ref_ov, fps=args.fps, macro_block_size=1)
            wandb.log({
                f"sample_{i:02d}_gen": wandb.Video(str(fn_ov), fps=args.fps, format="mp4"),
                f"sample_{i:02d}_gt": wandb.Video(str(fn_ref), fps=args.fps, format="mp4"),
                "sample": i, "caption": s["caption"][:200], "picked": str(info),
            })
        except Exception as e:
            print(f"[2trk]   gt fail: {e}", flush=True)
            wandb.log({
                f"sample_{i:02d}_gen": wandb.Video(str(fn_ov), fps=args.fps, format="mp4"),
                "sample": i, "caption": s["caption"][:200], "picked": str(info),
            })

    wandb.finish()
    print("[2trk] DONE", flush=True)


if __name__ == "__main__":
    main()
