#!/usr/bin/env python3
"""CFG-ablation eval on a trackwan checkpoint.

Reproduces ``fastvideo/train/callbacks/track_validation.py::_sample`` denoise
formula EXACTLY, but with configurable ``w_text`` and ``w_motion`` so we can
isolate which CFG branch is causing artifacts.

Runs each val sample under N configs; uploads all videos to one wandb run for
side-by-side visual comparison.

Usage (single-GPU, on any held alloc):
  python data_pipeline/cfg_ablation.py \
    --model-dir /mnt/lustre/vlm-s4duan/exports/synth_stage2_paperLR_ckpt400 \
    --yaml examples/train/scenario/worldmodel/finetune_wantrack_synth_stage2_paperLR.yaml \
    --val-parquet /mnt/lustre/vlm-s4duan/val_examples_mixed/combined_parquet_dataset \
    --wandb-run-name cfg_abl_paperLR_ckpt400
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
        "text_embedding": batch["text_embedding"][i:i + 1].clone(),
        "text_attention_mask": batch["text_attention_mask"][i:i + 1].clone(),
        "first_frame_latent": batch["first_frame_latent"][i:i + 1].clone(),
        "clip_feature": batch["clip_feature"][i:i + 1].clone(),
        "vae_latent": batch["vae_latent"][i:i + 1].clone(),
        "track_points": batch["track_points"][i:i + 1].clone(),
        "track_visibility": batch["track_visibility"][i:i + 1].clone(),
        "object_ids": batch["object_ids"][i:i + 1].clone() if "object_ids" in batch else None,
        "track_weights": batch["track_weights"][i:i + 1].clone() if "track_weights" in batch else None,
        "caption": str(infos[i].get("caption", "") if i < len(infos) else ""),
    } for i in range(len(rows))]


@torch.no_grad()
def generate_with_cfg(model, sample: dict, *, w_text: float, w_motion: float,
                       num_steps: int, seed: int,
                       null_text: torch.Tensor | None = None,
                       null_text_mask: torch.Tensor | None = None,
                       no_track: bool = False) -> np.ndarray:
    """Reproduce track_validation._sample denoise EXACTLY, w/ configurable CFG.

    If ``null_text`` is provided, use it as the ∅ text (canonical Wan/T5 convention
    = UMT5-encoded empty string). Otherwise fall back to zeros_like(txt).
    If ``no_track`` is True, pass tp=None/tv=None to every forward (matches the
    val callback's `track_val/no_track` panel).
    """
    device = model.device
    dtype = torch.bfloat16

    ff = sample["first_frame_latent"].to(device, dtype)
    txt = sample["text_embedding"].to(device, dtype)
    mask = sample["text_attention_mask"].to(device, dtype)
    clip = sample["clip_feature"].to(device, dtype)
    if no_track:
        tp = None
        tv = None
    else:
        # CRITICAL: sparse-sample tracks the same way training + track_validation do.
        # Feeding the raw 2500 tracks is out-of-distribution and produces melted output.
        tp_raw = sample["track_points"].float()
        tv_raw = sample["track_visibility"].float()
        oid = sample.get("object_ids")
        tw = sample.get("track_weights")
        aug_gen = torch.Generator(device=device).manual_seed(int(seed))
        tp_s, tv_s = model._augment_tracks(tp_raw, tv_raw, aug_gen, object_ids=oid, track_weights=tw)
        if tp_s is None or tv_s is None:  # motion_drop fired -> fall back to raw
            tp_s, tv_s = tp_raw, tv_raw
        tp = tp_s.to(device, dtype)
        tv = tv_s.to(device, dtype)
    cond20 = model._build_i2v_cond_concat(ff)

    _, _, T, H, W = ff.shape
    gen = torch.Generator(device="cpu").manual_seed(int(seed))
    latents = torch.randn((1, 16, T, H, W), generator=gen, dtype=torch.float32).to(device)
    sched = FlowMatchEulerDiscreteScheduler(shift=float(model.timestep_shift))
    sched.set_timesteps(int(num_steps), device=device)

    do_text_cfg = (w_text != 1.0)
    do_motion_cfg = (w_motion != 1.0) and not no_track  # no tracks -> motion CFG is a no-op

    def _fwd(text_e, tp_e, tv_e, mi, ts_):
        with torch.autocast(device.type, dtype=dtype), \
             set_forward_context(current_timestep=ts_, attn_metadata=None):
            return model.transformer(hidden_states=mi, encoder_hidden_states=text_e,
                                     encoder_attention_mask=mask, timestep=ts_,
                                     encoder_hidden_states_image=clip,
                                     track_points=tp_e, track_visibility=tv_e,
                                     return_dict=False)

    if do_text_cfg:
        if null_text is not None:
            # Broadcast the [seq_len, dim] null embedding to match txt shape [B, seq_len, dim]
            nl = null_text.to(device, dtype)
            if nl.dim() == 2:
                nl = nl.unsqueeze(0).expand_as(txt).contiguous()
            txt_null = nl
        else:
            txt_null = torch.zeros_like(txt)
    else:
        txt_null = None

    for tt in sched.timesteps:
        mi = torch.cat([latents.to(dtype), cond20], dim=1)
        ts_ = tt.reshape(1).to(device, dtype)
        v_full = _fwd(txt, tp, tv, mi, ts_)
        if not do_text_cfg and not do_motion_cfg:
            v = v_full
        elif do_text_cfg and not do_motion_cfg:
            v_no_text = _fwd(txt_null, tp, tv, mi, ts_)
            v = v_no_text + w_text * (v_full - v_no_text)
        elif do_motion_cfg and not do_text_cfg:
            v_no_motion = _fwd(txt, None, None, mi, ts_)
            v = v_no_motion + w_motion * (v_full - v_no_motion)
        else:
            # joint text+motion CFG (matches track_validation._sample)
            v_no_text = _fwd(txt_null, tp, tv, mi, ts_)
            v_no_motion = _fwd(txt, None, None, mi, ts_)
            alpha = w_text / max(w_text + w_motion, 1e-6)
            v_base = alpha * v_no_text + (1.0 - alpha) * v_no_motion
            v = v_base + w_text * (v_full - v_no_text) + w_motion * (v_full - v_no_motion)
        latents = sched.step(v.float(), tt, latents.float(), return_dict=False)[0]

    return twi.decode_to_pixels(model, latents)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model-dir", required=True)
    p.add_argument("--yaml", required=True)
    p.add_argument("--val-parquet", required=True)
    p.add_argument("--wandb-project", default="wantrack-bidir")
    p.add_argument("--wandb-run-name", required=True)
    p.add_argument("--out-dir", default="/mnt/lustre/vlm-s4duan/cfg_ablation")
    p.add_argument("--steps", type=int, default=30)
    p.add_argument("--seed", type=int, default=1000)
    p.add_argument("--fps", type=int, default=24)
    p.add_argument("--configs", default="3.0,1.5;1.0,1.5;1.0,1.0",
                   help="';'-sep list of 'w_text,w_motion' pairs")
    p.add_argument("--null-text-pt",
                   help="Path to precomputed UMT5(``'') embedding .pt file. If given, this is "
                        "used as the ∅ text branch instead of zeros_like(txt).")
    p.add_argument("--no-track", action="store_true",
                   help="Force track_points=None throughout denoise. Reproduces the val "
                        "callback's `track_val/no_track` panel.")
    args = p.parse_args()

    null_text = None
    null_text_mask = None
    if args.null_text_pt:
        blob = torch.load(args.null_text_pt, map_location="cpu")
        null_text = blob["embedding"]
        null_text_mask = blob.get("attention_mask")
        print(f"[abl] loaded null-text embedding: shape={list(null_text.shape)} "
              f"norm={null_text.norm().item():.3f}", flush=True)

    import wandb
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    print(f"[abl] loading model {args.model_dir}", flush=True)
    model, tc = twi.load_trackwan(args.model_dir, args.yaml)
    text_len = int(getattr(tc.data, "text_padding_length", 256))

    print(f"[abl] loading val samples from {args.val_parquet}", flush=True)
    samples = load_val_samples(args.val_parquet, text_len)
    print(f"[abl] {len(samples)} val samples", flush=True)

    configs = [tuple(float(x) for x in pair.split(","))
               for pair in args.configs.split(";")]
    print(f"[abl] configs: {configs}", flush=True)

    run = wandb.init(project=args.wandb_project, name=args.wandb_run_name,
                     config={"model_dir": args.model_dir, "steps": args.steps,
                             "configs": [list(c) for c in configs]})

    for i, s in enumerate(samples):
        sid = s["id"] or f"clip{i:03d}"
        print(f"[abl] === sample {i}: {sid} ===", flush=True)
        wandb_log = {"sample": i, "caption": s["caption"][:120]}
        for wt, wm in configs:
            t0 = time.time()
            frames = generate_with_cfg(model, s, w_text=wt, w_motion=wm,
                                        num_steps=args.steps, seed=args.seed + i,
                                        null_text=null_text, null_text_mask=null_text_mask,
                                        no_track=args.no_track)
            dt = time.time() - t0
            tag = f"wt{wt}_wm{wm}".replace(".", "p")
            fn = Path(args.out_dir) / f"{sid}_{tag}.mp4"
            imageio.mimsave(str(fn), frames, fps=args.fps, macro_block_size=1)
            wandb_log[f"gen_{tag}"] = wandb.Video(str(fn), fps=args.fps, format="mp4")
            print(f"[abl]   {tag}: {dt:.1f}s -> {fn.name}", flush=True)
        wandb.log(wandb_log)

    # also render GT reference once
    for i, s in enumerate(samples):
        try:
            ref = twi.decode_reference(model, s["vae_latent"].to(model.device, torch.bfloat16))
            fn = Path(args.out_dir) / f"{s['id']}_gt.mp4"
            imageio.mimsave(str(fn), ref, fps=args.fps, macro_block_size=1)
            wandb.log({"sample": i, "reference_gt": wandb.Video(str(fn), fps=args.fps, format="mp4")})
        except Exception as e:
            print(f"[abl] gt render failed for sample {i}: {e}", flush=True)

    wandb.finish()
    print("[abl] DONE", flush=True)


if __name__ == "__main__":
    main()
