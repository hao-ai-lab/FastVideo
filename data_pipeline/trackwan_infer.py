# SPDX-License-Identifier: Apache-2.0
"""Standalone TrackWan inference: generate a video from (first frame + text + tracks).

Loads a checkpoint exported with ``dcp_to_diffusers`` into the *same*
``WanTrackModel`` wrapper used for training (so train/inference parity is exact),
then runs a flow-matching denoise loop feeding the point-track control. Used by
the controllability tests and the Gradio app.

Backend only -- no track authoring here (see ``synthetic_tracks.py``) and no
metrics (see ``fastvideo/eval/metrics/motion/cotracker_epe``).
"""
from __future__ import annotations

import os
from typing import Any

import numpy as np
import torch


def _ensure_dist_env() -> None:
    for k, v in [("RANK", "0"), ("LOCAL_RANK", "0"), ("WORLD_SIZE", "1"),
                 ("MASTER_ADDR", "127.0.0.1"), ("MASTER_PORT", "29600")]:
        os.environ.setdefault(k, v)


def load_trackwan(export_dir: str, yaml_path: str) -> tuple[Any, Any]:
    """Build a ``WanTrackModel`` with the exported (trained) weights on 1 GPU."""
    _ensure_dist_env()
    from fastvideo.distributed import (
        maybe_init_distributed_environment_and_model_parallel, )
    from fastvideo.train.utils.config import load_run_config
    from fastvideo.train.models.wantrack.wantrack import WanTrackModel

    maybe_init_distributed_environment_and_model_parallel(tp_size=1, sp_size=1)

    cfg = load_run_config(yaml_path)
    tc = cfg.training
    tc.distributed.tp_size = 1
    tc.distributed.sp_size = 1
    tc.distributed.num_gpus = 1
    tc.distributed.hsdp_replicate_dim = 1
    tc.distributed.hsdp_shard_dim = 1
    tc.model_path = export_dir  # vae + components loaded from the export

    model = WanTrackModel(
        init_from=export_dir,
        training_config=tc,
        trainable=False,
        flow_shift=float(tc.pipeline_config.flow_shift),
        enable_gradient_checkpointing_type=None,
    )
    model.init_preprocessors(tc)
    model.transformer.eval()
    return model, tc


def load_conditioning_from_parquet(data_path: str, indices: list[int], text_len: int
                                   ) -> list[dict[str, Any]]:
    """Pull (first_frame_latent, text, vae_latent, GT tracks, caption) for clips."""
    import glob

    import pyarrow.parquet as pq

    from fastvideo.dataset.dataloader.schema import pyarrow_schema_i2v_track
    from fastvideo.dataset.utils import collate_rows_from_parquet_schema

    files = sorted(glob.glob(os.path.join(data_path, "**", "*.parquet"), recursive=True))
    rows: list[dict[str, Any]] = []
    for f in files:
        rows.extend(pq.read_table(f).to_pylist())
    sel = [rows[i] for i in indices]
    batch = collate_rows_from_parquet_schema(
        sel, pyarrow_schema_i2v_track, text_padding_length=int(text_len), cfg_rate=0.0)
    infos = batch.get("info_list") or [{} for _ in sel]
    out = []
    for i in range(len(sel)):
        out.append({
            "text_embedding": batch["text_embedding"][i:i + 1].clone(),
            "text_attention_mask": batch["text_attention_mask"][i:i + 1].clone(),
            "vae_latent": batch["vae_latent"][i:i + 1].clone(),
            "first_frame_latent": batch["first_frame_latent"][i:i + 1].clone(),
            "track_points": batch["track_points"][i:i + 1].clone(),         # [1,T,N,2] normalized
            "track_visibility": batch["track_visibility"][i:i + 1].clone(),  # [1,T,N]
            "caption": str(infos[i].get("caption", "")),
        })
    return out


@torch.no_grad()
def generate(model: Any, *, first_frame_latent: torch.Tensor,
             text_embedding: torch.Tensor, text_attention_mask: torch.Tensor,
             track_points: torch.Tensor | None, track_visibility: torch.Tensor | None,
             num_steps: int = 30, seed: int = 0) -> torch.Tensor:
    """Denoise from noise -> normalized latents [1,16,T,H,W]. Tracks may be None."""
    from fastvideo.forward_context import set_forward_context
    from fastvideo.models.schedulers.scheduling_flow_match_euler_discrete import (
        FlowMatchEulerDiscreteScheduler, )

    device = model.device
    dtype = torch.bfloat16
    ff = first_frame_latent.to(device, dtype)                       # [1,16,T,H,W] (already normalized)
    cond20 = model._build_i2v_cond_concat(ff)                       # [1,20,T,H,W]
    txt = text_embedding.to(device, dtype)
    mask = text_attention_mask.to(device, dtype)
    tp = track_points.to(device, dtype) if track_points is not None else None
    tv = track_visibility.to(device, dtype) if track_visibility is not None else None

    _, _, T, H, W = ff.shape
    gen = torch.Generator(device="cpu").manual_seed(int(seed))
    latents = torch.randn((1, 16, T, H, W), generator=gen, dtype=torch.float32).to(device)

    sched = FlowMatchEulerDiscreteScheduler(shift=float(model.timestep_shift))
    sched.set_timesteps(int(num_steps), device=device)
    for t in sched.timesteps:
        model_in = torch.cat([latents.to(dtype), cond20], dim=1)    # [1,36,T,H,W]
        ts = t.reshape(1).to(device, dtype)
        with torch.autocast(device.type, dtype=dtype), set_forward_context(
                current_timestep=ts, attn_metadata=None):
            v = model.transformer(
                hidden_states=model_in,
                encoder_hidden_states=txt,
                encoder_attention_mask=mask,
                timestep=ts,
                track_points=tp,
                track_visibility=tv,
                return_dict=False,
            )
        latents = sched.step(v.float(), t, latents.float(), return_dict=False)[0]
    return latents


@torch.no_grad()
def decode_to_pixels(model: Any, latents: torch.Tensor) -> np.ndarray:
    """Normalized latents [1,16,T,H,W] -> uint8 frames [T,H,W,3]."""
    px = model.decode_latents(latents.permute(0, 2, 1, 3, 4))[0]    # [3,T,H,W] in [0,1]
    video = (px.clamp(0, 1).float().cpu().numpy() * 255.0).astype(np.uint8)
    return np.transpose(video, (1, 2, 3, 0))                        # [T,H,W,3]


@torch.no_grad()
def decode_reference(model: Any, vae_latent: torch.Tensor) -> np.ndarray:
    """Decode the raw GT vae_latent (needs normalize first) -> uint8 [T,H,W,3]."""
    from fastvideo.training.training_utils import normalize_dit_input
    raw = vae_latent.to(model.device, torch.bfloat16)
    norm = normalize_dit_input("wan", raw, model.vae)
    px = model.decode_latents(norm.permute(0, 2, 1, 3, 4))[0]
    video = (px.clamp(0, 1).float().cpu().numpy() * 255.0).astype(np.uint8)
    return np.transpose(video, (1, 2, 3, 0))
