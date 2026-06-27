# SPDX-License-Identifier: Apache-2.0
"""Track-conditioned validation callback for WanTrack overfit/finetune runs.

Unlike the generic ``ValidationCallback`` (which drives a stock text/I2V
pipeline that knows nothing about point tracks), this callback runs a
self-contained flow-matching denoising loop on the *training* model in-process
and feeds it the exact MotionStream conditioning: text embedding, first-frame
latent (I2V), and point tracks. It then decodes the result and overlays the
ground-truth tracks so you can eyeball whether the generation follows them.

It reuses a few fixed samples pulled deterministically from the preprocessed
parquet (so during an overfit run you are generating the very clips you trained
on). Both the generation *and* a one-time VAE-decoded ground-truth reference are
logged to wandb with the tracks drawn on top.

Everything is read from the YAML ``callbacks.track_validation`` section and the
model is accessed via ``method.student`` -- no checkpoint loading, no pipeline.
"""
from __future__ import annotations

import colorsys
import glob
import os
from typing import Any, TYPE_CHECKING

import imageio.v2 as imageio
import numpy as np
import torch
from PIL import Image, ImageDraw

from fastvideo.forward_context import set_forward_context
from fastvideo.logger import init_logger
from fastvideo.models.schedulers.scheduling_flow_match_euler_discrete import (
    FlowMatchEulerDiscreteScheduler, )
from fastvideo.train.callbacks.callback import Callback
from fastvideo.training.trackers import DummyTracker

if TYPE_CHECKING:
    from fastvideo.train.methods.base import TrainingMethod

logger = init_logger(__name__)


# ----------------------------------------------------------------------
# Track overlay helpers (mirrors data_pipeline/visualize_tracks.py).
# ----------------------------------------------------------------------
def _grid_colors(grid_size: int, stride: int) -> np.ndarray:
    idx = np.arange(0, grid_size, stride)
    gy, gx = np.meshgrid(idx, idx, indexing="ij")
    nx = gx.reshape(-1) / max(grid_size - 1, 1)
    ny = gy.reshape(-1) / max(grid_size - 1, 1)
    cols = np.empty((nx.shape[0], 3), dtype=np.uint8)
    for i, (x, y) in enumerate(zip(nx, ny, strict=True)):
        r, g, b = colorsys.hsv_to_rgb(float(x), 1.0, 0.5 + 0.5 * float(y))
        cols[i] = (int(r * 255), int(g * 255), int(b * 255))
    return cols


def _subsample(tracks: np.ndarray, vis: np.ndarray, grid_size: int, stride: int):
    t = tracks.shape[0]
    if tracks.shape[1] != grid_size * grid_size:
        return tracks, vis
    tr = tracks.reshape(t, grid_size, grid_size, 2)[:, ::stride, ::stride, :].reshape(t, -1, 2)
    vs = vis.reshape(t, grid_size, grid_size)[:, ::stride, ::stride].reshape(t, -1)
    return tr, vs


def _draw_overlay(frames, tracks, vis, colors, tail, radius, vis_thresh) -> list[np.ndarray]:
    t, h, w, _ = frames.shape
    n = tracks.shape[1]
    out = []
    for fi in range(t):
        img = Image.fromarray(frames[fi]).convert("RGB")
        draw = ImageDraw.Draw(img)
        t0 = max(0, fi - tail)
        for pi in range(n):
            col = tuple(int(c) for c in colors[pi])
            pts = []
            for tj in range(t0, fi + 1):
                if vis[tj, pi] >= vis_thresh:
                    x, y = float(tracks[tj, pi, 0]), float(tracks[tj, pi, 1])
                    if 0 <= x < w and 0 <= y < h:
                        pts.append((x, y))
                else:
                    pts = []
            if len(pts) >= 2:
                draw.line(pts, fill=col, width=1)
            if vis[fi, pi] >= vis_thresh:
                x, y = float(tracks[fi, pi, 0]), float(tracks[fi, pi, 1])
                if 0 <= x < w and 0 <= y < h:
                    draw.ellipse([x - radius, y - radius, x + radius, y + radius], fill=col)
        out.append(np.asarray(img))
    return out


class TrackValidationCallback(Callback):
    """Generate + log track-overlaid videos from the training model."""

    def __init__(
        self,
        *,
        every_steps: int = 100,
        num_val_samples: int = 2,
        num_inference_steps: int = 30,
        guidance_scale: float = 1.0,
        output_dir: str | None = None,
        fps: int = 24,
        grid_stride: int = 3,
        tail: int = 12,
        radius: int = 2,
        vis_thresh: float = 0.5,
        validate_at_start: bool = True,
        seed: int = 0,
    ) -> None:
        self.every_steps = int(every_steps)
        self.num_val_samples = int(num_val_samples)
        self.num_inference_steps = int(num_inference_steps)
        self.guidance_scale = float(guidance_scale)
        self.output_dir = str(output_dir) if output_dir is not None else None
        self.fps = int(fps)
        self.grid_stride = int(grid_stride)
        self.tail = int(tail)
        self.radius = int(radius)
        self.vis_thresh = float(vis_thresh)
        self.validate_at_start = bool(validate_at_start)
        self.seed = int(seed)

        self.tracker: Any = DummyTracker()
        self._samples: list[dict[str, Any]] = []
        self._ref_logged = False
        self._did_start_val = False

    # ------------------------------------------------------------------
    # Hooks
    # ------------------------------------------------------------------
    def on_train_start(self, method: TrainingMethod, iteration: int = 0) -> None:
        tracker = getattr(method, "tracker", None)
        if tracker is not None:
            self.tracker = tracker
        from fastvideo.distributed import get_world_group
        self._is_main = get_world_group().rank == 0
        try:
            self._load_validation_samples(method)
            logger.info("TrackValidation: loaded %d validation samples", len(self._samples))
        except Exception as exc:  # noqa: BLE001 - never break training on val setup
            logger.warning("TrackValidation: failed to load samples (%s); disabling", exc)
            self._samples = []

    def on_validation_begin(self, method: TrainingMethod, iteration: int = 0) -> None:
        if not self._samples:
            return
        run = False
        if self.validate_at_start and not self._did_start_val:
            run = True
        if self.every_steps > 0 and iteration % self.every_steps == 0:
            run = True
        if not run:
            return
        self._did_start_val = True
        try:
            self._run(method, iteration)
        except Exception as exc:  # noqa: BLE001 - validation must not kill training
            logger.warning("TrackValidation: generation failed at step %d: %s", iteration, exc)

    # ------------------------------------------------------------------
    # Sample loading (deterministic, straight from parquet)
    # ------------------------------------------------------------------
    def _load_validation_samples(self, method: TrainingMethod) -> None:
        import pyarrow.parquet as pq

        from fastvideo.dataset.dataloader.schema import pyarrow_schema_i2v_track
        from fastvideo.dataset.utils import collate_rows_from_parquet_schema

        tc = self.training_config
        data_path = str(tc.data.data_path)
        files = sorted(glob.glob(os.path.join(data_path, "**", "*.parquet"), recursive=True))
        if not files:
            raise FileNotFoundError(f"no parquet under {data_path}")

        rows: list[dict[str, Any]] = []
        for f in files:
            tbl = pq.read_table(f)
            rows.extend(tbl.to_pylist())
            if len(rows) >= self.num_val_samples:
                break
        rows = rows[:self.num_val_samples]

        text_len = int(tc.pipeline_config.text_encoder_configs[0].arch_config.text_len)
        batch = collate_rows_from_parquet_schema(
            rows, pyarrow_schema_i2v_track, text_padding_length=text_len, cfg_rate=0.0, seed=self.seed)

        n = batch["text_embedding"].shape[0]
        infos = batch.get("info_list") or [{} for _ in range(n)]
        for i in range(n):
            self._samples.append({
                "text_embedding": batch["text_embedding"][i:i + 1].clone(),
                "text_attention_mask": batch["text_attention_mask"][i:i + 1].clone(),
                "vae_latent": batch["vae_latent"][i:i + 1].clone(),
                "first_frame_latent": batch["first_frame_latent"][i:i + 1].clone(),
                "track_points": batch["track_points"][i:i + 1].clone(),
                "track_visibility": batch["track_visibility"][i:i + 1].clone(),
                "caption": str(infos[i].get("caption", "") if i < len(infos) else ""),
            })

    # ------------------------------------------------------------------
    # Core
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _run(self, method: TrainingMethod, step: int) -> None:
        student = method.student
        transformer = student.transformer
        was_training = bool(getattr(transformer, "training", False))
        transformer.eval()
        out_dir = self.output_dir or os.path.join(self.training_config.checkpoint.output_dir, "track_validation")
        if self._is_main:
            os.makedirs(out_dir, exist_ok=True)
        try:
            gen_logs: list[Any] = []
            ref_logs: list[Any] = []
            for i, s in enumerate(self._samples):
                gen_latents = self._sample(student, transformer, s)          # [1,16,T,H,W] normalized
                gen_px = student.decode_latents(gen_latents.permute(0, 2, 1, 3, 4))[0]  # [3,T,H,W] in [0,1]
                gen_frames = self._overlay_tracks(gen_px, s)
                if self._is_main:
                    fn = os.path.join(out_dir, f"step{step:06d}_sample{i}_gen.mp4")
                    imageio.mimsave(fn, gen_frames, fps=self.fps, macro_block_size=1)
                    art = self.tracker.video(fn, caption=f"[{step}] {s['caption'][:120]}")
                    if art is not None:
                        gen_logs.append(art)

                # One-time ground-truth reference (VAE round-trip + same tracks).
                if not self._ref_logged and self._is_main:
                    ref_px = self._decode_reference(student, s)
                    ref_frames = self._overlay_tracks(ref_px, s)
                    fn_ref = os.path.join(out_dir, f"reference_sample{i}_gt.mp4")
                    imageio.mimsave(fn_ref, ref_frames, fps=self.fps, macro_block_size=1)
                    art = self.tracker.video(fn_ref, caption=f"GT {s['caption'][:120]}")
                    if art is not None:
                        ref_logs.append(art)

            if self._is_main and gen_logs:
                logs: dict[str, Any] = {"track_val/generated": gen_logs}
                if ref_logs and not self._ref_logged:
                    logs["track_val/reference_gt"] = ref_logs
                self.tracker.log_artifacts(logs, step)
                self._ref_logged = True
                logger.info("TrackValidation: logged %d generated videos at step %d", len(gen_logs), step)
        finally:
            if was_training:
                transformer.train()

    def _sample(self, student: Any, transformer: torch.nn.Module, s: dict[str, Any]) -> torch.Tensor:
        device = student.device
        dtype = torch.bfloat16
        flow_shift = float(student.timestep_shift)

        ff = s["first_frame_latent"].to(device, dtype)                 # [1,16,T,H,W] (raw, matches training)
        cond20 = student._build_i2v_cond_concat(ff)                    # [1,20,T,H,W]
        txt = s["text_embedding"].to(device, dtype)
        mask = s["text_attention_mask"].to(device, dtype)
        tp = s["track_points"].to(device, dtype)                       # [1,T,N,2] normalized
        tv = s["track_visibility"].to(device, dtype)                   # [1,T,N]

        _, _, T, H, W = ff.shape
        gen = torch.Generator(device="cpu").manual_seed(self.seed)
        latents = torch.randn((1, 16, T, H, W), generator=gen, dtype=torch.float32).to(device)

        sched = FlowMatchEulerDiscreteScheduler(shift=flow_shift)
        sched.set_timesteps(self.num_inference_steps, device=device)
        for t in sched.timesteps:
            model_in = torch.cat([latents.to(dtype), cond20], dim=1)   # [1,36,T,H,W]
            ts = t.reshape(1).to(device, dtype)
            with torch.autocast(device.type, dtype=dtype), set_forward_context(
                    current_timestep=ts, attn_metadata=None):
                # WanTransformer3DModel.forward returns a bare [B,C,T,H,W] tensor.
                v = transformer(
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

    def _decode_reference(self, student: Any, s: dict[str, Any]) -> torch.Tensor:
        from fastvideo.training.training_utils import normalize_dit_input
        device = student.device
        dtype = torch.bfloat16
        raw = s["vae_latent"].to(device, dtype)                        # [1,16,T,H,W] raw VAE latent
        norm = normalize_dit_input("wan", raw, student.vae)
        return student.decode_latents(norm.permute(0, 2, 1, 3, 4))[0]  # [3,T,H,W]

    def _overlay_tracks(self, px: torch.Tensor, s: dict[str, Any]) -> list[np.ndarray]:
        # px: [3,T,H,W] in [0,1]
        video = (px.clamp(0, 1).float().cpu().numpy() * 255.0).astype(np.uint8)
        frames = np.transpose(video, (1, 2, 3, 0))                    # [T,H,W,3]
        T, H, W, _ = frames.shape

        tp = s["track_points"][0].float().cpu().numpy()               # [Tt,N,2] normalized
        tv = s["track_visibility"][0].float().cpu().numpy()           # [Tt,N]
        tt = min(T, tp.shape[0])
        frames, tp, tv = frames[:tt], tp[:tt], tv[:tt]

        grid = int(round(tp.shape[1] ** 0.5))
        tp, tv = _subsample(tp, tv, grid, self.grid_stride)
        colors = _grid_colors(grid, self.grid_stride)
        if colors.shape[0] != tp.shape[1]:
            colors = _grid_colors(int(round(tp.shape[1] ** 0.5)) or 1, 1)[:tp.shape[1]]

        # Normalized [0,1] -> pixel coords of the generated frame.
        tp_px = tp.copy()
        tp_px[..., 0] = tp[..., 0] * W
        tp_px[..., 1] = tp[..., 1] * H
        return _draw_overlay(frames, tp_px, tv, colors, self.tail, self.radius, self.vis_thresh)
