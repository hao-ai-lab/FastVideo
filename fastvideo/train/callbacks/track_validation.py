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
        paired_no_track: bool = True,
        motion_guidance_scale: float = 1.0,
    ) -> None:
        self.every_steps = int(every_steps)
        self.num_val_samples = int(num_val_samples)
        self.num_inference_steps = int(num_inference_steps)
        # ``guidance_scale`` is TEXT CFG (v_uncond + s_text*(v_text - v_uncond)); at 1.0 == disabled.
        # ``motion_guidance_scale`` is MotionStream MOTION CFG on top of that (s_motion*(v_full - v_text)).
        self.guidance_scale = float(guidance_scale)
        self.motion_guidance_scale = float(motion_guidance_scale)
        self.output_dir = str(output_dir) if output_dir is not None else None
        self.fps = int(fps)
        self.grid_stride = int(grid_stride)
        self.tail = int(tail)
        self.radius = int(radius)
        self.vis_thresh = float(vis_thresh)
        self.validate_at_start = bool(validate_at_start)
        self.seed = int(seed)
        # Also generate a NO-TRACK counterfactual per sample (same image + prompt, tracks=None).
        # If it matches the with-track generation, the model is ignoring the tracks.
        self.paired_no_track = bool(paired_no_track)

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
        batch = collate_rows_from_parquet_schema(rows,
                                                 pyarrow_schema_i2v_track,
                                                 text_padding_length=text_len,
                                                 cfg_rate=0.0,
                                                 seed=self.seed)

        n = batch["text_embedding"].shape[0]
        infos = batch.get("info_list") or [{} for _ in range(n)]

        def _opt(key: str, i: int) -> Any:
            # optional per-sample tensor (object_ids / track_weights); None if absent/empty
            t = batch.get(key)
            if t is None or not torch.is_tensor(t) or t.numel() == 0 or t.shape[0] <= i:
                return None
            row = t[i:i + 1].clone()
            return row if row.numel() > 0 else None

        for i in range(n):
            self._samples.append({
                "text_embedding": batch["text_embedding"][i:i + 1].clone(),
                "text_attention_mask": batch["text_attention_mask"][i:i + 1].clone(),
                "vae_latent": batch["vae_latent"][i:i + 1].clone(),
                "first_frame_latent": batch["first_frame_latent"][i:i + 1].clone(),
                "clip_feature": batch["clip_feature"][i:i + 1].clone(),
                "track_points": batch["track_points"][i:i + 1].clone(),
                "track_visibility": batch["track_visibility"][i:i + 1].clone(),
                "object_ids": _opt("object_ids", i),
                "track_weights": _opt("track_weights", i),
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
            notrack_logs: list[Any] = []
            ref_logs: list[Any] = []
            for i, s in enumerate(self._samples):
                # Apply the SAME training-time sampler so the model sees the sampled subset
                # (not the full 2500 grid) and the overlay shows those sampled points.
                tp_s, tv_s = self._sampled_tracks(student, s)
                gen_latents = self._sample(student, transformer, s, tp_s, tv_s)  # [1,16,T,H,W] normalized
                gen_px = student.decode_latents(gen_latents.permute(0, 2, 1, 3, 4))[0]  # [3,T,H,W] in [0,1]
                gen_frames = self._overlay_tracks(gen_px, s, tp_s, tv_s)
                if self._is_main:
                    fn = os.path.join(out_dir, f"step{step:06d}_sample{i}_gen.mp4")
                    imageio.mimsave(fn, gen_frames, fps=self.fps, macro_block_size=1)
                    art = self.tracker.video(fn, caption=f"[{step}] WITH-track {s['caption'][:110]}")
                    if art is not None:
                        gen_logs.append(art)

                # NO-TRACK counterfactual: same image + prompt, tracks=None (zero track map).
                # Overlaid with the SAME sampled tracks for side-by-side comparison -- if this
                # matches the with-track video, the model is ignoring the tracks.
                if self.paired_no_track:
                    nt_latents = self._sample(student, transformer, s, None, None)
                    nt_px = student.decode_latents(nt_latents.permute(0, 2, 1, 3, 4))[0]
                    nt_frames = self._overlay_tracks(nt_px, s, tp_s, tv_s)
                    if self._is_main:
                        fn_nt = os.path.join(out_dir, f"step{step:06d}_sample{i}_notrack.mp4")
                        imageio.mimsave(fn_nt, nt_frames, fps=self.fps, macro_block_size=1)
                        art = self.tracker.video(fn_nt, caption=f"[{step}] NO-track {s['caption'][:110]}")
                        if art is not None:
                            notrack_logs.append(art)

                # One-time ground-truth reference (VAE round-trip + same sampled tracks).
                if not self._ref_logged and self._is_main:
                    ref_px = self._decode_reference(student, s)
                    ref_frames = self._overlay_tracks(ref_px, s, tp_s, tv_s)
                    fn_ref = os.path.join(out_dir, f"reference_sample{i}_gt.mp4")
                    imageio.mimsave(fn_ref, ref_frames, fps=self.fps, macro_block_size=1)
                    art = self.tracker.video(fn_ref, caption=f"GT {s['caption'][:120]}")
                    if art is not None:
                        ref_logs.append(art)

            if self._is_main and gen_logs:
                logs: dict[str, Any] = {"track_val/generated": gen_logs}
                if notrack_logs:
                    logs["track_val/no_track"] = notrack_logs
                if ref_logs and not self._ref_logged:
                    logs["track_val/reference_gt"] = ref_logs
                self.tracker.log_artifacts(logs, step)
                self._ref_logged = True
                logger.info("TrackValidation: logged %d gen + %d no-track videos at step %d", len(gen_logs),
                            len(notrack_logs), step)
        finally:
            if was_training:
                transformer.train()

    def _sampled_tracks(self, student: Any, s: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the training-time track sampler (_augment_tracks) on this val sample, deterministically
        (fixed seed), so the model is conditioned on the sampled subset exactly like training. Falls
        back to the full tracks if the sampler is disabled (WANTRACK_AUG=0) or drops motion."""
        device = student.device
        tp = s["track_points"].to(device)
        tv = s["track_visibility"].to(device)
        n = tp.shape[2]
        oid = s.get("object_ids")
        oid = oid.to(device).long() if (oid is not None and oid.shape[-1] == n) else None
        tw = s.get("track_weights")
        tw = tw.to(device).float() if (tw is not None and tw.shape[-1] == n) else None
        gen = torch.Generator(device=device).manual_seed(self.seed)
        tp_s, tv_s = student._augment_tracks(tp.float(), tv.float(), gen, object_ids=oid, track_weights=tw)
        if tp_s is None or tv_s is None:  # motion-drop (off in val) -> keep full tracks
            return tp, tv
        return tp_s, tv_s

    def _sample(self, student: Any, transformer: torch.nn.Module, s: dict[str, Any], tp_in: torch.Tensor,
                tv_in: torch.Tensor) -> torch.Tensor:
        device = student.device
        dtype = torch.bfloat16
        flow_shift = float(student.timestep_shift)

        ff = s["first_frame_latent"].to(device, dtype)  # [1,16,T,H,W] (raw, matches training)
        cond20 = student._build_i2v_cond_concat(ff)  # [1,20,T,H,W]
        txt = s["text_embedding"].to(device, dtype)
        mask = s["text_attention_mask"].to(device, dtype)
        # tp_in/tv_in None => no-track counterfactual (transformer builds a zero track map).
        tp = tp_in.to(device, dtype) if tp_in is not None else None  # [1,T,N,2] normalized sampled subset
        tv = tv_in.to(device, dtype) if tv_in is not None else None  # [1,T,N] sampled visibility
        img = s["clip_feature"].to(device, dtype)  # [1,SeqLen,Dim] CLIP frame-0

        _, _, T, H, W = ff.shape
        gen = torch.Generator(device="cpu").manual_seed(self.seed)
        latents = torch.randn((1, 16, T, H, W), generator=gen, dtype=torch.float32).to(device)

        sched = FlowMatchEulerDiscreteScheduler(shift=flow_shift)
        sched.set_timesteps(self.num_inference_steps, device=device)

        wt = float(self.guidance_scale)
        wm = float(self.motion_guidance_scale)
        cfg_on = (wt != 1.0) or (wm != 1.0)  # (1.0, 1.0) collapses to plain v_full
        # "unconditional" text = zero embed (matches WANTRACK_TEXT_DROP training).
        txt_null = torch.zeros_like(txt) if cfg_on else None

        def _fwd(text_e: torch.Tensor, tp_e: torch.Tensor | None, tv_e: torch.Tensor | None, mi: torch.Tensor,
                 tsv: torch.Tensor) -> torch.Tensor:
            with torch.autocast(device.type, dtype=dtype), set_forward_context(current_timestep=tsv,
                                                                               attn_metadata=None):
                return transformer(hidden_states=mi,
                                   encoder_hidden_states=text_e,
                                   encoder_attention_mask=mask,
                                   timestep=tsv,
                                   encoder_hidden_states_image=img,
                                   track_points=tp_e,
                                   track_visibility=tv_e,
                                   return_dict=False)

        for t in sched.timesteps:
            model_in = torch.cat([latents.to(dtype), cond20], dim=1)  # [1,36,T,H,W]
            ts = t.reshape(1).to(device, dtype)
            v_full = _fwd(txt, tp, tv, model_in, ts)  # v(c_t, c_m)
            if not cfg_on:
                v = v_full
            elif tp is None:
                # No-track counterfactual: motion CFG undefined -> plain text CFG only.
                v_uncond = _fwd(txt_null, None, None, model_in, ts)
                v = v_uncond + wt * (v_full - v_uncond)
            else:
                # MotionStream Eq. 2: joint text+motion CFG (3 NFE / step).
                #   v_no_text   = v(∅, c_m)   (drop text, keep tracks)
                #   v_no_motion = v(c_t, ∅)   (keep text, drop tracks)
                #   v_base      = α · v_no_text + (1-α) · v_no_motion,  α = wt / (wt+wm)
                #   v̂           = v_base + wt·(v_full - v_no_text) + wm·(v_full - v_no_motion)
                v_no_text = _fwd(txt_null, tp, tv, model_in, ts)
                v_no_motion = _fwd(txt, None, None, model_in, ts)
                alpha = wt / (wt + wm) if (wt + wm) > 0 else 0.5
                v_base = alpha * v_no_text + (1.0 - alpha) * v_no_motion
                v = v_base + wt * (v_full - v_no_text) + wm * (v_full - v_no_motion)
            latents = sched.step(v.float(), t, latents.float(), return_dict=False)[0]
        return latents

    def _decode_reference(self, student: Any, s: dict[str, Any]) -> torch.Tensor:
        from fastvideo.training.training_utils import normalize_dit_input
        device = student.device
        dtype = torch.bfloat16
        raw = s["vae_latent"].to(device, dtype)  # [1,16,T,H,W] raw VAE latent
        norm = normalize_dit_input("wan", raw, student.vae)
        return student.decode_latents(norm.permute(0, 2, 1, 3, 4))[0]  # [3,T,H,W]

    def _overlay_tracks(self, px: torch.Tensor, s: dict[str, Any], tp_in: torch.Tensor,
                        tv_in: torch.Tensor) -> list[np.ndarray]:
        # px: [3,T,H,W] in [0,1]. Draws ONLY the sampled tracks (those the sampler kept), gated by
        # the sampled visibility, so the overlay matches what the model was conditioned on.
        video = (px.clamp(0, 1).float().cpu().numpy() * 255.0).astype(np.uint8)
        frames = np.transpose(video, (1, 2, 3, 0))  # [T,H,W,3]
        T, H, W, _ = frames.shape

        tp = tp_in[0].float().cpu().numpy()  # [Tt,N,2] normalized
        tv = tv_in[0].float().cpu().numpy()  # [Tt,N] sampled visibility
        tt = min(T, tp.shape[0])
        frames, tp, tv = frames[:tt], tp[:tt], tv[:tt]
        n = tp.shape[1]
        grid = int(round(n**0.5))

        colors = _grid_colors(grid, 1)  # one color per grid cell (row-major)
        if colors.shape[0] < n:
            colors = np.resize(colors, (n, 3))
        colors = colors[:n]

        # keep only tracks that are ever visible after sampling (the sampled subset)
        ever = tv.max(axis=0) >= self.vis_thresh
        if not ever.any():
            ever = np.ones(n, dtype=bool)
        tp, tv, colors = tp[:, ever], tv[:, ever], colors[ever]

        # Normalized [0,1] -> pixel coords of the generated frame.
        tp_px = tp.copy()
        tp_px[..., 0] = tp[..., 0] * W
        tp_px[..., 1] = tp[..., 1] * H
        return _draw_overlay(frames, tp_px, tv, colors, self.tail, self.radius, self.vis_thresh)
