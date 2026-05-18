"""Synchformer DeSync — average audio-video desynchronization in seconds.

Per-sample. Ports ``hkchengrex/av-benchmark``'s ``DeSync`` against the
``24-01-04T16-39-21`` Synchformer checkpoint (the one MMAudio reports
against). ``argmax`` over the 21-class grid in ``[-2, +2]`` seconds for
the first 14 and last 14 segments separately; lower is better, 0 = ideal.

Synchformer's transformer carries a fixed positional embedding sized for
~14 segments per modality, so keep input clips in the 3-10 s range —
shorter clips raise in ``_segment_video`` and longer clips ignore the
middle.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from fastvideo.eval.metrics.base import BaseMetric
from fastvideo.eval.models import ensure_checkpoint
from fastvideo.eval.registry import register
from fastvideo.eval.types import MetricResult

_SYNCHFORMER_URL = "https://github.com/hkchengrex/MMAudio/releases/download/v0.1/synchformer_state_dict.pth"
_SYNCHFORMER_NAME = "synchformer_state_dict.pth"

_SYNC_SIZE = 224
_SYNC_FPS = 25.0
_AUDIO_SR = 16000
_VIDEO_SEG_FRAMES = 16
_VIDEO_SEG_STEP = 8
_AUDIO_SEG_SAMPLES = 10240
_AUDIO_SEG_STEP = 5120
_GRID_LOW, _GRID_HIGH, _GRID_SIZE = -2.0, 2.0, 21
_NUM_SEG_PER_DIRECTION = 14
_AUDIO_MEL_FRAMES = 66
_AUDIO_MEAN, _AUDIO_STD = -4.2677393, 4.5689974


def _resample_video(frames: torch.Tensor, target_fps: float, src_fps: float) -> torch.Tensor:
    """Nearest-neighbor temporal resample of ``(T, C, H, W)`` to *target_fps*,
    preserving the source clip's actual duration (no padding, no truncation).
    """
    if abs(src_fps - target_fps) < 1e-6:
        return frames
    src_t = frames.shape[0]
    duration = src_t / src_fps
    target_t = max(_VIDEO_SEG_FRAMES, int(round(duration * target_fps)))
    idx = (np.arange(target_t) * (src_fps / target_fps)).astype(np.int64)
    idx = np.clip(idx, 0, src_t - 1)
    return frames[idx]


def _video_transform(frames: torch.Tensor) -> torch.Tensor:
    """``(T, C, H, W)`` in [0,1] → resize-shortest-to-224, center-crop 224, normalize."""
    t, c, h, w = frames.shape
    scale = _SYNC_SIZE / min(h, w)
    new_h, new_w = int(round(h * scale)), int(round(w * scale))
    frames = F.interpolate(frames, size=(new_h, new_w), mode="bicubic", align_corners=False, antialias=True)
    top = (new_h - _SYNC_SIZE) // 2
    left = (new_w - _SYNC_SIZE) // 2
    frames = frames[:, :, top:top + _SYNC_SIZE, left:left + _SYNC_SIZE]
    return (frames - 0.5) / 0.5  # mean/std=0.5 per channel


def _segment_video(frames: torch.Tensor) -> torch.Tensor:
    """``(T, C, 224, 224)`` → ``(S, T_seg, C, 224, 224)`` with seg_size=16, step=8."""
    t = frames.shape[0]
    num_segments = (t - _VIDEO_SEG_FRAMES) // _VIDEO_SEG_STEP + 1
    if num_segments <= 0:
        raise ValueError(f"video too short for Synchformer segmenting (got {t} frames)")
    segs = [frames[i * _VIDEO_SEG_STEP:i * _VIDEO_SEG_STEP + _VIDEO_SEG_FRAMES] for i in range(num_segments)]
    return torch.stack(segs, dim=0)


def _load_audio_waveform(audio_path: str, resample_cache: dict[int, Any]) -> torch.Tensor:
    """Load *audio_path* at 16 kHz mono, native length. Returns ``(N,)``."""
    import torchaudio
    waveform, sr = torchaudio.load(audio_path)
    waveform = waveform.mean(dim=0)
    if sr != _AUDIO_SR:
        if sr not in resample_cache:
            resample_cache[sr] = torchaudio.transforms.Resample(sr, _AUDIO_SR)
        waveform = resample_cache[sr](waveform)
    return waveform


def _pad_or_truncate_mel(x: torch.Tensor, target_t: int) -> torch.Tensor:
    """Pad along the time dim with the per-position min, or truncate to *target_t*."""
    t = x.shape[-1]
    if t == target_t:
        return x
    if t > target_t:
        return x[..., :target_t]
    pad_value = x.min().item()
    pad_len = target_t - t
    return F.pad(x, (0, pad_len), value=pad_value)


def _encode_audio_segments(synchformer: Any, waveform: torch.Tensor, mel_transform: Any) -> torch.Tensor:
    """Ports ``av_bench/extract.py::encode_audio_with_sync``.

    waveform: ``(T,)``. Returns Synchformer audio features
    ``(1, S, ta, D)`` aligned to the video segmentation.
    """
    waveform = waveform.unsqueeze(0)  # (1, T)
    t = waveform.shape[1]
    num_segments = max(1, (t - _AUDIO_SEG_SAMPLES) // _AUDIO_SEG_STEP + 1)
    segments = []
    for i in range(num_segments):
        seg = waveform[:, i * _AUDIO_SEG_STEP:i * _AUDIO_SEG_STEP + _AUDIO_SEG_SAMPLES]
        if seg.shape[1] < _AUDIO_SEG_SAMPLES:
            seg = F.pad(seg, (0, _AUDIO_SEG_SAMPLES - seg.shape[1]))
        segments.append(seg)
    x = torch.stack(segments, dim=1)  # (1, S, T_seg)
    x = mel_transform(x)
    x = torch.log(x + 1e-6)
    x = _pad_or_truncate_mel(x, _AUDIO_MEL_FRAMES)
    x = (x - _AUDIO_MEAN) / (2 * _AUDIO_STD)
    # Synchformer's extract_afeats expects (B, S, 1, F, T)
    return synchformer.extract_afeats(x.unsqueeze(2))


@register("audio.desync")
class DeSyncMetric(BaseMetric):
    """Per-sample Synchformer DeSync magnitude in seconds (lower is better)."""

    name = "audio.desync"
    requires_reference = False
    higher_is_better = False
    needs_gpu = True
    is_set_metric = False
    backbone = "synchformer"
    dependencies = ["torchaudio"]

    def __init__(self, src_fps: float | None = None) -> None:
        super().__init__()
        # Defaults to 25 fps (Synchformer's target) when the pool's decode fps is unknown.
        self._src_fps = src_fps
        self._model: Any = None
        self._mel: Any = None
        self._grid: torch.Tensor | None = None
        self._resample_cache: dict[int, Any] = {}

    def to(self, device):
        super().to(device)
        if self._model is not None:
            self._model = self._model.to(self.device)
        if self._mel is not None:
            self._mel = self._mel.to(self.device)
        return self

    def setup(self) -> None:
        if self._model is not None:
            return
        import torchaudio
        from fastvideo.third_party.eval.synchformer import Synchformer, make_class_grid

        ckpt = ensure_checkpoint(_SYNCHFORMER_NAME, _SYNCHFORMER_URL)
        model = Synchformer().to(self.device).eval()
        model.load_state_dict(torch.load(ckpt, weights_only=True, map_location=self.device))
        self._model = model

        self._mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=_AUDIO_SR,
            win_length=400,
            hop_length=160,
            n_fft=1024,
            n_mels=128,
        ).to(self.device)
        self._grid = make_class_grid(_GRID_LOW, _GRID_HIGH, _GRID_SIZE)

    @torch.no_grad()
    def compute(self, sample: dict) -> MetricResult:
        if self._model is None:
            self.setup()

        video = sample.get("video")
        audio_path = sample.get("audio")
        if video is None or audio_path is None:
            return self._skip(sample, "missing 'video' or 'audio'")

        # Prefer the per-sample fps the pool decoded at; fall back to the
        # constructor override. Without either, the audio/video windows can't
        # be aligned, so skip rather than guess.
        sample_fps = sample.get("fps")
        src_fps = float(sample_fps) if sample_fps is not None else self._src_fps
        if src_fps is None:
            return self._skip(sample, "missing 'fps' (set sample fps or pass src_fps=)")

        # Video preprocessing → (S, T_seg, C, 224, 224)
        frames = video.float().to(self.device)
        frames = _resample_video(frames, _SYNC_FPS, src_fps)
        frames = _video_transform(frames)
        vsegs = _segment_video(frames).unsqueeze(0)  # (B=1, S, T_seg, C, H, W)
        # Synchformer's extract_vfeats wants (B, S, T_seg, C, H, W); fold (B, S) → (B*S, 1, ...).
        b, s = vsegs.shape[:2]
        from einops import rearrange
        vsegs_for_model = rearrange(vsegs, "b s t c h w -> (b s) 1 t c h w")
        vfeats = self._model.extract_vfeats(vsegs_for_model)
        vfeats = rearrange(vfeats, "(b s) 1 t d -> b s t d", b=b)  # (1, S, t_v, D)

        # Audio preprocessing → Synchformer audio features (1, S, t_a, D)
        waveform = _load_audio_waveform(audio_path, self._resample_cache).to(self.device)
        assert self._mel is not None
        afeats = _encode_audio_segments(self._model, waveform, self._mel)

        # Compare first-14 and last-14 segments; argmax over the 21-class grid;
        # |grid|-value → average across the two directions.
        sync_grid = self._grid.to(self.device) if self._grid is not None else None
        assert sync_grid is not None
        # Synchformer's transformer carries a fixed 198-token positional embedding
        # (~14 segments × (tv+ta) + 2 special tokens). Anything shorter would
        # shape-mismatch on the pos_emb add inside compare_v_a, so skip.
        n_vsegs = int(vfeats.shape[1])
        n_asegs = int(afeats.shape[1])
        s_used = min(n_vsegs, n_asegs)
        if s_used < _NUM_SEG_PER_DIRECTION:
            return self._skip(
                sample,
                f"too few segments for Synchformer pos_emb "
                f"(need {_NUM_SEG_PER_DIRECTION}, got v={n_vsegs} a={n_asegs}); "
                f"use clips of at least ~5 s",
            )
        s_used = _NUM_SEG_PER_DIRECTION
        front_logits = self._model.compare_v_a(vfeats[:, :s_used], afeats[:, :s_used])
        back_logits = self._model.compare_v_a(vfeats[:, -s_used:], afeats[:, -s_used:])
        front_id = int(torch.argmax(front_logits, dim=-1).item())
        back_id = int(torch.argmax(back_logits, dim=-1).item())
        front_d = abs(float(sync_grid[front_id].item()))
        back_d = abs(float(sync_grid[back_id].item()))
        desync = (front_d + back_d) / 2.0
        return MetricResult(
            name=self.name,
            score=desync,
            details={
                "front_desync_s": front_d,
                "back_desync_s": back_d,
                "num_segments_used": s_used,
            },
        )
