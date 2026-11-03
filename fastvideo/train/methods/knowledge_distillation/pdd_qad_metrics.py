# SPDX-License-Identifier: Apache-2.0
"""Per-forward (T1) error and reference-fidelity bookkeeping for PDD QAD.

All quantities are on the target rows only (conditioning rows are inputs).
``x0`` is the scheduler's own denoised estimate, ``x_t + sigma_t * v`` with
``sigma_t = 1 - timestep``, so x0 error is ``sigma_t`` times velocity error.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any

import torch
import torch.distributed as dist

MODALITIES = ("video", "audio")


def target_slices(prepared: Any) -> dict[str, slice]:
    return {
        "video": slice(prepared.layout.num_condition_video_rows, None),
        "audio": slice(prepared.layout.num_condition_audio_rows, None)
    }


def denoised(state: torch.Tensor, velocity: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
    """x0 exactly as ``MiniMaxH3Scheduler.step`` forms it (fp32)."""
    return state.float() + (1 - timestep.to(torch.float32)) * velocity.float()


def scheduler_step(scheduler: Any, rung: int, velocity: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
    """``scheduler.step`` at forward ``rung`` without relying on its internal counter."""
    scheduler._step_index = rung
    return scheduler.step(velocity.float(), scheduler.timesteps[rung], state, return_dict=False)[0]


def step_packed(schedulers: tuple[Any, Any], prepared: Any, rung: int, video: torch.Tensor, audio: torch.Tensor,
                video_v: torch.Tensor, audio_v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Advance the packed target rows by one PDD forward (conditioning rows untouched), as the sampler does."""
    cut = target_slices(prepared)
    video, audio = video.clone(), audio.clone()
    video[cut["video"]] = scheduler_step(schedulers[0], rung, video_v[cut["video"]], video[cut["video"]])
    audio[cut["audio"]] = scheduler_step(schedulers[1], rung, audio_v[cut["audio"]], audio[cut["audio"]])
    return video, audio


def x0_pair(prepared: Any, schedulers: tuple[Any, Any], rung: int, video: torch.Tensor, audio: torch.Tensor,
            outputs: dict[str, tuple[torch.Tensor, torch.Tensor]]) -> dict[str, dict[str, torch.Tensor]]:
    """``{modality: {"student": x0, "teacher": x0, "student_v": v, "teacher_v": v}}`` on target rows."""
    cut = target_slices(prepared)
    states = {"video": video, "audio": audio}
    result: dict[str, dict[str, torch.Tensor]] = {}
    for index, modality in enumerate(MODALITIES):
        timestep = schedulers[index].timesteps[rung]
        entry = {}
        for role, (video_v, audio_v) in outputs.items():
            velocity = (video_v, audio_v)[index][cut[modality]]
            entry[role] = denoised(states[modality][cut[modality]], velocity, timestep)
            entry[f"{role}_v"] = velocity.float()
        result[modality] = entry
    return result


class RungStats:
    """Sums of per-rung error terms, reduced over data-parallel groups (one SP rank contributes)."""

    KEYS = ("mse", "err_sq", "ref_sq", "verr_sq", "vref_sq", "dot", "s_sq")
    EXTRA_KEYS = ("endpoint.video", "endpoint.audio", "keyframe.vs_teacher.first", "keyframe.vs_teacher.last",
                  "keyframe.student.first", "keyframe.teacher.first", "keyframe.student.last", "keyframe.teacher.last")

    def __init__(self, num_rungs: int) -> None:
        self.num_rungs = num_rungs
        self.sums: dict[str, torch.Tensor] = {}
        self.counts: dict[str, float] = defaultdict(float)
        # Ranks hold different rows (cases), so every rank reduces this same fixed key list.
        self.all_keys = [f"{m}.{r}.{k}" for m in MODALITIES for r in range(num_rungs) for k in self.KEYS]
        self.all_keys += list(self.EXTRA_KEYS)

    def _add(self, key: str, value: torch.Tensor) -> None:
        self.sums[key] = self.sums.get(key, 0.0) + value.detach().double()

    def add_rung(self, rung: int, pair: dict[str, dict[str, torch.Tensor]]) -> None:
        for modality, entry in pair.items():
            s, t, sv, tv = entry["student"], entry["teacher"], entry["student_v"], entry["teacher_v"]
            prefix = f"{modality}.{rung}"
            self._add(f"{prefix}.mse", (s - t).pow(2).mean())
            self._add(f"{prefix}.err_sq", (s - t).pow(2).sum())
            self._add(f"{prefix}.ref_sq", t.pow(2).sum())
            self._add(f"{prefix}.verr_sq", (sv - tv).pow(2).sum())
            self._add(f"{prefix}.vref_sq", tv.pow(2).sum())
            self._add(f"{prefix}.dot", (s * t).sum())
            self._add(f"{prefix}.s_sq", s.pow(2).sum())
            for key in self.KEYS:
                self.counts[f"{prefix}.{key}"] += 1

    def add_scalar(self, key: str, value: torch.Tensor | float) -> None:
        if key not in self.EXTRA_KEYS:
            raise KeyError(f"{key} is not a reduced RungStats key")
        self._add(key, torch.as_tensor(value, dtype=torch.float64, device=self._device()))
        self.counts[key] += 1

    def _device(self) -> torch.device:
        return torch.device("cuda", torch.cuda.current_device()) if torch.cuda.is_available() else torch.device("cpu")

    def reduce(self, contribute: bool) -> dict[str, float]:
        """All-reduce sums and counts (call on every rank in the same key order); returns means."""
        keys = self.all_keys
        device = self._device()
        values = torch.tensor([float(self.sums.get(k, 0.0)) for k in keys] + [self.counts.get(k, 0.0) for k in keys],
                              dtype=torch.float64,
                              device=device)
        if not contribute:
            values.zero_()
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(values)
        sums, counts = values[:len(keys)], values[len(keys):]
        return {key: float(sums[i]) / float(counts[i]) for i, key in enumerate(keys) if float(counts[i]) > 0}


def summarize(means: dict[str, float], num_rungs: int, weights: dict[str, float]) -> dict[str, float]:
    """Per-rung rel-L2 / cosine / MSE plus the weighted mean x0 rel-L2 ``score`` (lower is better)."""
    out: dict[str, float] = {}
    score_num = score_den = 0.0
    for modality in MODALITIES:
        rel_all = []
        for rung in range(num_rungs):
            p = f"{modality}.{rung}"
            if f"{p}.ref_sq" not in means:
                continue
            rel = (means[f"{p}.err_sq"] / max(means[f"{p}.ref_sq"], 1e-30))**0.5
            vrel = (means[f"{p}.verr_sq"] / max(means[f"{p}.vref_sq"], 1e-30))**0.5
            cos = means[f"{p}.dot"] / max((means[f"{p}.s_sq"] * means[f"{p}.ref_sq"])**0.5, 1e-30)
            out[f"x0_rel_l2/{modality}/rung{rung}"] = rel
            out[f"v_rel_l2/{modality}/rung{rung}"] = vrel
            out[f"x0_cos/{modality}/rung{rung}"] = cos
            out[f"x0_mse/{modality}/rung{rung}"] = means[f"{p}.mse"]
            rel_all.append(rel)
        if rel_all:
            out[f"x0_rel_l2/{modality}/mean"] = sum(rel_all) / len(rel_all)
            score_num += weights[modality] * out[f"x0_rel_l2/{modality}/mean"]
            score_den += weights[modality]
    out["score"] = score_num / max(score_den, 1e-30)
    for key, value in means.items():
        if key.startswith(("endpoint.", "keyframe.")):
            out[key.replace(".", "/")] = value
    return out


def keyframe_latents(prepared: Any, final_video: torch.Tensor) -> dict[str, tuple[torch.Tensor, torch.Tensor | None]]:
    """Generated keyframe latents ``[C, H, W]`` (first; last for first_last_frame) and their reference image.

    OmniRef encodes keyframe references at high resolution with the target's
    aspect ratio, so the reference latent is area-resized to the target grid: a
    rough proxy, logged only. The guarded keyframe metric compares the student's
    keyframe with the teacher's (``keyframe/vs_teacher``).
    """
    import torch.nn.functional as F

    from fastvideo.pipelines.basic.minimax_h3.packing import unpatchify_video_tokens

    channels, frames, height, width = prepared.latent_shape
    patch = (1, 2, 2)
    cut = target_slices(prepared)["video"]
    video = unpatchify_video_tokens(final_video[cut][None].float(), frames, height, width, channels, patch)[0]
    images: list[torch.Tensor | None] = []
    offset = 0
    for ref in prepared.references:
        if ref.media_type == "audio":
            continue
        count = ref.num_latent_frames * (ref.latent_height // 2) * (ref.latent_width // 2)
        image = None
        same_aspect = abs(ref.latent_height * width - ref.latent_width * height) <= 0.02 * ref.latent_width * height
        if ref.media_type == "image" and same_aspect:
            rows = prepared.reference_video_rows[offset:offset + count].to(video.device)
            latent = unpatchify_video_tokens(rows[None].float(), 1, ref.latent_height, ref.latent_width, channels,
                                             patch)[:, :, 0]
            if (ref.latent_height, ref.latent_width) != (height, width):
                latent = F.interpolate(latent, size=(height, width), mode="area")
            image = latent[0]
        images.append(image)
        offset += count
    pairs = {"first_frame": [("first", 0, 0)], "first_last_frame": [("first", 0, 0), ("last", -1, 1)]}
    return {
        name: (video[:, frame], images[ref] if ref < len(images) else None)
        for name, frame, ref in pairs.get(prepared.spec.case, [])
    }


def rel_l2(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return (a.float() - b.float()).norm() / b.float().norm().clamp_min(1e-12)


__all__ = ["RungStats", "keyframe_latents", "rel_l2", "step_packed", "summarize", "target_slices", "x0_pair"]
