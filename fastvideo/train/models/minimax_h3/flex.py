# SPDX-License-Identifier: Apache-2.0
"""Pika H3 Flex (six-forward re-noise sampler) as a per-forward model for NVFP4 QAD.

Same contract as ``omniref_pdd.MiniMaxH3OmniRefPDDModel``, so ``OmniRefPDDQADMethod`` runs unchanged: one rung per
sampler forward, ``rung_scope`` / ``forward_rung`` / ``schedulers`` / ``prepare`` / ``quantization``. ``init_from``
is a merged Flex model directory written by ``scripts/distill/pika_h3_flex_qad/merge_pika_adapter.py``.

Differences from OmniRef:

- Dense attention (``FLASH_ATTN``): Flex has no VSA compression gate, so there is no tile selection to freeze and the
  FP4 sparse-attention emulator does not apply. QAD covers the NVFP4 linears only.
- No PDD head fusion: every rung is a plain transformer forward over Pika's clean-time knots
  (1000, 750, 500, 250, 100, 40), shifted per modality (video 12, audio 3), starting from unit noise.
- Re-noise steps: ``x_next = (1 - sigma_next) x0 + sigma_next eps``. ``eps`` is a function of (row seed, forward,
  modality) only, so the teacher-forced pass, the student rollout and the seed+1 noise-floor eval of a row all draw
  the same noise. It is not Pika's sampler stream; the distribution is.

``MiniMaxH3FlexRef2VAModel`` reads Ref2VA rows (``omniref_data``) on ``transformer_ref``;
``MiniMaxH3FlexFL2VAModel`` reads T2VA / I2VA / L2VA / FL2VA keyframe-anchor rows on ``transformer``.
"""
from __future__ import annotations

import contextlib
import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch

from fastvideo.configs.pipelines.pika_h3_flex import (PIKA_AUDIO_SCHEDULER_SHIFT, PIKA_DENOISING_STEPS,
                                                      PIKA_VIDEO_SCHEDULER_SHIFT)
from fastvideo.forward_context import set_forward_context
from fastvideo.logger import init_logger
from fastvideo.models.schedulers.scheduling_minimax_h3 import MiniMaxH3Scheduler
from fastvideo.train.models.minimax_h3.omniref_pdd import MiniMaxH3OmniRefPDDModel, _pin_amax

if TYPE_CHECKING:
    from fastvideo.train.models.minimax_h3.omniref_data import PreparedRow
    from fastvideo.train.utils.training_config import TrainingConfig

logger = init_logger(__name__)
CONTRACT_FILE = "pika_flex_qad.json"
CONTRACT_SCHEMA = "pika-flex-qad-v1"
TASK_TO_TRUNK = {"fl2va": "transformer", "ref2va": "transformer_ref"}
# Block projections and FFN linears: the set the NVFP4 export quantizes. Norms, AdaLN, the token refiner,
# embeddings and the output heads stay frozen.
FLEX_TRAINABLE = r"transformer_blocks\.\d+\.(attn\.(to_q|to_k|to_v|to_out)|ff\.(fc_in|fc_out))\."
_VIDEO_STREAM, _AUDIO_STREAM = 0, 1


@dataclass(frozen=True)
class FlexContract:
    """The merged checkpoint's sampling contract (``pika_flex_qad.json``)."""
    task: str
    trunk: str
    denoising_steps: tuple[int, ...]
    video_shift: float
    audio_shift: float

    @classmethod
    def read(cls, model_path: str) -> FlexContract:
        path = Path(model_path) / CONTRACT_FILE
        if not path.is_file():
            raise ValueError(f"{model_path} has no {CONTRACT_FILE}; build it with merge_pika_adapter.py")
        data = json.loads(path.read_text())
        if data.get("schema") != CONTRACT_SCHEMA or data.get("sampler") != "renoise":
            raise ValueError(f"{path} is not a {CONTRACT_SCHEMA} re-noise contract")
        contract = cls(str(data["task"]), str(data["trunk"]), tuple(int(s) for s in data["denoising_steps"]),
                       float(data["video_scheduler_shift"]), float(data["audio_scheduler_shift"]))
        released = (PIKA_DENOISING_STEPS, PIKA_VIDEO_SCHEDULER_SHIFT, PIKA_AUDIO_SCHEDULER_SHIFT)
        if (contract.denoising_steps, contract.video_shift, contract.audio_shift) != released:
            raise ValueError(f"{path} does not carry the released Pika H3 Flex ladder {released}")
        if TASK_TO_TRUNK.get(contract.task) != contract.trunk:
            raise ValueError(f"{path}: task {contract.task!r} does not run on trunk {contract.trunk!r}")
        return contract


def flex_sigmas(steps: tuple[int, ...], shift: float) -> torch.Tensor:
    """Shifted sigma grid over ``steps`` plus the terminal zero, as ``MiniMaxH3DenoisingStage._set_shifted_schedule``."""
    base = torch.tensor([step / 1000.0 for step in steps] + [0.0], dtype=torch.float32)
    return shift * base / (1 + (shift - 1) * base)


def renoise_seed(row_seed: int, forward: int, stream: int) -> int:
    """Distinct generator seed per (row seed, sampler forward, modality); forwards < 8, two streams."""
    return (int(row_seed) + 1) * 16 + 2 * int(forward) + int(stream)


class RowRenoiseScheduler(MiniMaxH3Scheduler):
    """H3 re-noise scheduler whose fresh noise at forward ``k`` is fixed by (row seed, ``k``, modality)."""

    def __init__(self, shift: float, row_seed: int, stream: int) -> None:
        super().__init__(shift=shift, stochastic_sampling=True, shift_provided_sigmas=True)
        self._row_seed = int(row_seed)
        self._stream = int(stream)

    def step(self,
             model_output: torch.FloatTensor,
             timestep: float | torch.FloatTensor,
             sample: torch.FloatTensor,
             return_dict: bool = True) -> Any:
        if self._step_index is None:
            raise ValueError("RowRenoiseScheduler needs an explicit step index (use pdd_qad_metrics.scheduler_step)")
        generator = torch.Generator(device=sample.device)
        generator.manual_seed(renoise_seed(self._row_seed, self._step_index, self._stream))
        self.set_noise_generator(generator)
        return super().step(model_output, timestep, sample, return_dict=return_dict)


class MiniMaxH3FlexRef2VAModel(MiniMaxH3OmniRefPDDModel):
    """Teacher (merged bf16 Flex) or NVFP4 QAD student on the six-forward Flex ladder, Ref2VA rows."""

    _transformer_module_type = "transformer_ref"
    _attention_backend = "FLASH_ATTN"
    _task = "ref2va"

    def __init__(self,
                 *,
                 init_from: str,
                 training_config: TrainingConfig,
                 trainable: bool = True,
                 enable_gradient_checkpointing_type: str | None = None,
                 trainable_patterns: str = FLEX_TRAINABLE,
                 qad: dict[str, Any] | None = None,
                 data: dict[str, Any] | None = None,
                 frozen_bf16: bool = False) -> None:
        super().__init__(init_from=init_from,
                         training_config=training_config,
                         trainable=trainable,
                         enable_gradient_checkpointing_type=enable_gradient_checkpointing_type,
                         trainable_patterns=trainable_patterns,
                         qad=qad,
                         data=data,
                         frozen_bf16=frozen_bf16)

    # ------------------------------------------------------------------ construction hooks
    def _read_contract(self, model_path: str) -> Any:
        contract = FlexContract.read(model_path)
        if contract.task != self._task or contract.trunk != self._transformer_module_type:
            raise ValueError(f"{model_path} is a Flex {contract.task} merge on {contract.trunk}; "
                             f"{type(self).__name__} needs {self._task} on {self._transformer_module_type}")
        return contract

    def _check_tile(self) -> None:
        """Dense attention: no VSA tile to check."""

    def _install_qad(self) -> None:
        """NVFP4 W4A4 linears with the calibrated static FFN scales; attention stays BF16 dense."""
        from fastvideo.layers.quantization.nvfp4_qad import NVFP4QADPlan, install_nvfp4_qad, load_amax_table

        if dict(self.qad_config.get("fp4_attention") or {}).get("enabled", False):
            raise ValueError("Pika H3 Flex runs dense attention; the FP4 attention emulator models VSA-H3 tiles. "
                             "Set qad.fp4_attention.enabled: false")
        linears = dict(self.qad_config.get("linears") or {})
        if not linears.get("enabled", True):
            return
        amax_path = linears.get("amax_json")
        table = load_amax_table(amax_path) if amax_path else None
        if amax_path:
            self.amax_sha256 = _pin_amax(amax_path, linears.get("amax_sha256"),
                                         self.training_config.checkpoint.output_dir)
        self.provisional_rows = int(linears.get("provisional_calibration_rows", 0))
        if table is None and self.provisional_rows <= 0:
            raise ValueError("qad.linears.amax_json (the Flex calibration table) is required")
        # Flex checkpoints have no compression gate, so ``gate`` has nothing to quantize.
        plan = NVFP4QADPlan(quantize_attention=bool(linears.get("attention", True)),
                            quantize_gate=False,
                            quantize_ffn=bool(linears.get("ffn", True)),
                            skip_blocks=tuple(int(b) for b in linears.get("skip_blocks", ())))
        self.qad_linears = install_nvfp4_qad(self.transformer, table, plan, allow_uncalibrated=table is None)
        logger.info("Flex %s QAD student: %d NVFP4 QAD linears, BF16 dense attention, %.2fB trainable parameters",
                    self._task, len(self.qad_linears), self.num_trainable / 1e9)

    @contextlib.contextmanager
    def quantization(self, *, linears: bool = True, attention: bool = True) -> Iterator[None]:
        """Toggle the NVFP4 linears (``attention`` is accepted for the method's contract; it is always BF16)."""
        from fastvideo.layers.quantization.nvfp4_qad import set_nvfp4_qad_enabled

        set_nvfp4_qad_enabled(self.qad_linears, linears)
        try:
            yield
        finally:
            set_nvfp4_qad_enabled(self.qad_linears, True)

    # ------------------------------------------------------------------ Flex ladder
    @property
    def plan(self) -> Any:  # pragma: no cover - PDD-only
        raise AttributeError("Pika H3 Flex has no PDD plan")

    @property
    def num_rungs(self) -> int:
        return len(self.contract.denoising_steps)

    def schedulers(self, prepared: PreparedRow | None = None) -> tuple[Any, Any]:
        """Re-noise schedulers on the shifted Flex grids; their noise is fixed by ``prepared``'s seed."""
        if prepared is None:
            raise ValueError("Flex re-noise schedulers need the prepared row (its seed fixes the noise)")
        seed = int(prepared.spec.seed)
        out = []
        for shift, stream in ((self.contract.video_shift, _VIDEO_STREAM), (self.contract.audio_shift, _AUDIO_STREAM)):
            scheduler = RowRenoiseScheduler(shift, seed, stream)
            scheduler.set_timesteps(sigmas=flex_sigmas(self.contract.denoising_steps, shift), device=self.device)
            out.append(scheduler)
        return out[0], out[1]

    def prepare(self, spec: Any, row: dict[str, Any]) -> PreparedRow:
        from fastvideo.train.models.minimax_h3.omniref_data import prepare_row

        # Knot 1000 is sigma 1 for both modalities: target rows start as unit noise (no PDD start scaling).
        return prepare_row(spec,
                           row,
                           patch_size=self.patch_size,
                           latent_channels=24,
                           audio_channels=32,
                           pdd_start_sigmas=(1.0, 1.0),
                           device=self.device)

    @contextlib.contextmanager
    def rung_scope(self, prepared: PreparedRow, rung: int, timesteps: tuple[Any, Any]) -> Iterator[Any]:
        """Forward context of Flex forward ``rung``; keep backward inside it (activation recompute reads it)."""
        from fastvideo.pipelines.basic.minimax_h3.packing import MINIMAX_H3_KEYFRAME_NOISE_AUG, build_row_timesteps
        video_t, audio_t = float(timesteps[0][rung].item()), float(timesteps[1][rung].item())
        unique, inverse = build_row_timesteps(prepared.layout,
                                              video_timestep=video_t,
                                              audio_timestep=audio_t,
                                              condition_video_timestep=max(video_t, MINIMAX_H3_KEYFRAME_NOISE_AUG),
                                              condition_audio_timestep=1.0)
        with set_forward_context(current_timestep=rung, attn_metadata=None):
            yield unique.to(self.device), inverse.to(self.device)


# Keyframe anchors of each FL2VA task, in packing order (Pika's first/last-frame conditioning).
FL2VA_ANCHORS: dict[str, tuple[str, ...]] = {
    "t2va": (),
    "i2va": ("first", ),
    "l2va": ("last", ),
    "fl2va": ("first", "last")
}


def prepare_fl2va_row(spec: Any, row: dict[str, Any], *, patch_size: tuple[int, int, int], latent_channels: int,
                      audio_channels: int, device: torch.device) -> PreparedRow:
    """Initial packed FL2VA state: keyframe anchors at the canvas size, noised to the keyframe clean time.

    Mirrors ``MiniMaxH3LatentPreparationStage``: the row's image references are the keyframes (one latent frame at
    the target's latent size each, clean), packed as ``build_packed_sequence`` condition rows in anchor order and
    noised with ``keyframe_condition_noise`` before the target's unit noise, all from one row-seeded generator.
    """
    from diffusers.utils.torch_utils import randn_tensor

    from fastvideo.models.schedulers.scheduling_minimax_h3 import MiniMaxH3Scheduler
    from fastvideo.pipelines.basic.minimax_h3.packing import (MINIMAX_H3_AUDIO_CHANNELS, MINIMAX_H3_KEYFRAME_NOISE_AUG,
                                                              audio_latent_num_frames, build_packed_sequence,
                                                              keyframe_condition_noise, patchify_video_latents,
                                                              video_latent_num_frames)
    from fastvideo.train.models.minimax_h3.omniref_data import PreparedRow, _tensor, prepared_references

    anchors = FL2VA_ANCHORS.get(spec.case)
    if anchors is None:
        raise ValueError(f"{spec.id}: {spec.case!r} is not an FL2VA task ({sorted(FL2VA_ANCHORS)})")
    num_frames, height, width = int(row["num_frames"]), int(row["height"]), int(row["width"])
    frames, lat_h, lat_w = video_latent_num_frames(num_frames), height // 16, width // 16
    references = prepared_references(row)
    if len(references) != len(anchors) or any(ref.media_type != "image" or (ref.num_latent_frames, ref.latent_height,
                                                                            ref.latent_width) != (1, lat_h, lat_w)
                                              for ref in references):
        raise ValueError(f"{spec.id}: {spec.case} needs {len(anchors)} canvas-size keyframe(s), got "
                         f"{[(r.media_type, r.latent_height, r.latent_width) for r in references]}")
    num_audio = audio_latent_num_frames(num_frames)
    tags = _tensor(row, "text_token_tags").round().to(torch.long)
    embeds = _tensor(row, "text_embedding").to(device=device, dtype=torch.bfloat16)[None]
    layout = build_packed_sequence(tags, frames, lat_h, lat_w, num_audio, patch_size, keyframe_anchors=anchors)
    generator = torch.Generator("cpu").manual_seed(spec.seed)
    clean = _tensor(row, "reference_video_latent_rows")
    condition = clean[:0].to(device)
    if anchors:
        noise = keyframe_condition_noise(((1, lat_h, lat_w), ) * len(anchors),
                                         patch_size,
                                         latent_channels,
                                         generator=generator,
                                         device=device)
        condition = MiniMaxH3Scheduler().scale_noise(clean.to(device), MINIMAX_H3_KEYFRAME_NOISE_AUG, noise)
    video_noise = randn_tensor((1, latent_channels, frames, lat_h, lat_w),
                               generator=generator,
                               device=device,
                               dtype=torch.float32)
    audio = randn_tensor((num_audio * MINIMAX_H3_AUDIO_CHANNELS, audio_channels),
                         generator=generator,
                         device=device,
                         dtype=torch.float32)
    video = torch.cat((condition, patchify_video_latents(video_noise, patch_size)))
    if video.shape[0] != layout.video_indices.numel() or audio.shape[0] != layout.audio_indices.numel():
        raise ValueError(f"{spec.id}: packed rows do not match the FL2VA layout")
    return PreparedRow(spec, layout, embeds, video, audio, (latent_channels, frames, lat_h, lat_w), references, clean)


class MiniMaxH3FlexFL2VAModel(MiniMaxH3FlexRef2VAModel):
    """The FL2VA adapter on ``transformer``: T2VA / I2VA / L2VA / FL2VA rows (keyframe anchors, not references).

    Rows come from ``build_flex_ref2va_rows.py --mode fl2va`` manifests (``case`` in ``FL2VA_ANCHORS``) and are
    planned, held out and streamed exactly as the Ref2VA rows (``omniref_data``).
    """

    _transformer_module_type = "transformer"
    _task = "fl2va"

    def prepare(self, spec: Any, row: dict[str, Any]) -> PreparedRow:
        return prepare_fl2va_row(spec,
                                 row,
                                 patch_size=self.patch_size,
                                 latent_channels=24,
                                 audio_channels=32,
                                 device=self.device)


__all__ = [
    "FL2VA_ANCHORS", "FLEX_TRAINABLE", "FlexContract", "prepare_fl2va_row", "MiniMaxH3FlexFL2VAModel",
    "MiniMaxH3FlexRef2VAModel", "RowRenoiseScheduler", "flex_sigmas", "renoise_seed"
]
