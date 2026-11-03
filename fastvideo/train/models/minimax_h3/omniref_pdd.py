# SPDX-License-Identifier: Apache-2.0
"""FastH3 OmniRef PDD transformer as a per-forward (rung) model for distillation.

One instance wraps ``transformer_ref`` of a FastH3 OmniRef PDD export, loaded
through the training FSDP loader with VSA-H3 attention at the trained tile
size. It exposes the eight PDD forwards of the deployed sampler one at a
time: ``rung_scope`` installs exactly what ``MiniMaxH3DenoisingStage`` sets
for forward ``k`` (VSA-H3 metadata with the reference keep rate, the fused
PDD output heads, per-row timesteps), ``forward_rung`` runs the transformer,
and ``step`` applies the scheduler's Euler step. Teacher and student load
the same way, so a student whose quantization is disabled is the teacher.

The student may carry NVFP4 QAD linears (``qad.linears``) and the sparse-FP4
attention emulator (``qad.fp4_attention``); see ``fastvideo.layers.
quantization.nvfp4_qad`` and ``fastvideo.attention.backends.fp4_vsa_qat``.
"""
from __future__ import annotations

import contextlib
import json
import re
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch

from fastvideo.distributed import get_sp_group, get_world_group
from fastvideo.forward_context import set_forward_context
from fastvideo.logger import init_logger
from fastvideo.train.models.base import ModelBase
from fastvideo.train.utils.activation_checkpoint import apply_activation_checkpointing, resolve_checkpointing_type
from fastvideo.train.utils.moduleloader import load_module_from_path

if TYPE_CHECKING:
    from fastvideo.train.models.minimax_h3.omniref_data import PreparedRow
    from fastvideo.train.utils.training_config import TrainingConfig

logger = init_logger(__name__)

# Block linears and norms; AdaLN, embeddings, the token refiner and the output heads stay frozen.
DEFAULT_TRAINABLE = (r"transformer_blocks\.\d+\.(attn\.(to_q|to_k|to_v|to_out|to_gate_compress|norm_q|norm_k)"
                     r"|norm1|norm2|ff\.(fc_in|fc_out))\.")
INFERENCE_FILE = "fastvideo_inference.json"


@dataclass(frozen=True)
class PDDContract:
    """The checkpoint's deployed sampling contract (``fastvideo_inference.json``)."""
    pdd_step_indices: tuple[int, ...]
    video_shift: float
    audio_shift: float
    vsa_sparsity: float
    vsa_tile_size: int
    vsa_ref_keep_rate: float | None

    @classmethod
    def read(cls, model_path: str) -> PDDContract:
        data = json.loads((Path(model_path) / INFERENCE_FILE).read_text())
        if data.get("model_type") != "ref2va" or "pdd_step_indices" not in data:
            raise ValueError(f"{model_path} is not a FastH3 Ref2VA PDD export")
        return cls(tuple(int(i) for i in data["pdd_step_indices"]), float(data["video_scheduler_shift"]),
                   float(data["audio_scheduler_shift"]), float(data["vsa_sparsity"]), int(data["vsa_tile_size"]),
                   data.get("vsa_ref_keep_rate"))


class MiniMaxH3OmniRefPDDModel(ModelBase):
    """Teacher or QAD student for per-forward regression on the OmniRef PDD ladder."""

    _transformer_cls_name = "MiniMaxH3Transformer3DModel"
    _transformer_module_type = "transformer_ref"

    def __init__(self,
                 *,
                 init_from: str,
                 training_config: TrainingConfig,
                 trainable: bool = True,
                 enable_gradient_checkpointing_type: str | None = None,
                 trainable_patterns: str = DEFAULT_TRAINABLE,
                 qad: dict[str, Any] | None = None,
                 data: dict[str, Any] | None = None,
                 frozen_bf16: bool = False) -> None:
        super().__init__(trainable=trainable, attention_backend="VIDEO_SPARSE_ATTN_H3")
        # Keep a frozen role's masters in bf16 (the checkpoint's dtype) instead of an fp32 upcast: compute is bf16 either
        # way, so outputs are bit-identical, and the role needs half the memory.
        self._frozen_bf16 = bool(frozen_bf16)
        if training_config.pipeline_config is None:
            raise ValueError("MiniMaxH3OmniRefPDDModel requires a resolved MiniMax-H3 pipeline config")
        if str(training_config.dit_precision) != "fp32" and trainable:
            raise ValueError("QAD needs fp32 master weights (training.dit_precision: fp32); BF16 masters "
                             "round away updates at QAD learning rates")
        self.contract = self._read_contract(init_from)
        self._check_tile()
        # Inference keeps proj_in/out, audio_proj_in/out and the time embedder in FP32; FSDP mixed precision would
        # compute them in BF16 and the harness would no longer be the deployed operating point (E1: -7.6% v_rel at
        # forward 0). They are frozen here, so they stay FP32 and replicated outside FSDP.
        training_config.pipeline_config.dit_config.uniform_parameter_dtype = False  # type: ignore[attr-defined]
        training_config.pipeline_config.dit_config.replicate_fp32_modules = True  # type: ignore[attr-defined]
        self._init_from = str(init_from)
        self.training_config = training_config
        self.qad_config = dict(qad or {})
        self.data_config = dict(data or {})
        with self._load_precision(trainable):
            self.transformer = load_module_from_path(model_path=self._init_from,
                                                     module_type=self._transformer_module_type,
                                                     training_config=training_config,
                                                     override_transformer_cls_name=self._transformer_cls_name,
                                                     attention_backend=self.attention_backend)
        checkpointing = resolve_checkpointing_type(enable_gradient_checkpointing_type, training_config)
        if trainable and checkpointing:
            self.transformer = apply_activation_checkpointing(self.transformer, checkpointing_type=checkpointing)
        self.num_trainable = self._apply_trainable(trainable_patterns if trainable else None)
        self._check_fp32_modules_frozen()
        self.patch_size = tuple(int(v) for v in self.transformer.patch_size)
        self._plan: Any = None
        self._vsa_builder: Any = None
        self.qad_linears: dict[str, Any] = {}
        self.fp4_numerics: Any = None
        self.provisional_rows = 0
        self.amax_sha256 = ""
        self.eval_rows: list[Any] = []
        self.train_rows: list[Any] = []
        if self.qad_config.get("enabled", False):
            self._install_qad()
        self.dataloader: Any = None
        self.start_step = 0

    # ------------------------------------------------------------------ setup
    def _load_precision(self, trainable: bool) -> contextlib.AbstractContextManager[None]:
        """Context for the transformer load: a frozen role may keep bf16 masters (``frozen_bf16``)."""
        if trainable or not self._frozen_bf16:
            return contextlib.nullcontext()
        return self._bf16_load()

    @contextlib.contextmanager
    def _bf16_load(self) -> Iterator[None]:
        config = self.training_config
        pipeline_config = config.pipeline_config
        saved = (config.dit_precision, pipeline_config.dit_precision)
        config.dit_precision = pipeline_config.dit_precision = "bf16"
        try:
            yield
        finally:
            config.dit_precision, pipeline_config.dit_precision = saved

    def _read_contract(self, model_path: str) -> Any:
        return PDDContract.read(model_path)

    def _check_tile(self) -> None:
        if self.contract.vsa_tile_size != 128:
            raise ValueError(f"the FP4 VSA emulator models tile 128; this export uses {self.contract.vsa_tile_size}")

    def load_row(self, spec: Any) -> PreparedRow:
        from fastvideo.train.models.minimax_h3.omniref_data import read_row
        return self.prepare(spec, read_row(spec.parquet))

    def _apply_trainable(self, pattern: str | None) -> int:
        regex = re.compile(pattern) if pattern else None
        count = 0
        for name, parameter in self.transformer.named_parameters():
            clean = name.replace("_checkpoint_wrapped_module.", "")
            wanted = regex is not None and regex.search(clean) is not None
            parameter.requires_grad_(wanted)
            count += parameter.numel() if wanted else 0
        self.transformer.train(regex is not None)
        if regex is not None and count == 0:
            raise ValueError(f"trainable_patterns {pattern!r} matched no parameters")
        return count

    def _check_fp32_modules_frozen(self) -> None:
        """FP32 boundary modules are replicated outside FSDP: training them would desynchronize the ranks."""
        selector = getattr(self.transformer, "_get_parameter_dtype", None)
        for name, parameter in self.transformer.named_parameters():
            clean = name.replace("_checkpoint_wrapped_module.", "")
            if callable(selector) and selector(clean, torch.bfloat16) == torch.float32:
                if parameter.requires_grad:
                    raise ValueError(f"{clean} is an FP32 replicated module and must stay frozen")
                if parameter.dtype != torch.float32:
                    raise ValueError(f"{clean} should be FP32 (as inference), got {parameter.dtype}")

    def _install_qad(self) -> None:
        from fastvideo.attention.backends.fp4_vsa_qat import FP4AttentionNumerics, install_fp4_vsa_attention
        from fastvideo.layers.quantization.nvfp4_qad import NVFP4QADPlan, install_nvfp4_qad, load_amax_table

        linears = dict(self.qad_config.get("linears") or {})
        if linears.get("enabled", True):
            amax_path = linears.get("amax_json")
            table = load_amax_table(amax_path) if amax_path else None
            if amax_path:
                self.amax_sha256 = _pin_amax(amax_path, linears.get("amax_sha256"),
                                             self.training_config.checkpoint.output_dir)
            plan = NVFP4QADPlan(quantize_attention=bool(linears.get("attention", True)),
                                quantize_gate=bool(linears.get("gate", True)),
                                quantize_ffn=bool(linears.get("ffn", True)),
                                skip_blocks=tuple(int(b) for b in linears.get("skip_blocks", ())))
            # Smoke tests only: without Stage A's table, max-calibrate the FFN scales on a few rows at step 0.
            self.provisional_rows = int(linears.get("provisional_calibration_rows", 0))
            if table is None and self.provisional_rows <= 0:
                raise ValueError("qad.linears.amax_json (Stage A calibration) is required")
            self.qad_linears = install_nvfp4_qad(self.transformer, table, plan, allow_uncalibrated=table is None)
        attention = dict(self.qad_config.get("fp4_attention") or {})
        if attention.get("enabled", True):
            floor = attention.get("first_block_max_floor")
            self.fp4_numerics = FP4AttentionNumerics(quantize=True,
                                                     two_level_p=bool(attention.get("two_level_p", False)),
                                                     smooth_k=bool(attention.get("smooth_k", False)),
                                                     first_block_max_floor=None if floor is None else float(floor))
        elif self._trainable:
            # Tile 128 has no other grad-capable kernel: train through the emulator with exact softmax.
            self.fp4_numerics = FP4AttentionNumerics(quantize=False)
        install_fp4_vsa_attention(self.transformer, self.fp4_numerics)
        logger.info("OmniRef QAD student: %d NVFP4 QAD linears, attention %s, %.2fB trainable parameters",
                    len(self.qad_linears), self.fp4_numerics, self.num_trainable / 1e9)

    @contextlib.contextmanager
    def quantization(self, *, linears: bool = True, attention: bool = True) -> Iterator[None]:
        """Temporarily enable/disable the student's fake quantization (all off = the teacher's numerics)."""
        from fastvideo.attention.backends.fp4_vsa_qat import install_fp4_vsa_attention
        from fastvideo.layers.quantization.nvfp4_qad import set_nvfp4_qad_enabled

        set_nvfp4_qad_enabled(self.qad_linears, linears)
        if not attention:
            install_fp4_vsa_attention(self.transformer, None)
        try:
            yield
        finally:
            set_nvfp4_qad_enabled(self.qad_linears, True)
            install_fp4_vsa_attention(self.transformer, self.fp4_numerics)

    def init_preprocessors(self, training_config: TrainingConfig) -> None:
        from fastvideo.train.models.minimax_h3.omniref_data import (OMNIREF_CASES, RowStream, interleave_by_group,
                                                                    limit_frames, load_eval_rows, load_manifest_groups,
                                                                    select_rows, top_up_eval_rows)
        cfg = self.data_config
        cases = tuple(cfg.get("cases", OMNIREF_CASES))
        resolutions = tuple(cfg.get("eval_resolutions", ("480p", )))
        eval_per_group = cfg.get("eval_per_group", 2)
        eval_max_frames = int(cfg.get("eval_max_frames", 0))
        eval_rows = load_eval_rows(cfg["eval_manifest"], cases, resolutions, eval_per_group, eval_max_frames)
        groups = load_manifest_groups(list(cfg["manifests"]), cases, int(cfg.get("max_frames", 243)))
        missing = sorted(set(cases) - {case for case, _ in groups})
        if missing:
            logger.warning("OmniRef cases with no rows in the given manifests: %s", missing)
        if isinstance(eval_per_group, dict):
            # More held-out rows than the precomputed eval manifest holds: draw them before the training plan,
            # under the same eval length cap as the manifest rows.
            eval_groups = limit_frames({
                key: entries
                for key, entries in groups.items() if key[1] in resolutions
            }, eval_max_frames)
            eval_rows = top_up_eval_rows(eval_rows, eval_groups, int(cfg.get("eval_seed", 20261008)),
                                         dict(eval_per_group))
        else:
            eval_rows = interleave_by_group(eval_rows)
        plan = select_rows(groups, int(cfg.get("seed", 20261007)), dict(cfg.get("per_group", {"480p": 400})),
                           frozenset(row.source for row in eval_rows))
        if int(cfg.get("max_train_rows", 0)) > 0:  # smoke tests: overfit a few rows
            plan = plan[:int(cfg["max_train_rows"])]
        world = get_world_group()
        sp_size = int(training_config.distributed.sp_size or 1)
        self.eval_rows = eval_rows
        self.train_rows = plan
        self.dataloader = RowStream(plan,
                                    dp_rank=world.rank // sp_size,
                                    dp_size=world.world_size // sp_size,
                                    seed=int(training_config.data.seed or 0))
        logger.info("OmniRef QAD rows: %d train (%s), %d held-out eval", len(plan), dict(sorted(_count(plan).items())),
                    len(eval_rows))

    # ------------------------------------------------------------------ PDD ladder
    @property
    def plan(self) -> Any:
        if self._plan is None:
            from fastvideo.layers.pdd import PDDModalitySchedule, build_pdd_sampling_plan
            schedules = {
                "video": PDDModalitySchedule(shift=self.contract.video_shift),
                "audio": PDDModalitySchedule(shift=self.contract.audio_shift)
            }
            self._plan = build_pdd_sampling_plan(self.contract.pdd_step_indices, schedules, device=self.device)
        return self._plan

    @property
    def num_rungs(self) -> int:
        return int(self.plan.num_steps)

    def schedulers(self) -> tuple[Any, Any]:
        """Fresh video/audio schedulers on the PDD node sigmas (as the denoising stage sets them)."""
        from fastvideo.models.schedulers.scheduling_minimax_h3 import MiniMaxH3Scheduler
        video = MiniMaxH3Scheduler(shift=self.contract.video_shift)
        audio = MiniMaxH3Scheduler(shift=self.contract.audio_shift)
        video.set_timesteps(sigmas=self.plan.node_sigmas["video"].to(torch.float32), device=self.device)
        audio.set_timesteps(sigmas=self.plan.node_sigmas["audio"].to(torch.float32), device=self.device)
        return video, audio

    def prepare(self, spec: Any, row: dict[str, Any]) -> PreparedRow:
        from fastvideo.train.models.minimax_h3.omniref_data import prepare_row
        start = (float(self.plan.node_sigmas["video"][0]), float(self.plan.node_sigmas["audio"][0]))
        return prepare_row(spec,
                           row,
                           patch_size=self.patch_size,
                           latent_channels=24,
                           audio_channels=32,
                           pdd_start_sigmas=start,
                           device=self.device)

    def _metadata(self, prepared: PreparedRow, rung: int) -> Any:
        from fastvideo.attention.backends.video_sparse_attn_h3 import MiniMaxH3VSAMetadataBuilder
        from fastvideo.pipelines.basic.minimax_h3.stages.minimax_h3_denoising import (_h3_vsa_ref2va_segments,
                                                                                      _h3_vsa_single_region_segments)
        if self._vsa_builder is None:
            self._vsa_builder = MiniMaxH3VSAMetadataBuilder()
        keep = self.contract.vsa_ref_keep_rate
        segments = (_h3_vsa_single_region_segments(prepared.layout, self.patch_size)
                    if keep is None else _h3_vsa_ref2va_segments(prepared.layout, self.patch_size))
        return self._vsa_builder.build(current_timestep=rung,
                                       patch_size=self.patch_size,
                                       VSA_sparsity=self.contract.vsa_sparsity,
                                       packed_segments=segments,
                                       device=self.device,
                                       exempt=True,
                                       dense_layers=(),
                                       tile_size=self.contract.vsa_tile_size,
                                       ref_keep_rate=keep)

    @contextlib.contextmanager
    def rung_scope(self, prepared: PreparedRow, rung: int, timesteps: tuple[Any, Any]) -> Iterator[Any]:
        """Forward context of PDD forward ``rung``; keep backward inside it (activation recompute reads it)."""
        from fastvideo.pipelines.basic.minimax_h3.packing import MINIMAX_H3_KEYFRAME_NOISE_AUG, build_row_timesteps
        video_t, audio_t = float(timesteps[0][rung].item()), float(timesteps[1][rung].item())
        unique, inverse = build_row_timesteps(prepared.layout,
                                              video_timestep=video_t,
                                              audio_timestep=audio_t,
                                              condition_video_timestep=max(video_t, MINIMAX_H3_KEYFRAME_NOISE_AUG),
                                              condition_audio_timestep=1.0)
        start, end = self.plan.block(rung)
        fusion = self.transformer.fuse_pdd_block(start, end, self.plan.integration_weights, torch.float32)
        with fusion, set_forward_context(current_timestep=rung, attn_metadata=self._metadata(prepared, rung)):
            yield unique.to(self.device), inverse.to(self.device)

    def forward_rung(self, prepared: PreparedRow, video: torch.Tensor, audio: torch.Tensor,
                     row_timesteps: tuple[torch.Tensor, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Packed velocities ``([Lv, Cv], [La, Ca])`` for one forward; call inside ``rung_scope``."""
        layout = prepared.layout
        device = self.device
        # No autocast: like inference, the transformer casts its inputs to each weight's dtype (BF16 under
        # FSDP mixed precision), so Q/K/V reach attention in BF16 exactly as on the deployed path. Autocast
        # would run RMSNorm/RoPE in fp32 and change what the FP4 quantizers see.
        video_v, audio_v = self.transformer(hidden_states=video[None],
                                            audio_hidden_states=audio[None],
                                            encoder_hidden_states=prepared.prompt_embeds,
                                            timestep=row_timesteps[0],
                                            timestep_indices=row_timesteps[1],
                                            token_tags=layout.token_tags.to(device),
                                            position_ids=layout.position_ids.to(device),
                                            video_indices=layout.video_indices.to(device),
                                            audio_indices=layout.audio_indices.to(device),
                                            text_indices=layout.text_indices.to(device))
        return video_v[0], audio_v[0]

    # ------------------------------------------------------------------ ModelBase contract (unused paths)
    def prepare_batch(self, raw_batch: dict[str, Any], **kwargs: Any) -> Any:  # noqa: D102
        raise NotImplementedError("use prepare() + rung_scope() + forward_rung()")

    def add_noise(self, clean_latents: torch.Tensor, noise: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("PDD rungs start from the generator's noise; see prepare()")

    def predict_noise(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError("use forward_rung()")

    def backward(self, loss: torch.Tensor, ctx: Any, *, grad_accum_rounds: int) -> None:
        raise NotImplementedError("the PDD QAD method backpropagates inside rung_scope()")

    def sp_group(self) -> Any:
        return get_sp_group()


def _pin_amax(path: str, expected_sha256: str | None, output_dir: str) -> str:
    """Check the calibration table's sha256 (when pinned) and keep a copy next to the run's checkpoints.

    The static FFN scales are fixed buffers that QAD learns around, so a run must never pick up a table that was
    rewritten in place.
    """
    import hashlib
    import shutil

    digest = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    if expected_sha256 and digest != expected_sha256:
        raise ValueError(f"{path} has sha256 {digest}, but the config pins {expected_sha256}")
    if output_dir and get_world_group().rank == 0:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, Path(output_dir) / f"amax-{digest[:12]}.json")
    logger.info("NVFP4 calibration table %s sha256 %s", path, digest)
    return digest


def _count(rows: list[Any]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        key = f"{row.case}/{row.resolution}"
        counts[key] = counts.get(key, 0) + 1
    return counts


__all__ = ["DEFAULT_TRAINABLE", "MiniMaxH3OmniRefPDDModel", "PDDContract"]
