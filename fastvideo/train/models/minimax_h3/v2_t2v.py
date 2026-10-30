# SPDX-License-Identifier: Apache-2.0
"""FastH3 V2 (8-step DMD, text-to-video+audio) as a per-forward model for NVFP4 QAD with optional token compression.

Same contract as ``omniref_pdd.MiniMaxH3OmniRefPDDModel`` so ``OmniRefPDDQADMethod`` runs unchanged: eight forwards
("rungs", one per DMD step), ``rung_scope`` / ``forward_rung`` / ``schedulers`` / ``prepare`` / ``quantization``.

Differences from OmniRef: plain ``transformer`` module (no PDD head fusion), V2 sampling contract (video shift 10,
audio shift 3, ``dmd_denoising_steps``), VSA tile 64 with sparsity 0.8 (BF16 Triton kernels by default;
``qad.fp4_attention`` installs the same FP4 emulator as the shared recipe), text-only rows (``v2_data``), and the
optional middle-block spatial token compression (``v2_compression``).

``quantization(linears=, attention=)``: ``linears`` toggles the NVFP4 QAD linears; ``attention=False`` switches off
the FP4 attention emulator and the token compression, so ``quantization(False, False)`` is the BF16 teacher.
"""
from __future__ import annotations

import contextlib
import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch

from fastvideo.distributed import get_sp_world_size, get_world_group, model_parallel_is_initialized
from fastvideo.forward_context import set_forward_context
from fastvideo.logger import init_logger
from fastvideo.train.models.minimax_h3.omniref_pdd import DEFAULT_TRAINABLE, MiniMaxH3OmniRefPDDModel, _pin_amax
from fastvideo.train.models.minimax_h3.v2_compression import (CompressionSpec, TokenCompression, block_flops_share,
                                                              compressed_grid)

if TYPE_CHECKING:
    from fastvideo.train.models.minimax_h3.omniref_data import PreparedRow
    from fastvideo.train.models.minimax_h3.v2_data import V2RowSpec
    from fastvideo.train.utils.training_config import TrainingConfig

logger = init_logger(__name__)
INFERENCE_FILE = "fastvideo_inference.json"


@dataclass(frozen=True)
class V2Contract:
    """The checkpoint's deployed sampling contract (``fastvideo_inference.json``)."""
    dmd_steps: tuple[int, ...]
    video_shift: float
    audio_shift: float
    vsa_sparsity: float
    vsa_tile_size: int

    @classmethod
    def read(cls, model_path: str) -> V2Contract:
        data = json.loads((Path(model_path) / INFERENCE_FILE).read_text())
        if "dmd_denoising_steps" not in data or data.get("task") != "t2av":
            raise ValueError(f"{model_path} is not a FastH3 T2AV DMD export")
        return cls(tuple(int(i) for i in data["dmd_denoising_steps"]), float(data["video_scheduler_shift"]),
                   float(data["audio_scheduler_shift"]), float(data["vsa_sparsity"]), int(data["vsa_tile_size"]))


class MiniMaxH3V2T2VModel(MiniMaxH3OmniRefPDDModel):
    """Teacher (bf16 V2) or NVFP4 QAD student (+ token compression) on the 8-step V2 ladder."""

    _transformer_module_type = "transformer"

    def __init__(self,
                 *,
                 init_from: str,
                 training_config: TrainingConfig,
                 trainable: bool = True,
                 enable_gradient_checkpointing_type: str | None = None,
                 trainable_patterns: str = DEFAULT_TRAINABLE,
                 qad: dict[str, Any] | None = None,
                 compression: dict[str, Any] | None = None,
                 frozen_bf16: bool = False,
                 data: dict[str, Any] | None = None) -> None:
        self._compression_cfg = dict(compression or {})
        super().__init__(init_from=init_from,
                         training_config=training_config,
                         trainable=trainable,
                         enable_gradient_checkpointing_type=enable_gradient_checkpointing_type,
                         trainable_patterns=trainable_patterns,
                         qad=qad,
                         data=data,
                         frozen_bf16=frozen_bf16)
        self._vsa_builder: Any = self._vsa_builder
        self.compression: TokenCompression | None = None
        spec = CompressionSpec.from_config(self._compression_cfg)
        if spec is not None:
            if model_parallel_is_initialized() and get_sp_world_size() > 1:
                raise ValueError("token compression needs sp_size 1")
            self.compression = TokenCompression(self.transformer, spec)
            logger.info("V2 token compression: blocks [%d, %d) mode %s", spec.start, spec.end, spec.mode)
        self._compression_on = True

    # ------------------------------------------------------------------ construction hooks
    def _read_contract(self, model_path: str) -> Any:
        return V2Contract.read(model_path)

    def _check_tile(self) -> None:
        if self.contract.vsa_tile_size != 64:
            raise ValueError(f"the V2 QAD harness expects VSA tile 64, got {self.contract.vsa_tile_size}")

    def _install_qad(self) -> None:
        """NVFP4 linears plus the optional FP4 VSA attention emulator.

        With ``qad.fp4_attention.enabled`` the student trains against the same FP4 attention numerics
        used by the fast inference lane; without it V2 keeps the native bf16 VSA path.
        """
        from fastvideo.attention.backends.fp4_vsa_qat import FP4AttentionNumerics, install_fp4_vsa_attention
        from fastvideo.layers.quantization.nvfp4_qad import NVFP4QADPlan, install_nvfp4_qad, load_amax_table

        linears = dict(self.qad_config.get("linears") or {})
        if not linears.get("enabled", True):
            return
        amax_path = linears.get("amax_json")
        if not amax_path:
            raise ValueError("qad.linears.amax_json (the Consumer export's static FFN scales) is required")
        table = load_amax_table(amax_path)
        self.amax_sha256 = _pin_amax(amax_path, linears.get("amax_sha256"), self.training_config.checkpoint.output_dir)
        plan = NVFP4QADPlan(quantize_attention=bool(linears.get("attention", True)),
                            quantize_gate=bool(linears.get("gate", True)),
                            quantize_ffn=bool(linears.get("ffn", True)),
                            skip_blocks=tuple(int(b) for b in linears.get("skip_blocks", ())))
        self.qad_linears = install_nvfp4_qad(self.transformer, table, plan)
        self.fp4_numerics = None
        attention = dict(self.qad_config.get("fp4_attention") or {})
        if attention.get("enabled", False):
            floor = attention.get("first_block_max_floor")
            self.fp4_numerics = FP4AttentionNumerics(quantize=True,
                                                     two_level_p=bool(attention.get("two_level_p", False)),
                                                     smooth_k=bool(attention.get("smooth_k", False)),
                                                     first_block_max_floor=None if floor is None else float(floor))
            install_fp4_vsa_attention(self.transformer, self.fp4_numerics)
        logger.info("V2 QAD student: %d NVFP4 QAD linears, attention %s, %.2fB trainable parameters",
                    len(self.qad_linears), self.fp4_numerics, self.num_trainable / 1e9)

    @contextlib.contextmanager
    def quantization(self, *, linears: bool = True, attention: bool = True) -> Iterator[None]:
        from fastvideo.attention.backends.fp4_vsa_qat import install_fp4_vsa_attention
        from fastvideo.layers.quantization.nvfp4_qad import set_nvfp4_qad_enabled

        set_nvfp4_qad_enabled(self.qad_linears, linears)
        fp4 = getattr(self, "fp4_numerics", None)
        if fp4 is not None and not attention:
            install_fp4_vsa_attention(self.transformer, None)
        previous = self._compression_on
        self._compression_on = attention
        try:
            yield
        finally:
            set_nvfp4_qad_enabled(self.qad_linears, True)
            if fp4 is not None:
                install_fp4_vsa_attention(self.transformer, fp4)
            self._compression_on = previous

    # ------------------------------------------------------------------ data
    def init_preprocessors(self, training_config: TrainingConfig) -> None:
        from fastvideo.train.models.minimax_h3.omniref_data import RowStream
        from fastvideo.train.models.minimax_h3.v2_data import load_row_specs

        cfg = self.data_config
        size = (int(cfg.get("width", 832)), int(cfg.get("height", 480)), int(cfg.get("num_frames", 124)))
        eval_rows = load_row_specs(cfg["eval_manifest"], *size)
        eval_ids = {row.id for row in eval_rows}
        train_rows = [row for row in load_row_specs(cfg["train_manifest"], *size) if row.id not in eval_ids]
        if int(cfg.get("max_train_rows", 0)) > 0:
            train_rows = train_rows[:int(cfg["max_train_rows"])]
        if int(cfg.get("max_eval_rows", 0)) > 0:
            eval_rows = eval_rows[:int(cfg["max_eval_rows"])]
        world = get_world_group()
        sp_size = int(training_config.distributed.sp_size or 1)
        self.eval_rows, self.train_rows = eval_rows, train_rows
        self.dataloader = RowStream(train_rows,
                                    dp_rank=world.rank // sp_size,
                                    dp_size=world.world_size // sp_size,
                                    seed=int(training_config.data.seed or 0))
        logger.info("V2 QAD rows: %d train, %d held-out eval", len(train_rows), len(eval_rows))

    def load_row(self, spec: V2RowSpec) -> PreparedRow:
        from fastvideo.train.models.minimax_h3.v2_data import prepare_row
        return prepare_row(spec, patch_size=self.patch_size, latent_channels=24, audio_channels=32, device=self.device)

    # ------------------------------------------------------------------ DMD ladder
    @property
    def plan(self) -> Any:  # pragma: no cover - PDD-only
        raise AttributeError("V2 has no PDD plan")

    @property
    def num_rungs(self) -> int:
        return len(self.contract.dmd_steps)

    def schedulers(self) -> tuple[Any, Any]:
        """Fresh video/audio schedulers on the DMD sigma grid, as ``MiniMaxH3DenoisingStage._set_dmd_schedule``."""
        from fastvideo.models.schedulers.scheduling_minimax_h3 import MiniMaxH3Scheduler
        base = torch.tensor([step / 1000.0 for step in self.contract.dmd_steps] + [0.0], dtype=torch.float32)
        out = []
        for shift in (self.contract.video_shift, self.contract.audio_shift):
            scheduler = MiniMaxH3Scheduler(shift=shift)
            scheduler.set_timesteps(sigmas=shift * base / (1 + (shift - 1) * base), device=self.device)
            out.append(scheduler)
        return out[0], out[1]

    def _token_grid(self, prepared: PreparedRow) -> tuple[int, int, int]:
        layout = self.patch_size
        return (prepared.layout.num_video_latent_frames // layout[0], prepared.layout.latent_height // layout[1],
                prepared.layout.latent_width // layout[2])

    def _metadata(self, prepared: PreparedRow, rung: int, grid: tuple[int, int, int] | None = None) -> Any:
        """VSA-H3 metadata for the full video grid, or for a (compressed) token ``grid``."""
        from fastvideo.attention.backends.video_sparse_attn_h3 import MiniMaxH3VSAMetadataBuilder
        from fastvideo.pipelines.basic.minimax_h3.stages.minimax_h3_denoising import _h3_vsa_single_region_segments
        if self._vsa_builder is None:
            self._vsa_builder = MiniMaxH3VSAMetadataBuilder()
        layout = prepared.layout
        if grid is None:
            segments = _h3_vsa_single_region_segments(layout, self.patch_size)
        else:
            p = self.patch_size
            segments = (int(layout.text_indices.numel()), int(layout.audio_indices.numel()),
                        (grid[0] * p[0], grid[1] * p[1], grid[2] * p[2]))
        return self._vsa_builder.build(current_timestep=rung,
                                       patch_size=self.patch_size,
                                       VSA_sparsity=self.contract.vsa_sparsity,
                                       packed_segments=segments,
                                       device=self.device,
                                       exempt=True,
                                       dense_layers=(),
                                       tile_size=self.contract.vsa_tile_size)

    @contextlib.contextmanager
    def rung_scope(self, prepared: PreparedRow, rung: int, timesteps: tuple[Any, Any]) -> Iterator[Any]:
        """Forward context of DMD forward ``rung``; keep backward inside it (activation recompute reads it)."""
        from fastvideo.pipelines.basic.minimax_h3.packing import MINIMAX_H3_KEYFRAME_NOISE_AUG, build_row_timesteps
        video_t, audio_t = float(timesteps[0][rung].item()), float(timesteps[1][rung].item())
        unique, inverse = build_row_timesteps(prepared.layout,
                                              video_timestep=video_t,
                                              audio_timestep=audio_t,
                                              condition_video_timestep=max(video_t, MINIMAX_H3_KEYFRAME_NOISE_AUG),
                                              condition_audio_timestep=1.0)
        with contextlib.ExitStack() as stack:
            stack.enter_context(set_forward_context(current_timestep=rung, attn_metadata=self._metadata(prepared,
                                                                                                        rung)))
            first_steps = self.compression.spec.first_steps if self.compression is not None else None
            if self.compression is not None and self._compression_on and (first_steps is None or rung < first_steps):
                grid = self._token_grid(prepared)
                sub = compressed_grid(self.compression.spec.mode, grid)
                stack.enter_context(self.compression.scope(grid, self._metadata(prepared, rung, sub)))
            yield unique.to(self.device), inverse.to(self.device)

    def set_compression(self, spec: CompressionSpec | None) -> None:
        """Replace the token-compression spec (probes and sweeps); ``None`` removes it."""
        if self.compression is not None:
            self.compression.remove()
        self.compression = None if spec is None else TokenCompression(self.transformer, spec)

    def compression_summary(self, prepared: PreparedRow) -> dict[str, float]:
        """Token and FLOP accounting of the configured compression for one row."""
        spec = self.compression.spec if self.compression else None
        grid = self._token_grid(prepared)
        other = prepared.layout.sequence_length - grid[0] * grid[1] * grid[2]
        out = block_flops_share(spec, grid, len(self.transformer.transformer_blocks), other)
        out["video_tokens"] = float(grid[0] * grid[1] * grid[2])
        out["sequence_tokens"] = float(prepared.layout.sequence_length)
        return out

    # ------------------------------------------------------------------ PDD-only paths of the parent
    def forward_rung(self, prepared: PreparedRow, video: torch.Tensor, audio: torch.Tensor,
                     row_timesteps: tuple[torch.Tensor, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        return super().forward_rung(prepared, video, audio, row_timesteps)


__all__ = ["MiniMaxH3V2T2VModel", "V2Contract"]
