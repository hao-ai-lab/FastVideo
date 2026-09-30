# SPDX-License-Identifier: Apache-2.0
"""Pipeline configuration for MiniMax H3 joint video/audio generation."""

from __future__ import annotations

from dataclasses import dataclass, field

from fastvideo.configs.models import EncoderConfig, VAEConfig
from fastvideo.configs.models.dits.minimax_h3 import MiniMaxH3Config
from fastvideo.configs.models.encoders.minimax_h3_qwen3_vl import MiniMaxH3Qwen3VLConfig
from fastvideo.configs.models.vaes.minimax_h3_audio import MiniMaxH3AudioVAEConfig
from fastvideo.configs.models.vaes.minimax_h3_video import MiniMaxH3VideoVAEConfig
from fastvideo.configs.pipelines.base import PipelineConfig

# Ref2VA VSA policy that sparsifies every reference video as its own region.
MINIMAX_H3_VSA_REF_POLICY_P2 = "p2_multi_region"


@dataclass
class MiniMaxH3PipelineConfig(PipelineConfig):
    """Component and precision policy shared by T2VA, FL2VA, and Ref2VA."""

    dit_config: MiniMaxH3Config = field(default_factory=MiniMaxH3Config)
    vae_config: VAEConfig = field(default_factory=MiniMaxH3VideoVAEConfig)
    audio_vae_config: VAEConfig = field(default_factory=MiniMaxH3AudioVAEConfig)
    text_encoder_configs: tuple[EncoderConfig, ...] = field(default_factory=lambda: (MiniMaxH3Qwen3VLConfig(), ))

    flow_shift: float | None = None
    embedded_cfg_scale: float | None = None
    dit_precision: str = "bf16"
    vae_precision: str = "fp32"
    text_encoder_precisions: tuple[str, ...] = field(default_factory=lambda: ("bf16", ))
    vae_sp: bool = False
    # Parallel Decoding Distillation (PDD) students: fine-grid node indices of
    # the fused blocks one request runs (``num_inference_steps + 1`` values,
    # 0 to the transformer's ``pdd_steps``). A PDD export's
    # ``fastvideo_inference.json`` sets it; unset uses the balanced partition.
    pdd_step_indices: list[int] | None = None
    # Ref2VA reference-video sparsity under VIDEO_SPARSE_ATTN_H3.
    # "p2_multi_region" tiles each reference video as its own sparse region
    # that keeps ``vsa_ref_keep_rate`` (in (0, 1)) of its tiles (unset: the
    # target's keep rate); None keeps every conditioning row dense.
    vsa_ref_policy: str | None = None
    vsa_ref_keep_rate: float | None = None

    def check_pipeline_config(self) -> None:
        super().check_pipeline_config()
        if self.flow_shift is not None:
            raise ValueError("MiniMax-H3 uses separate checkpoint-defined video/audio scheduler shifts; "
                             "flow_shift must remain unset.")
        if self.pdd_step_indices is not None and (not isinstance(self.pdd_step_indices, list | tuple) or any(
                isinstance(index, bool) or not isinstance(index, int) for index in self.pdd_step_indices)):
            raise ValueError(f"MiniMax-H3 pdd_step_indices must be a list of ints, got {self.pdd_step_indices!r}.")
        if self.vsa_ref_policy not in (None, MINIMAX_H3_VSA_REF_POLICY_P2):
            raise ValueError(f"MiniMax-H3 vsa_ref_policy must be None or {MINIMAX_H3_VSA_REF_POLICY_P2!r}, "
                             f"got {self.vsa_ref_policy!r}.")
        keep_rate = self.vsa_ref_keep_rate
        # A keep rate of 1 would leave reference-video attention dense.
        if keep_rate is not None and (isinstance(keep_rate, bool) or not isinstance(keep_rate, int | float)
                                      or not 0.0 < float(keep_rate) < 1.0):
            raise ValueError(f"MiniMax-H3 vsa_ref_keep_rate must be in (0, 1), got {keep_rate!r}.")
        if keep_rate is not None and self.vsa_ref_policy is None:
            raise ValueError("MiniMax-H3 vsa_ref_keep_rate requires vsa_ref_policy.")


__all__ = ["MINIMAX_H3_VSA_REF_POLICY_P2", "MiniMaxH3PipelineConfig"]
