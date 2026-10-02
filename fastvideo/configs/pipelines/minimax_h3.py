# SPDX-License-Identifier: Apache-2.0
"""Pipeline configuration for MiniMax H3 joint video/audio generation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from fastvideo.configs.models import EncoderConfig, VAEConfig
from fastvideo.configs.models.dits.minimax_h3 import MiniMaxH3Config
from fastvideo.configs.models.encoders.minimax_h3_qwen3_vl import MiniMaxH3Qwen3VLConfig
from fastvideo.configs.models.vaes.minimax_h3_audio import MiniMaxH3AudioVAEConfig
from fastvideo.configs.models.vaes.minimax_h3_video import MiniMaxH3VideoVAEConfig
from fastvideo.configs.pipelines.base import PipelineConfig
from fastvideo.logger import init_logger
from fastvideo.utils import read_optional_model_json

if TYPE_CHECKING:
    from fastvideo.api.sampling_param import SamplingParam
    from fastvideo.api.schema import GenerationRequest
    from fastvideo.fastvideo_args import FastVideoArgs

logger = init_logger(__name__)

# The checkpoint file that records how a distilled FastH3 export must be sampled.
FASTH3_INFERENCE_FILE = "fastvideo_inference.json"
FASTH3_INFERENCE_SCHEMA = "fasth3-inference-contract-v1"
# The fields a FastH3 Ref2VA PDD export writes to FASTH3_INFERENCE_FILE. Missing
# and unknown fields are both errors, so no field is silently ignored.
_PDD_INFERENCE_FIELDS = frozenset({
    "schema_version",
    "schema",
    "model_type",
    "transformer_component",
    "conditioning",
    "base_model_revision",
    "pdd_steps",
    "pdd_step_indices",
    "num_inference_steps",
    "transformer_forwards",
    "grid_max_t",
    "video_scheduler_shift",
    "audio_scheduler_shift",
    "guidance_scale",
    "attention_backend",
    "vsa_sparsity",
    "vsa_tile_size",
    "vsa_ref_policy",
    "vsa_ref_keep_rate",
})
# Fields that repeat a value FastVideo fixes in code: the Ref2VA pipeline, its
# conditioning, sampling without classifier-free guidance, and the per-reference
# sparse regions that only VIDEO_SPARSE_ATTN_H3 builds.
_PDD_FIXED_FIELDS: dict[str, Any] = {
    "schema_version": FASTH3_INFERENCE_SCHEMA,
    "schema": FASTH3_INFERENCE_SCHEMA,
    "model_type": "ref2va",
    "transformer_component": "transformer_ref",
    "conditioning": "fixed_ordered_references_target_only_flow",
    "guidance_scale": 1.0,
    "attention_backend": "VIDEO_SPARSE_ATTN_H3",
    "vsa_ref_policy": "p2_multi_region",
}

# The base snapshot an export was distilled against: "hf://<repo id>@<revision>".
_BASE_MODEL_REVISION_PREFIX = "hf://"


def parse_base_model_revision(value: Any) -> tuple[str, str]:
    """Split a contract's ``base_model_revision``, ``hf://<repo id>@<revision>``, into repo id and revision."""
    if isinstance(value, str) and value.startswith(_BASE_MODEL_REVISION_PREFIX):
        repo, separator, revision = value[len(_BASE_MODEL_REVISION_PREFIX):].partition("@")
        if separator and repo.strip() and revision.strip() and "@" not in revision:
            return repo, revision
    raise ValueError(f"FastH3 base_model_revision={value!r} must be hf://<repo id>@<revision>.")


def _require_fraction(name: str, value: Any, *, allow_zero: bool) -> None:
    """Raise unless *value* is a real number in [0, 1) (``allow_zero``) or in (0, 1)."""
    is_real = isinstance(value, int | float) and not isinstance(value, bool)
    if not (is_real and (value >= 0.0 if allow_zero else value > 0.0) and value < 1.0):
        raise ValueError(f"{name} must be in {'[0, 1)' if allow_zero else '(0, 1)'}, got {value!r}.")


def _read_pdd_inference_file(model_path: str, revision: str | None) -> dict[str, Any] | None:
    """Read and validate a FastH3 Ref2VA PDD checkpoint's fastvideo_inference.json; None for other checkpoints.

    The file must carry exactly the PDD fields. Fields that FastVideo fixes in
    code must hold that value, and fields that repeat a value stored in another
    checkpoint file must equal that file: ``pdd_steps`` the transformer config,
    the shifts the scheduler configs, and the step counts the partition, which
    must increase strictly from 0 to ``pdd_steps``.
    """
    inference_file = read_optional_model_json(model_path, FASTH3_INFERENCE_FILE, revision)
    if inference_file is None or "pdd_steps" not in inference_file:
        return None
    source = f"{model_path}/{FASTH3_INFERENCE_FILE}"
    missing = sorted(_PDD_INFERENCE_FIELDS - inference_file.keys())
    unknown = sorted(inference_file.keys() - _PDD_INFERENCE_FIELDS)
    if missing or unknown:
        raise ValueError(f"{source} must carry exactly the FastH3 PDD fields; missing {missing}, unknown {unknown}.")
    for key, expected in _PDD_FIXED_FIELDS.items():
        if inference_file[key] != expected:
            raise ValueError(f"{source} {key}={inference_file[key]!r} is unsupported; expected {expected!r}.")
    parse_base_model_revision(inference_file["base_model_revision"])
    from fastvideo.layers.pdd import PDD_GRID_MAX_T
    if inference_file["grid_max_t"] != PDD_GRID_MAX_T:
        raise ValueError(f"{source} grid_max_t={inference_file['grid_max_t']!r} is unsupported; the PDD fine grid "
                         f"ends at {PDD_GRID_MAX_T}.")

    owner_files = {}
    for owner in ("transformer_ref/config.json", "scheduler/scheduler_config.json",
                  "audio_scheduler/scheduler_config.json"):
        owner_files[owner] = read_optional_model_json(model_path, owner, revision)
        if owner_files[owner] is None:
            raise ValueError(f"{model_path} has {FASTH3_INFERENCE_FILE} but no {owner}.")
    pdd_steps = owner_files["transformer_ref/config.json"].get("pdd_steps")
    if inference_file["pdd_steps"] != pdd_steps:
        raise ValueError(f"{source} pdd_steps={inference_file['pdd_steps']!r} disagrees with "
                         f"transformer_ref/config.json pdd_steps={pdd_steps!r}.")
    for key, owner in (("video_scheduler_shift", "scheduler/scheduler_config.json"),
                       ("audio_scheduler_shift", "audio_scheduler/scheduler_config.json")):
        shift = owner_files[owner].get("shift")
        if inference_file[key] != shift:
            raise ValueError(f"{source} {key}={inference_file[key]!r} disagrees with {owner} shift={shift!r}.")
    indices = inference_file["pdd_step_indices"]
    if (not isinstance(indices, list) or len(indices) < 2 or any(type(index) is not int for index in indices)
            or indices[0] != 0 or indices[-1] != pdd_steps
            or any(left >= right for left, right in zip(indices, indices[1:], strict=False))):
        raise ValueError(f"{source} pdd_step_indices={indices!r} must increase strictly from 0 to "
                         f"pdd_steps={pdd_steps}.")
    for key in ("num_inference_steps", "transformer_forwards"):
        if inference_file[key] != len(indices) - 1:
            raise ValueError(f"{source} {key}={inference_file[key]!r} disagrees with the {len(indices) - 1} blocks "
                             "of pdd_step_indices.")

    # Trained VSA settings. A run's override of a tunable one is checked where it is applied.
    from fastvideo.attention.backends.video_sparse_attn_h3 import VSA_H3_TILE_SHAPES
    tile_size = inference_file["vsa_tile_size"]
    if type(tile_size) is not int or tile_size not in VSA_H3_TILE_SHAPES:
        raise ValueError(f"{source} vsa_tile_size={tile_size!r} must be one of {sorted(VSA_H3_TILE_SHAPES)}.")
    _require_fraction(f"{source} vsa_sparsity", inference_file["vsa_sparsity"], allow_zero=True)
    # A keep rate of 1 would leave reference-video attention dense.
    _require_fraction(f"{source} vsa_ref_keep_rate", inference_file["vsa_ref_keep_rate"], allow_zero=False)
    return inference_file


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
    # the fused blocks, one transformer forward each. resolve_checkpoint_settings
    # sets it from the checkpoint's fastvideo_inference.json; None means the
    # checkpoint is not a PDD student.
    pdd_step_indices: tuple[int, ...] | None = None
    # Fraction of each reference video's VSA tiles that every video query keeps,
    # for PDD students. None takes the checkpoint's trained value; another value
    # overrides it with a warning.
    vsa_ref_keep_rate: float | None = None

    def check_pipeline_config(self) -> None:
        super().check_pipeline_config()
        if self.flow_shift is not None:
            raise ValueError("MiniMax-H3 uses separate checkpoint-defined video/audio scheduler shifts; "
                             "flow_shift must remain unset.")

    def resolve_checkpoint_settings(self, fastvideo_args: FastVideoArgs) -> None:
        """Apply a FastH3 Ref2VA PDD checkpoint's fastvideo_inference.json to this run.

        The file is the one source of the PDD partition and the trained VSA
        settings, and a setting the run leaves unset takes the file's value.
        The partition, attention backend and tile size are fixed by training,
        so a different run value raises; VSA sparsity and the reference keep
        rate may differ, with a warning. For checkpoints without PDD fields
        (base MiniMax-H3, DMD exports) this method sets nothing.
        """
        inference_file = _read_pdd_inference_file(fastvideo_args.model_path, fastvideo_args.revision)
        if inference_file is None:
            for name in ("pdd_step_indices", "vsa_ref_keep_rate"):
                if getattr(self, name) is not None:
                    raise ValueError(f"{name} applies only to FastH3 PDD checkpoints, whose {FASTH3_INFERENCE_FILE} "
                                     f"sets it; {fastvideo_args.model_path} has no PDD fields.")
            return

        # Training-fixed settings: an unset run value takes the file's value; a different one raises.
        from fastvideo.attention.selector import coerce_attn_backend
        trained_backend = inference_file["attention_backend"]
        if fastvideo_args.attention_backend is None:
            fastvideo_args.attention_backend = trained_backend
        elif coerce_attn_backend(fastvideo_args.attention_backend) != coerce_attn_backend(trained_backend):
            raise ValueError(f"This FastH3 PDD checkpoint was trained with attention_backend={trained_backend}; "
                             f"this run requests {fastvideo_args.attention_backend} (argument or "
                             "FASTVIDEO_ATTENTION_BACKEND). Leave it unset or pass the trained value.")
        trained_tile = inference_file["vsa_tile_size"]
        if fastvideo_args.VSA_tile_size is None:
            fastvideo_args.VSA_tile_size = trained_tile
        elif fastvideo_args.VSA_tile_size != trained_tile:
            raise ValueError(f"This FastH3 PDD checkpoint was trained with VSA_tile_size={trained_tile}; this run "
                             f"requests {fastvideo_args.VSA_tile_size}. Leave it unset or pass the trained value.")
        indices = tuple(inference_file["pdd_step_indices"])
        if self.pdd_step_indices is not None and tuple(self.pdd_step_indices) != indices:
            raise ValueError(f"pdd_step_indices comes from {FASTH3_INFERENCE_FILE} ({list(indices)}); this run sets "
                             f"{list(self.pdd_step_indices)}.")
        if self.dmd_denoising_steps is not None:
            raise ValueError(f"{fastvideo_args.model_path} is a FastH3 PDD checkpoint; dmd_denoising_steps must be "
                             "unset.")
        self.pdd_step_indices = indices

        # Tunable settings: an unset run value takes the trained value; a different one wins, with a warning.
        trained_sparsity = inference_file["vsa_sparsity"]
        if fastvideo_args.VSA_sparsity is None:
            fastvideo_args.VSA_sparsity = trained_sparsity
        elif fastvideo_args.VSA_sparsity != trained_sparsity:
            _require_fraction("VSA_sparsity", fastvideo_args.VSA_sparsity, allow_zero=True)
            logger.warning("FastH3 PDD checkpoint was trained with VSA_sparsity=%s; this run uses %s.",
                           trained_sparsity, fastvideo_args.VSA_sparsity)
        trained_keep_rate = inference_file["vsa_ref_keep_rate"]
        if self.vsa_ref_keep_rate is None:
            self.vsa_ref_keep_rate = trained_keep_rate
        elif self.vsa_ref_keep_rate != trained_keep_rate:
            _require_fraction("vsa_ref_keep_rate", self.vsa_ref_keep_rate, allow_zero=False)
            logger.warning("FastH3 PDD checkpoint was trained with vsa_ref_keep_rate=%s; this run uses %s.",
                           trained_keep_rate, self.vsa_ref_keep_rate)

        logger.info(
            "FastH3 PDD checkpoint %s: %d fused blocks %s of a %d-interval grid; attention_backend=%s, "
            "VSA_tile_size=%s, VSA_sparsity=%s, vsa_ref_keep_rate=%s", fastvideo_args.model_path,
            len(indices) - 1, list(indices), indices[-1], fastvideo_args.attention_backend,
            fastvideo_args.VSA_tile_size, fastvideo_args.VSA_sparsity, self.vsa_ref_keep_rate)

    def apply_request_constraints(self, request: GenerationRequest, sampling_param: SamplingParam) -> SamplingParam:
        """Fix ``num_inference_steps`` to the block count of a PDD checkpoint.

        A PDD checkpoint runs exactly one transformer forward per trained fused
        block: a request that leaves ``num_inference_steps`` unset gets the block
        count, and a request that sets another count raises. Checkpoints without
        a PDD partition return ``sampling_param`` unchanged.
        """
        if self.pdd_step_indices is None:
            return sampling_param
        from fastvideo.api.compat import explicit_request_updates

        num_blocks = len(self.pdd_step_indices) - 1
        requested_steps = explicit_request_updates(request).get("num_inference_steps", num_blocks)
        if requested_steps != num_blocks:
            raise ValueError(f"This FastH3 PDD checkpoint runs exactly {num_blocks} transformer forwards; the request "
                             f"sets num_inference_steps={requested_steps}. Pass num_inference_steps={num_blocks}. Only "
                             "a request parsed from a mapping or a config file can leave it unset; a "
                             "GenerationRequest built in Python counts every field as set.")
        sampling_param.num_inference_steps = num_blocks
        return sampling_param


__all__ = ["FASTH3_INFERENCE_FILE", "FASTH3_INFERENCE_SCHEMA", "MiniMaxH3PipelineConfig", "parse_base_model_revision"]
