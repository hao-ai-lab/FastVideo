# SPDX-License-Identifier: Apache-2.0
"""PAN2 pipeline configuration: text-to-video, and image-to-video from a first frame.

The same configuration samples FastPAN2-style Parallel Decoding Distillation (PDD) students, whose
``fastvideo_inference.json`` fixes the sampler (fused blocks of the PDD grid, no guidance) and the sparse attention.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar

from fastvideo.configs.models import DiTConfig, EncoderConfig, VAEConfig
from fastvideo.configs.models.dits import PAN2VideoConfig
from fastvideo.configs.models.encoders import Qwen3_5TextConfig
from fastvideo.configs.pipelines.base import PipelineConfig
from fastvideo.logger import init_logger
from fastvideo.models.wan.vae_config import WanVAEConfig
from fastvideo.utils import read_optional_model_json

if TYPE_CHECKING:
    from fastvideo.api.sampling_param import SamplingParam
    from fastvideo.api.schema import GenerationRequest
    from fastvideo.fastvideo_args import FastVideoArgs

logger = init_logger(__name__)

PAN2_SYSTEM_PROMPT = "You are a helpful assistant. Generate the video with the following instructions."

# The checkpoint file that records how a distilled PAN2 export (e.g. IFM/FastPAN2) must be sampled.
PAN2_INFERENCE_FILE = "fastvideo_inference.json"
PAN2_PDD_INFERENCE_SCHEMA = "pan2-pdd-inference-contract-v1"
# The fields a PAN2 PDD export writes to PAN2_INFERENCE_FILE. Missing and unknown fields are both errors, so no field
# is silently ignored, except the optional fields below.
_PDD_INFERENCE_FIELDS = frozenset({
    "schema",
    "model_type",
    "transformer_component",
    "conditioning",
    "pdd_steps",
    "pdd_step_indices",
    "num_inference_steps",
    "transformer_forwards",
    "grid_max_t",
    "video_scheduler_shift",
    "guidance_scale",
    "precision_decoding",
    "attention_backend",
    "vsa_sparsity",
    "vsa_tile_size",
    "vsa_tile_policy",
    "vsa_compress_text_keys",
})
# Fields that files written before them lack, with the value such a file means: a student exported before VSA2 trained
# with VSA-H3, and one exported before dense VSA layers had none. A writer leaves out an empty dense-layer list.
_PDD_OPTIONAL_FIELDS: dict[str, Any] = {"vsa_variant": "vsa", "vsa_dense_layers": []}
# Fields that repeat a value FastVideo fixes in code: the PAN2 pipeline, its first-latent-frame conditioning stack
# (zeros for text-to-video), sampling without classifier-free guidance, and the fp32 fused-block integration.
_PDD_FIXED_FIELDS: dict[str, Any] = {
    "schema": PAN2_PDD_INFERENCE_SCHEMA,
    "model_type": "pan2",
    "transformer_component": "transformer",
    "conditioning": "first_latent_frame",
    "guidance_scale": 1.0,
    "precision_decoding": "float32",
}
_SPARSE_BACKEND = "VIDEO_SPARSE_ATTN_H3"


def _is_real(value: Any) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool) and math.isfinite(value)


def _validate_pdd_fields(inference_file: dict[str, Any], source: str) -> None:
    """Check every field's value; ``source`` names the file or writer in errors."""
    missing = sorted(_PDD_INFERENCE_FIELDS - inference_file.keys())
    unknown = sorted(inference_file.keys() - _PDD_INFERENCE_FIELDS - _PDD_OPTIONAL_FIELDS.keys())
    if missing or unknown:
        raise ValueError(f"{source} must carry exactly the PAN2 PDD fields; missing {missing}, unknown {unknown}.")
    for key, expected in _PDD_FIXED_FIELDS.items():
        if inference_file[key] != expected:
            raise ValueError(f"{source} {key}={inference_file[key]!r} is unsupported; expected {expected!r}.")
    from fastvideo.layers.pdd import PDD_GRID_MAX_T
    if inference_file["grid_max_t"] != PDD_GRID_MAX_T:
        raise ValueError(f"{source} grid_max_t={inference_file['grid_max_t']!r} is unsupported; the PDD fine grid "
                         f"ends at {PDD_GRID_MAX_T}.")
    pdd_steps = inference_file["pdd_steps"]
    if type(pdd_steps) is not int or pdd_steps < 2:
        raise ValueError(f"{source} pdd_steps={pdd_steps!r} must be an int >= 2.")
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
    shift = inference_file["video_scheduler_shift"]
    if not _is_real(shift) or shift <= 0:
        raise ValueError(f"{source} video_scheduler_shift={shift!r} must be positive and finite.")
    backend = inference_file["attention_backend"]
    if backend is not None:
        from fastvideo.attention.selector import backend_name_to_enum
        if not isinstance(backend, str) or backend_name_to_enum(backend) is None:
            raise ValueError(f"{source} attention_backend={backend!r} is not an attention backend name.")
    from fastvideo.attention.backends.video_sparse_attn_h3 import (VSA_H3_AXIS_POLICIES, VSA_H3_TILE_SHAPES,
                                                                   VSA_H3_VARIANTS)
    sparsity = inference_file["vsa_sparsity"]
    if not _is_real(sparsity) or not 0.0 <= sparsity < 1.0:
        raise ValueError(f"{source} vsa_sparsity={sparsity!r} must be in [0, 1).")
    tile_size = inference_file["vsa_tile_size"]
    if type(tile_size) is not int or tile_size not in VSA_H3_TILE_SHAPES:
        raise ValueError(f"{source} vsa_tile_size={tile_size!r} must be one of {sorted(VSA_H3_TILE_SHAPES)}.")
    policy = inference_file["vsa_tile_policy"]
    if not isinstance(policy, list) or len(policy) != 3 or any(axis not in VSA_H3_AXIS_POLICIES for axis in policy):
        raise ValueError(f"{source} vsa_tile_policy={policy!r} must give one of {VSA_H3_AXIS_POLICIES} per (t, h, w) "
                         "axis.")
    if not isinstance(inference_file["vsa_compress_text_keys"], bool):
        raise ValueError(f"{source} vsa_compress_text_keys must be a bool.")
    variant = inference_file.get("vsa_variant", _PDD_OPTIONAL_FIELDS["vsa_variant"])
    if variant not in VSA_H3_VARIANTS:
        raise ValueError(f"{source} vsa_variant={variant!r} must be one of {VSA_H3_VARIANTS}.")
    dense = inference_file.get("vsa_dense_layers", _PDD_OPTIONAL_FIELDS["vsa_dense_layers"])
    if (not isinstance(dense, list) or any(type(layer) is not int or layer < 0 for layer in dense)
            or any(left >= right for left, right in zip(dense, dense[1:], strict=False))):
        raise ValueError(f"{source} vsa_dense_layers={dense!r} must list block indices >= 0 in increasing order.")


def build_pan2_pdd_inference_file(**trained_fields: Any) -> dict[str, Any]:
    """The fastvideo_inference.json of a PAN2 PDD export: the code-fixed fields plus *trained_fields*.

    *trained_fields* must supply exactly the remaining PDD fields, so a writer cannot drift from what
    ``PAN2PipelineConfig.resolve_checkpoint_settings`` reads back.
    """
    overlap = sorted(_PDD_FIXED_FIELDS.keys() & trained_fields.keys())
    if overlap:
        raise ValueError(f"A PAN2 PDD {PAN2_INFERENCE_FILE} takes only the trained fields; {overlap} are fixed.")
    inference_file = {**_PDD_FIXED_FIELDS, **trained_fields}
    _validate_pdd_fields(inference_file, f"A PAN2 PDD {PAN2_INFERENCE_FILE}")
    return inference_file


def read_pan2_pdd_inference_file(model_path: str, revision: str | None = None) -> dict[str, Any] | None:
    """Read and validate a PAN2 PDD checkpoint's fastvideo_inference.json; None for a checkpoint without one.

    Besides its own values, fields that repeat a value stored in another checkpoint file must equal that file:
    ``pdd_steps`` the transformer config and ``video_scheduler_shift`` the scheduler config the export reuses.
    """
    inference_file = read_optional_model_json(model_path, PAN2_INFERENCE_FILE, revision)
    if inference_file is None:
        return None
    source = f"{model_path}/{PAN2_INFERENCE_FILE}"
    if not isinstance(inference_file, dict) or inference_file.get("schema") != PAN2_PDD_INFERENCE_SCHEMA:
        raise ValueError(f"{source} is not a PAN2 PDD inference file (schema {PAN2_PDD_INFERENCE_SCHEMA}).")
    _validate_pdd_fields(inference_file, source)
    for owner, key, owner_key in (("transformer/config.json", "pdd_steps", "pdd_steps"),
                                  ("scheduler/scheduler_config.json", "video_scheduler_shift", "shift")):
        owner_file = read_optional_model_json(model_path, owner, revision)
        if owner_file is None:
            raise ValueError(f"{model_path} has {PAN2_INFERENCE_FILE} but no {owner}.")
        if inference_file[key] != owner_file.get(owner_key):
            raise ValueError(f"{source} {key}={inference_file[key]!r} disagrees with {owner} "
                             f"{owner_key}={owner_file.get(owner_key)!r}.")
    return inference_file


@dataclass
class PAN2PipelineConfig(PipelineConfig):
    """One configuration serves text-to-video and, given a first frame, image-to-video."""

    dit_config: DiTConfig = field(default_factory=PAN2VideoConfig)
    # The Wan2.2 48-channel VAE; its architecture and latent statistics come from vae/config.json at load time.
    vae_config: VAEConfig = field(default_factory=WanVAEConfig)
    text_encoder_configs: tuple[EncoderConfig, ...] = field(default_factory=lambda: (Qwen3_5TextConfig(), ))
    text_encoder_precisions: tuple[str, ...] = field(default_factory=lambda: ("bf16", ))

    dit_precision: str = "bf16"
    vae_precision: str = "fp32"
    # Decode the video with fastvideo-kernel's Wan2.2 decoder (B200 / GB200 / B300; a 704 x 1248 video, batch 1): bf16
    # with fp32 accumulation, about 7x faster than the fp32 decode on a B300, at PSNR 61-63 dB and SSIM 0.998 against
    # it. Other devices and sizes, and False, keep the fp32 decode.
    vae_decode_kernel: bool = True
    # With vae_decode_kernel, split the decode over the sequence-parallel GPUs (2, 4 or 8) rather than decoding the
    # whole video on each of them. On B300s a 20 s video decodes in 3.0 s on one GPU, 1.55 s on 2 (the same video, bit
    # for bit), 0.79 s on 4 and 0.47 s on 8 (rounded differently, at the same PSNR against the fp32 decode).
    vae_decode_kernel_parallel: bool = True
    # Run the DiT's fused Triton kernels (Q/K RMSNorm + RoPE, LayerNorm + modulation): 21% less denoising time on 8
    # B300s, as far from the eager path's samples as bf16 rounding differences carry them.
    dit_fused_kernels: bool = True
    # A sparse student's VSA scatters its tiles straight into the layout its kernels read, as the process-wide
    # FASTVIDEO_H3_VSA_HEADS_FIRST_TILE does: the same video bit for bit, and 8-10% less denoising time for the
    # eight-call VSA2 student on 8 B300s.
    vsa_heads_first_tile: bool = True
    # Unset: the checkpoint's scheduler config decides, which is shift 9 for IFM/PAN2, the reference sampler's
    # fixed value. A fixed default here would override the shift the checkpoint declares.
    flow_shift: float | None = None
    # Image-to-video: the first latent frame (the conditioning image) takes the positive prediction instead of the
    # guided one, as the reference sampler does by default (``--i2v_skip_first_latent_cfg``).
    i2v_skip_first_latent_cfg: bool = True
    # PAN2 samples on fp32 latents; keep the Euler update out of autocast.
    scheduler_step_in_fp32: bool = True

    # Check the prompt and every generated video with the PAN 2 guardrails (the `pan2-guardrail` package), as the
    # vLLM-Omni and Diffusers integrations do by default. A blocked prompt or video raises
    # `PAN2GuardrailViolationError`. False turns both checks off, at your own risk.
    enable_guardrails: bool = True
    # Keep the guard models on CPU and move each to the GPU only for its check.
    offload_guardrail_models: bool = False

    # A single space is PAN2's trained unconditional prompt; normalization must not turn it into an empty string.
    preserve_single_space_negative_prompt: ClassVar[bool] = True
    # Prompt encoding: PAN2 conditions on the tokens after the chat template's user header, taken from the text
    # encoder's hidden states with its last two layers skipped (``hidden_states[-3]``).
    text_system_prompt: str = PAN2_SYSTEM_PROMPT
    text_encoder_skip_layers: int = 2
    text_max_sequence_length: int = 1536

    # Parallel Decoding Distillation (PDD) students: fine-grid node indices of the fused blocks, one transformer
    # forward each. resolve_checkpoint_settings sets it, and the student's VSA-H3 layout of the joint [video; text]
    # sequence, from the checkpoint's fastvideo_inference.json; None means the checkpoint is not a PDD student.
    pdd_step_indices: tuple[int, ...] | None = None
    vsa_tile_policy: tuple[str, ...] | None = None
    vsa_compress_text_keys: bool | None = None
    vsa_variant: str | None = None
    vsa_dense_layers: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        self.vae_config.load_encoder = True
        self.vae_config.load_decoder = True
        self.text_encoder_configs[0].arch_config.output_hidden_states = True

    def resolve_checkpoint_settings(self, fastvideo_args: FastVideoArgs) -> None:
        """Apply a PAN2 PDD checkpoint's fastvideo_inference.json to this run.

        The file is the one source of the PDD partition and the student's sparse-attention layout. A sparse student
        (VIDEO_SPARSE_ATTN_H3) fixes the attention backend and tile size, so a different run value raises; an unset
        VSA sparsity takes the trained value and another one wins with a warning. A dense student leaves the run's
        dense backend alone. For checkpoints without the file this method sets nothing.
        """
        inference_file = read_pan2_pdd_inference_file(fastvideo_args.model_path, fastvideo_args.revision)
        if inference_file is None:
            if self.pdd_step_indices is not None:
                raise ValueError(f"pdd_step_indices applies only to PAN2 PDD checkpoints, whose {PAN2_INFERENCE_FILE} "
                                 f"sets it; {fastvideo_args.model_path} has none.")
            return
        if self.flow_shift is not None and self.flow_shift != inference_file["video_scheduler_shift"]:
            raise ValueError(f"This PAN2 PDD checkpoint was trained with flow shift "
                             f"{inference_file['video_scheduler_shift']}; this run sets flow_shift={self.flow_shift}.")
        if self.dmd_denoising_steps is not None:
            raise ValueError(
                f"{fastvideo_args.model_path} is a PAN2 PDD checkpoint; dmd_denoising_steps must be unset.")
        indices = tuple(inference_file["pdd_step_indices"])
        if self.pdd_step_indices is not None and tuple(self.pdd_step_indices) != indices:
            raise ValueError(f"pdd_step_indices comes from {PAN2_INFERENCE_FILE} ({list(indices)}); this run sets "
                             f"{list(self.pdd_step_indices)}.")
        self.pdd_step_indices = indices
        self.vsa_tile_policy = tuple(inference_file["vsa_tile_policy"])
        self.vsa_compress_text_keys = inference_file["vsa_compress_text_keys"]
        self.vsa_variant = inference_file.get("vsa_variant", _PDD_OPTIONAL_FIELDS["vsa_variant"])
        self.vsa_dense_layers = tuple(inference_file.get("vsa_dense_layers", _PDD_OPTIONAL_FIELDS["vsa_dense_layers"]))

        if inference_file["attention_backend"] == _SPARSE_BACKEND:
            from fastvideo.attention.selector import coerce_attn_backend
            if fastvideo_args.attention_backend is None:
                fastvideo_args.attention_backend = _SPARSE_BACKEND
            elif coerce_attn_backend(fastvideo_args.attention_backend) != coerce_attn_backend(_SPARSE_BACKEND):
                raise ValueError(f"This PAN2 PDD checkpoint was trained with attention_backend={_SPARSE_BACKEND}; "
                                 f"this run requests {fastvideo_args.attention_backend} (argument or "
                                 "FASTVIDEO_ATTENTION_BACKEND). Leave it unset or pass the trained value.")
            trained_tile = inference_file["vsa_tile_size"]
            if fastvideo_args.VSA_tile_size is None:
                fastvideo_args.VSA_tile_size = trained_tile
            elif fastvideo_args.VSA_tile_size != trained_tile:
                raise ValueError(f"This PAN2 PDD checkpoint was trained with VSA_tile_size={trained_tile}; this run "
                                 f"requests {fastvideo_args.VSA_tile_size}. Leave it unset or pass the trained value.")
            trained_sparsity = inference_file["vsa_sparsity"]
            if fastvideo_args.VSA_sparsity is None:
                fastvideo_args.VSA_sparsity = trained_sparsity
            elif fastvideo_args.VSA_sparsity != trained_sparsity:
                logger.warning("PAN2 PDD checkpoint was trained with VSA_sparsity=%s; this run uses %s.",
                               trained_sparsity, fastvideo_args.VSA_sparsity)
        logger.info("PAN2 PDD checkpoint %s: %d fused blocks %s of a %d-interval grid; attention_backend=%s",
                    fastvideo_args.model_path,
                    len(indices) - 1, list(indices), indices[-1], inference_file["attention_backend"])

    def apply_request_constraints(self, request: GenerationRequest, sampling_param: SamplingParam) -> SamplingParam:
        """Fix ``num_inference_steps`` to a PDD checkpoint's block count and sample it without CFG.

        A PDD student runs exactly one transformer forward per trained fused block and no unconditional forward: a
        request that leaves ``num_inference_steps`` or ``guidance_scale`` unset gets the block count and 1, and a
        request that sets another value raises. Other checkpoints return ``sampling_param`` unchanged.
        """
        if self.pdd_step_indices is None:
            return sampling_param
        from fastvideo.api.compat import explicit_request_updates

        explicit = explicit_request_updates(request)
        num_blocks = len(self.pdd_step_indices) - 1
        if explicit.get("num_inference_steps", num_blocks) != num_blocks:
            raise ValueError(f"This PAN2 PDD checkpoint runs exactly {num_blocks} transformer forwards; the request "
                             f"sets num_inference_steps={explicit['num_inference_steps']}. Pass "
                             f"num_inference_steps={num_blocks} or leave it unset.")
        if explicit.get("guidance_scale", 1.0) != 1.0:
            raise ValueError(f"This PAN2 PDD checkpoint samples without classifier-free guidance; the request sets "
                             f"guidance_scale={explicit['guidance_scale']}. Pass 1.0 or leave it unset.")
        sampling_param.num_inference_steps = num_blocks
        sampling_param.guidance_scale = 1.0
        return sampling_param


__all__ = [
    "PAN2_INFERENCE_FILE",
    "PAN2_PDD_INFERENCE_SCHEMA",
    "PAN2PipelineConfig",
    "PAN2_SYSTEM_PROMPT",
    "build_pan2_pdd_inference_file",
    "read_pan2_pdd_inference_file",
]
