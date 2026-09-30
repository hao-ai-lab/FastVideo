# SPDX-License-Identifier: Apache-2.0
"""FastVideo composed pipelines for MiniMax H3."""

from __future__ import annotations

import gc
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeGuard

import torch

from fastvideo.attention.backends.video_sparse_attn_h3 import VSA_H3_TILE_SHAPES
from fastvideo.attention.selector import coerce_attn_backend, get_env_variable_attn_backend
from fastvideo.configs.models.vaes.minimax_h3_audio import MiniMaxH3AudioVAEArchConfig
from fastvideo.configs.models.vaes.minimax_h3_video import MiniMaxH3VideoVAEArchConfig
from fastvideo.configs.pipelines.minimax_h3 import MINIMAX_H3_VSA_REF_POLICY_P2, MiniMaxH3PipelineConfig
from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.layers.pdd import PDD_GRID_MAX_T, validate_grid_indices, validate_pdd_steps
from fastvideo.logger import init_logger
from fastvideo.models.hf_transformer_utils import get_diffusers_config
from fastvideo.pipelines.basic.minimax_h3.stages import (
    MiniMaxH3AudioDecodingStage,
    MiniMaxH3ConditioningStage,
    MiniMaxH3DenoisingStage,
    MiniMaxH3InputPreparationStage,
    MiniMaxH3LatentPreparationStage,
    MiniMaxH3VideoDecodingStage,
)
from fastvideo.pipelines.composed_pipeline_base import ComposedPipelineBase
from fastvideo.pipelines.lazy_module import is_lazy_module
from fastvideo.pipelines.lora_pipeline import LoRAPipeline
from fastvideo.pipelines.pipeline_batch_info import ForwardBatch
from fastvideo.platforms import AttentionBackendEnum

logger = init_logger(__name__)

# Same split as the MLX runtime: condition, release the ~66 GB Qwen3-VL stack,
# then load DiT + VAEs. Keeping them resident together OOMs unified-memory
# boxes (GB10 / Spark) even though host offload is correctly disabled there.
_DENOISE_MODULE_NAMES = ("vae", "audio_vae", "transformer")

_FASTH3_CONTRACT_SCHEMA = "fasth3-inference-contract-v1"
# Every key a Parallel Decoding Distillation (PDD) export may carry. Unknown
# keys are rejected so a newer recipe never runs with part of its contract
# silently ignored.
_PDD_CONTRACT_KEYS = frozenset({
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
# Ref2VA: ordered references are clean conditions; only target rows follow the flow.
_REF2VA_CONDITIONING = "fixed_ordered_references_target_only_flow"


def _is_real_number(value: Any) -> TypeGuard[int | float]:
    return not isinstance(value, bool) and isinstance(value, int | float) and math.isfinite(value)


def _requested_attention_backend(fastvideo_args: FastVideoArgs) -> AttentionBackendEnum | None:
    """The backend the transformer loader builds with: the run's request, else the environment.

    ``None`` means automatic selection, which never picks VIDEO_SPARSE_ATTN_H3.
    """
    requested = coerce_attn_backend(getattr(fastvideo_args, "attention_backend", None))
    return requested if requested is not None else get_env_variable_attn_backend()


def _checkpoint_has_vsa_gates(transformer_dir: Path) -> bool:
    """Whether a transformer checkpoint ships VSA-H3 compression gates (``to_gate_compress``).

    Reads only the shard index, or the safetensors headers when there is none.
    """
    indexes = sorted(transformer_dir.glob("*.safetensors.index.json"))
    if indexes:
        return any(".to_gate_compress." in name for index in indexes
                   for name in json.loads(index.read_text(encoding="utf-8")).get("weight_map", {}))
    from safetensors import safe_open

    for path in sorted(transformer_dir.glob("*.safetensors")):
        with safe_open(str(path), framework="pt") as handle:
            if any(".to_gate_compress." in name for name in handle.keys()):  # noqa: SIM118
                return True
    return False


def _require_pdd_attention_backend(contract: dict[str, Any], transformer_dir: Path,
                                   requested: AttentionBackendEnum | None) -> None:
    """Fail with a clear error when a PDD export that needs VIDEO_SPARSE_ATTN_H3 runs with another backend.

    Its trained compression gates exist only under that backend and the H3
    transformer loads strictly, so any other backend would otherwise fail
    later as a missing ``to_gate_compress`` parameter.
    """
    trained = contract.get("attention_backend")
    if trained is not None and coerce_attn_backend(trained) == AttentionBackendEnum.VIDEO_SPARSE_ATTN_H3:
        reason = "it was trained with VIDEO_SPARSE_ATTN_H3"
    elif _checkpoint_has_vsa_gates(transformer_dir):
        reason = f"{transformer_dir.name} carries VSA compression gates (to_gate_compress)"
    else:
        return
    if requested == AttentionBackendEnum.VIDEO_SPARSE_ATTN_H3:
        return
    settings = ", ".join(f"{name}={contract[key]}"
                         for key, name in (("vsa_sparsity", "VSA_sparsity"), ("vsa_tile_size", "VSA_tile_size"))
                         if key in contract)
    raise ValueError(
        f"This FastH3 PDD checkpoint needs attention_backend=VIDEO_SPARSE_ATTN_H3: {reason}, and the gates load "
        f"only under that backend. This run requests "
        f"{'automatic selection' if requested is None else requested.name}. Select VIDEO_SPARSE_ATTN_H3"
        f"{f' with {settings}' if settings else ''}; basic_fasth3_omniref_pdd.py passes the contract's values.")


@dataclass(frozen=True)
class _H3VideoGeometry:
    spatial_compression_ratio: int
    latent_channels: int


@dataclass(frozen=True)
class _H3AudioGeometry:
    sampling_rate: int


def _default_video_geometry() -> _H3VideoGeometry:
    arch = MiniMaxH3VideoVAEArchConfig()
    return _H3VideoGeometry(
        spatial_compression_ratio=int(arch.spatial_compression_ratio),
        latent_channels=int(arch.latent_channels),
    )


def _default_audio_geometry() -> _H3AudioGeometry:
    return _H3AudioGeometry(sampling_rate=int(MiniMaxH3AudioVAEArchConfig().sampling_rate))


def _apply_h3_checkpoint_arch_configs(model_path: str, fastvideo_args: FastVideoArgs,
                                      extra_config_module_map: dict[str, str]) -> None:
    """Overlay checkpoint config.json onto pipeline configs without loading weights."""
    root = Path(model_path)
    vae_dir = root / extra_config_module_map.get("vae", "vae")
    if (vae_dir / "config.json").is_file():
        fastvideo_args.pipeline_config.vae_config.update_model_arch(get_diffusers_config(str(vae_dir)))
    audio_vae_dir = root / extra_config_module_map.get("audio_vae", "audio_vae")
    audio_vae_config = getattr(fastvideo_args.pipeline_config, "audio_vae_config", None)
    if audio_vae_config is not None and (audio_vae_dir / "config.json").is_file():
        audio_vae_config.update_model_arch(get_diffusers_config(str(audio_vae_dir)))
    transformer_dir = root / extra_config_module_map.get("transformer", "transformer")
    if (transformer_dir / "config.json").is_file():
        fastvideo_args.pipeline_config.dit_config.update_model_arch(get_diffusers_config(str(transformer_dir)))
    dit_config = fastvideo_args.pipeline_config.dit_config
    vae_arch = getattr(fastvideo_args.pipeline_config.vae_config, "arch_config", None)
    patch_size = getattr(dit_config, "patch_size", None)
    if patch_size is not None and vae_arch is not None:
        logger.info(
            "MiniMax-H3 geometry from config: patch_size=%s spatial_compression_ratio=%s latent_channels=%s",
            tuple(patch_size),
            int(getattr(vae_arch, "spatial_compression_ratio", 0)),
            int(getattr(vae_arch, "latent_channels", 0)),
        )


def _use_taeh3_t2va(fastvideo_args: FastVideoArgs | None, *, ref2va: bool) -> bool:
    return (not ref2va) and getattr(fastvideo_args, "video_decode_backend", "h3-vae") == "taeh3"


class MiniMaxH3BasePipeline(LoRAPipeline, ComposedPipelineBase):
    """Shared loading and target-generation path for MiniMax H3.

    Inherits ``LoRAPipeline`` so acceleration and distillation adapters can be merged
    in; without it every adapter is rejected with "pipeline is not a LoRAPipeline".
    """

    # The linears every published H3 adapter targets. Left unset, ``LoRAPipeline``
    # wraps *every* linear in the DiT -- including ``proj_in``, whose ``.weight`` the
    # forward pass reads directly. ``BaseLayerWithLoRA`` exposes no ``.weight``, so
    # that wrapping turns generation into an AttributeError before the first step.
    lora_target_modules = [
        "attn.to_q",
        "attn.to_k",
        "attn.to_v",
        "attn.to_out",
        "ff.fc_in",
        "ff.fc_out",
        "adaln_proj.linear",
        # The final AdaLN projection. Published community adapters (larryvrh's Turbo)
        # target it as `final_layer.adaln_proj.linear`.
        "norm_out.linear",
    ]

    pipeline_config_cls: type[MiniMaxH3PipelineConfig] = MiniMaxH3PipelineConfig
    _ref2va_default = False
    _required_config_modules = [
        "text_encoder",
        "tokenizer",
        "processor",
        "vae",
        "audio_vae",
        "transformer",
        "scheduler",
        "audio_scheduler",
    ]
    # Deferral is safe here: geometry scalars come from checkpoint config.json
    # (applied in initialize_pipeline without loading weights), no stage
    # constructor reads a deferred component, and initialize_pipeline only
    # inspects the schedulers, which are never deferred.
    _lazy_module_names = ("text_encoder", "transformer", "vae", "audio_vae")

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self._ref2va = getattr(self, "_ref2va_default", False)
        self._denoise_stages_ready = False
        super().__init__(*args, **kwargs)

    @classmethod
    def get_hf_download_component_dirs(cls) -> tuple[str, ...]:
        return tuple(sorted(cls._extra_config_module_map.get(name, name) for name in cls._required_config_modules))

    @classmethod
    def get_hf_download_allow_patterns(cls) -> list[str]:
        patterns = super().get_hf_download_allow_patterns()
        assert patterns is not None
        # Keep the optional distilled schedule even when downloading only the
        # selected transformer partition. Otherwise Hub and local loads differ.
        return [*patterns, "fastvideo_inference.json"]

    def initialize_pipeline(self, fastvideo_args: FastVideoArgs) -> None:
        _apply_h3_checkpoint_arch_configs(self.model_path, fastvideo_args, self._extra_config_module_map)
        # Each modality's scheduler_config.json owns its shift. Base H3 keeps
        # 12/3; a distilled checkpoint can serialize a different trained pair
        # (for example 10/3) without being silently rewritten to base defaults.
        for module_name, modality in (("scheduler", "video"), ("audio_scheduler", "audio")):
            shift = getattr(self.get_module(module_name), "shift", None)
            if shift is None or not math.isfinite(float(shift)) or float(shift) <= 0:
                raise ValueError(f"MiniMax-H3 {modality} scheduler must expose a positive finite shift, got {shift}.")
        self._load_checkpoint_schedule(fastvideo_args)
        policy = getattr(fastvideo_args.pipeline_config, "vsa_ref_policy", None)
        if policy is not None and self._transformer_attention_backend(fastvideo_args) != (
                AttentionBackendEnum.VIDEO_SPARSE_ATTN_H3):
            raise ValueError(f"vsa_ref_policy={policy!r} sparsifies reference videos under VIDEO_SPARSE_ATTN_H3; "
                             "select that attention backend or unset vsa_ref_policy.")

    def _transformer_attention_backend(self, fastvideo_args: FastVideoArgs) -> AttentionBackendEnum | None:
        """The loaded transformer's recorded backend, else the one it will be built with."""
        transformer = getattr(self, "modules", {}).get("transformer")
        if transformer is not None and not is_lazy_module(transformer):
            recorded = getattr(getattr(transformer, "config", None), "_resolved_attention_backend", None)
            if recorded is not None:
                return recorded
        return _requested_attention_backend(fastvideo_args)

    def _checkpoint_contract(self) -> Any:
        """The export's ``fastvideo_inference.json``, or None when there is none."""
        path = Path(self.model_path) / "fastvideo_inference.json"
        return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None

    def _transformer_dir(self) -> Path:
        return Path(self.model_path) / self._extra_config_module_map.get("transformer", "transformer")

    def _load_checkpoint_schedule(self, fastvideo_args: FastVideoArgs) -> None:
        """A distilled export's schedule is explicit; never silently use a uniform grid."""
        contract = self._checkpoint_contract()
        if contract is None:
            return
        if not isinstance(contract, dict) or contract.get("schema_version") != "fasth3-inference-contract-v1":
            raise ValueError("Unsupported FastH3 fastvideo_inference.json schema.")
        if "pdd_steps" in contract:
            self._load_pdd_contract(contract, fastvideo_args)
            return
        steps = contract.get("dmd_denoising_steps")
        if (not isinstance(steps, list) or not steps
                or any(type(step) is not int or not 0 < step <= 1000 for step in steps)
                or any(left <= right for left, right in zip(steps, steps[1:], strict=False))):
            raise ValueError("FastH3 checkpoint DMD rungs must be strictly decreasing integers in (0, 1000].")
        if (type(contract.get("num_inference_steps")) is not int or contract["num_inference_steps"] != len(steps) + 1
                or type(contract.get("transformer_forwards")) is not int
                or contract["transformer_forwards"] != len(steps)):
            raise ValueError(
                "FastH3 checkpoint DMD rung count must match transformer_forwards and num_inference_steps - 1.")
        for name, key in (("scheduler", "video_scheduler_shift"), ("audio_scheduler", "audio_scheduler_shift")):
            if key not in contract:
                # Earlier exports (the four-step Preview v1 checkpoints) carry only the ladder; the
                # scheduler configs are the sole source of the shifts for them.
                continue
            declared = contract[key]
            if (isinstance(declared, bool) or not isinstance(declared, int | float) or not math.isfinite(declared)
                    or declared <= 0 or float(self.get_module(name).shift) != float(declared)):
                raise ValueError(f"FastH3 checkpoint {key}={declared!r} disagrees with {name}/scheduler_config.json.")
        config = fastvideo_args.pipeline_config
        if config.dmd_denoising_steps is not None and config.dmd_denoising_steps != steps:
            raise ValueError("Explicit DMD schedule disagrees with the checkpoint's trained DMD rungs.")
        config.dmd_denoising_steps = list(steps)
        logger.info("FastH3 checkpoint schedule: %d transformer forwards, DMD rungs=%s, video/audio shifts=%s/%s",
                    len(steps), steps,
                    self.get_module("scheduler").shift,
                    self.get_module("audio_scheduler").shift)

    def _load_pdd_contract(self, contract: dict[str, Any], fastvideo_args: FastVideoArgs) -> None:
        """Load a Parallel Decoding Distillation (PDD) export's sampling contract.

        A PDD student's transformer carries ``pdd_steps`` widened heads; the
        contract names the fine-grid partition it was trained to sample
        (``pdd_step_indices``, one fused block per transformer forward) and,
        for Ref2VA, the reference-video sparsity policy. Both are applied to
        the pipeline config. Attention backend, sparsity, and tile size stay
        explicit run settings: an export trained with VIDEO_SPARSE_ATTN_H3
        requires that backend, and a sparsity or tile-size mismatch is logged.
        """
        unknown = sorted(set(contract) - _PDD_CONTRACT_KEYS)
        if unknown:
            raise ValueError(f"FastH3 PDD fastvideo_inference.json carries unsupported keys {unknown}.")
        if contract.get("schema", _FASTH3_CONTRACT_SCHEMA) != _FASTH3_CONTRACT_SCHEMA:
            raise ValueError(f"FastH3 PDD contract schema={contract['schema']!r} disagrees with its schema_version.")
        expected_model_type = "ref2va" if getattr(self, "_ref2va", self._ref2va_default) else "t2va"
        model_type = contract.get("model_type", expected_model_type)
        if model_type != expected_model_type:
            hint = " Select MiniMaxH3Ref2VAModularPipeline." if model_type == "ref2va" else ""
            raise ValueError(f"FastH3 checkpoint is a {model_type!r} PDD export; this pipeline runs "
                             f"{expected_model_type!r}.{hint}")
        transformer_dir = self._extra_config_module_map.get("transformer", "transformer")
        if contract.get("transformer_component", transformer_dir) != transformer_dir:
            raise ValueError(f"FastH3 PDD contract targets {contract['transformer_component']!r}; this pipeline "
                             f"loads {transformer_dir!r}.")
        if model_type == "ref2va" and contract.get("conditioning", _REF2VA_CONDITIONING) != _REF2VA_CONDITIONING:
            raise ValueError(f"Unsupported FastH3 Ref2VA conditioning {contract['conditioning']!r}.")
        revision = contract.get("base_model_revision")
        if revision is not None and (not isinstance(revision, str) or not revision.strip()):
            raise ValueError(f"FastH3 PDD base_model_revision must be a non-empty string, got {revision!r}.")

        pdd_steps = validate_pdd_steps(contract["pdd_steps"])
        transformer_steps = fastvideo_args.pipeline_config.dit_config.arch_config.pdd_steps
        if transformer_steps != pdd_steps:
            raise ValueError(f"FastH3 PDD contract pdd_steps={pdd_steps} disagrees with "
                             f"{transformer_dir}/config.json pdd_steps={transformer_steps!r}.")
        indices = contract.get("pdd_step_indices")
        if not isinstance(indices, list) or any(type(index) is not int for index in indices):
            raise ValueError(f"FastH3 PDD pdd_step_indices must be a list of integers, got {indices!r}.")
        forwards = len(indices) - 1
        if forwards < 1:
            raise ValueError(f"FastH3 PDD pdd_step_indices must name at least one block, got {indices!r}.")
        validate_grid_indices(indices, pdd_steps, forwards)
        for key in ("num_inference_steps", "transformer_forwards"):
            if type(contract.get(key)) is not int or contract[key] != forwards:
                raise ValueError(f"FastH3 PDD {key}={contract.get(key)!r} must equal the {forwards} fused blocks "
                                 "named by pdd_step_indices.")
        grid_max_t = contract.get("grid_max_t", PDD_GRID_MAX_T)
        if not _is_real_number(grid_max_t) or float(grid_max_t) != PDD_GRID_MAX_T:
            raise ValueError(f"FastH3 PDD grid_max_t={grid_max_t!r} is unsupported; the fine grid ends at "
                             f"{PDD_GRID_MAX_T}.")
        for name, key in (("scheduler", "video_scheduler_shift"), ("audio_scheduler", "audio_scheduler_shift")):
            # Required: the fused blocks' node sigmas are these shifts applied to the fine grid.
            declared = contract.get(key)
            if (not _is_real_number(declared) or declared <= 0
                    or float(self.get_module(name).shift) != float(declared)):
                raise ValueError(f"FastH3 checkpoint {key}={declared!r} disagrees with {name}/scheduler_config.json.")
        guidance_scale = contract.get("guidance_scale", 1.0)
        if not _is_real_number(guidance_scale) or float(guidance_scale) != 1.0:
            raise ValueError(f"FastH3 PDD guidance_scale={guidance_scale!r} is unsupported; MiniMax-H3 samples "
                             "without classifier-free guidance.")

        trained_backend = contract.get("attention_backend")
        if trained_backend is not None:
            trained_backend = coerce_attn_backend(trained_backend)
        sparsity = contract.get("vsa_sparsity")
        if sparsity is not None and (not _is_real_number(sparsity) or not 0.0 <= sparsity < 1.0):
            raise ValueError(f"FastH3 PDD vsa_sparsity must be in [0, 1), got {sparsity!r}.")
        tile_size = contract.get("vsa_tile_size")
        if tile_size is not None and (type(tile_size) is not int or tile_size not in VSA_H3_TILE_SHAPES):
            raise ValueError(
                f"FastH3 PDD vsa_tile_size must be one of {sorted(VSA_H3_TILE_SHAPES)}, got {tile_size!r}.")
        policy = contract.get("vsa_ref_policy")
        keep_rate = contract.get("vsa_ref_keep_rate")
        if policy is not None:
            if policy != MINIMAX_H3_VSA_REF_POLICY_P2 or model_type != "ref2va":
                raise ValueError(f"Unsupported FastH3 PDD vsa_ref_policy {policy!r} for a {model_type!r} export.")
            # A keep rate of 1 would leave reference-video attention dense.
            if not _is_real_number(keep_rate) or not 0.0 < keep_rate < 1.0:
                raise ValueError(f"FastH3 PDD vsa_ref_keep_rate must be in (0, 1), got {keep_rate!r}.")
            keep_rate = float(keep_rate)
        elif keep_rate is not None:
            raise ValueError("FastH3 PDD vsa_ref_keep_rate requires vsa_ref_policy.")

        config = fastvideo_args.pipeline_config
        if not isinstance(config, MiniMaxH3PipelineConfig):
            raise TypeError(f"FastH3 PDD checkpoints need MiniMaxH3PipelineConfig, got {type(config).__name__}.")
        if config.dmd_denoising_steps is not None:
            raise ValueError("A FastH3 PDD checkpoint samples fused blocks of its fine grid; dmd_denoising_steps "
                             "must be unset.")
        explicit_indices = None if config.pdd_step_indices is None else list(config.pdd_step_indices)
        explicit_settings = (
            ("pdd_step_indices", explicit_indices, indices),
            ("vsa_ref_policy", config.vsa_ref_policy, policy),
            ("vsa_ref_keep_rate", config.vsa_ref_keep_rate, keep_rate),
        )
        for field_name, explicit, trained in explicit_settings:
            if explicit is not None and explicit != trained:
                raise ValueError(f"Explicit {field_name}={explicit!r} disagrees with the checkpoint's {trained!r}.")
        requested = self._transformer_attention_backend(fastvideo_args)
        _require_pdd_attention_backend(contract, self._transformer_dir(), requested)
        config.pdd_step_indices = list(indices)
        config.vsa_ref_policy = policy
        config.vsa_ref_keep_rate = keep_rate

        if trained_backend is not None and requested is not None and requested != trained_backend:
            logger.warning("FastH3 PDD checkpoint was trained with %s attention; this run requests %s.",
                           trained_backend.name, requested.name)
        if requested == AttentionBackendEnum.VIDEO_SPARSE_ATTN_H3:
            for trained, run_name, run_value in (
                (sparsity, "VSA_sparsity", getattr(fastvideo_args, "VSA_sparsity", None)),
                (tile_size, "VSA_tile_size", getattr(fastvideo_args, "VSA_tile_size", None)),
            ):
                if trained is not None and run_value is not None and float(run_value) != float(trained):
                    logger.warning("FastH3 PDD checkpoint was trained with %s=%s; this run uses %s.", run_name, trained,
                                   run_value)
        logger.info(
            "FastH3 PDD checkpoint: %d fused blocks %s of a %d-interval grid, video/audio shifts=%s/%s, "
            "reference VSA policy=%s (keep rate %s)", forwards, indices, pdd_steps,
            self.get_module("scheduler").shift,
            self.get_module("audio_scheduler").shift, policy, keep_rate)

    def _defer_denoise_modules(self, fastvideo_args: FastVideoArgs) -> bool:
        if not fastvideo_args.inference_mode or bool(getattr(fastvideo_args, "training_mode", False)):
            return False
        # Both mechanisms defer the same four modules and both decide when to
        # free them. Running them together strips DiT/VAEs from the first load
        # (sequential) while the base wraps the encoder in a proxy (lazy), so
        # post_init's VAE compile transform has nothing to attach to. Lazy is
        # the more general owner — including auto-on for unified memory — so it
        # wins whenever it is on. Sequential remains the H3-only fallback when
        # lazy is off.
        if bool(getattr(fastvideo_args, "lazy_module_load", False)):
            logger.info("MiniMax-H3 sequential module load off: lazy_module_load owns deferral")
            return False
        requested = fastvideo_args.h3_sequential_load
        if requested is True:
            return True
        if requested is False:
            return False
        from fastvideo.pipelines import composed_pipeline_base
        from fastvideo.platforms import current_platform

        device = composed_pipeline_base.get_local_torch_device()
        device_id = 0 if device.index is None else int(device.index)
        unified = bool(current_platform.has_unified_memory(device_id))
        logger.info("MiniMax-H3 sequential module load auto=%s (unified_memory=%s)", unified, unified)
        return unified

    def _denoise_module_names(self, fastvideo_args: FastVideoArgs | None = None) -> tuple[str, ...]:
        args = fastvideo_args if fastvideo_args is not None else getattr(self, "fastvideo_args", None)
        if _use_taeh3_t2va(args, ref2va=self._ref2va):
            return tuple(name for name in _DENOISE_MODULE_NAMES if name != "vae")
        return _DENOISE_MODULE_NAMES

    def _denoise_modules_loaded(self) -> bool:
        return all(self.get_module(name) is not None for name in self._denoise_module_names())

    def load_modules(self,
                     fastvideo_args: FastVideoArgs,
                     loaded_modules: dict[str, torch.nn.Module] | None = None) -> dict[str, Any]:
        """Load the Qwen3-VL conditioner first; defer DiT and VAEs until after encode."""
        if loaded_modules is None or "transformer" not in loaded_modules:
            # Before any weights load: a PDD export that needs VIDEO_SPARSE_ATTN_H3
            # would otherwise fail only when its transformer loads.
            contract = self._checkpoint_contract()
            if isinstance(contract, dict) and "pdd_steps" in contract:
                _require_pdd_attention_backend(contract, self._transformer_dir(),
                                               _requested_attention_backend(fastvideo_args))
        if not self._defer_denoise_modules(fastvideo_args):
            if _use_taeh3_t2va(fastvideo_args, ref2va=self._ref2va):
                saved = list(self.required_config_modules)
                self._required_config_modules = [name for name in saved if name != "vae"]
                try:
                    return super().load_modules(fastvideo_args, loaded_modules)
                finally:
                    self._required_config_modules = saved
            return super().load_modules(fastvideo_args, loaded_modules)
        if loaded_modules is not None and all(name in loaded_modules
                                              for name in self._denoise_module_names(fastvideo_args)):
            return super().load_modules(fastvideo_args, loaded_modules)

        saved = list(self.required_config_modules)
        # Always defer the full denoise set on the first load. TAEH3 T2VA then
        # omits the video VAE from the second load via `_denoise_module_names`.
        self._required_config_modules = [name for name in saved if name not in _DENOISE_MODULE_NAMES]
        try:
            logger.info("Loading MiniMax-H3 condition modules first: %s", self._required_config_modules)
            return super().load_modules(fastvideo_args, loaded_modules)
        finally:
            self._required_config_modules = saved

    def _load_denoise_modules(self, fastvideo_args: FastVideoArgs) -> None:
        if self._denoise_modules_loaded():
            return
        saved = list(self.required_config_modules)
        denoise_names = self._denoise_module_names(fastvideo_args)
        self._required_config_modules = [name for name in saved if name != "text_encoder"]
        if _use_taeh3_t2va(fastvideo_args, ref2va=self._ref2va):
            self._required_config_modules = [name for name in self._required_config_modules if name != "vae"]
        try:
            logger.info("Loading MiniMax-H3 denoise modules after releasing the text encoder: %s",
                        [name for name in self._required_config_modules if name in denoise_names])
            loaded = super().load_modules(fastvideo_args, loaded_modules=self.modules)
            for name, module in loaded.items():
                self.add_module(name, module)
            self._apply_inference_compile(tuple(name for name in loaded if name in _DENOISE_MODULE_NAMES))
        finally:
            self._required_config_modules = saved

    def _release_text_encoder(self) -> None:
        stage = self._stage_name_mapping.get("conditioning_stage")
        if stage is not None:
            stage.conditioner = None
        encoder = self.modules.pop("text_encoder", None)
        if encoder is None:
            return
        logger.info("Released MiniMax-H3 text encoder after conditioning")
        del encoder
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _ensure_text_encoder(self, fastvideo_args: FastVideoArgs) -> None:
        """Reload Qwen3-VL after `_release_text_encoder` so a later request can encode."""
        encoder = self.get_module("text_encoder")
        stage = self._stage_name_mapping.get("conditioning_stage")
        if encoder is not None:
            if stage is not None and getattr(stage, "conditioner", None) is None:
                stage.conditioner = encoder
            return
        saved = list(self.required_config_modules)
        self._required_config_modules = ["text_encoder"]
        try:
            logger.info("Reloading MiniMax-H3 text encoder for a subsequent request")
            loaded = super().load_modules(fastvideo_args, loaded_modules=self.modules)
            for name, module in loaded.items():
                self.add_module(name, module)
            self._apply_inference_compile(("text_encoder", ))
        finally:
            self._required_config_modules = saved
        if stage is not None:
            stage.conditioner = self.get_module("text_encoder")

    def _run_condition_then_denoise(self, batch: ForwardBatch, fastvideo_args: FastVideoArgs) -> ForwardBatch:
        for name in ("input_preparation_stage", "conditioning_stage"):
            batch = self._stage_name_mapping[name](batch, fastvideo_args)
        self._release_text_encoder()
        self._load_denoise_modules(fastvideo_args)
        if not self._denoise_stages_ready:
            self._add_denoise_stages(ref2va=self._ref2va)
        for name in (
                "latent_preparation_stage",
                "denoising_stage",
                "video_decoding_stage",
                "audio_decoding_stage",
        ):
            batch = self._stage_name_mapping[name](batch, fastvideo_args)
        return batch

    def _input_video_geometry(self, fastvideo_args: FastVideoArgs) -> Any:
        """Read canvas scalars from checkpoint JSON, not a live VAE proxy."""
        arch = getattr(getattr(fastvideo_args.pipeline_config, "vae_config", None), "arch_config", None)
        if arch is not None:
            return arch
        return _default_video_geometry()

    def _input_vae(self) -> Any:
        live = self.get_module("vae")
        if live is not None:
            return live
        return self._input_video_geometry(self.fastvideo_args)

    def _input_audio_vae(self, fastvideo_args: FastVideoArgs, *, ref2va: bool) -> Any | None:
        if not ref2va:
            return None
        arch = getattr(getattr(fastvideo_args.pipeline_config, "audio_vae_config", None), "arch_config", None)
        if arch is not None:
            return arch
        return _default_audio_geometry()

    def _add_condition_stages(self, fastvideo_args: FastVideoArgs, *, ref2va: bool) -> None:
        self.add_stage(
            "input_preparation_stage",
            MiniMaxH3InputPreparationStage(
                vae=self._input_video_geometry(fastvideo_args),
                audio_vae=self._input_audio_vae(fastvideo_args, ref2va=ref2va),
                ref2va=ref2va,
            ),
        )
        self.add_stage(
            "conditioning_stage",
            MiniMaxH3ConditioningStage(
                conditioner=self.get_module("text_encoder"),
                tokenizer=self.get_module("tokenizer"),
                processor=self.get_module("processor"),
                ref2va=ref2va,
            ),
        )

    def _add_denoise_stages(self, *, ref2va: bool) -> None:
        transformer = self.get_module("transformer")
        vae = self.get_module("vae")
        audio_vae = self.get_module("audio_vae")
        scheduler = self.get_module("scheduler")
        audio_scheduler = self.get_module("audio_scheduler")
        use_taeh3 = _use_taeh3_t2va(getattr(self, "fastvideo_args", None), ref2va=ref2va)
        if transformer is None or audio_vae is None:
            raise RuntimeError("MiniMax-H3 denoise stages require transformer and audio_vae to be loaded.")
        if not use_taeh3 and vae is None:
            raise RuntimeError("MiniMax-H3 full-VAE decode requires the video VAE to be loaded.")
        encode_vae = vae if vae is not None else self._input_vae()
        self.add_stage(
            "latent_preparation_stage",
            MiniMaxH3LatentPreparationStage(
                vae=encode_vae,
                audio_vae=audio_vae,
                scheduler=scheduler,
                ref2va=ref2va,
            ),
        )
        self.add_stage(
            "denoising_stage",
            MiniMaxH3DenoisingStage(
                transformer=transformer,
                scheduler=scheduler,
                audio_scheduler=audio_scheduler,
            ),
        )
        self.add_stage("video_decoding_stage", MiniMaxH3VideoDecodingStage(vae=None if use_taeh3 else vae))
        self.add_stage("audio_decoding_stage", MiniMaxH3AudioDecodingStage(audio_vae=audio_vae))
        self._denoise_stages_ready = True

    def _add_stages(self, fastvideo_args: FastVideoArgs, *, ref2va: bool) -> None:
        self._ref2va = ref2va
        self._add_condition_stages(fastvideo_args, ref2va=ref2va)
        if self._denoise_modules_loaded():
            self._add_denoise_stages(ref2va=ref2va)

    def forward(self, batch: ForwardBatch, fastvideo_args: FastVideoArgs) -> ForwardBatch:
        if not self.post_init_called:
            self.post_init()
        indices = getattr(fastvideo_args.pipeline_config, "pdd_step_indices", None)
        if indices is not None and batch.num_inference_steps != len(indices) - 1:
            # Checked again by the denoising stage; failing here skips conditioning and VAE work.
            raise ValueError(f"This PDD checkpoint runs {len(indices) - 1} fused blocks (pdd_step_indices={indices}); "
                             f"num_inference_steps counts transformer forwards and must be {len(indices) - 1}, "
                             f"got {batch.num_inference_steps}.")

        # Sequential encode-then-release is the H3-only fallback. Lazy and the
        # fully-resident discrete-GPU path both keep a complete stage list and
        # must use the base forward so abort cleanup and text_encoder_cpu_offload
        # still apply. Releasing Qwen on every request was re-reading it from disk
        # when neither deferral flag was on.
        if self._defer_denoise_modules(fastvideo_args):
            try:
                self._ensure_text_encoder(fastvideo_args)
                if self._denoise_stages_ready:
                    logger.info("Running MiniMax-H3 condition stages before denoise (subsequent request)")
                else:
                    logger.info("Running MiniMax-H3 condition stages before loading DiT/VAE weights")
                return self._run_condition_then_denoise(batch, fastvideo_args)
            except BaseException:
                self._release_all_lazy_modules()
                raise
        return super().forward(batch, fastvideo_args)


class MiniMaxH3Pipeline(MiniMaxH3BasePipeline):
    """One-request joint video/stereo-audio pipeline for T2VA and FL2VA."""

    def create_pipeline_stages(self, fastvideo_args: FastVideoArgs) -> None:
        self._add_stages(fastvideo_args, ref2va=False)


class MiniMaxH3RefPipeline(MiniMaxH3BasePipeline):
    """Ordered-reference joint video/stereo-audio pipeline for Ref2VA."""

    _extra_config_module_map = {"transformer": "transformer_ref"}
    _ref2va_default = True

    def create_pipeline_stages(self, fastvideo_args: FastVideoArgs) -> None:
        self._add_stages(fastvideo_args, ref2va=True)


class MiniMaxH3ModularPipeline(MiniMaxH3Pipeline):
    """Public T2VA/FL2VA entry matching the official manifest class name."""


class MiniMaxH3Ref2VAModularPipeline(MiniMaxH3RefPipeline):
    """Public Ref2VA entry using the checkpoint's ``transformer_ref`` partition."""


EntryClass = [MiniMaxH3ModularPipeline, MiniMaxH3Ref2VAModularPipeline]

__all__ = [
    "EntryClass",
    "MiniMaxH3BasePipeline",
    "MiniMaxH3ModularPipeline",
    "MiniMaxH3Pipeline",
    "MiniMaxH3Ref2VAModularPipeline",
    "MiniMaxH3RefPipeline",
]
