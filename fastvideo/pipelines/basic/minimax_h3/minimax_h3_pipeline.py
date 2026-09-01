# SPDX-License-Identifier: Apache-2.0
"""FastVideo composed pipelines for MiniMax H3."""

from __future__ import annotations

import gc
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from fastvideo.configs.models.vaes.minimax_h3_audio import MiniMaxH3AudioVAEArchConfig
from fastvideo.configs.models.vaes.minimax_h3_video import MiniMaxH3VideoVAEArchConfig
from fastvideo.configs.pipelines.minimax_h3 import MiniMaxH3PipelineConfig
from fastvideo.fastvideo_args import FastVideoArgs
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
from fastvideo.pipelines.lora_pipeline import LoRAPipeline
from fastvideo.pipelines.pipeline_batch_info import ForwardBatch

logger = init_logger(__name__)

# Same split as the MLX runtime: condition, release the ~66 GB Qwen3-VL stack,
# then load DiT + VAEs. Keeping them resident together OOMs unified-memory
# boxes (GB10 / Spark) even though host offload is correctly disabled there.
_DENOISE_MODULE_NAMES = ("vae", "audio_vae", "transformer")


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

    def initialize_pipeline(self, fastvideo_args: FastVideoArgs) -> None:
        _apply_h3_checkpoint_arch_configs(self.model_path, fastvideo_args, self._extra_config_module_map)
        for module_name, modality, expected_shift in (
            ("scheduler", "video", 12.0),
            ("audio_scheduler", "audio", 3.0),
        ):
            shift = getattr(self.get_module(module_name), "shift", None)
            if shift is None or float(shift) != expected_shift:
                raise ValueError(f"MiniMax-H3 {modality} scheduler must expose shift={expected_shift:g}, got {shift}.")

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
