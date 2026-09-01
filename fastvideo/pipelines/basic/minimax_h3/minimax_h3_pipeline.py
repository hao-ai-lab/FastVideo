# SPDX-License-Identifier: Apache-2.0
"""FastVideo composed pipelines for MiniMax H3."""

from __future__ import annotations

import gc
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from fastvideo.configs.models.vaes.minimax_h3_audio import MiniMaxH3AudioVAEArchConfig
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


def _apply_h3_checkpoint_arch_configs(model_path: str, fastvideo_args: FastVideoArgs,
                                      extra_config_module_map: dict[str, str]) -> None:
    """Overlay checkpoint config.json onto pipeline configs without loading weights."""
    root = Path(model_path)
    vae_dir = root / "vae"
    if (vae_dir / "config.json").is_file():
        fastvideo_args.pipeline_config.vae_config.update_model_arch(get_diffusers_config(str(vae_dir)))
    transformer_dir = root / extra_config_module_map.get("transformer", "transformer")
    if (transformer_dir / "config.json").is_file():
        fastvideo_args.pipeline_config.dit_config.update_model_arch(get_diffusers_config(str(transformer_dir)))
    dit_arch = getattr(fastvideo_args.pipeline_config.dit_config, "arch_config", None)
    vae_arch = getattr(fastvideo_args.pipeline_config.vae_config, "arch_config", None)
    logger.info(
        "MiniMax-H3 geometry from config: patch_size=%s spatial_compression_ratio=%s latent_channels=%s",
        getattr(dit_arch, "patch_size", None),
        getattr(vae_arch, "spatial_compression_ratio", None),
        getattr(vae_arch, "latent_channels", None),
    )


def _use_taeh3_t2va(fastvideo_args: FastVideoArgs | None, *, ref2va: bool) -> bool:
    return (not ref2va) and getattr(fastvideo_args, "video_decode_backend", "h3-vae") == "taeh3"


@dataclass(frozen=True)
class _H3AudioGeometry:
    sampling_rate: int


def _default_audio_geometry() -> _H3AudioGeometry:
    return _H3AudioGeometry(sampling_rate=int(MiniMaxH3AudioVAEArchConfig().sampling_rate))


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

    def _input_vae(self) -> Any:
        live = self.get_module("vae")
        if live is not None:
            return live
        return self.fastvideo_args.pipeline_config.vae_config.arch_config

    def _input_audio_vae(self, *, ref2va: bool) -> Any | None:
        if not ref2va:
            return None
        return self.get_module("audio_vae") or _default_audio_geometry()

    def _add_condition_stages(self, *, ref2va: bool) -> None:
        self.add_stage(
            "input_preparation_stage",
            MiniMaxH3InputPreparationStage(
                vae=self._input_vae(),
                audio_vae=self._input_audio_vae(ref2va=ref2va),
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

    def _add_stages(self, *, ref2va: bool) -> None:
        self._ref2va = ref2va
        self._add_condition_stages(ref2va=ref2va)
        if self._denoise_modules_loaded():
            self._add_denoise_stages(ref2va=ref2va)

    def forward(self, batch: ForwardBatch, fastvideo_args: FastVideoArgs) -> ForwardBatch:
        if not self.post_init_called:
            self.post_init()

        if self._denoise_stages_ready:
            return super().forward(batch, fastvideo_args)

        logger.info("Running MiniMax-H3 condition stages before loading DiT/VAE weights")
        for stage in self.stages:
            batch = stage(batch, fastvideo_args)
        self._release_text_encoder()
        self._load_denoise_modules(fastvideo_args)
        self._add_denoise_stages(ref2va=self._ref2va)
        for name in (
                "latent_preparation_stage",
                "denoising_stage",
                "video_decoding_stage",
                "audio_decoding_stage",
        ):
            batch = self._stage_name_mapping[name](batch, fastvideo_args)
        return batch


class MiniMaxH3Pipeline(MiniMaxH3BasePipeline):
    """One-request joint video/stereo-audio pipeline for T2VA and FL2VA."""

    def create_pipeline_stages(self, fastvideo_args: FastVideoArgs) -> None:
        del fastvideo_args
        self._add_stages(ref2va=False)


class MiniMaxH3RefPipeline(MiniMaxH3BasePipeline):
    """Ordered-reference joint video/stereo-audio pipeline for Ref2VA."""

    _extra_config_module_map = {"transformer": "transformer_ref"}
    _ref2va_default = True

    def create_pipeline_stages(self, fastvideo_args: FastVideoArgs) -> None:
        del fastvideo_args
        self._add_stages(ref2va=True)


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
