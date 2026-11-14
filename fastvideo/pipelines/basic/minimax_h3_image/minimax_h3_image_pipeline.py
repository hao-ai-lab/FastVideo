# SPDX-License-Identifier: Apache-2.0
"""MiniMax-H3-Image pipeline.

Loads ``transformer_diffusers/`` and ``vae_diffusers/``. The text encoder is
the H3 Qwen3-VL conditioner, so serialized NVFP4/FP8 and
``override_text_encoder_quant`` apply unchanged. DiT FFN linears keep the
names the NVFP4 filter already matches.
"""

from __future__ import annotations

import gc
from typing import Any

import torch

from fastvideo.configs.pipelines.minimax_h3_image import MiniMaxH3ImagePipelineConfig
from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.logger import init_logger
from fastvideo.pipelines.basic.minimax_h3_image.stages import (
    MiniMaxH3ImageConditioningStage,
    MiniMaxH3ImageInputStage,
    MiniMaxH3ImageSamplingStage,
)
from fastvideo.pipelines.composed_pipeline_base import ComposedPipelineBase
from fastvideo.pipelines.lora_pipeline import LoRAPipeline
from fastvideo.pipelines.pipeline_batch_info import ForwardBatch
from fastvideo.utils import verify_model_config_and_directory

logger = init_logger(__name__)

_DENOISE_MODULE_NAMES = ("transformer", "vae")


class MiniMaxH3ImageModularPipeline(LoRAPipeline, ComposedPipelineBase):
    """Text-to-image and reference-edit pipeline for MiniMax-H3-Image."""

    # Same linear names the video adapters target. ``proj_in`` and the PDD
    # heads stay unwrapped: the forward reads ``.weight`` on ``proj_in``.
    lora_target_modules = [
        "attn.to_q",
        "attn.to_k",
        "attn.to_v",
        "attn.to_out",
        "ff.fc_in",
        "ff.fc_out",
        "adaln_proj.linear",
        "norm_out.linear",
    ]

    pipeline_config_cls: type[MiniMaxH3ImagePipelineConfig] = MiniMaxH3ImagePipelineConfig
    _required_config_modules = [
        "text_encoder",
        "tokenizer",
        "processor",
        "vae",
        "transformer",
        "scheduler",
    ]
    _extra_config_module_map = {
        "transformer": "transformer_diffusers",
        "vae": "vae_diffusers",
    }
    _lazy_module_names = ("text_encoder", "transformer", "vae")

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self._sampling_stage_ready = False
        super().__init__(*args, **kwargs)

    @classmethod
    def get_hf_download_component_dirs(cls) -> tuple[str, ...]:
        return tuple(sorted(cls._extra_config_module_map.get(name, name) for name in cls._required_config_modules))

    def _load_config(self, model_path: str) -> dict[str, Any]:
        # Require the diffusers folders, not the native ``transformer/`` tree.
        config = verify_model_config_and_directory(
            model_path,
            required_component_dirs=self.get_hf_download_component_dirs(),
        )
        cleaned: dict[str, Any] = {}
        for key, value in config.items():
            if key.startswith("_"):
                cleaned[key] = value
                continue
            if isinstance(value, list) and value and isinstance(value[0], str):
                cleaned[key] = value[:2]
        scheduler = cleaned.get("scheduler")
        if not isinstance(scheduler, list) or not scheduler or scheduler[0] is None:
            cleaned["scheduler"] = ["diffusers", "MiniMaxH3Scheduler"]
        if "_class_name" not in cleaned or "_diffusers_version" not in cleaned:
            raise ValueError(f"{model_path} is missing a Diffusers manifest.")
        return cleaned

    def initialize_pipeline(self, fastvideo_args: FastVideoArgs) -> None:
        area = int(fastvideo_args.pipeline_config.min_generation_area)
        logger.info("MiniMax-H3-Image min_generation_area=%d", area)

    def _defer_denoise_modules(self, fastvideo_args: FastVideoArgs) -> bool:
        if not fastvideo_args.inference_mode or bool(getattr(fastvideo_args, "training_mode", False)):
            return False
        if bool(getattr(fastvideo_args, "lazy_module_load", False)):
            logger.info("MiniMax-H3-Image sequential module load off: lazy_module_load owns deferral")
            return False
        # The release is a full Qwen3-VL stack plus the DiT. Default to
        # encode-then-release so they are not resident together. ``None`` and
        # ``True`` both take this path; pass ``--no-h3-sequential-load`` to
        # keep both loaded.
        return fastvideo_args.h3_sequential_load is not False

    def _denoise_modules_loaded(self) -> bool:
        return all(self.get_module(name) is not None for name in _DENOISE_MODULE_NAMES)

    def load_modules(self,
                     fastvideo_args: FastVideoArgs,
                     loaded_modules: dict[str, torch.nn.Module] | None = None) -> dict[str, Any]:
        if not self._defer_denoise_modules(fastvideo_args):
            return super().load_modules(fastvideo_args, loaded_modules)
        if loaded_modules is not None and all(name in loaded_modules for name in _DENOISE_MODULE_NAMES):
            return super().load_modules(fastvideo_args, loaded_modules)
        saved = list(self.required_config_modules)
        self._required_config_modules = [name for name in saved if name not in _DENOISE_MODULE_NAMES]
        try:
            logger.info("Loading MiniMax-H3-Image condition modules first: %s", self._required_config_modules)
            return super().load_modules(fastvideo_args, loaded_modules)
        finally:
            self._required_config_modules = saved

    def _load_denoise_modules(self, fastvideo_args: FastVideoArgs) -> None:
        if self._denoise_modules_loaded():
            return
        saved = list(self.required_config_modules)
        self._required_config_modules = [name for name in saved if name != "text_encoder"]
        try:
            logger.info("Loading MiniMax-H3-Image DiT and VAE after releasing the text encoder")
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
        logger.info("Released MiniMax-H3-Image text encoder after conditioning")
        del encoder
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _ensure_text_encoder(self, fastvideo_args: FastVideoArgs) -> None:
        encoder = self.get_module("text_encoder")
        stage = self._stage_name_mapping.get("conditioning_stage")
        if encoder is not None:
            if stage is not None and getattr(stage, "conditioner", None) is None:
                stage.conditioner = encoder
            return
        saved = list(self.required_config_modules)
        self._required_config_modules = ["text_encoder"]
        try:
            logger.info("Reloading MiniMax-H3-Image text encoder")
            loaded = super().load_modules(fastvideo_args, loaded_modules=self.modules)
            for name, module in loaded.items():
                self.add_module(name, module)
            self._apply_inference_compile(("text_encoder", ))
        finally:
            self._required_config_modules = saved
        if stage is not None:
            stage.conditioner = self.get_module("text_encoder")

    def _add_condition_stages(self, fastvideo_args: FastVideoArgs) -> None:
        area = int(fastvideo_args.pipeline_config.min_generation_area)
        self.add_stage("input_stage", MiniMaxH3ImageInputStage(area))
        self.add_stage(
            "conditioning_stage",
            MiniMaxH3ImageConditioningStage(
                conditioner=self.get_module("text_encoder"),
                tokenizer=self.get_module("tokenizer"),
                processor=self.get_module("processor"),
            ),
        )

    def _add_sampling_stage(self, fastvideo_args: FastVideoArgs) -> None:
        if self._sampling_stage_ready:
            return
        self.add_stage(
            "sampling_stage",
            MiniMaxH3ImageSamplingStage(
                vae=self.get_module("vae"),
                transformer=self.get_module("transformer"),
                scheduler=self.get_module("scheduler"),
                min_generation_area=int(fastvideo_args.pipeline_config.min_generation_area),
            ),
        )
        self._sampling_stage_ready = True

    def create_pipeline_stages(self, fastvideo_args: FastVideoArgs) -> None:
        self._add_condition_stages(fastvideo_args)
        if self._denoise_modules_loaded():
            self._add_sampling_stage(fastvideo_args)

    def _run_condition_then_denoise(self, batch: ForwardBatch, fastvideo_args: FastVideoArgs) -> ForwardBatch:
        for name in ("input_stage", "conditioning_stage"):
            batch = self._stage_name_mapping[name](batch, fastvideo_args)
        self._release_text_encoder()
        self._load_denoise_modules(fastvideo_args)
        self._add_sampling_stage(fastvideo_args)
        return self._stage_name_mapping["sampling_stage"](batch, fastvideo_args)

    def forward(self, batch: ForwardBatch, fastvideo_args: FastVideoArgs) -> ForwardBatch:
        if self._defer_denoise_modules(fastvideo_args):
            if not self.post_init_called:
                self.post_init()
            try:
                self._ensure_text_encoder(fastvideo_args)
                return self._run_condition_then_denoise(batch, fastvideo_args)
            except BaseException:
                self._release_all_lazy_modules()
                raise
        return super().forward(batch, fastvideo_args)


EntryClass = MiniMaxH3ImageModularPipeline
