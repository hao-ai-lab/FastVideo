# SPDX-License-Identifier: Apache-2.0
"""
Stable Audio pipeline for text-to-audio generation.

Supports loading from:
- Unified format: model.safetensors or model.ckpt at model root + model_config.json
- HuggingFace: stabilityai/stable-audio-open-1.0 (uses unified when available)
"""
from __future__ import annotations

import json
import os
from typing import Any

from fastvideo.configs.pipelines.stable_audio import StableAudioPipelineConfig
from fastvideo.configs.sample.stable_audio import StableAudioSamplingParam
from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.logger import init_logger
from fastvideo.models.loader.component_loader import PipelineComponentLoader
from fastvideo.models.stable_audio.conditioner import StableAudioConditioner
from fastvideo.models.stable_audio.pretransform import StableAudioPretransform
from fastvideo.pipelines.composed_pipeline_base import ComposedPipelineBase
from fastvideo.pipelines.stages import (
    StableAudioConditioningStage,
    StableAudioDecodingStage,
    StableAudioDenoisingStage,
    StableAudioInputValidationStage,
    StableAudioLatentPreparationStage,
)

logger = init_logger(__name__)

UNIFIED_CHECKPOINT_NAMES = ("model.safetensors", "model.ckpt")


def _detect_unified_checkpoint(model_root: str) -> str | None:
    """Return path to unified checkpoint if present, else None."""
    for name in UNIFIED_CHECKPOINT_NAMES:
        path = os.path.join(model_root, name)
        if os.path.isfile(path):
            return path
    return None


def _is_unified_format(model_path: str) -> bool:
    """True if model root has unified checkpoint + model_config.json."""
    model_root = model_path.rstrip(os.sep)
    config_path = os.path.join(model_root, "model_config.json")
    if not os.path.isfile(config_path):
        return False
    return _detect_unified_checkpoint(model_root) is not None


class StableAudioPipeline(ComposedPipelineBase):
    """
    Text-to-audio pipeline using Stable Audio Open.

    Loads from unified checkpoint (model.safetensors / model.ckpt + model_config.json)
    or from HuggingFace-style diffusers layout when unified is not available.
    """

    pipeline_config_cls = StableAudioPipelineConfig
    sampling_params_cls = StableAudioSamplingParam

    _required_config_modules = [
        "conditioner",
        "pretransform",
        "transformer",
    ]

    def create_pipeline_stages(self, fastvideo_args: FastVideoArgs) -> None:
        self.add_stage(
            stage_name="input_validation_stage",
            stage=StableAudioInputValidationStage(),
        )
        self.add_stage(
            stage_name="conditioning_stage",
            stage=StableAudioConditioningStage(
                conditioner=self.get_module("conditioner")),
        )
        self.add_stage(
            stage_name="latent_preparation_stage",
            stage=StableAudioLatentPreparationStage(
                pretransform=self.get_module("pretransform")),
        )
        self.add_stage(
            stage_name="denoising_stage",
            stage=StableAudioDenoisingStage(
                transformer=self.get_module("transformer"), ),
        )
        self.add_stage(
            stage_name="decoding_stage",
            stage=StableAudioDecodingStage(
                pretransform=self.get_module("pretransform"), ),
        )

    def load_modules(
        self,
        fastvideo_args: FastVideoArgs,
        loaded_modules: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        model_path = self.model_path.rstrip(os.sep)
        config_path = os.path.join(model_path, "model_config.json")
        unified_ckpt = _detect_unified_checkpoint(model_path)

        if unified_ckpt is not None and os.path.isfile(config_path):
            return self._load_unified(model_path, config_path, unified_ckpt,
                                      fastvideo_args)
        return self._load_diffusers(model_path, fastvideo_args, loaded_modules)

    def _load_unified(
        self,
        model_path: str,
        config_path: str,
        checkpoint_path: str,
        fastvideo_args: FastVideoArgs,
    ) -> dict[str, Any]:
        """Load from unified checkpoint (model.safetensors / model.ckpt + model_config.json)."""
        with open(config_path, encoding="utf-8") as f:
            model_config = json.load(f)

        pretransform = StableAudioPretransform(
            model_config=model_config,
            checkpoint_path=checkpoint_path,
        )
        conditioner = StableAudioConditioner(
            model_config=model_config,
            checkpoint_path=checkpoint_path,
        )

        transformer_path = os.path.join(model_path, "transformer")
        transformer = PipelineComponentLoader.load_module(
            module_name="transformer",
            component_model_path=transformer_path,
            transformers_or_diffusers="diffusers",
            fastvideo_args=fastvideo_args,
        )

        sample_rate = getattr(fastvideo_args.pipeline_config, "sample_rate",
                              44100)
        logger.info(
            "Loaded Stable Audio (unified) from %s, sample_rate=%d",
            checkpoint_path,
            sample_rate,
        )

        return {
            "conditioner": conditioner,
            "pretransform": pretransform,
            "transformer": transformer,
        }

    def _load_diffusers(
        self,
        model_path: str,
        fastvideo_args: FastVideoArgs,
        loaded_modules: dict[str, Any] | None,
    ) -> dict[str, Any]:
        """Load from HuggingFace diffusers layout (model_index.json + subdirs)."""
        model_index = self._load_config(model_path)
        logger.info("Loading Stable Audio (diffusers) from %s", model_index)

        model_index.pop("_class_name")
        model_index.pop("_diffusers_version")
        model_index.pop("_name_or_path", None)
        model_index.pop("workload_type", None)

        config_path = os.path.join(model_path, "model_config.json")
        if not os.path.isfile(config_path):
            raise ValueError(
                "Diffusers layout requires model_config.json for Stable Audio. "
                "Ensure model_config.json exists, or use a unified checkpoint "
                "(model.safetensors + model_config.json).")
        with open(config_path, encoding="utf-8") as f:
            json.load(f)  # Validate model_config.json is valid

        unified_ckpt = _detect_unified_checkpoint(model_path)
        if unified_ckpt is None:
            raise ValueError(
                "Stable Audio requires a unified checkpoint (model.safetensors or model.ckpt) "
                "at the model root. The diffusers layout alone is not supported yet."
            )

        return self._load_unified(model_path, config_path, unified_ckpt,
                                  fastvideo_args)


EntryClass = StableAudioPipeline
