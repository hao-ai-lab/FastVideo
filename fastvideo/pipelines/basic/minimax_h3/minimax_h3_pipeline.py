# SPDX-License-Identifier: Apache-2.0
"""Private FastVideo composed pipelines for MiniMax H3."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, cast

import torch

from fastvideo.configs.pipelines.base import PipelineConfig
from fastvideo.configs.pipelines.minimax_h3 import MiniMaxH3PipelineConfig
from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.pipelines.basic.minimax_h3.stages import (
    MiniMaxH3AudioDecodingStage,
    MiniMaxH3ConditioningStage,
    MiniMaxH3DenoisingStage,
    MiniMaxH3FL2VALayoutPreparationStage,
    MiniMaxH3InputPreparationStage,
    MiniMaxH3KeyframeEncodingStage,
    MiniMaxH3LatentPreparationStage,
    MiniMaxH3Ref2VAConditioningStage,
    MiniMaxH3Ref2VALayoutPreparationStage,
    MiniMaxH3ReferenceEncodingStage,
    MiniMaxH3ReferencePreparationStage,
    MiniMaxH3TimestepPreparationStage,
    MiniMaxH3VideoDecodingStage,
)
from fastvideo.pipelines.composed_pipeline_base import ComposedPipelineBase
from fastvideo.utils import maybe_download_model


def _normalize_modular_model_index(config: dict[str, Any]) -> dict[str, Any]:
    """Translate H3's modular component specs at the family loading boundary."""
    normalized: dict[str, Any] = {}
    for name, entry in config.items():
        if name.startswith("_") and name not in {"_class_name", "_diffusers_version"}:
            continue
        if not isinstance(entry, list | tuple):
            normalized[name] = entry
            continue
        if len(entry) == 2:
            normalized[name] = list(entry)
            continue
        if len(entry) != 3 or not isinstance(entry[2], dict):
            raise ValueError(f"Invalid modular component specification for {name!r}: {entry!r}")

        library, class_name, loading = entry
        type_hint = loading.get("type_hint", (library, class_name))
        if not isinstance(type_hint, list | tuple) or len(type_hint) != 2:
            raise ValueError(f"Invalid modular type_hint for {name!r}: {type_hint!r}")
        normalized[name] = list(type_hint)

        subfolder = loading.get("subfolder")
        if isinstance(subfolder, str) and subfolder != name:
            raise ValueError(f"Modular component {name!r} uses subfolder {subfolder!r}; "
                             "FastVideo requires component names and subfolders to match.")

    if "_diffusers_version" not in normalized:
        raise ValueError("modular_model_index.json does not contain _diffusers_version")
    return normalized


class MiniMaxH3BasePipeline(ComposedPipelineBase):
    """Shared private loading and target-generation path for MiniMax H3."""

    pipeline_config_cls: type[MiniMaxH3PipelineConfig] = MiniMaxH3PipelineConfig
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

    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        device: str | None = None,
        torch_dtype: torch.dtype | None = None,
        pipeline_config: str | PipelineConfig | None = None,
        args: argparse.Namespace | None = None,
        required_config_modules: list[str] | None = None,
        loaded_modules: dict[str, torch.nn.Module] | None = None,
        **kwargs: Any,
    ) -> MiniMaxH3BasePipeline:
        """Load the private pipeline without requiring a public config detector."""
        if pipeline_config is None:
            pipeline_config = MiniMaxH3PipelineConfig()
        elif isinstance(pipeline_config, str):
            config_path = pipeline_config
            pipeline_config = MiniMaxH3PipelineConfig()
            pipeline_config.load_from_json(config_path)
            pipeline_config.pipeline_config_path = config_path
        pipeline = super().from_pretrained(
            model_path=model_path,
            device=device,
            torch_dtype=torch_dtype,
            pipeline_config=pipeline_config,
            args=args,
            required_config_modules=required_config_modules,
            loaded_modules=loaded_modules,
            **kwargs,
        )
        return cast("MiniMaxH3BasePipeline", pipeline)

    def _load_config(self, model_path: str) -> dict[str, Any]:
        """Load a canonical or modular Diffusers component manifest."""
        resolved = Path(maybe_download_model(model_path, revision=getattr(self.fastvideo_args, "revision", None)))
        self.model_path = str(resolved)
        if (resolved / "model_index.json").is_file():
            return super()._load_config(str(resolved))

        modular_path = resolved / "modular_model_index.json"
        if not modular_path.is_file():
            raise ValueError(f"MiniMax-H3 model directory {resolved} has no component manifest.")
        with modular_path.open(encoding="utf-8") as file:
            return _normalize_modular_model_index(json.load(file))

    def initialize_pipeline(self, fastvideo_args: FastVideoArgs) -> None:
        del fastvideo_args
        for module_name, modality, expected_shift in (
            ("scheduler", "video", 12.0),
            ("audio_scheduler", "audio", 3.0),
        ):
            shift = getattr(self.get_module(module_name), "shift", None)
            if shift is None or float(shift) != expected_shift:
                raise ValueError(f"MiniMax-H3 {modality} scheduler must expose shift={expected_shift:g}, got {shift}.")

    def _add_target_generation_stages(
        self,
        transformer: Any,
        vae: Any,
        audio_vae: Any,
        scheduler: Any,
        audio_scheduler: Any,
    ) -> None:
        self.add_stage(
            "latent_preparation_stage",
            MiniMaxH3LatentPreparationStage(transformer=transformer, vae=vae, audio_vae=audio_vae),
        )
        self.add_stage(
            "timestep_preparation_stage",
            MiniMaxH3TimestepPreparationStage(scheduler=scheduler, audio_scheduler=audio_scheduler),
        )
        self.add_stage(
            "denoising_stage",
            MiniMaxH3DenoisingStage(
                transformer=transformer,
                scheduler=scheduler,
                audio_scheduler=audio_scheduler,
            ),
        )
        self.add_stage("video_decoding_stage", MiniMaxH3VideoDecodingStage(vae=vae, transformer=transformer))
        self.add_stage("audio_decoding_stage", MiniMaxH3AudioDecodingStage(audio_vae=audio_vae))


class MiniMaxH3Pipeline(MiniMaxH3BasePipeline):
    """One-request joint video/stereo-audio pipeline for T2VA and FL2VA."""

    def create_pipeline_stages(self, fastvideo_args: FastVideoArgs) -> None:
        del fastvideo_args
        transformer = self.get_module("transformer")
        vae = self.get_module("vae")
        audio_vae = self.get_module("audio_vae")
        scheduler = self.get_module("scheduler")
        audio_scheduler = self.get_module("audio_scheduler")

        self.add_stage("input_preparation_stage", MiniMaxH3InputPreparationStage(vae=vae))
        self.add_stage(
            "conditioning_stage",
            MiniMaxH3ConditioningStage(
                conditioner=self.get_module("text_encoder"),
                tokenizer=self.get_module("tokenizer"),
                processor=self.get_module("processor"),
            ),
        )
        self.add_stage(
            "keyframe_encoding_stage",
            MiniMaxH3KeyframeEncodingStage(vae=vae, transformer=transformer, scheduler=scheduler),
        )
        self.add_stage(
            "layout_preparation_stage",
            MiniMaxH3FL2VALayoutPreparationStage(transformer=transformer),
        )
        self._add_target_generation_stages(
            transformer,
            vae,
            audio_vae,
            scheduler,
            audio_scheduler,
        )


class MiniMaxH3RefPipeline(MiniMaxH3BasePipeline):
    """Ordered-reference joint video/stereo-audio pipeline for Ref2VA."""

    _extra_config_module_map = {"transformer": "transformer_ref"}

    def create_pipeline_stages(self, fastvideo_args: FastVideoArgs) -> None:
        del fastvideo_args
        transformer = self.get_module("transformer")
        vae = self.get_module("vae")
        audio_vae = self.get_module("audio_vae")
        scheduler = self.get_module("scheduler")
        audio_scheduler = self.get_module("audio_scheduler")

        self.add_stage(
            "reference_preparation_stage",
            MiniMaxH3ReferencePreparationStage(vae=vae, audio_vae=audio_vae),
        )
        self.add_stage(
            "conditioning_stage",
            MiniMaxH3Ref2VAConditioningStage(
                conditioner=self.get_module("text_encoder"),
                tokenizer=self.get_module("tokenizer"),
                processor=self.get_module("processor"),
            ),
        )
        self.add_stage(
            "reference_encoding_stage",
            MiniMaxH3ReferenceEncodingStage(
                vae=vae,
                audio_vae=audio_vae,
                transformer=transformer,
                scheduler=scheduler,
            ),
        )
        self.add_stage(
            "layout_preparation_stage",
            MiniMaxH3Ref2VALayoutPreparationStage(transformer=transformer),
        )
        self._add_target_generation_stages(
            transformer,
            vae,
            audio_vae,
            scheduler,
            audio_scheduler,
        )


__all__ = ["MiniMaxH3BasePipeline", "MiniMaxH3Pipeline", "MiniMaxH3RefPipeline"]
