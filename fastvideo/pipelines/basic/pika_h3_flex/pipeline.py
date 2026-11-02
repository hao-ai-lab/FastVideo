# SPDX-License-Identifier: Apache-2.0
"""Pika H3 Flex pipeline: MiniMax-H3 components, Pika adapter and sampler."""

from __future__ import annotations

import math
from typing import Any

from fastvideo.configs.pipelines.pika_h3_flex import (
    PIKA_AUDIO_SCHEDULER_SHIFT,
    PIKA_VIDEO_SCHEDULER_SHIFT,
    PikaH3FlexPipelineConfig,
)
from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.logger import init_logger
from fastvideo.pipelines.basic.minimax_h3.minimax_h3_pipeline import MiniMaxH3BasePipeline
from fastvideo.pipelines.basic.pika_h3_flex.adapter import adapter_hf_patterns, load_adapter, task_for_pipeline
from fastvideo.pipelines.basic.pika_h3_flex.stages.denoising import PikaH3FlexDenoisingStage
from fastvideo.pipelines.basic.pika_h3_flex.stages.input_preparation import PikaH3FlexInputPreparationStage

logger = init_logger(__name__)


class PikaH3FlexBasePipeline(MiniMaxH3BasePipeline):
    """Pika H3 Flex's adapter, bucket geometry, and six-forward re-noise sampler."""

    pipeline_config_cls = PikaH3FlexPipelineConfig
    input_preparation_stage_cls = PikaH3FlexInputPreparationStage
    denoising_stage_cls = PikaH3FlexDenoisingStage
    _ref2va_default = False
    _pika_adapter: Any = None

    @classmethod
    def get_hf_download_allow_patterns(cls) -> list[str]:
        patterns = super().get_hf_download_allow_patterns()
        # The concrete class picks the adapter, so a T2VA download does not
        # fetch the 2.7 GB Ref2VA adapter (and vice versa).
        task = "ref2va" if issubclass(cls, PikaH3FlexRefPipeline) else "fl2va"
        return [*patterns, *adapter_hf_patterns(task)]

    def _resolve_pika_adapter(self) -> None:
        """Point ``lora_path`` at the released adapter before LoRA layers attach.

        ``_load_config`` runs before ``LoRAPipeline`` reads ``lora_path``, so the
        automatic adapter behaves exactly like an explicitly passed one,
        including the lazy/sequential transformer paths.
        """
        task = task_for_pipeline(self._ref2va)
        explicit = self.fastvideo_args.lora_path
        source = explicit if explicit is not None else self.model_path
        revision = None if explicit is not None else self.fastvideo_args.revision
        adapter = load_adapter(source, task=task, revision=revision)
        source_kind = "explicit" if explicit is not None else "released"
        logger.info("Pika H3 Flex: loaded the %s %s adapter from %s", source_kind, task, adapter.path)
        self._pika_adapter = adapter
        self.fastvideo_args.lora_path = adapter.path

    def _load_config(self, model_path: str) -> dict[str, Any]:
        config = super()._load_config(model_path)
        self._resolve_pika_adapter()
        return config

    def initialize_pipeline(self, fastvideo_args: FastVideoArgs) -> None:
        super().initialize_pipeline(fastvideo_args)
        expected = {"video": PIKA_VIDEO_SCHEDULER_SHIFT, "audio": PIKA_AUDIO_SCHEDULER_SHIFT}
        for module_name, modality in (("scheduler", "video"), ("audio_scheduler", "audio")):
            actual = float(self.get_module(module_name).shift)
            target = expected[modality]
            if not math.isfinite(actual) or abs(actual - target) > 1e-9:
                raise ValueError(f"Pika H3 Flex samples with {modality} shift {target:g}, but "
                                 f"{module_name}/scheduler_config.json declares {actual:g}.")

    def create_pipeline_stages(self, fastvideo_args: FastVideoArgs) -> None:
        self._add_stages(fastvideo_args, ref2va=self._ref2va_default)


class PikaH3FlexPipeline(PikaH3FlexBasePipeline):
    """T2VA/I2VA/L2VA/FL2VA pipeline through the ``fl2va`` adapter."""

    _ref2va_default = False


class PikaH3FlexRefPipeline(PikaH3FlexBasePipeline):
    """Ordered-reference pipeline through the ``ref2va`` adapter and ``transformer_ref``."""

    _extra_config_module_map = {"transformer": "transformer_ref"}
    _ref2va_default = True


class PikaH3FlexModularPipeline(PikaH3FlexPipeline):
    """Public entry matching the released ``modular_model_index.json`` class name."""


class PikaH3FlexRef2VAModularPipeline(PikaH3FlexRefPipeline):
    """Public Ref2VA entry matching the released modular class name."""


EntryClass = [PikaH3FlexModularPipeline, PikaH3FlexRef2VAModularPipeline]

__all__ = [
    "EntryClass",
    "PikaH3FlexBasePipeline",
    "PikaH3FlexModularPipeline",
    "PikaH3FlexPipeline",
    "PikaH3FlexRef2VAModularPipeline",
    "PikaH3FlexRefPipeline",
]
