# SPDX-License-Identifier: Apache-2.0
# Adapted from https://github.com/vllm-project/vllm/blob/v0.6.4.post1/vllm/model_executor/models/registry.py
# and https://github.com/sgl-project/sglang/blob/v0.4.3/python/sglang/srt/models/registry.py

import importlib
import pkgutil
from dataclasses import dataclass, field
from functools import lru_cache
from typing import AbstractSet, Dict, Optional, Tuple, Type, Union

from fastvideo.v1.logger import init_logger
from fastvideo.v1.pipelines.composed_pipeline_base import ComposedPipelineBase
from fastvideo.v1.pipelines.lora_pipeline import LoRAPipeline

logger = init_logger(__name__)


@dataclass
class _PipelineRegistry:
    # Keyed by pipeline_arch
    pipelines: Dict[str, Optional[Type[ComposedPipelineBase]]] = field(
        default_factory=dict)

    def get_supported_archs(self) -> AbstractSet[str]:
        return self.pipelines.keys()

    def _try_load_pipeline_cls(
            self, pipeline_arch: str) -> Optional[Type[ComposedPipelineBase]]:
        if pipeline_arch not in self.pipelines:
            return None

        return self.pipelines[pipeline_arch]

    def resolve_pipeline_cls(
        self,
        architecture: str,
    ) -> Tuple[Union[Type[ComposedPipelineBase], Type[LoRAPipeline]], str]:
        if not architecture:
            logger.warning("No pipeline architecture is specified")

        pipeline_cls = self._try_load_pipeline_cls(architecture)
        if pipeline_cls is not None:
            return (pipeline_cls, architecture)

        supported_archs = self.get_supported_archs()
        raise ValueError(
            f"Pipeline architectures {architecture} are not supported for now. "
            f"Supported architectures: {supported_archs}")


@lru_cache
def import_pipeline_classes():
    pipeline_arch_name_to_cls = {}
    package_name = "fastvideo.v1.pipelines"
    package = importlib.import_module(package_name)
    for _, name, ispkg in pkgutil.iter_modules(package.__path__,
                                               package_name + "."):
        if ispkg:
            if name.split(".")[-1] == "stages":
                continue
            sub_package_name = name
            sub_package = importlib.import_module(sub_package_name)
            for _, name, ispkg in pkgutil.iter_modules(sub_package.__path__,
                                                       sub_package_name + "."):
                try:
                    module = importlib.import_module(name)
                except Exception as e:
                    logger.warning("Ignore import error when loading %s. %s",
                                   name, e)
                    continue
                if hasattr(module, "EntryClass"):
                    entry = module.EntryClass
                    if isinstance(
                            entry, list
                    ):  # To support multiple pipeline classes in one module
                        for tmp in entry:
                            assert (
                                tmp.__name__ not in pipeline_arch_name_to_cls
                            ), f"Duplicated pipeline implementation for {tmp.__name__}"
                            pipeline_arch_name_to_cls[tmp.__name__] = tmp
                    else:
                        assert (
                            entry.__name__ not in pipeline_arch_name_to_cls
                        ), f"Duplicated pipeline implementation for {entry.__name__}"
                        pipeline_arch_name_to_cls[entry.__name__] = entry
    return pipeline_arch_name_to_cls


PipelineRegistry = _PipelineRegistry(import_pipeline_classes())
