
# Adapted from https://github.com/vllm-project/vllm/blob/v0.6.4.post1/vllm/model_executor/models/registry.py
# and https://github.com/sgl-project/sglang/blob/v0.4.3/python/sglang/srt/models/registry.py

import importlib
import pkgutil
from dataclasses import dataclass, field
from functools import lru_cache
from typing import AbstractSet, Dict, List, Optional, Tuple, Type, Union

from fastvideo.logger import init_logger

import torch.nn as nn

logger = init_logger(__name__)


@dataclass
class _DiTRegistry:
    # Keyed by pipeline_arch
    dits : Dict[str, Union[Type[nn.Module], str]] = field(default_factory=dict)

    def get_supported_archs(self) -> AbstractSet[str]:
        return self.dits.keys()

    def _raise_for_unsupported(self, architectures: List[str]):
        all_supported_archs = self.get_supported_archs()

        if any(arch in all_supported_archs for arch in architectures):
            raise ValueError(
                f"DiT architectures {architectures} failed "
                "to be inspected. Please check the logs for more details."
            )

        raise ValueError(
            f"DiT architectures {architectures} are not supported for now. "
            f"Supported architectures: {all_supported_archs}"
        )

    def _try_load_dit_cls(self, dit_arch: str) -> Optional[Type[nn.Module]]:
        if dit_arch not in self.dits:
            return None

        return self.dits[dit_arch]

    def resolve_dit_cls(
        self,
        architecture: str,
    ) -> Tuple[Type[nn.Module], str]:
        if not architecture:
            logger.warning("No dit architecture is specified")

        dit_cls = self._try_load_dit_cls(architecture)
        if dit_cls is not None:
            return (dit_cls, architecture)

        return self._raise_for_unsupported(architecture)


@lru_cache()
def import_dit_classes():
    dit_arch_name_to_cls = {}
    package_name = "fastvideo.models.dits"
    package = importlib.import_module(package_name)
    for _, name, ispkg in pkgutil.iter_modules(package.__path__, package_name + "."):
        if ispkg:
            try:
                module = importlib.import_module(name)
            except Exception as e:
                logger.warning(f"Ignore import error when loading {name}. " f"{e}")
                continue
            logger.info("ok")
            logger.info(f"{module}")
            logger.info(f"Has EntryClass: {hasattr(module, 'EntryClass')}")
            from fastvideo.utils.logging_ import ForkedPdb
            ForkedPdb().set_trace()
            if hasattr(module, "EntryClass"):
                entry = module.EntryClass
                print(entry)
                print(entry.__name__)
                if isinstance(
                    entry, list
                ):  # To support multiple pipeline classes in one module
                    for tmp in entry:
                        assert (
                            tmp.__name__ not in dit_arch_name_to_cls
                        ), f"Duplicated dit implementation for {tmp.__name__}"
                        dit_arch_name_to_cls[tmp.__name__] = tmp
                else:
                    assert (
                        entry.__name__ not in dit_arch_name_to_cls
                    ), f"Duplicated dit implementation for {entry.__name__}"
                    dit_arch_name_to_cls[entry.__name__] = entry
    return dit_arch_name_to_cls


DiTRegistry = _DiTRegistry(import_dit_classes())