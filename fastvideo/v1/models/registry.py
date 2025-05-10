# SPDX-License-Identifier: Apache-2.0
# Adapted from vllm: https://github.com/vllm-project/vllm/blob/v0.7.3/vllm/model_executor/models/registry.py

import importlib
import os
import pickle
import subprocess
import sys
import tempfile
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from functools import lru_cache
from typing import (AbstractSet, Callable, Dict, List, NoReturn, Optional,
                    Tuple, Type, TypeVar, Union, cast)

import cloudpickle
from torch import nn

from fastvideo.v1.logger import logger

# huggingface class name: (component_name, fastvideo module name, fastvideo class name)
_TEXT_TO_VIDEO_DIT_MODELS = {
    "HunyuanVideoTransformer3DModel":
    ("dits", "hunyuanvideo", "HunyuanVideoTransformer3DModel"),
    "WanTransformer3DModel": ("dits", "wanvideo", "WanTransformer3DModel"),
    "StepVideoModel": ("dits", "stepvideo", "StepVideoModel")
}

_IMAGE_TO_VIDEO_DIT_MODELS = {
    # "HunyuanVideoTransformer3DModel": ("dits", "hunyuanvideo", "HunyuanVideoDiT"),
    "WanTransformer3DModel": ("dits", "wanvideo", "WanTransformer3DModel"),
}

_TEXT_ENCODER_MODELS = {
    "CLIPTextModel": ("encoders", "clip", "CLIPTextModel"),
    "LlamaModel": ("encoders", "llama", "LlamaModel"),
    "UMT5EncoderModel": ("encoders", "t5", "UMT5EncoderModel"),
    "STEP1TextEncoder": ("encoders", "stepllm", "STEP1TextEncoder"),
    "BertModel": ("encoders", "clip", "CLIPTextModel"),
}

_IMAGE_ENCODER_MODELS: dict[str, tuple] = {
    # "HunyuanVideoTransformer3DModel": ("image_encoder", "hunyuanvideo", "HunyuanVideoImageEncoder"),
    "CLIPVisionModelWithProjection": ("encoders", "clip", "CLIPVisionModel"),
}

_VAE_MODELS = {
    "AutoencoderKLHunyuanVideo":
    ("vaes", "hunyuanvae", "AutoencoderKLHunyuanVideo"),
    "AutoencoderKLWan": ("vaes", "wanvae", "AutoencoderKLWan"),
    "AutoencoderKLStepvideo": ("vaes", "stepvideovae", "AutoencoderKLStepvideo")
}

_SCHEDULERS = {
    "FlowMatchEulerDiscreteScheduler":
    ("schedulers", "scheduling_flow_match_euler_discrete",
     "FlowMatchDiscreteScheduler"),
    "UniPCMultistepScheduler":
    ("schedulers", "scheduling_unipc_multistep", "UniPCMultistepScheduler"),
}

_FAST_VIDEO_MODELS = {
    **_TEXT_TO_VIDEO_DIT_MODELS,
    **_IMAGE_TO_VIDEO_DIT_MODELS,
    **_TEXT_ENCODER_MODELS,
    **_IMAGE_ENCODER_MODELS,
    **_VAE_MODELS,
    **_SCHEDULERS,
}

_SUBPROCESS_COMMAND = [
    sys.executable, "-m", "fastvideo.v1.models.dits.registry"
]

_T = TypeVar("_T")


@dataclass(frozen=True)
class _ModelInfo:
    architecture: str

    @staticmethod
    def from_model_cls(model: Type[nn.Module]) -> "_ModelInfo":
        return _ModelInfo(architecture=model.__name__, )


class _BaseRegisteredModel(ABC):

    @abstractmethod
    def inspect_model_cls(self) -> _ModelInfo:
        raise NotImplementedError

    @abstractmethod
    def load_model_cls(self) -> Type[nn.Module]:
        raise NotImplementedError


@dataclass(frozen=True)
class _RegisteredModel(_BaseRegisteredModel):
    """
    Represents a model that has already been imported in the main process.
    """

    interfaces: _ModelInfo
    model_cls: Type[nn.Module]

    @staticmethod
    def from_model_cls(model_cls: Type[nn.Module]):
        return _RegisteredModel(
            interfaces=_ModelInfo.from_model_cls(model_cls),
            model_cls=model_cls,
        )

    def inspect_model_cls(self) -> _ModelInfo:
        return self.interfaces

    def load_model_cls(self) -> Type[nn.Module]:
        return self.model_cls


def _run_in_subprocess(fn: Callable[[], _T]) -> _T:
    # NOTE: We use a temporary directory instead of a temporary file to avoid
    # issues like https://stackoverflow.com/questions/23212435/permission-denied-to-write-to-my-temporary-file
    with tempfile.TemporaryDirectory() as tempdir:
        output_filepath = os.path.join(tempdir, "registry_output.tmp")

        # `cloudpickle` allows pickling lambda functions directly
        input_bytes = cloudpickle.dumps((fn, output_filepath))

        # cannot use `sys.executable __file__` here because the script
        # contains relative imports
        returned = subprocess.run(_SUBPROCESS_COMMAND,
                                  input=input_bytes,
                                  capture_output=True)

        # check if the subprocess is successful
        try:
            returned.check_returncode()
        except Exception as e:
            # wrap raised exception to provide more information
            raise RuntimeError(f"Error raised in subprocess:\n"
                               f"{returned.stderr.decode()}") from e

        with open(output_filepath, "rb") as f:
            return cast(_T, pickle.load(f))


@dataclass(frozen=True)
class _LazyRegisteredModel(_BaseRegisteredModel):
    """
    Represents a model that has not been imported in the main process.
    """
    module_name: str
    component_name: str
    class_name: str

    # Performed in another process to avoid initializing CUDA
    def inspect_model_cls(self) -> _ModelInfo:
        return _run_in_subprocess(
            lambda: _ModelInfo.from_model_cls(self.load_model_cls()))

    def load_model_cls(self) -> Type[nn.Module]:
        mod = importlib.import_module(self.module_name)
        return cast(Type[nn.Module], getattr(mod, self.class_name))


@lru_cache(maxsize=128)
def _try_load_model_cls(
    model_arch: str,
    model: _BaseRegisteredModel,
) -> Optional[Type[nn.Module]]:
    from fastvideo.v1.platforms import current_platform
    current_platform.verify_model_arch(model_arch)
    try:
        return model.load_model_cls()
    except Exception:
        logger.exception("Error in loading model architecture '%s'", model_arch)
        return None


@lru_cache(maxsize=128)
def _try_inspect_model_cls(
    model_arch: str,
    model: _BaseRegisteredModel,
) -> Optional[_ModelInfo]:
    try:
        return model.inspect_model_cls()
    except Exception:
        logger.exception("Error in inspecting model architecture '%s'",
                         model_arch)
        return None


@dataclass
class _ModelRegistry:
    # Keyed by model_arch
    models: Dict[str, _BaseRegisteredModel] = field(default_factory=dict)

    def get_supported_archs(self) -> AbstractSet[str]:
        return self.models.keys()

    def register_model(
        self,
        model_arch: str,
        model_cls: Union[Type[nn.Module], str],
    ) -> None:
        """
        Register an external model to be used in vLLM.

        :code:`model_cls` can be either:

        - A :class:`torch.nn.Module` class directly referencing the model.
        - A string in the format :code:`<module>:<class>` which can be used to
          lazily import the model. This is useful to avoid initializing CUDA
          when importing the model and thus the related error
          :code:`RuntimeError: Cannot re-initialize CUDA in forked subprocess`.
        """
        if model_arch in self.models:
            logger.warning(
                "Model architecture %s is already registered, and will be "
                "overwritten by the new model class %s.", model_arch, model_cls)

        if isinstance(model_cls, str):
            split_str = model_cls.split(":")
            if len(split_str) != 2:
                msg = "Expected a string in the format `<module>:<class>`"
                raise ValueError(msg)

            model = _LazyRegisteredModel(*split_str)
        else:
            model = _RegisteredModel.from_model_cls(model_cls)

        self.models[model_arch] = model

    def _raise_for_unsupported(self, architectures: List[str]) -> NoReturn:
        all_supported_archs = self.get_supported_archs()

        if any(arch in all_supported_archs for arch in architectures):
            raise ValueError(
                f"Model architectures {architectures} failed "
                "to be inspected. Please check the logs for more details.")

        raise ValueError(
            f"Model architectures {architectures} are not supported for now. "
            f"Supported architectures: {all_supported_archs}")

    def _try_load_model_cls(self, model_arch: str) -> Optional[Type[nn.Module]]:
        if model_arch not in self.models:
            return None

        return _try_load_model_cls(model_arch, self.models[model_arch])

    def _try_inspect_model_cls(self, model_arch: str) -> Optional[_ModelInfo]:
        if model_arch not in self.models:
            return None

        return _try_inspect_model_cls(model_arch, self.models[model_arch])

    def _normalize_archs(
        self,
        architectures: Union[str, List[str]],
    ) -> List[str]:
        if isinstance(architectures, str):
            architectures = [architectures]
        if not architectures:
            logger.warning("No model architectures are specified")

        normalized_arch = []
        for model in architectures:
            if model not in self.models:
                model = "TransformersModel"
            normalized_arch.append(model)
        return normalized_arch

    def inspect_model_cls(
        self,
        architectures: Union[str, List[str]],
    ) -> Tuple[_ModelInfo, str]:
        architectures = self._normalize_archs(architectures)

        for arch in architectures:
            model_info = self._try_inspect_model_cls(arch)
            if model_info is not None:
                return (model_info, arch)

        return self._raise_for_unsupported(architectures)

    def resolve_model_cls(
        self,
        architectures: Union[str, List[str]],
    ) -> Tuple[Type[nn.Module], str]:
        architectures = self._normalize_archs(architectures)

        for arch in architectures:
            model_cls = self._try_load_model_cls(arch)
            if model_cls is not None:
                return (model_cls, arch)

        return self._raise_for_unsupported(architectures)


ModelRegistry = _ModelRegistry({
    model_arch:
    _LazyRegisteredModel(
        module_name=f"fastvideo.v1.models.{component_name}.{mod_relname}",
        component_name=component_name,
        class_name=cls_name,
    )
    for model_arch, (component_name, mod_relname,
                     cls_name) in _FAST_VIDEO_MODELS.items()
})
