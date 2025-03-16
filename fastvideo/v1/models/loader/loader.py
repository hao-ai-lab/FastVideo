# SPDX-License-Identifier: Apache-2.0

# Adapted from vllm
# Copyright 2023 The vLLM Authors.
# Copyright 2025 The FastVideo Authors.

# ruff: noqa: SIM117
import collections
import copy
import dataclasses
import fnmatch
import glob
import inspect
import itertools
import math
import os
import time
import warnings
from abc import ABC, abstractmethod
from contextlib import contextmanager
from typing import (Any, Callable, Dict, Generator, Iterable, List, Optional,
                    Tuple, cast)

import huggingface_hub
import torch
from torch import nn
from transformers.utils import SAFE_WEIGHTS_INDEX_NAME

from fastvideo.v1.logger import init_logger
from fastvideo.v1.models.loader.utils import set_default_torch_dtype
from fastvideo.v1.models.loader.weight_utils import (
    download_safetensors_index_file_from_hf, download_weights_from_hf,
    filter_duplicate_safetensors_files, filter_files_not_needed_for_inference,
    pt_weights_iterator,
    safetensors_weights_iterator)
from fastvideo.v1.platforms import current_platform
from fastvideo.v1.models.registry import ModelRegistry

from fastvideo.v1.inference_args import InferenceArgs

@contextmanager
def device_loading_context(module: torch.nn.Module,
                           target_device: torch.device):
    if target_device.type == "cpu":
        # If target is CPU, no need to move anything
        yield module
        return

    original_device_states: Dict[str, torch.device] = {}

    # Store original device states and move parameters to GPU if they're on CPU
    for name, p in module.named_parameters():
        if p.device.type == "cpu":
            original_device_states[name] = p.device
            p.data = p.data.to(target_device)
        # Parameters already on target device are not touched

    try:
        yield module

    finally:
        # Restore parameters to their original devices, ignoring new parameters
        pin_memory = True
        for name, p in module.named_parameters():
            if name in original_device_states:
                original_device: torch.device = original_device_states[name]
                if original_device.type == "cpu":
                    # `torch.empty_like` does not support `pin_memory` argument
                    cpu_data = torch.empty_strided(
                        size=p.data.size(),
                        stride=p.data.stride(),
                        dtype=p.data.dtype,
                        layout=p.data.layout,
                        device="cpu",
                        pin_memory=pin_memory,
                    )
                    cpu_data.copy_(p.data)
                    p.data = cpu_data
                else:
                    p.data = p.data.to(original_device)
        # New parameters or parameters already on target device are untouched


logger = init_logger(__name__)


def _initialize_model(
    model_path: str,
    model_config,
) -> nn.Module:
    """Initialize a model with the given configurations.
    
    Args:
        model_name: The name of the model architecture
        model_path: Path to the model directory
        component_name: Optional component name (e.g., "text_encoder", "vae")
            If provided, will load the config from the component subdirectory
    
    Returns:
        The initialized model
    """
    architectures = getattr(model_config, "architectures", [])
    if len(architectures) == 0:
        raise ValueError("Model config does not contain a valid model architecture")
        logger.info("Trying to load model from diffusers format")
        class_name = model_config.pop("_class_name")
        architectures = [class_name]
    if architectures is None:
        raise ValueError("Model config does not contain a valid model architecture")
    model_cls, _ = ModelRegistry.resolve_model_cls(architectures)
    logger.info(f"Loading model_cls: {model_cls}")

    return model_cls(model_config)

    
class BaseModelLoader(ABC):
    """Base class for model loaders."""

    @abstractmethod
    def download_model(self, inference_args: InferenceArgs) -> None:
        """Download a model so that it can be immediately loaded."""
        raise NotImplementedError

    @abstractmethod
    def load_model(self, *, inference_args: InferenceArgs) -> nn.Module:
        """Load a model with the given configurations."""
        raise NotImplementedError


class DefaultModelLoader(BaseModelLoader):
    """Model loader that can load different file types from disk."""

    @dataclasses.dataclass
    class Source:
        """A source for weights."""

        model_or_path: str
        """The model ID or path."""

        revision: Optional[str]
        """The optional model revision."""

        prefix: str = ""
        """A prefix to prepend to all weights."""

        fall_back_to_pt: bool = True
        """Whether .pt weights can be used."""

        allow_patterns_overrides: Optional[list[str]] = None
        """If defined, weights will load exclusively using these patterns."""

    counter_before_loading_weights: float = 0.0
    counter_after_loading_weights: float = 0.0

    def _prepare_weights(
        self,
        model_name_or_path: str,
        revision: Optional[str],
        fall_back_to_pt: bool,
        allow_patterns_overrides: Optional[list[str]],
    ) -> Tuple[str, List[str], bool]:
        """Prepare weights for the model.

        If the model is not local, it will be downloaded."""
        # model_name_or_path = (self._maybe_download_from_modelscope(
        #     model_name_or_path, revision) or model_name_or_path)

        is_local = os.path.isdir(model_name_or_path)
        assert is_local, "Model path must be a local directory"
        # load_format = self.load_config.load_format
        # load_format = LoadFormat.AUTO
        use_safetensors = False
        index_file = SAFE_WEIGHTS_INDEX_NAME
        allow_patterns = ["*.safetensors", "*.bin"]
        # Some quantized models use .pt files for storing the weights.
        # if load_format == LoadFormat.AUTO:
        #     allow_patterns = ["*.safetensors", "*.bin"]
        # elif load_format == LoadFormat.SAFETENSORS:
        #     use_safetensors = True
        #     allow_patterns = ["*.safetensors"]
        # elif load_format == LoadFormat.MISTRAL:
        #     use_safetensors = True
        #     allow_patterns = ["consolidated*.safetensors"]
        #     index_file = "consolidated.safetensors.index.json"
        # elif load_format == LoadFormat.PT:
        #     allow_patterns = ["*.pt"]
        # elif load_format == LoadFormat.NPCACHE:
        #     allow_patterns = ["*.bin"]
        # else:
        #     raise ValueError(f"Unknown load_format: {load_format}")

        if fall_back_to_pt:
            allow_patterns += ["*.pt"]

        if allow_patterns_overrides is not None:
            allow_patterns = allow_patterns_overrides

        if not is_local:
            hf_folder = download_weights_from_hf(
                model_name_or_path,
                self.load_config.download_dir,
                allow_patterns,
                revision,
                ignore_patterns=self.load_config.ignore_patterns,
            )
        else:
            hf_folder = model_name_or_path

        hf_weights_files: List[str] = []
        for pattern in allow_patterns:
            hf_weights_files += glob.glob(os.path.join(hf_folder, pattern))
            if len(hf_weights_files) > 0:
                if pattern == "*.safetensors":
                    use_safetensors = True
                break

        if use_safetensors:
            # For models like Mistral-7B-Instruct-v0.3
            # there are both sharded safetensors files and a consolidated
            # safetensors file. Using both breaks.
            # Here, we download the `model.safetensors.index.json` and filter
            # any files not found in the index.
            if not is_local:
                download_safetensors_index_file_from_hf(
                    model_name_or_path,
                    index_file,
                    self.load_config.download_dir,
                    revision,
                )
            hf_weights_files = filter_duplicate_safetensors_files(
                hf_weights_files, hf_folder, index_file)
        else:
            hf_weights_files = filter_files_not_needed_for_inference(
                hf_weights_files)

        if len(hf_weights_files) == 0:
            raise RuntimeError(
                f"Cannot find any model weights with `{model_name_or_path}`")

        return hf_folder, hf_weights_files, use_safetensors

    def _get_weights_iterator(
            self, source: "Source"
    ) -> Generator[Tuple[str, torch.Tensor], None, None]:
        """Get an iterator for the model weights based on the load format."""
        hf_folder, hf_weights_files, use_safetensors = self._prepare_weights(
            source.model_or_path, source.revision, source.fall_back_to_pt,
            source.allow_patterns_overrides)
        if use_safetensors:
            weights_iterator = safetensors_weights_iterator(hf_weights_files)
        else:
            weights_iterator = pt_weights_iterator(hf_weights_files)

        # if current_platform.is_tpu():
        #     # In PyTorch XLA, we should call `xm.mark_step` frequently so that
        #     # not too many ops are accumulated in the XLA program.
        #     import torch_xla.core.xla_model as xm

        #     def _xla_weights_iterator(iterator: Generator):
        #         for weights in iterator:
        #             yield weights
        #             xm.mark_step()

        #     weights_iterator = _xla_weights_iterator(weights_iterator)

        if self.counter_before_loading_weights == 0.0:
            self.counter_before_loading_weights = time.perf_counter()
        # Apply the prefix.
        return ((source.prefix + name, tensor)
                for (name, tensor) in weights_iterator)

    def _get_all_weights(
        self,
        model_config: Dict[str, Any],
        model: nn.Module,
    ) -> Generator[Tuple[str, torch.Tensor], None, None]:
        primary_weights = DefaultModelLoader.Source(
            model_config.model,
            "",
            prefix="",
            fall_back_to_pt=getattr(model, "fall_back_to_pt_during_load",
                                    True),
            allow_patterns_overrides=getattr(model, "allow_patterns_overrides",
                                             None),
        )
        yield from self._get_weights_iterator(primary_weights)

        secondary_weights = cast(
            Iterable[DefaultModelLoader.Source],
            getattr(model, "secondary_weights", ()),
        )
        for source in secondary_weights:
            yield from self._get_weights_iterator(source)

    def download_model(self, inference_args: InferenceArgs) -> None:
        raise NotImplementedError
        # self._prepare_weights(inference_args.model,
        #                       inference_args.revision,
        #                       fall_back_to_pt=True,
        #                       allow_patterns_overrides=None)

    def load_model(self, model_path: str,
                   model_config: Dict[str, Any],
                   inference_args: InferenceArgs) -> nn.Module:
        logger.info(f"Loading model on device: {inference_args.device_str}")
        target_device = torch.device(inference_args.device_str)
        # TODO(will): add support for other dtypes
        with set_default_torch_dtype(torch.float16):
            with target_device:
                model = _initialize_model(model_path, model_config)
                
            weights_to_load = {name for name, _ in model.named_parameters()}
            model_config.model = model_path
            loaded_weights = model.load_weights(
                self._get_all_weights(model_config, model))
            self.counter_after_loading_weights = time.perf_counter()
            logger.info(
                "Loading weights took %.2f seconds",
                self.counter_after_loading_weights -
                self.counter_before_loading_weights)
            # We only enable strict check for non-quantized models
            # that have loaded weights tracking currently.
            # if loaded_weights is not None:
            weights_not_loaded = weights_to_load - loaded_weights
            if weights_not_loaded:
                raise ValueError(
                    "Following weights were not initialized from "
                    f"checkpoint: {weights_not_loaded}")

        # TODO(will): add support for training/finetune
        return model.eval()





def get_model_loader() -> BaseModelLoader:
    """Get a model loader based on the load format."""
    # if isinstance(load_config.load_format, type):
    #     return load_config.load_format(load_config)

    return DefaultModelLoader()