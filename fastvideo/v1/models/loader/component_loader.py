# SPDX-License-Identifier: Apache-2.0

import dataclasses
import glob
import json
import os
import time
from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Any, Generator, Iterable, List, Optional, Tuple, cast

import torch
import torch.nn as nn
from safetensors.torch import load_file as safetensors_load_file
from transformers import AutoImageProcessor, AutoTokenizer
from transformers.utils import SAFE_WEIGHTS_INDEX_NAME

from fastvideo.v1.configs.models import EncoderConfig
from fastvideo.v1.distributed import get_torch_device
from fastvideo.v1.fastvideo_args import FastVideoArgs
from fastvideo.v1.logger import init_logger
from fastvideo.v1.models.hf_transformer_utils import get_diffusers_config
from fastvideo.v1.models.loader.fsdp_load import maybe_load_fsdp_model
from fastvideo.v1.models.loader.utils import set_default_torch_dtype
from fastvideo.v1.models.loader.weight_utils import (
    filter_duplicate_safetensors_files, filter_files_not_needed_for_inference,
    pt_weights_iterator, safetensors_weights_iterator)
from fastvideo.v1.models.registry import ModelRegistry
from fastvideo.v1.utils import PRECISION_TO_TYPE

logger = init_logger(__name__)


class ComponentLoader(ABC):
    """Base class for loading a specific type of model component."""

    def __init__(self, device=None) -> None:
        self.device = device

    @abstractmethod
    def load(self, model_path: str, architecture: str,
             fastvideo_args: FastVideoArgs):
        """
        Load the component based on the model path, architecture, and inference args.
        
        Args:
            model_path: Path to the component model
            architecture: Architecture of the component model
            fastvideo_args: FastVideoArgs
            
        Returns:
            The loaded component
        """
        raise NotImplementedError

    @classmethod
    def for_module_type(cls, module_type: str,
                        transformers_or_diffusers: str) -> 'ComponentLoader':
        """
        Factory method to create a component loader for a specific module type.
        
        Args:
            module_type: Type of module (e.g., "vae", "text_encoder", "transformer", "scheduler")
            transformers_or_diffusers: Whether the module is from transformers or diffusers
            
        Returns:
            A component loader for the specified module type
        """
        # Map of module types to their loader classes and expected library
        module_loaders = {
            "scheduler": (SchedulerLoader, "diffusers"),
            "transformer": (TransformerLoader, "diffusers"),
            "vae": (VAELoader, "diffusers"),
            "text_encoder": (TextEncoderLoader, "transformers"),
            "text_encoder_2": (TextEncoderLoader, "transformers"),
            "tokenizer": (TokenizerLoader, "transformers"),
            "tokenizer_2": (TokenizerLoader, "transformers"),
            "image_processor": (ImageProcessorLoader, "transformers"),
            "image_encoder": (ImageEncoderLoader, "transformers"),
        }

        if module_type in module_loaders:
            loader_cls, expected_library = module_loaders[module_type]
            # Assert that the library matches what's expected for this module type
            assert transformers_or_diffusers == expected_library, f"{module_type} must be loaded from {expected_library}, got {transformers_or_diffusers}"
            return loader_cls()

        # For unknown module types, use a generic loader
        logger.warning(
            "No specific loader found for module type: %s. Using generic loader.",
            module_type)
        return GenericComponentLoader(transformers_or_diffusers)


class TextEncoderLoader(ComponentLoader):
    """Loader for text encoders."""

    @dataclasses.dataclass
    class Source:
        """A source for weights."""

        model_or_path: str
        """The model ID or path."""

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
        fall_back_to_pt: bool,
        allow_patterns_overrides: Optional[list[str]],
    ) -> Tuple[str, List[str], bool]:
        """Prepare weights for the model.

        If the model is not local, it will be downloaded."""
        # model_name_or_path = (self._maybe_download_from_modelscope(
        #     model_name_or_path, revision) or model_name_or_path)

        is_local = os.path.isdir(model_name_or_path)
        assert is_local, "Model path must be a local directory"

        use_safetensors = False
        index_file = SAFE_WEIGHTS_INDEX_NAME
        allow_patterns = ["*.safetensors", "*.bin"]

        if fall_back_to_pt:
            allow_patterns += ["*.pt"]

        if allow_patterns_overrides is not None:
            allow_patterns = allow_patterns_overrides

        hf_folder = model_name_or_path

        hf_weights_files: List[str] = []
        for pattern in allow_patterns:
            hf_weights_files += glob.glob(os.path.join(hf_folder, pattern))
            if len(hf_weights_files) > 0:
                if pattern == "*.safetensors":
                    use_safetensors = True
                break

        if use_safetensors:
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
            source.model_or_path, source.fall_back_to_pt,
            source.allow_patterns_overrides)
        if use_safetensors:
            weights_iterator = safetensors_weights_iterator(hf_weights_files)
        else:
            weights_iterator = pt_weights_iterator(hf_weights_files)

        if self.counter_before_loading_weights == 0.0:
            self.counter_before_loading_weights = time.perf_counter()
        # Apply the prefix.
        return ((source.prefix + name, tensor)
                for (name, tensor) in weights_iterator)

    def _get_all_weights(
        self,
        model_config: Any,
        model: nn.Module,
        model_path: str,
    ) -> Generator[Tuple[str, torch.Tensor], None, None]:
        primary_weights = TextEncoderLoader.Source(
            model_path,
            prefix="",
            fall_back_to_pt=getattr(model, "fall_back_to_pt_during_load", True),
            allow_patterns_overrides=getattr(model, "allow_patterns_overrides",
                                             None),
        )
        yield from self._get_weights_iterator(primary_weights)

        secondary_weights = cast(
            Iterable[TextEncoderLoader.Source],
            getattr(model, "secondary_weights", ()),
        )
        for source in secondary_weights:
            yield from self._get_weights_iterator(source)

    def load(self, model_path: str, architecture: str,
             fastvideo_args: FastVideoArgs):
        """Load the text encoders based on the model path, architecture, and inference args."""
        # model_config: PretrainedConfig = get_hf_config(
        #     model=model_path,
        #     trust_remote_code=fastvideo_args.trust_remote_code,
        #     revision=fastvideo_args.revision,
        #     model_override_args=None,
        # )
        model_config = get_diffusers_config(model=model_path)
        model_config.pop("_name_or_path", None)
        model_config.pop("transformers_version", None)
        model_config.pop("model_type", None)
        model_config.pop("tokenizer_class", None)
        model_config.pop("torch_dtype", None)
        logger.info("HF Model config: %s", model_config)

        # @TODO(Wei): Better way to handle this?
        try:
            encoder_config = fastvideo_args.pipeline_config.text_encoder_configs[
                0]
            encoder_config.update_model_arch(model_config)
            encoder_precision = fastvideo_args.pipeline_config.text_encoder_precisions[
                0]
        except Exception:
            encoder_config = fastvideo_args.pipeline_config.text_encoder_configs[
                1]
            encoder_config.update_model_arch(model_config)
            encoder_precision = fastvideo_args.pipeline_config.text_encoder_precisions[
                1]

        target_device = get_torch_device()
        # TODO(will): add support for other dtypes
        return self.load_model(model_path, encoder_config, target_device,
                               encoder_precision)

    def load_model(self,
                   model_path: str,
                   model_config: EncoderConfig,
                   target_device: torch.device,
                   dtype: str = "fp16"):
        with set_default_torch_dtype(PRECISION_TO_TYPE[dtype]):
            with target_device:
                architectures = getattr(model_config, "architectures", [])
                model_cls, _ = ModelRegistry.resolve_model_cls(architectures)
                model = model_cls(model_config)

            weights_to_load = {name for name, _ in model.named_parameters()}
            loaded_weights = model.load_weights(
                self._get_all_weights(model_config, model, model_path))
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
                raise ValueError("Following weights were not initialized from "
                                 f"checkpoint: {weights_not_loaded}")

        return model.eval()


class ImageEncoderLoader(TextEncoderLoader):

    def load(self, model_path: str, architecture: str,
             fastvideo_args: FastVideoArgs):
        """Load the text encoders based on the model path, architecture, and inference args."""
        # model_config: PretrainedConfig = get_hf_config(
        #     model=model_path,
        #     trust_remote_code=fastvideo_args.trust_remote_code,
        #     revision=fastvideo_args.revision,
        #     model_override_args=None,
        # )
        with open(os.path.join(model_path, "config.json")) as f:
            model_config = json.load(f)
        model_config.pop("_name_or_path", None)
        model_config.pop("transformers_version", None)
        model_config.pop("torch_dtype", None)
        model_config.pop("model_type", None)
        logger.info("HF Model config: %s", model_config)

        encoder_config = fastvideo_args.pipeline_config.image_encoder_config
        encoder_config.update_model_arch(model_config)

        target_device = get_torch_device()
        # TODO(will): add support for other dtypes
        return self.load_model(
            model_path, encoder_config, target_device,
            fastvideo_args.pipeline_config.image_encoder_precision)


class ImageProcessorLoader(ComponentLoader):
    """Loader for image processor."""

    def load(self, model_path: str, architecture: str,
             fastvideo_args: FastVideoArgs):
        """Load the image processor based on the model path, architecture, and inference args."""
        logger.info("Loading image processor from %s", model_path)

        image_processor = AutoImageProcessor.from_pretrained(model_path, )
        logger.info("Loaded image processor: %s",
                    image_processor.__class__.__name__)
        return image_processor


class TokenizerLoader(ComponentLoader):
    """Loader for tokenizers."""

    def load(self, model_path: str, architecture: str,
             fastvideo_args: FastVideoArgs):
        """Load the tokenizer based on the model path, architecture, and inference args."""
        logger.info("Loading tokenizer from %s", model_path)

        tokenizer = AutoTokenizer.from_pretrained(
            model_path,  # "<path to model>/tokenizer"
            # in v0, this was same string as encoder_name "ClipTextModel"
            # TODO(will): pass these tokenizer kwargs from inference args? Maybe
            # other method of config?
            padding_size='right',
        )
        logger.info("Loaded tokenizer: %s", tokenizer.__class__.__name__)
        return tokenizer


class VAELoader(ComponentLoader):
    """Loader for VAE."""

    def load(self, model_path: str, architecture: str,
             fastvideo_args: FastVideoArgs):
        """Load the VAE based on the model path, architecture, and inference args."""
        config = get_diffusers_config(model=model_path)
        class_name = config.pop("_class_name")
        assert class_name is not None, "Model config does not contain a _class_name attribute. Only diffusers format is supported."

        vae_config = fastvideo_args.pipeline_config.vae_config
        vae_config.update_model_arch(config)

        with set_default_torch_dtype(PRECISION_TO_TYPE[
                fastvideo_args.pipeline_config.vae_precision]):
            vae_cls, _ = ModelRegistry.resolve_model_cls(class_name)
            vae = vae_cls(vae_config).to(get_torch_device())

        # Find all safetensors files
        safetensors_list = glob.glob(
            os.path.join(str(model_path), "*.safetensors"))
        # TODO(PY)
        assert len(
            safetensors_list
        ) == 1, f"Found {len(safetensors_list)} safetensors files in {model_path}"
        loaded = safetensors_load_file(safetensors_list[0])
        vae.load_state_dict(
            loaded, strict=False)  # We might only load encoder or decoder

        return vae.eval()


class TransformerLoader(ComponentLoader):
    """Loader for transformer."""

    def load(self, model_path: str, architecture: str,
             fastvideo_args: FastVideoArgs):
        """Load the transformer based on the model path, architecture, and inference args."""
        config = get_diffusers_config(model=model_path)
        hf_config = deepcopy(config)
        cls_name = config.pop("_class_name")
        if cls_name is None:
            raise ValueError(
                "Model config does not contain a _class_name attribute. "
                "Only diffusers format is supported.")

        # Config from Diffusers supersedes fastvideo's model config
        dit_config = fastvideo_args.pipeline_config.dit_config
        dit_config.update_model_arch(config)

        model_cls, _ = ModelRegistry.resolve_model_cls(cls_name)

        # Find all safetensors files
        safetensors_list = glob.glob(
            os.path.join(str(model_path), "*.safetensors"))
        if not safetensors_list:
            raise ValueError(f"No safetensors files found in {model_path}")

        logger.info("Loading model from %s safetensors files in %s",
                    len(safetensors_list), model_path)

        default_dtype = PRECISION_TO_TYPE[
            fastvideo_args.pipeline_config.dit_precision]

        # Load the model using FSDP loader
        logger.info("Loading model from %s, default_dtype: %s", cls_name,
                    default_dtype)
        assert fastvideo_args.hsdp_shard_dim is not None
        model = maybe_load_fsdp_model(
            model_cls=model_cls,
            init_params={
                "config": dit_config,
                "hf_config": hf_config
            },
            weight_dir_list=safetensors_list,
            device=get_torch_device(),
            hsdp_replicate_dim=fastvideo_args.hsdp_replicate_dim,
            hsdp_shard_dim=fastvideo_args.hsdp_shard_dim,
            cpu_offload=fastvideo_args.use_cpu_offload,
            fsdp_inference=fastvideo_args.use_fsdp_inference,
            default_dtype=default_dtype,
            # TODO(will): make these configurable
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.float32,
            output_dtype=None,
            training_mode=fastvideo_args.training_mode)
        if fastvideo_args.enable_torch_compile:
            logger.info("Torch Compile enabled for DiT")
            for n, m in reversed(list(model.named_modules())):
                if any([
                        compile_condition(n, m)
                        for compile_condition in model._compile_conditions
                ]):
                    parts = n.split(".")
                    parent = model
                    attr = parts[-1]
                    for part in parts[:-1]:
                        if part.isdigit():
                            parent = parent[int(part)]
                        else:
                            parent = getattr(parent, part)
                    if attr.isdigit():
                        parent[int(attr)] = torch.compile(m)
                    else:
                        setattr(parent, attr, torch.compile(m))

        total_params = sum(p.numel() for p in model.parameters())
        logger.info("Loaded model with %.2fB parameters", total_params / 1e9)

        dtypes = set(param.dtype for param in model.parameters())
        if len(dtypes) > 1:
            model = model.to(default_dtype)
        model = model.eval()
        return model


class SchedulerLoader(ComponentLoader):
    """Loader for scheduler."""

    def load(self, model_path: str, architecture: str,
             fastvideo_args: FastVideoArgs):
        """Load the scheduler based on the model path, architecture, and inference args."""
        config = get_diffusers_config(model=model_path)

        class_name = config.pop("_class_name")
        assert class_name is not None, "Model config does not contain a _class_name attribute. Only diffusers format is supported."

        scheduler_cls, _ = ModelRegistry.resolve_model_cls(class_name)

        scheduler = scheduler_cls(**config)
        if fastvideo_args.pipeline_config.flow_shift is not None:
            scheduler.set_shift(fastvideo_args.pipeline_config.flow_shift)
        if fastvideo_args.pipeline_config.timesteps_scale is not None:
            scheduler.set_timesteps_scale(
                fastvideo_args.pipeline_config.timesteps_scale)
        return scheduler


class GenericComponentLoader(ComponentLoader):
    """Generic loader for components that don't have a specific loader."""

    def __init__(self, library="transformers") -> None:
        super().__init__()
        self.library = library

    def load(self, model_path: str, architecture: str,
             fastvideo_args: FastVideoArgs):
        """Load a generic component based on the model path, architecture, and inference args."""
        logger.warning("Using generic loader for %s with library %s",
                       model_path, self.library)

        if self.library == "transformers":
            from transformers import AutoModel

            model = AutoModel.from_pretrained(
                model_path,
                trust_remote_code=fastvideo_args.trust_remote_code,
                revision=fastvideo_args.revision,
            )
            logger.info("Loaded generic transformers model: %s",
                        model.__class__.__name__)
            return model
        elif self.library == "diffusers":
            logger.warning(
                "Generic loading for diffusers components is not fully implemented"
            )

            model_config = get_diffusers_config(model=model_path)
            logger.info("Diffusers Model config: %s", model_config)
            # This is a placeholder - in a real implementation, you'd need to handle this properly
            return None
        else:
            raise ValueError(f"Unsupported library: {self.library}")


class PipelineComponentLoader:
    """
    Utility class for loading pipeline components.
    This replaces the chain of if-else statements in load_pipeline_module.
    """

    @staticmethod
    def load_module(module_name: str, component_model_path: str,
                    transformers_or_diffusers: str, architecture: str,
                    fastvideo_args: FastVideoArgs):
        """
        Load a pipeline module.
        
        Args:
            module_name: Name of the module (e.g., "vae", "text_encoder", "transformer", "scheduler")
            component_model_path: Path to the component model
            transformers_or_diffusers: Whether the module is from transformers or diffusers
            architecture: Architecture of the component model
            pipeline_args: Inference arguments
            
        Returns:
            The loaded module
        """
        logger.info(
            "Loading %s using %s from %s",
            module_name,
            transformers_or_diffusers,
            component_model_path,
        )

        # Get the appropriate loader for this module type
        loader = ComponentLoader.for_module_type(module_name,
                                                 transformers_or_diffusers)

        # Load the module
        return loader.load(component_model_path, architecture, fastvideo_args)
