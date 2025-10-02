# SPDX-License-Identifier: Apache-2.0
"""
Base class for composed pipelines.

This module defines the base class for pipelines that are composed of multiple stages.
"""

import argparse
import os
from abc import ABC, abstractmethod
from typing import Any, cast

import torch

from fastvideo.configs.pipelines import PipelineConfig
from fastvideo.distributed import (
    maybe_init_distributed_environment_and_model_parallel, get_world_group)
from fastvideo.fastvideo_args import FastVideoArgs, TrainingArgs
from fastvideo.logger import init_logger
from fastvideo.profiler import get_or_create_profiler
from fastvideo.models.loader.component_loader import PipelineComponentLoader
from fastvideo.pipelines.pipeline_batch_info import ForwardBatch
from fastvideo.pipelines.stages import PipelineStage
import fastvideo.envs as envs
from fastvideo.utils import (maybe_download_model,
                             verify_model_config_and_directory)

logger = init_logger(__name__)


class ComposedPipelineBase(ABC):
    """
    Base class for pipelines composed of multiple stages.
    
    This class provides the framework for creating pipelines by composing multiple
    stages together. Each stage is responsible for a specific part of the diffusion
    process, and the pipeline orchestrates the execution of these stages.
    """

    is_video_pipeline: bool = False  # To be overridden by video pipelines
    _required_config_modules: list[str] = []
    _extra_config_module_map: dict[str, str] = {}
    training_args: TrainingArgs | None = None
    fastvideo_args: FastVideoArgs | TrainingArgs | None = None
    modules: dict[str, Any] = {}
    post_init_called: bool = False

    # TODO(will): args should support both inference args and training args
    def __init__(self,
                 model_path: str,
                 fastvideo_args: FastVideoArgs | TrainingArgs,
                 required_config_modules: list[str] | None = None,
                 loaded_modules: dict[str, torch.nn.Module] | None = None):
        """
        Initialize the pipeline. After __init__, the pipeline should be ready to
        use. The pipeline should be stateless and not hold any batch state.
        """
        self.fastvideo_args = fastvideo_args

        self.model_path: str = model_path
        self._stages: list[PipelineStage] = []
        self._stage_name_mapping: dict[str, PipelineStage] = {}

        if required_config_modules is not None:
            self._required_config_modules = required_config_modules

        if self._required_config_modules is None:
            raise NotImplementedError(
                "Subclass must set _required_config_modules")

        maybe_init_distributed_environment_and_model_parallel(
            fastvideo_args.tp_size, fastvideo_args.sp_size)

        # Torch profiler. Enabled and configured through env vars:
        # FASTVIDEO_TORCH_PROFILER_DIR=/path/to/save/trace
        trace_dir = envs.FASTVIDEO_TORCH_PROFILER_DIR
        self.profiler_controller = get_or_create_profiler(trace_dir)
        self.profiler = self.profiler_controller.profiler

        self.local_rank = get_world_group().local_rank

        # Load modules directly in initialization
        logger.info("Loading pipeline modules...")
        with self.profiler_controller.region("profiler_region_model_loading"):
            self.modules = self.load_modules(fastvideo_args, loaded_modules)

    def set_trainable(self) -> None:
        # Only train DiT
        if getattr(self.fastvideo_args, "training_mode", False):
            for name, module in self.modules.items():
                if not isinstance(module, torch.nn.Module):
                    continue
                if "transformer" in name:
                    module.requires_grad_(True)
                else:
                    module.requires_grad_(False)

    def post_init(self) -> None:
        assert self.fastvideo_args is not None, "fastvideo_args must be set"
        if self.post_init_called:
            return
        self.post_init_called = True
        if self.fastvideo_args.training_mode:
            assert isinstance(self.fastvideo_args, TrainingArgs)
            self.training_args = self.fastvideo_args
            assert self.training_args is not None
            self.initialize_training_pipeline(self.training_args)
            if self.training_args.log_validation:
                self.initialize_validation_pipeline(self.training_args)

        self.initialize_pipeline(self.fastvideo_args)
        if self.fastvideo_args.enable_torch_compile:
            transformer_module = self.modules["transformer"]
            if self.fastvideo_args.training_mode:
                logger.info(
                    "Torch Compile enabled via FSDP loader for training; skipping additional pipeline compile"
                )
            else:
                fsdp_module_cls = None
                try:
                    from torch.distributed.fsdp import FSDPModule  # type: ignore
                    fsdp_module_cls = FSDPModule
                except Exception:  # pragma: no cover - FSDP not always available
                    fsdp_module_cls = None
                if fsdp_module_cls is not None and isinstance(
                        transformer_module, fsdp_module_cls):
                    logger.info(
                        "Transformer is already FSDP-wrapped; skipping torch.compile in pipeline"
                    )
                else:
                    compile_kwargs = self.fastvideo_args.torch_compile_kwargs or {}
                    logger.info("Enabling torch.compile for DiT with kwargs=%s",
                                compile_kwargs)
                    self.modules["transformer"] = torch.compile(
                        transformer_module, **compile_kwargs)
                    logger.info("Torch Compile enabled for DiT")

        if not self.fastvideo_args.training_mode:
            logger.info("Creating pipeline stages...")
            self.create_pipeline_stages(self.fastvideo_args)

    def initialize_training_pipeline(self, training_args: TrainingArgs):
        raise NotImplementedError(
            "if training_mode is True, the pipeline must implement this method")

    def initialize_validation_pipeline(self, training_args: TrainingArgs):
        raise NotImplementedError(
            "if log_validation is True, the pipeline must implement this method"
        )

    @classmethod
    def from_pretrained(cls,
                        model_path: str,
                        device: str | None = None,
                        torch_dtype: torch.dtype | None = None,
                        pipeline_config: str | PipelineConfig | None = None,
                        args: argparse.Namespace | None = None,
                        required_config_modules: list[str] | None = None,
                        loaded_modules: dict[str, torch.nn.Module]
                        | None = None,
                        **kwargs) -> "ComposedPipelineBase":
        """
        Load a pipeline from a pretrained model.
        loaded_modules: Optional[Dict[str, torch.nn.Module]] = None,
        If provided, loaded_modules will be used instead of loading from config/pretrained weights.
        """
        if args is None or args.inference_mode:

            kwargs['model_path'] = model_path
            fastvideo_args = FastVideoArgs.from_kwargs(**kwargs)
        else:
            assert args is not None, "args must be provided for training mode"
            fastvideo_args = TrainingArgs.from_cli_args(args)
            # TODO(will): fix this so that its not so ugly
            fastvideo_args.model_path = model_path
            for key, value in kwargs.items():
                setattr(fastvideo_args, key, value)

            fastvideo_args.dit_cpu_offload = False
            # we hijack the precision to be the master weight type so that the
            # model is loaded with the correct precision. Subsequently we will
            # use FSDP2's MixedPrecisionPolicy to set the precision for the
            # fwd, bwd, and other operations' precision.
            assert fastvideo_args.pipeline_config.dit_precision == 'fp32', 'only fp32 is supported for training'

        logger.info("fastvideo_args in from_pretrained: %s", fastvideo_args)

        pipe = cls(model_path,
                   fastvideo_args,
                   required_config_modules=required_config_modules,
                   loaded_modules=loaded_modules)
        pipe.post_init()
        return pipe

    def get_module(self, module_name: str, default_value: Any = None) -> Any:
        if module_name not in self.modules:
            return default_value
        return self.modules[module_name]

    def add_module(self, module_name: str, module: Any):
        self.modules[module_name] = module

    def _load_config(self, model_path: str) -> dict[str, Any]:
        model_path = maybe_download_model(self.model_path)
        self.model_path = model_path
        # fastvideo_args.downloaded_model_path = model_path
        logger.info("Model path: %s", model_path)
        config = verify_model_config_and_directory(model_path)
        return cast(dict[str, Any], config)

    @property
    def required_config_modules(self) -> list[str]:
        """
        List of modules that are required by the pipeline. The names should match
        the diffusers directory and model_index.json file. These modules will be
        loaded using the PipelineComponentLoader and made available in the
        modules dictionary. Access these modules using the get_module method.

        class ConcretePipeline(ComposedPipelineBase):
            _required_config_modules = ["vae", "text_encoder", "transformer", "scheduler", "tokenizer"]
            

            @property
            def required_config_modules(self):
                return self._required_config_modules
        """
        return self._required_config_modules

    @property
    def stages(self) -> list[PipelineStage]:
        """
        List of stages in the pipeline.
        """
        return self._stages

    @abstractmethod
    def create_pipeline_stages(self, fastvideo_args: FastVideoArgs):
        """
        Create the inference pipeline stages.
        """
        raise NotImplementedError

    def create_training_stages(self, training_args: TrainingArgs):
        """
        Create the training pipeline stages.
        """
        raise NotImplementedError

    def initialize_pipeline(self, fastvideo_args: FastVideoArgs):
        """
        Initialize the pipeline.
        """
        return

    def load_modules(
        self,
        fastvideo_args: FastVideoArgs,
        loaded_modules: dict[str, torch.nn.Module] | None = None
    ) -> dict[str, Any]:
        """
        Load the modules from the config.
        loaded_modules: Optional[Dict[str, torch.nn.Module]] = None, 
        If provided, loaded_modules will be used instead of loading from config/pretrained weights.
        """

        model_index = self._load_config(self.model_path)
        logger.info("Loading pipeline modules from config: %s", model_index)

        # remove keys that are not pipeline modules
        model_index.pop("_class_name")
        model_index.pop("_diffusers_version")
        if "boundary_ratio" in model_index and model_index[
                "boundary_ratio"] is not None:
            logger.info(
                "MoE pipeline detected. Adding transformer_2 to self.required_config_modules..."
            )
            self.required_config_modules.append("transformer_2")
            logger.info("MoE pipeline detected. Setting boundary ratio to %s",
                        model_index["boundary_ratio"])
            fastvideo_args.pipeline_config.dit_config.boundary_ratio = model_index[
                "boundary_ratio"]

        model_index.pop("boundary_ratio", None)
        # used by Wan2.2 ti2v
        model_index.pop("expand_timesteps", None)

        # some sanity checks
        assert len(
            model_index
        ) > 1, "model_index.json must contain at least one pipeline module"

        for module_name in self.required_config_modules:
            if module_name not in model_index and module_name in self._extra_config_module_map:
                extra_module_value = self._extra_config_module_map[module_name]
                logger.warning(
                    "model_index.json does not contain a %s module, but found {%s: %s} in _extra_config_module_map, adding to model_index.",
                    module_name, module_name, extra_module_value)
                if extra_module_value in model_index:
                    logger.info("Using module %s for %s", extra_module_value,
                                module_name)
                    model_index[module_name] = model_index[extra_module_value]
                    continue
                else:
                    raise ValueError(
                        f"Required module key: {module_name} value: {model_index.get(module_name)} was not found in loaded modules {model_index.keys()}"
                    )

        # all the component models used by the pipeline
        required_modules = self.required_config_modules
        logger.info("Loading required modules: %s", required_modules)

        modules = {}
        for module_name, (transformers_or_diffusers,
                          architecture) in model_index.items():
            if transformers_or_diffusers is None:
                logger.warning(
                    "Module %s in model_index.json has null value, removing from required_config_modules",
                    module_name)
                if module_name in self.required_config_modules:
                    self.required_config_modules.remove(module_name)
                continue
            if module_name not in required_modules:
                logger.info("Skipping module %s", module_name)
                continue
            if loaded_modules is not None and module_name in loaded_modules:
                logger.info("Using module %s already provided", module_name)
                modules[module_name] = loaded_modules[module_name]
                continue

            # we load the module from the extra config module map if it exists
            if module_name in self._extra_config_module_map:
                load_module_name = self._extra_config_module_map[module_name]
            else:
                load_module_name = module_name

            component_model_path = os.path.join(self.model_path,
                                                load_module_name)
            module = PipelineComponentLoader.load_module(
                module_name=load_module_name,
                component_model_path=component_model_path,
                transformers_or_diffusers=transformers_or_diffusers,
                fastvideo_args=fastvideo_args,
            )
            logger.info("Loaded module %s from %s", module_name,
                        component_model_path)

            if module_name in modules:
                logger.warning("Overwriting module %s", module_name)
            modules[module_name] = module

        # Check if all required modules were loaded
        for module_name in required_modules:
            if module_name not in modules or modules[module_name] is None:
                raise ValueError(
                    f"Required module key: {module_name} value: {modules.get(module_name)} was not found in loaded modules {modules.keys()}"
                )

        return modules

    def add_stage(self, stage_name: str, stage: PipelineStage):
        assert self.modules is not None, "No modules are registered"
        self._stages.append(stage)
        self._stage_name_mapping[stage_name] = stage
        setattr(self, stage_name, stage)

    def profile(self, is_start: bool = True):
        if self.profiler is None:
            raise RuntimeError("Profiler is not enabled.")
        if is_start:
            self.profiler.start()
        else:
            self.profiler.stop()
            # only print profiler results on rank 0
            if self.local_rank == 0:
                print(self.profiler.key_averages().table(
                    sort_by="self_cuda_time_total"))

    # TODO(will): don't hardcode no_grad
    @torch.no_grad()
    def forward(
        self,
        batch: ForwardBatch,
        fastvideo_args: FastVideoArgs,
    ) -> ForwardBatch:
        """
        Generate a video or image using the pipeline.
        
        Args:
            batch: The batch to generate from.
            fastvideo_args: The inference arguments.
        Returns:
            ForwardBatch: The batch with the generated video or image.
        """
        if not self.post_init_called:
            self.post_init()

        # Execute each stage
        logger.info("Running pipeline stages: %s",
                    self._stage_name_mapping.keys())
        # logger.info("Batch: %s", batch)
        for stage in self.stages:
            batch = stage(batch, fastvideo_args)

        # Return the output
        return batch

    def train(self) -> None:
        raise NotImplementedError(
            "if training_mode is True, the pipeline must implement this method")
