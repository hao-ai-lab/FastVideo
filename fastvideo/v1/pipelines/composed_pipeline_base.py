# SPDX-License-Identifier: Apache-2.0
"""
Base class for composed pipelines.

This module defines the base class for pipelines that are composed of multiple stages.
"""

import argparse
import os
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union, cast

import torch

from fastvideo.v1.configs.pipelines import PipelineConfig
from fastvideo.v1.distributed import (
    maybe_init_distributed_environment_and_model_parallel)
from fastvideo.v1.fastvideo_args import FastVideoArgs, TrainingArgs
from fastvideo.v1.logger import init_logger
from fastvideo.v1.models.loader.component_loader import PipelineComponentLoader
from fastvideo.v1.pipelines.pipeline_batch_info import ForwardBatch
from fastvideo.v1.pipelines.stages import PipelineStage
from fastvideo.v1.utils import (maybe_download_model,
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
    _required_config_modules: List[str] = []
    training_args: Optional[TrainingArgs] = None
    fastvideo_args: Optional[FastVideoArgs] = None
    modules: Dict[str, torch.nn.Module] = {}

    # TODO(will): args should support both inference args and training args
    def __init__(self,
                 model_path: str,
                 fastvideo_args: Union[FastVideoArgs, TrainingArgs],
                 required_config_modules: Optional[List[str]] = None,
                 loaded_modules: Optional[Dict[str, torch.nn.Module]] = None):
        """
        Initialize the pipeline. After __init__, the pipeline should be ready to
        use. The pipeline should be stateless and not hold any batch state.
        """
        self.fastvideo_args = fastvideo_args

        self.model_path: str = model_path
        self._stages: List[PipelineStage] = []
        self._stage_name_mapping: Dict[str, PipelineStage] = {}

        if required_config_modules is not None:
            self._required_config_modules = required_config_modules

        if self._required_config_modules is None:
            raise NotImplementedError(
                "Subclass must set _required_config_modules")

        maybe_init_distributed_environment_and_model_parallel(
            fastvideo_args.tp_size, fastvideo_args.sp_size)

        # Load modules directly in initialization
        logger.info("Loading pipeline modules...")
        self.modules = self.load_modules(fastvideo_args, loaded_modules)

        if fastvideo_args.training_mode:
            assert isinstance(fastvideo_args, TrainingArgs)
            self.training_args = fastvideo_args
            assert self.training_args is not None
            if self.training_args.log_validation:
                self.initialize_validation_pipeline(self.training_args)
            self.initialize_training_pipeline(self.training_args)

        self.initialize_pipeline(fastvideo_args)

        if not fastvideo_args.training_mode:
            logger.info("Creating pipeline stages...")
            self.create_pipeline_stages(fastvideo_args)

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
                        device: Optional[str] = None,
                        torch_dtype: Optional[torch.dtype] = None,
                        pipeline_config: Optional[
                            Union[str
                                  | PipelineConfig]] = None,
                        args: Optional[argparse.Namespace] = None,
                        required_config_modules: Optional[List[str]] = None,
                        loaded_modules: Optional[Dict[str,
                                                      torch.nn.Module]] = None,
                        **kwargs) -> "ComposedPipelineBase":
        """
        Load a pipeline from a pretrained model.
        loaded_modules: Optional[Dict[str, torch.nn.Module]] = None,
        If provided, loaded_modules will be used instead of loading from config/pretrained weights.
        """
        if args is None or args.inference_mode:

            kwargs['model_path'] = model_path
            fastvideo_args = FastVideoArgs.from_kwargs(kwargs)
        else:
            assert args is not None, "args must be provided for training mode"
            fastvideo_args = TrainingArgs.from_cli_args(args)
            # TODO(will): fix this so that its not so ugly
            fastvideo_args.model_path = model_path
            for key, value in kwargs.items():
                setattr(fastvideo_args, key, value)

            fastvideo_args.use_cpu_offload = False
            # make sure we are in training mode
            fastvideo_args.inference_mode = False
            # we hijack the precision to be the master weight type so that the
            # model is loaded with the correct precision. Subsequently we will
            # use FSDP2's MixedPrecisionPolicy to set the precision for the
            # fwd, bwd, and other operations' precision.
            # fastvideo_args.precision = fastvideo_args.master_weight_type
            assert fastvideo_args.pipeline_config.dit_precision == 'fp32', 'only fp32 is supported for training'
            # assert fastvideo_args.precision == 'fp32', 'only fp32 is supported for training'

        logger.info("fastvideo_args in from_pretrained: %s", fastvideo_args)

        return cls(model_path,
                   fastvideo_args,
                   required_config_modules=required_config_modules,
                   loaded_modules=loaded_modules)

    def get_module(self, module_name: str, default_value: Any = None) -> Any:
        if module_name not in self.modules:
            return default_value
        return self.modules[module_name]

    def add_module(self, module_name: str, module: Any):
        self.modules[module_name] = module

    def _load_config(self, model_path: str) -> Dict[str, Any]:
        model_path = maybe_download_model(self.model_path)
        self.model_path = model_path
        # fastvideo_args.downloaded_model_path = model_path
        logger.info("Model path: %s", model_path)
        config = verify_model_config_and_directory(model_path)
        return cast(Dict[str, Any], config)

    @property
    def required_config_modules(self) -> List[str]:
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
    def stages(self) -> List[PipelineStage]:
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
        loaded_modules: Optional[Dict[str, torch.nn.Module]] = None
    ) -> Dict[str, Any]:
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

        # some sanity checks
        assert len(
            model_index
        ) > 1, "model_index.json must contain at least one pipeline module"

        for module_name in self.required_config_modules:
            if module_name not in model_index:
                raise ValueError(
                    f"model_index.json must contain a {module_name} module")

        # all the component models used by the pipeline
        required_modules = self.required_config_modules
        logger.info("Loading required modules: %s", required_modules)

        modules = {}
        for module_name, (transformers_or_diffusers,
                          architecture) in model_index.items():
            if module_name not in required_modules:
                logger.info("Skipping module %s", module_name)
                continue
            if loaded_modules is not None and module_name in loaded_modules:
                logger.info("Using module %s already provided", module_name)
                modules[module_name] = loaded_modules[module_name]
                continue
            component_model_path = os.path.join(self.model_path, module_name)
            module = PipelineComponentLoader.load_module(
                module_name=module_name,
                component_model_path=component_model_path,
                transformers_or_diffusers=transformers_or_diffusers,
                architecture=architecture,
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
                    f"Required module {module_name} was not loaded properly")

        return modules

    def add_stage(self, stage_name: str, stage: PipelineStage):
        assert self.modules is not None, "No modules are registered"
        self._stages.append(stage)
        self._stage_name_mapping[stage_name] = stage
        setattr(self, stage_name, stage)

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
