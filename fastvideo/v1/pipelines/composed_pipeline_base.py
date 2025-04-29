# SPDX-License-Identifier: Apache-2.0
"""
Base class for composed pipelines.

This module defines the base class for pipelines that are composed of multiple stages.
"""

import os
from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Any, Dict, List, Optional, cast

import torch

from fastvideo.v1.fastvideo_args import FastVideoArgs
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

    # TODO(will): args should support both inference args and training args
    def __init__(self,
                 model_path: str,
                 fastvideo_args: FastVideoArgs,
                 config: Optional[Dict[str, Any]] = None):
        """
        Initialize the pipeline. After __init__, the pipeline should be ready to
        use. The pipeline should be stateless and not hold any batch state.
        """
        self.model_path = model_path
        self._stages: List[PipelineStage] = []
        self._stage_name_mapping: Dict[str, PipelineStage] = {}

        if self._required_config_modules is None:
            raise NotImplementedError(
                "Subclass must set _required_config_modules")

        if config is None:
            # Load configuration
            logger.info("Loading pipeline configuration...")
            self.config = self._load_config(model_path)
        else:
            self.config = config

        # Load modules directly in initialization
        logger.info("Loading pipeline modules...")
        self.modules = self.load_modules(fastvideo_args)

        self.initialize_pipeline(fastvideo_args)

        logger.info("Creating pipeline stages...")
        self.create_pipeline_stages(fastvideo_args)

    def get_module(self, module_name: str) -> Any:
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
        Create the pipeline stages.
        """
        raise NotImplementedError

    def initialize_pipeline(self, fastvideo_args: FastVideoArgs):
        """
        Initialize the pipeline.
        """
        return

    def load_modules(self, fastvideo_args: FastVideoArgs) -> Dict[str, Any]:
        """
        Load the modules from the config.
        """
        logger.info("Loading pipeline modules from config: %s", self.config)
        modules_config = deepcopy(self.config)

        # remove keys that are not pipeline modules
        modules_config.pop("_class_name")
        modules_config.pop("_diffusers_version")

        # some sanity checks
        assert len(
            modules_config
        ) > 1, "model_index.json must contain at least one pipeline module"

        required_modules = [
            "vae", "text_encoder", "transformer", "scheduler", "tokenizer"
        ]
        for module_name in required_modules:
            if module_name not in modules_config:
                raise ValueError(
                    f"model_index.json must contain a {module_name} module")
        logger.info("Diffusers config passed sanity checks")

        # all the component models used by the pipeline
        modules = {}
        for module_name, (transformers_or_diffusers,
                          architecture) in modules_config.items():
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

        required_modules = self.required_config_modules
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
        logger.info("Batch: %s", batch)
        for stage in self.stages:
            batch = stage(batch, fastvideo_args)

        # Return the output
        return batch
