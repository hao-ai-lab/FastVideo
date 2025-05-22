# SPDX-License-Identifier: Apache-2.0
"""
Base class for composed pipelines.

This module defines the base class for pipelines that are composed of multiple stages.
"""

import argparse
import math
import os
from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Any, Dict, List, Optional, Union, cast

import torch
from diffusers.optimization import get_scheduler
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

import wandb
from fastvideo.dataset.latent_datasets import (LatentDataset,
                                               latent_collate_function)
from fastvideo.distill.solver import EulerSolver
from fastvideo.utils.dataset_utils import LengthGroupedSampler
from fastvideo.v1.configs.pipelines import (PipelineConfig,
                                            get_pipeline_config_cls_for_name)
from fastvideo.v1.distributed import (get_sp_group,
                                      init_distributed_environment,
                                      initialize_model_parallel,
                                      model_parallel_is_initialized)
from fastvideo.v1.fastvideo_args import FastVideoArgs, TrainingArgs
from fastvideo.v1.logger import init_logger
from fastvideo.v1.models.loader.component_loader import PipelineComponentLoader
from fastvideo.v1.pipelines.pipeline_batch_info import ForwardBatch
from fastvideo.v1.pipelines.stages import PipelineStage
from fastvideo.v1.utils import (maybe_download_model, shallow_asdict,
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
        self.fastvideo_args = fastvideo_args
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

        self.maybe_init_distributed_environment(fastvideo_args)

        # Load modules directly in initialization
        logger.info("Loading pipeline modules...")
        self.modules = self.load_modules(fastvideo_args)

        if fastvideo_args.training_mode:
            device = fastvideo_args.device
            sp_group = get_sp_group()
            world_size = sp_group.world_size
            rank = sp_group.rank

            assert isinstance(fastvideo_args, TrainingArgs)
            self.modules["transformer"].requires_grad_(True)
            self.modules["transformer"].train()

            # self.modules["text_encoder"].requires_grad_(False)
            # self.modules["vae"].requires_grad_(False)
            # self.modules["tokenizer"].requires_grad_(False)

            # self.modules["teacher_transformer"] = deepcopy(
            #     self.modules["transformer"])
            # self.modules["teacher_transformer"].requires_grad_(False)

            transformer = self.modules["transformer"]
            # teacher_transformer = self.modules["teacher_transformer"]
            args = fastvideo_args

            noise_scheduler = self.modules["scheduler"]
            solver = EulerSolver(
                noise_scheduler.sigmas.numpy()[::-1],
                noise_scheduler.config.num_train_timesteps,
                euler_timesteps=args.num_euler_timesteps,
            )
            solver.to(device)
            params_to_optimize = transformer.parameters()
            params_to_optimize = list(
                filter(lambda p: p.requires_grad, params_to_optimize))

            optimizer = torch.optim.AdamW(
                params_to_optimize,
                lr=args.learning_rate,
                betas=(0.9, 0.999),
                weight_decay=args.weight_decay,
                eps=1e-8,
            )

            init_steps = 0
            logger.info("optimizer: %s", optimizer)

            # todo add lr scheduler
            lr_scheduler = get_scheduler(
                args.lr_scheduler,
                optimizer=optimizer,
                num_warmup_steps=args.lr_warmup_steps * world_size,
                num_training_steps=args.max_train_steps * world_size,
                num_cycles=args.lr_num_cycles,
                power=args.lr_power,
                last_epoch=init_steps - 1,
            )

            train_dataset = LatentDataset(args.data_json_path,
                                          args.num_latent_t, args.cfg)
            uncond_prompt_embed = train_dataset.uncond_prompt_embed
            uncond_prompt_mask = train_dataset.uncond_prompt_mask
            sampler = (LengthGroupedSampler(
                args.train_batch_size,
                rank=rank,
                world_size=world_size,
                lengths=train_dataset.lengths,
                group_frame=args.group_frame,
                group_resolution=args.group_resolution,
            ) if (args.group_frame or args.group_resolution) else
                       DistributedSampler(train_dataset,
                                          rank=rank,
                                          num_replicas=world_size,
                                          shuffle=False))

            train_dataloader = DataLoader(
                train_dataset,
                sampler=sampler,
                collate_fn=latent_collate_function,
                pin_memory=True,
                batch_size=args.train_batch_size,
                num_workers=args.dataloader_num_workers,
                drop_last=True,
            )
            self.lr_scheduler = lr_scheduler
            self.train_dataset = train_dataset
            self.train_dataloader = train_dataloader
            self.init_steps = init_steps
            self.optimizer = optimizer
            self.noise_scheduler = noise_scheduler
            self.solver = solver
            self.uncond_prompt_embed = uncond_prompt_embed
            self.uncond_prompt_mask = uncond_prompt_mask
            # self.noise_random_generator = noise_random_generator

            num_update_steps_per_epoch = math.ceil(
                len(train_dataloader) / args.gradient_accumulation_steps *
                args.sp_size / args.train_sp_batch_size)
            args.num_train_epochs = math.ceil(args.max_train_steps /
                                              num_update_steps_per_epoch)

            if rank <= 0:
                project = args.tracker_project_name or "fastvideo"
                wandb.init(project=project, config=args)

        self.initialize_pipeline(fastvideo_args)

        # logger.info("Creating pipeline stages...")
        # self.create_pipeline_stages(fastvideo_args)

        if fastvideo_args.training_mode:
            logger.info("Creating training pipeline stages...")
            self.create_training_stages(fastvideo_args)
            # logger.info("Creating validation pipeline stages...")
            # self.create_validation_stages(fastvideo_args)
        else:
            logger.info("Creating pipeline stages...")
            self.create_pipeline_stages(fastvideo_args)

    @classmethod
    def from_pretrained(cls,
                        model_path: str,
                        device: Optional[str] = None,
                        torch_dtype: Optional[torch.dtype] = None,
                        pipeline_config: Optional[
                            Union[str
                                  | PipelineConfig]] = None,
                        args: Optional[argparse.Namespace] = None,
                        **kwargs) -> "ComposedPipelineBase":
        config = None
        # 1. If users provide a pipeline config, it will override the default pipeline config
        if isinstance(pipeline_config, PipelineConfig):
            config = pipeline_config
        else:
            config_cls = get_pipeline_config_cls_for_name(model_path)
            if config_cls is not None:
                config = config_cls()
                if isinstance(pipeline_config, str):
                    config.load_from_json(pipeline_config)

        # 2. If users also provide some kwargs, it will override the pipeline config.
        # The user kwargs shouldn't contain model config parameters!
        if config is None:
            logger.warning("No config found for model %s, using default config",
                           model_path)
            config_args = kwargs
        else:
            config_args = shallow_asdict(config)
            config_args.update(kwargs)

        if config_args.get("inference_mode"):
            fastvideo_args = FastVideoArgs(model_path=model_path,
                                           device_str=device or "cuda" if
                                           torch.cuda.is_available() else "cpu",
                                           **config_args)
        else:
            assert args is not None, "args must be provided for training mode"
            fastvideo_args = TrainingArgs.from_cli_args(args)

        fastvideo_args.model_path = model_path
        fastvideo_args.device_str = device or "cuda" if torch.cuda.is_available(
        ) else "cpu"
        for key, value in config_args.items():
            setattr(fastvideo_args, key, value)

        # fastvideo_args = FastVideoArgs(
        #     model_path=model_path,
        #     device_str=device or "cuda" if torch.cuda.is_available() else "cpu",
        #     **config_args)
        fastvideo_args.check_fastvideo_args()

        return cls(model_path, fastvideo_args)

    def maybe_init_distributed_environment(self, fastvideo_args: FastVideoArgs):
        assert model_parallel_is_initialized(
        ) == False, "Distributed environment already initialized"
        local_rank = int(os.environ.get("LOCAL_RANK", -1))
        world_size = int(os.environ.get("WORLD_SIZE", -1))
        rank = int(os.environ.get("RANK", -1))

        if local_rank == -1 or world_size == -1 or rank == -1:
            raise ValueError(
                "Local rank, world size, and rank must be set. Use torchrun to launch the script."
            )

        torch.cuda.set_device(local_rank)
        init_distributed_environment(world_size=world_size,
                                     rank=rank,
                                     local_rank=local_rank)
        initialize_model_parallel(
            tensor_model_parallel_size=fastvideo_args.tp_size,
            sequence_model_parallel_size=fastvideo_args.sp_size)
        device = torch.device(f"cuda:{local_rank}")
        fastvideo_args.device = device

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
        Create the pipeline stages.
        """
        raise NotImplementedError

    # @abstractmethod
    # def create_validation_stages(self, fastvideo_args: FastVideoArgs):
    #     """
    #     Create the validation pipeline stages.
    #     """
    #     raise NotImplementedError

    def create_training_stages(self, fastvideo_args: FastVideoArgs):
        """
        Create the training pipeline stages.
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

        # all the component models used by the pipeline
        required_modules = self.required_config_modules
        logger.info("Loading required modules: %s", required_modules)

        modules = {}
        for module_name, (transformers_or_diffusers,
                          architecture) in modules_config.items():
            if module_name not in required_modules:
                logger.info("Skipping module %s", module_name)
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
        logger.info("Batch: %s", batch)
        for stage in self.stages:
            batch = stage(batch, fastvideo_args)

        # Return the output
        return batch
