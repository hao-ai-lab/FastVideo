# SPDX-License-Identifier: Apache-2.0
"""
Base class for composed pipelines.

This module defines the base class for pipelines that are composed of multiple stages.
"""

import argparse
import os
from abc import ABC, abstractmethod
from collections.abc import Iterator
from functools import partial
from typing import Any, cast

import torch

from fastvideo.configs.pipelines import PipelineConfig
from fastvideo.distributed import (
    get_local_torch_device,
    get_world_group,
    maybe_init_distributed_environment_and_model_parallel,
)
from fastvideo.distributed.communication_op import (warmup_sequence_parallel_communication)
from fastvideo.fastvideo_args import FastVideoArgs, TrainingArgs
from fastvideo.hooks.activation_trace import attach_activation_trace, detach_activation_trace
from fastvideo.logger import init_logger
from fastvideo.profiler import get_or_create_profiler
from fastvideo.models.loader.component_loader import PipelineComponentLoader
from fastvideo.pipelines.lazy_module import LazyModule, is_lazy_module
from fastvideo.pipelines.pipeline_batch_info import ForwardBatch
from fastvideo.pipelines.stages import PipelineStage
import fastvideo.envs as envs
from fastvideo.utils import (maybe_download_model, verify_model_config_and_directory)

logger = init_logger(__name__)


def _iter_held_objects(stage: PipelineStage) -> Iterator[Any]:
    """Yield everything a stage holds, walking into nested stages.

    A stage can compose others rather than hold a component directly:
    ``Cosmos25AutoDenoisingStage`` keeps the transformer inside its ``_t2w``
    and ``_v2w`` children. A scan that stopped at the outer stage would call
    that transformer unreferenced and never free it, so the flag would quietly
    deliver less than it promises on those pipelines.
    """
    stack: list[Any] = [stage]
    visited: set[int] = set()
    while stack:
        obj = stack.pop()
        # is_lazy_module first: isinstance() on a proxy forwards __class__ and
        # would load every deferred component just to work out where to free it.
        if is_lazy_module(obj):
            yield obj
            continue
        if isinstance(obj, PipelineStage | list | tuple | dict):
            if id(obj) in visited:
                continue
            visited.add(id(obj))
        if isinstance(obj, PipelineStage):
            for key, value in vars(obj).items():
                if key == "_lazy_modules_to_release":
                    # Installed by this schedule, not a real use.
                    continue
                stack.append(value)
        elif isinstance(obj, list | tuple):
            stack.extend(obj)
        elif isinstance(obj, dict):
            stack.extend(obj.values())


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
    # do not need to include moe related transformers
    trainable_transformer_names: list[str] = ["transformer"]
    trainable_transformer_modules: dict[str, torch.nn.Module] = {}
    post_init_called: bool = False
    # Set once the deferred-release schedule has been derived from the stage
    # list, so a stage added afterwards can rebuild it instead of running
    # against a plan that predates it.
    _lazy_release_hooks_installed: bool = False
    # Components this pipeline allows ``lazy_module_load`` to defer and free.
    # Empty by default: deferral is opt-in per pipeline, because releasing a
    # component and loading it again is only safe when nothing outside the
    # loader has changed it. Two habits break that and neither raises:
    #
    #   * mutating a component after load. ``LongCatPipeline.initialize_pipeline``
    #     turns on block-sparse attention and writes parameters into every
    #     transformer block. That runs once, so a re-materialized component
    #     silently comes back with the feature off.
    #   * reading a component's attributes while building stages. The shared
    #     ``DenoisingStage.__init__`` derives the attention backend from
    #     ``transformer.hidden_size``, which materializes the DiT before the
    #     first request and defeats the deferral it was meant to gain.
    #
    # A pipeline opts in by listing the components it has checked. Names match
    # the diffusers manifest.
    _lazy_module_names: tuple[str, ...] = ()

    @classmethod
    def get_hf_download_component_dirs(cls) -> tuple[str, ...] | None:
        """Return component directories for an opt-in partial Hub download."""
        return None

    @classmethod
    def get_hf_download_allow_patterns(cls) -> list[str] | None:
        """Return Hub patterns for the manifest and selected components."""
        component_dirs = cls.get_hf_download_component_dirs()
        if component_dirs is None:
            return None
        return [
            "model_index.json",
            "modular_model_index.json",
            *(f"{component_dir}/**" for component_dir in component_dirs),
        ]

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
        self._trace_mgr = None

        if required_config_modules is not None:
            self._required_config_modules = required_config_modules

        if self._required_config_modules is None:
            raise NotImplementedError("Subclass must set _required_config_modules")

        maybe_init_distributed_environment_and_model_parallel(fastvideo_args.tp_size, fastvideo_args.sp_size)

        # VideoGenerator applies this in each Worker before building the
        # pipeline. Keep direct from_pretrained/build_pipeline callers aligned,
        # but only after distributed setup has selected this process's device.
        if fastvideo_args.inference_mode:
            local_device = get_local_torch_device()
            device_id = local_device.index if local_device.index is not None else 0
            fastvideo_args.finalize_device_offload_policy(device_id)

        # Torch profiler. Enabled and configured through env vars:
        # FASTVIDEO_TORCH_PROFILER_DIR=/path/to/save/trace
        trace_dir = envs.FASTVIDEO_TORCH_PROFILER_DIR
        self.profiler_controller = get_or_create_profiler(trace_dir)

        self.local_rank = get_world_group().local_rank

        # Load modules directly in initialization
        logger.info("Loading pipeline modules...")
        with self.profiler_controller.region("profiler_region_model_loading"):
            self.modules = self.load_modules(fastvideo_args, loaded_modules)

    def set_trainable(self) -> None:
        # Only train DiT
        if getattr(self.fastvideo_args, "training_mode", False):
            for name, module in self.trainable_transformer_modules.items():
                logger.info("Setting %s to requires_grad=True", name)
                if not isinstance(module, torch.nn.Module):
                    logger.info("Skipping %s because it is not a torch.nn.Module", name)
                    continue
                module.requires_grad_(True)
                module.train()

    @staticmethod
    def _compile_with_conditions(
        module: torch.nn.Module,
        compile_kwargs: dict[str, Any],
    ) -> int:
        """Compile submodules that match module._compile_conditions."""
        compile_conditions = getattr(module, "_compile_conditions", None)
        if not compile_conditions:
            return 0

        compiled_count = 0
        for name, submodule in module.named_modules():
            if not name:
                continue
            if any(cond(name, submodule) for cond in compile_conditions):
                submodule.forward = torch.compile(submodule.forward, **compile_kwargs)
                compiled_count += 1
        return compiled_count

    @staticmethod
    def _compile_pipeline_module_instance(
        module_name: str,
        module: torch.nn.Module,
        fsdp_module_cls: type | None,
        compile_kwargs: dict[str, Any],
    ) -> Any:
        """Apply pipeline-level compile setup to one loaded component."""
        if fsdp_module_cls is not None and isinstance(module, fsdp_module_cls):
            logger.info(
                "%s is already FSDP-wrapped; skipping torch.compile in pipeline",
                module_name.capitalize(),
            )
            return module

        prepare_for_compile = getattr(module, "prepare_for_compile", None)
        if callable(prepare_for_compile):
            logger.info("Running prepare_for_compile for %s", module_name)
            prepare_for_compile()

        compiled_count = ComposedPipelineBase._compile_with_conditions(module, compile_kwargs)
        if compiled_count > 0:
            logger.info(
                "Enabled torch.compile for %d submodules in %s via _compile_conditions with kwargs=%s",
                compiled_count,
                module_name,
                compile_kwargs,
            )
            return module

        # Backward-compatible fallback: compile full module if no condition matched.
        logger.info("Enabling torch.compile for %s with kwargs=%s", module_name, compile_kwargs)
        return torch.compile(module, **compile_kwargs)

    def _maybe_compile_pipeline_module(
        self,
        module_name: str,
        fsdp_module_cls: type | None,
        compile_kwargs: dict[str, Any],
    ) -> None:
        if module_name not in self.modules:
            return

        entry = self.modules[module_name]
        if is_lazy_module(entry):
            # Compilation is part of materialization, not a one-time mutation
            # of the first loaded instance. The proxy remains in the module
            # map, so whole-module and conditional compile both survive every
            # release/reload cycle without making initialization eager.
            entry.set_materialize_transform(
                partial(
                    ComposedPipelineBase._compile_pipeline_module_instance,
                    module_name,
                    fsdp_module_cls=fsdp_module_cls,
                    compile_kwargs=dict(compile_kwargs),
                ))
            logger.info("Configured torch.compile for every materialization of deferred %s", module_name)
            return

        self.modules[module_name] = self._compile_pipeline_module_instance(
            module_name,
            entry,
            fsdp_module_cls,
            compile_kwargs,
        )

    def _apply_inference_compile(self, module_names: tuple[str, ...] | None = None) -> None:
        """Attach pipeline-level compile to modules that are present now.

        Sequential MiniMax-H3 loads DiT/VAEs after ``post_init``, so this is
        also called once those modules appear. Lazy proxies register a
        materialize transform and can be configured at ``post_init``.
        """
        if self.fastvideo_args is None:
            return
        compile_requested = any((
            self.fastvideo_args.enable_torch_compile,
            self.fastvideo_args.enable_torch_compile_text_encoder,
            self.fastvideo_args.enable_torch_compile_vae,
            self.fastvideo_args.enable_torch_compile_audio_vae,
        ))
        if self.fastvideo_args.training_mode and compile_requested:
            logger.info("Torch Compile enabled via FSDP loader for training; skipping additional pipeline compile")
        if self.fastvideo_args.training_mode:
            return

        compile_transformer = self.fastvideo_args.enable_torch_compile
        compile_text_encoder = self.fastvideo_args.enable_torch_compile_text_encoder
        compile_vae = self.fastvideo_args.enable_torch_compile_vae
        compile_audio_vae = self.fastvideo_args.enable_torch_compile_audio_vae
        if not (compile_transformer or compile_text_encoder or compile_vae or compile_audio_vae):
            return

        wanted = None if module_names is None else set(module_names)

        def _want(name: str) -> bool:
            return wanted is None or name in wanted

        fsdp_module_cls = None
        try:
            from torch.distributed.fsdp import FSDPModule  # type: ignore
            fsdp_module_cls = FSDPModule
        except Exception:  # pragma: no cover - FSDP not always available
            fsdp_module_cls = None

        global_compile_kwargs = (self.fastvideo_args.torch_compile_kwargs or {})
        dit_compile_kwargs = (self.fastvideo_args.torch_compile_kwargs_dit or global_compile_kwargs)
        text_compile_kwargs = (self.fastvideo_args.torch_compile_kwargs_text_encoder or global_compile_kwargs)
        vae_compile_kwargs = (self.fastvideo_args.torch_compile_kwargs_vae or global_compile_kwargs)
        audio_vae_compile_kwargs = (self.fastvideo_args.torch_compile_kwargs_audio_vae or global_compile_kwargs)

        if compile_transformer and self.fastvideo_args.inference_torch_compile:
            logger.info("inference_torch_compile already compiled the DiT regions in the "
                        "loader; skipping the pipeline-level DiT compile")
            compile_transformer = False
        if compile_transformer and any(_want(name) for name in ("transformer", "transformer_refine", "transformer_2")):
            for name in ("transformer", "transformer_refine", "transformer_2"):
                if _want(name):
                    self._maybe_compile_pipeline_module(
                        module_name=name,
                        fsdp_module_cls=fsdp_module_cls,
                        compile_kwargs=dit_compile_kwargs,
                    )
            if any(name in self.modules for name in ("transformer", "transformer_refine", "transformer_2")):
                logger.info("Torch Compile enabled for DiT")

        if compile_text_encoder and any(_want(name) for name in ("text_encoder", "text_encoder_2")):
            for name in ("text_encoder", "text_encoder_2"):
                if _want(name):
                    self._maybe_compile_pipeline_module(
                        module_name=name,
                        fsdp_module_cls=fsdp_module_cls,
                        compile_kwargs=text_compile_kwargs,
                    )
            if any(name in self.modules for name in ("text_encoder", "text_encoder_2")):
                logger.info("Torch Compile enabled for text encoder")

        if compile_vae and _want("vae"):
            self._maybe_compile_pipeline_module(
                module_name="vae",
                fsdp_module_cls=fsdp_module_cls,
                compile_kwargs=vae_compile_kwargs,
            )
            if "vae" in self.modules:
                logger.info("Torch Compile enabled for VAE")

        if compile_audio_vae and _want("audio_vae"):
            self._maybe_compile_pipeline_module(
                module_name="audio_vae",
                fsdp_module_cls=fsdp_module_cls,
                compile_kwargs=audio_vae_compile_kwargs,
            )
            if "audio_vae" in self.modules:
                logger.info("Torch Compile enabled for audio VAE")

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
        self._apply_inference_compile()

        trace_target = self.modules.get("transformer")
        if is_lazy_module(trace_target):
            # The hook manager keeps a strong reference to every module it
            # wraps, so attaching here would materialize the DiT before the
            # first request and pin that instance past any release.
            if envs.FASTVIDEO_TRACE_ACTIVATIONS:
                logger.warning("Activation trace is not attached to a deferred transformer; "
                               "turn off lazy_module_load to trace it")
            trace_target = None
        self._trace_mgr = attach_activation_trace(trace_target)

        if not self.fastvideo_args.training_mode:
            logger.info("Creating pipeline stages...")
            self.create_pipeline_stages(self.fastvideo_args)

            if self._lazy_module_load_enabled(self.fastvideo_args) and self._lazy_module_names:
                self._install_lazy_release_hooks()

            # Warmup NCCL communicators for sequence parallelism to avoid
            # slow first forward pass due to lazy initialization
            warmup_sequence_parallel_communication()

    def initialize_training_pipeline(self, training_args: TrainingArgs):
        raise NotImplementedError("if training_mode is True, the pipeline must implement this method")

    def initialize_validation_pipeline(self, training_args: TrainingArgs):
        raise NotImplementedError("if log_validation is True, the pipeline must implement this method")

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
        previous = self.modules.get(module_name)
        self.modules[module_name] = module
        # The release schedule keys proxies by identity. Replacing a deferred
        # module (or swapping a proxy for a freshly loaded instance) leaves
        # stages holding the old object unless the schedule is rebuilt.
        if self._lazy_release_hooks_installed and (is_lazy_module(previous) or is_lazy_module(module)):
            self._install_lazy_release_hooks()

    def _load_config(self, model_path: str) -> dict[str, Any]:
        revision = getattr(self.fastvideo_args, "revision", None)
        model_path = maybe_download_model(
            self.model_path,
            revision=revision,
            allow_patterns=self.get_hf_download_allow_patterns(),
        )
        self.model_path = model_path
        # fastvideo_args.downloaded_model_path = model_path
        logger.info("Model path: %s", model_path)
        config = verify_model_config_and_directory(
            model_path,
            required_component_dirs=self.get_hf_download_component_dirs(),
        )
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

    def load_modules(self,
                     fastvideo_args: FastVideoArgs,
                     loaded_modules: dict[str, torch.nn.Module] | None = None) -> dict[str, Any]:
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
        model_index.pop("_name_or_path", None)
        model_index.pop("workload_type", None)
        if "boundary_ratio" in model_index and model_index["boundary_ratio"] is not None:
            logger.info("MoE pipeline detected. Adding transformer_2 to self.required_config_modules...")
            self.required_config_modules.append("transformer_2")
            logger.info("MoE pipeline detected. Setting boundary ratio to %s", model_index["boundary_ratio"])
            fastvideo_args.pipeline_config.dit_config.boundary_ratio = model_index["boundary_ratio"]

        model_index.pop("boundary_ratio", None)
        # used by Wan2.2 ti2v
        model_index.pop("expand_timesteps", None)
        # HF metadata (e.g. Flux2 Klein is_distilled); not a loadable module
        model_index.pop("is_distilled", None)

        # some sanity checks
        assert len(model_index) > 1, "model_index.json must contain at least one pipeline module"

        for module_name in self.required_config_modules:
            if module_name not in model_index and module_name in self._extra_config_module_map:
                extra_module_value = self._extra_config_module_map[module_name]
                logger.warning(
                    "model_index.json does not contain a %s module, but found {%s: %s} in _extra_config_module_map, adding to model_index.",
                    module_name, module_name, extra_module_value)
                if extra_module_value in model_index:
                    logger.info("Using module %s for %s", extra_module_value, module_name)
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
        for module_name, module_spec in model_index.items():
            if not isinstance(module_spec, list | tuple):
                logger.info(
                    "Skipping non-module config entry %s=%s",
                    module_name,
                    module_spec,
                )
                continue
            if len(module_spec) < 1:
                logger.warning(
                    "Skipping module %s due to invalid empty spec in model_index.json",
                    module_name,
                )
                continue
            transformers_or_diffusers = module_spec[0]
            if transformers_or_diffusers is None:
                logger.warning("Module %s in model_index.json has null value, removing from required_config_modules",
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

            component_model_path = os.path.join(self.model_path, load_module_name)

            def load_component(load_module_name: str = load_module_name,
                               component_model_path: str = component_model_path,
                               transformers_or_diffusers: str = transformers_or_diffusers) -> Any:
                return PipelineComponentLoader.load_module(
                    module_name=load_module_name,
                    component_model_path=component_model_path,
                    transformers_or_diffusers=transformers_or_diffusers,
                    fastvideo_args=fastvideo_args,
                )

            if self._lazy_module_load_enabled(fastvideo_args) and module_name in self._lazy_module_names:
                module = LazyModule(module_name, load_component)
                logger.info("Deferred module %s from %s", module_name, component_model_path)
            else:
                module = load_component()
                logger.info("Loaded module %s from %s", module_name, component_model_path)

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

    @staticmethod
    def _lazy_module_load_enabled(fastvideo_args: FastVideoArgs) -> bool:
        """Deferred loading is inference only; training needs every component."""
        if not fastvideo_args.lazy_module_load:
            return False
        if fastvideo_args.training_mode:
            logger.warning("lazy_module_load is not supported in training mode; loading all modules eagerly")
            return False
        return True

    def _build_lazy_release_schedule(self) -> dict[int, list[str]]:
        """Map each stage index to the deferred modules it is the last user of.

        Derived from what the stages actually hold rather than declared per
        pipeline, so a stage added later cannot have its module freed out from
        under it. A module no stage references is never released, which is the
        safe direction: it stays loaded rather than disappearing mid-run.
        """
        lazy_names_by_id = {id(module): name for name, module in self.modules.items() if is_lazy_module(module)}
        if not lazy_names_by_id:
            return {}

        last_use: dict[str, int] = {}
        for index, stage in enumerate(self._stages):
            for held in _iter_held_objects(stage):
                name = lazy_names_by_id.get(id(held))
                if name is not None:
                    last_use[name] = index

        schedule: dict[int, list[str]] = {}
        for name, index in sorted(last_use.items()):
            schedule.setdefault(index, []).append(name)

        unreferenced = sorted(set(lazy_names_by_id.values()) - set(last_use))
        if unreferenced:
            logger.info("Deferred modules held by no stage, so never released: %s", unreferenced)
        return schedule

    def _install_lazy_release_hooks(self) -> None:
        """Tell each stage which deferred modules to free once it returns."""
        if not self._lazy_module_names:
            # Unified-memory auto-enable turns the flag on for every pipeline.
            # Only opted-in families (currently MiniMax-H3) should log about it.
            self._lazy_release_hooks_installed = True
            return
        schedule = self._build_lazy_release_schedule()
        for index, stage in enumerate(self._stages):
            stage._lazy_modules_to_release = tuple(self.modules[name] for name in schedule.get(index, ()))

        if not schedule:
            # Deferring without releasing still lowers the load-time peak, but
            # it is not what the flag promises, so say so rather than let a
            # no-op look like a win.
            logger.warning(
                "lazy_module_load is on but no deferred module is held by a stage, so nothing will be "
                "freed mid-run. Pipeline %s may load its modules eagerly or hold them outside its stages.",
                type(self).__name__)
            self._lazy_release_hooks_installed = True
            return

        for index, names in sorted(schedule.items()):
            logger.info("Deferred modules to free after stage %d (%s): %s", index,
                        getattr(self._stages[index], "_pipeline_stage_name", "?"), names)
        self._lazy_release_hooks_installed = True

    def _release_all_lazy_modules(self) -> None:
        """Free every deferred component that is currently materialized."""
        for module_name, module in self.modules.items():
            if not is_lazy_module(module):
                continue
            try:
                module.release()
            except Exception:
                # Never let cleanup replace the exception being propagated.
                logger.exception("Failed to release deferred module %s", module_name)

    def add_stage(self, stage_name: str, stage: PipelineStage):
        assert self.modules is not None, "No modules are registered"
        # Preserve the pipeline-unique stage key for structured metrics.
        # Multiple stages can share the same class (for example LTX2 main
        # denoise and refine denoise), so class-name keys would collide.
        stage._pipeline_stage_name = stage_name
        self._stages.append(stage)
        self._stage_name_mapping[stage_name] = stage
        setattr(self, stage_name, stage)

        if self._lazy_release_hooks_installed:
            # The schedule maps each deferred module to its last holder. A
            # stage appended afterwards may hold a module an earlier stage has
            # already been told to free, which would hand it a released
            # component mid-run. Rebuild rather than trust the stale plan.
            # H3 sequential load adds denoise stages on the first request; that
            # is the designed path, so do not log it as a warning.
            logger.debug("Stage %s was added after the deferred-release schedule was built; rebuilding the schedule",
                         stage_name)
            self._install_lazy_release_hooks()

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
        logger.info("Running pipeline stages: %s", self._stage_name_mapping.keys())
        # logger.info("Batch: %s", batch)
        try:
            for stage in self.stages:
                batch = stage(batch, fastvideo_args)
        except BaseException:
            # A stage's own hook frees only what that stage was the last user
            # of. When the run aborts earlier, everything already materialized
            # stays for the life of the generator, and the retry a
            # memory-constrained caller is most likely to attempt starts from a
            # worse position than the request that just failed.
            self._release_all_lazy_modules()
            raise

        # Return the output
        return batch

    def train(self) -> None:
        raise NotImplementedError("if training_mode is True, the pipeline must implement this method")

    def close(self) -> None:
        detach_activation_trace(getattr(self, "_trace_mgr", None))
        self._trace_mgr = None

    def __del__(self):
        self.close()
