import json
import os
import functools
from typing import Union, List, Dict, Any, Optional, Callable
import pathlib
import torch
import torch.distributed.checkpoint as dist_cp
from safetensors.torch import load_file, save_file
from torch.distributed.checkpoint.default_planner import DefaultLoadPlanner, DefaultSavePlanner
from torch.distributed.fsdp import FullStateDictConfig
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import StateDictType
from torch.distributed.checkpoint.stateful import Stateful
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_model_state_dict,
    set_model_state_dict,
    get_optimizer_state_dict,
    set_optimizer_state_dict,
)


from fastvideo.utils.logging_ import main_print
import shutil
import logging
import time
from torch.distributed.fsdp._common_utils import _FSDPState
from contextlib import contextmanager

logger = logging.getLogger(__name__)


class ModelWrapper(torch.distributed.checkpoint.stateful.Stateful):
    def __init__(self, model: Union[torch.nn.Module, List[torch.nn.Module]]) -> None:
        self.model = [model] if isinstance(model, torch.nn.Module) else model

    def state_dict(self) -> Dict[str, Any]:
        return {k: v for sd in map(get_model_state_dict, self.model) for k, v in sd.items()}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        func = functools.partial(
            set_model_state_dict,
            model_state_dict=state_dict,
            options=StateDictOptions(strict=False),
        )
        list(map(func, self.model))


class OptimizerWrapper(torch.distributed.checkpoint.stateful.Stateful):
    def __init__(self, optimizer: torch.optim.Optimizer) -> None:
        self.optimizer = optimizer

    @property
    def param_groups(self):
        return self.optimizer.param_groups

    def state_dict(self) -> Dict[str, Any]:
        return {
            "state": self.optimizer.state_dict(),
            "param_groups": [
                {k: v for k, v in g.items() if k != "params"}
                for g in self.optimizer.param_groups
            ]
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self.optimizer.load_state_dict(state_dict["state"])
        for param_group, saved_group in zip(
            self.optimizer.param_groups, state_dict["param_groups"]
        ):
            param_group.update(saved_group)


class SchedulerWrapper(torch.distributed.checkpoint.stateful.Stateful):
    def __init__(self, scheduler) -> None:
        self.scheduler = scheduler

    def state_dict(self) -> Dict[str, Any]:
        return {"scheduler": self.scheduler.state_dict()}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self.scheduler.load_state_dict(state_dict["scheduler"])

    def get_last_lr(self) -> List[float]:
        return self.scheduler.get_last_lr()


def save_checkpoint_v1(transformer, optimizer, scheduler, rank, output_dir, step):
    """Save distributed checkpoint with full training state."""
    states = {
        "model": ModelWrapper(transformer),
        "optimizer": OptimizerWrapper(optimizer),
        "scheduler": SchedulerWrapper(scheduler)
    }

    checkpoint_dir = os.path.join(output_dir, f"checkpoint-{step}")
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Save full training state as distributed checkpoint
    begin_time = time.monotonic()
    torch.distributed.checkpoint.save(
        states,
        checkpoint_id=checkpoint_dir
    )
    end_time = time.monotonic()
    
    logger.info(f"Saved checkpoint in {end_time - begin_time:.2f} seconds at step {step}")

    # Save consolidated weights in safetensors format on rank 0
    if rank <= 0:
        state_dict = gather_state_dict_on_cpu_rank0(
            transformer, 
            device=transformer.device,
            is_main_process=True
        )
        save_file(state_dict, os.path.join(checkpoint_dir, "model.safetensors"))

        # Save config
        config_dict = transformer.hf_config
        if "dtype" in config_dict:
            del config_dict["dtype"]
        config_path = os.path.join(checkpoint_dir, "config.json")
        with open(config_path, "w") as f:
            json.dump(config_dict, f, indent=4)


def save_consolidated_checkpoint(transformer, rank, output_dir, step):
    """
    Save only the consolidated checkpoint in safetensors format.
    Useful for smaller models that don't require distributed checkpointing.
    
    Args:
        transformer: The transformer model to save
        rank: Current process rank
        output_dir: Directory to save checkpoint
        step: Current training step
    """
    # Configure FSDP to save full state dict
    FSDP.set_state_dict_type(
        transformer,
        state_dict_type=StateDictType.FULL_STATE_DICT,
        state_dict_config=FullStateDictConfig(offload_to_cpu=True, rank0_only=True),
    )

    # Save only on rank 0
    if rank <= 0:
        save_dir = os.path.join(output_dir, f"checkpoint-{step}")
        os.makedirs(save_dir, exist_ok=True)

        # Get and save state dict
        cpu_state = transformer.state_dict()
        weight_path = os.path.join(save_dir, "diffusion_pytorch_model.safetensors")
        save_file(cpu_state, weight_path)

        # Save config
        config_dict = transformer.hf_config
        if "dtype" in config_dict:
            del config_dict["dtype"]
        config_path = os.path.join(save_dir, "config.json")
        with open(config_path, "w") as f:
            json.dump(config_dict, f, indent=4)

    main_print(f"--> consolidated checkpoint saved at step {step}")


def load_checkpoint_v1(transformer, optimizer=None, scheduler=None, checkpoint_path=None):
    """Load a distributed checkpoint."""
    if not os.path.exists(checkpoint_path):
        return transformer

    states = {
        "model": ModelWrapper(transformer)
    }
    
    if optimizer is not None:
        states["optimizer"] = OptimizerWrapper(optimizer)
    if scheduler is not None:
        states["scheduler"] = SchedulerWrapper(scheduler)

    # Load distributed checkpoint
    torch.distributed.checkpoint.load(
        states,
        checkpoint_id=checkpoint_path
    )

    # Load config if available
    config_path = os.path.join(checkpoint_path, "config.json")
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)
        if hasattr(transformer, 'hf_config'):
            transformer.hf_config.update(config)

    return transformer


class FSDPCheckpointer:
    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler._LRScheduler,
        states: Dict[str, Any],
        output_dir: str,
        checkpointing_steps: int,
        enable: bool = True,
        _callback_fn: Optional[Callable] = None,
    ):
        self.states = states
        self.states.update({
            "model": ModelWrapper(model),
            "optimizer": OptimizerWrapper(optimizer),
            "scheduler": SchedulerWrapper(scheduler)
        })
        
        self.output_dir = pathlib.Path(output_dir)
        self.checkpointing_steps = checkpointing_steps
        self.enable = enable
        self._callback_fn = _callback_fn

        if enable and not self.output_dir.exists():
            self.output_dir.mkdir(parents=True, exist_ok=True)

    def save(self, step: int, *, device: torch.device, is_main_process: bool):
        if not self.enable:
            return None

        checkpoint_dir = self.output_dir / f"checkpoint-{step}"
        begin_time = time.monotonic()
        
        try:
            # Save model state dict
            model_state = {
                "model": self.states["model"].state_dict()
            }
            torch.distributed.checkpoint.save(
                model_state,
                storage_writer=dist_cp.FileSystemWriter(checkpoint_dir / "model"),
                planner=DefaultSavePlanner()
            )

            # Save optimizer and scheduler states separately
            if is_main_process:
                # Save optimizer state
                opt_state = {
                    "state": self.states["optimizer"].state_dict(),
                    "param_groups": [
                        {k: v for k, v in g.items() if k != "params"}
                        for g in self.states["optimizer"].param_groups
                    ]
                }
                torch.save(opt_state, checkpoint_dir / "optimizer.pt")

                # Save scheduler state
                scheduler_state = self.states["scheduler"].state_dict()
                torch.save(scheduler_state, checkpoint_dir / "scheduler.pt")

                # Save other states
                other_states = {
                    k: v for k, v in self.states.items()
                    if k not in ["model", "optimizer", "scheduler"]
                }
                if other_states:
                    torch.save(other_states, checkpoint_dir / "other_states.pt")

                # Save consolidated model weights in safetensors format
                model = self.states["model"].model[0]

                print(f"Gathering state dict on CPU rank 0")
                state_dict = gather_state_dict_on_cpu_rank0(
                    model=model,
                    device=device,
                    is_main_process=is_main_process
                )
                print(f"State dict gathered on CPU rank 0")

                save_file(state_dict, checkpoint_dir / "model.safetensors")
                print(f"State dict saved to checkpoint_dir / model.safetensors")
                if self._callback_fn is not None:
                    self._callback_fn(state_dict)

            end_time = time.monotonic()
            logger.info(f"Saved checkpoint in {end_time - begin_time:.2f} seconds at step {step}")
            return checkpoint_dir.as_posix()

        except Exception as e:
            logger.error(f"Failed to save checkpoint at step {step}: {str(e)}")
            if checkpoint_dir.exists():
                shutil.rmtree(checkpoint_dir)
            raise

    def load(self, checkpoint_dir: str, load_type: str = "safetensors"):
        if not self.enable:
            return False
            
        checkpoint_dir = pathlib.Path(checkpoint_dir)
        if not checkpoint_dir.exists():
            return False

        try:
            # Load model state
            model_state = {}
            torch.distributed.checkpoint.load(
                model_state,
                storage_reader=dist_cp.FileSystemReader(checkpoint_dir / "model"),
                planner=DefaultLoadPlanner()
            )
            self.states["model"].load_state_dict(model_state["model"])

            # Load optimizer and scheduler states on main process
            if self.is_main_process:
                # Load optimizer state
                opt_state = torch.load(checkpoint_dir / "optimizer.pt")
                self.states["optimizer"].load_state_dict(opt_state["state"])
                for param_group, saved_group in zip(
                    self.states["optimizer"].param_groups, opt_state["param_groups"]
                ):
                    param_group.update(saved_group)

                # Load scheduler state
                scheduler_state = torch.load(checkpoint_dir / "scheduler.pt")
                self.states["scheduler"].load_state_dict(scheduler_state)

                # Load other states
                if (checkpoint_dir / "other_states.pt").exists():
                    other_states = torch.load(checkpoint_dir / "other_states.pt")
                    self.states.update(other_states)

            return True
            
        except Exception as e:
            logger.error(f"Failed to load checkpoint from {checkpoint_dir}: {str(e)}")
            raise


def gather_state_dict_on_cpu_rank0(
    model, device: Optional[torch.device] = None, *, is_main_process: bool
) -> Dict[str, Any]:
    """Gather full state dict on CPU for rank 0"""
    cpu_state_dict = {}
    sharded_sd = model.state_dict()
    
    for param_name, param in sharded_sd.items():
        if param.is_cpu:
            # Move back to device if offloaded to CPU
            param = param.to(device)
        if hasattr(param, "_local_tensor"):
            # Gather DTensor
            print(f"Gathering DTensor on CPU rank 0")
            param = param.full_tensor()
        if is_main_process:
            print(f"Gathering state dict on CPU rank 0")
            cpu_state_dict[param_name] = param.cpu()

        print(f"Barrier on CPU rank 0")
        torch.distributed.barrier()
        print(f"Barrier on CPU rank 0 done")
        
    return cpu_state_dict
    