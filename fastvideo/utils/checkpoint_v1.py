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
)


from fastvideo.utils.logging_ import main_print
import shutil
import logging

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


def save_checkpoint_v1(transformer, optimizer, scheduler, rank, output_dir, step):
    """
    Save both consolidated and distributed checkpoints with full training state.
    """
    states = {
        "model": ModelWrapper(transformer),
        "optimizer": optimizer,
        "scheduler": scheduler
    }

    if rank <= 0:
        save_dir = os.path.join(output_dir, f"checkpoint-{step}")
        os.makedirs(save_dir, exist_ok=True)

        # 1. Save consolidated safetensors for model weights
        with FSDP.state_dict_type(
            transformer,
            StateDictType.FULL_STATE_DICT,
            state_dict_config=FullStateDictConfig(offload_to_cpu=True, rank0_only=True),
        ):
            model_state = transformer.state_dict()
            weight_path = os.path.join(save_dir, "diffusion_pytorch_model.safetensors")
            save_file(model_state, weight_path)

        # 2. Save full training state as distributed checkpoint
        dist_cp.save_state_dict(
            state_dict=states,
            storage_writer=dist_cp.FileSystemWriter(os.path.join(save_dir, "dist_cp")),
            planner=DefaultSavePlanner()
        )

        # Save config
        config_dict = transformer.hf_config
        if "dtype" in config_dict:
            del config_dict["dtype"]
        config_path = os.path.join(save_dir, "config.json")
        with open(config_path, "w") as f:
            json.dump(config_dict, f, indent=4)
            
    main_print(f"--> checkpoint saved at step {step}")


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


def load_checkpoint_v1(transformer, checkpoint_path, load_type="safetensors"):
    """
    Load a checkpoint into the transformer model.
    
    Args:
        transformer: The transformer model to load weights into
        checkpoint_path: Path to the checkpoint directory or file
        load_type: Either "safetensors" or "dist_cp" to specify which format to load
    """
    if os.path.isdir(checkpoint_path):
        if load_type == "safetensors":
            model_path = os.path.join(checkpoint_path, "diffusion_pytorch_model.safetensors")
            
            FSDP.set_state_dict_type(
                transformer,
                state_dict_type=StateDictType.FULL_STATE_DICT,
                state_dict_config=FullStateDictConfig(offload_to_cpu=True, rank0_only=True),
            )
            
            state_dict = load_file(model_path)
            transformer.load_state_dict(state_dict, strict=False)
            
        elif load_type == "dist_cp":
            dist_cp_dir = os.path.join(checkpoint_path, "dist_cp")
            
            FSDP.set_state_dict_type(
                transformer,
                state_dict_type=StateDictType.SHARDED_STATE_DICT
            )
            
            state_dict = {"model": transformer.state_dict()}
            dist_cp.load_state_dict(
                state_dict=state_dict,
                storage_reader=dist_cp.FileSystemReader(dist_cp_dir),
                planner=DefaultLoadPlanner()
            )
            transformer.load_state_dict(state_dict["model"])

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
            "optimizer": optimizer,
            "scheduler": scheduler
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
        
        try:
            # 1. Save distributed checkpoint (all states)
            dist_cp.save_state_dict(
                self.states,
                storage_writer=dist_cp.FileSystemWriter(checkpoint_dir / "dist_cp"),
                planner=DefaultSavePlanner()
            )

            # 2. Save consolidated safetensors on rank 0
            if is_main_process:
                model = self.states["model"].model[0]
                state_dict = gather_state_dict_on_cpu_rank0(
                    model=model,
                    device=device,
                    is_main_process=is_main_process
                )
                save_file(
                    state_dict,
                    checkpoint_dir / "model.safetensors"
                )

                if self._callback_fn is not None:
                    self._callback_fn(state_dict)

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
            if load_type == "safetensors":
                # Load consolidated weights
                with FSDP.state_dict_type(
                    self.states["model"].model[0],
                    StateDictType.FULL_STATE_DICT,
                    FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
                ):
                    state_dict = load_file(checkpoint_dir / "model.safetensors")
                    self.states["model"].load_state_dict(state_dict)
            else:
                # Load full training state from distributed checkpoint
                dist_cp.load_state_dict(
                    self.states,
                    storage_reader=dist_cp.FileSystemReader(checkpoint_dir / "dist_cp"),
                    planner=DefaultLoadPlanner()
                )
            return True
            
        except Exception as e:
            logger.error(f"Failed to load checkpoint from {checkpoint_dir}: {str(e)}")
            raise


def gather_state_dict_on_cpu_rank0(
    model: torch.nn.Module, 
    device: torch.device,
    *, 
    is_main_process: bool
) -> Dict[str, Any]:
    """
    Gathers a model's full state dict on CPU for rank 0 process using FSDP.
    
    Args:
        model: The FSDP-wrapped model
        device: Device to temporarily move tensors to during gathering
        is_main_process: Whether current process is rank 0
    
    Returns:
        Dict containing the full state dict (on rank 0) or empty dict (other ranks)
    """
    with FSDP.state_dict_type(
        model,
        StateDictType.FULL_STATE_DICT,
        FullStateDictConfig(offload_to_cpu=True, rank0_only=True),
    ):
        state_dict = model.state_dict()
        
    if is_main_process:
        # Move any GPU tensors to CPU
        return {k: v.cpu() if isinstance(v, torch.Tensor) else v 
                for k, v in state_dict.items()}
    return {}