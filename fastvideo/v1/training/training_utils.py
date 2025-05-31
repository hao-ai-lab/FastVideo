import json
import math
import os
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dist_cp
import torch.distributed.checkpoint.stateful
from safetensors.torch import save_file
from torch.distributed.checkpoint.state_dict import (StateDictOptions,
                                                     get_model_state_dict,
                                                     get_optimizer_state_dict,
                                                     set_model_state_dict,
                                                     set_optimizer_state_dict)
from torch.distributed.fsdp import FullStateDictConfig
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import StateDictType

from fastvideo.v1.logger import init_logger

logger = init_logger(__name__)

_HAS_ERRORED_CLIP_GRAD_NORM_WHILE_HANDLING_FAILING_DTENSOR_CASES = False


def gather_state_dict_on_cpu_rank0(model,
                                   device: Optional[torch.device] = None,
                                   *,
                                   is_main_process: bool) -> Dict[str, Any]:
    rank = int(os.environ.get("RANK", 0))
    cpu_state_dict = {}
    sharded_sd = model.state_dict()
    for param_name, param in sharded_sd.items():
        if param.is_cpu:
            # Move back to device if offloaded to CPU
            param = param.to(device)
        if hasattr(param, "_local_tensor"):
            # Gather DTensor
            #logger.info(f"rank: {rank}, Gathering DTensor for {param_name}", local_main_process_only=False)
            # logger.info(f"rank: {rank}, type of param: {type(param)}", local_main_process_only=False)
            # logger.info(f"rank: {rank}, param: {param}", local_main_process_only=False)
            param = param.full_tensor()
        # if is_main_process:
        if rank <= 0:
            #print(f"Moving to CPU for {param_name}")
            cpu_state_dict[param_name] = param.cpu()
            # logger.info(f"rank: {rank}, done moving to cpu for {param_name}", local_main_process_only=False)
        #print(f"Barrier for {param_name}")
        torch.distributed.barrier()
    return cpu_state_dict


def save_checkpoint_new(transformer,
                        rank,
                        output_dir,
                        step,
                        optimizer=None,
                        dataloader=None,
                        scheduler=None) -> None:
    """
    Save checkpoint following finetrainer's distributed checkpoint approach.
    Saves both distributed checkpoint and consolidated model weights.
    """
    save_dir = os.path.join(output_dir, f"checkpoint-{step}")
    os.makedirs(save_dir, exist_ok=True)

    states = {
        "model": ModelWrapper(transformer),
    }

    if optimizer is not None:
        states["optimizer"] = OptimizerWrapper(transformer, optimizer)

    if dataloader is not None:
        states["dataloader"] = dataloader

    if scheduler is not None:
        states["scheduler"] = SchedulerWrapper(scheduler)

    distcp_dir = os.path.join(save_dir, "distributed_checkpoint")
    logger.info("rank: %s, saving distributed checkpoint to %s",
                rank,
                distcp_dir,
                local_main_process_only=False)

    begin_time = time.perf_counter()
    dist_cp.save(states, checkpoint_id=distcp_dir)
    end_time = time.perf_counter()

    logger.info("rank: %s, distributed checkpoint saved in %.2f seconds",
                rank,
                end_time - begin_time,
                local_main_process_only=False)

    cpu_state = gather_state_dict_on_cpu_rank0(transformer,
                                               device=None,
                                               is_main_process=True)

    if rank <= 0:
        # Save model weights (consolidated)
        weight_path = os.path.join(save_dir,
                                   "diffusion_pytorch_model.safetensors")
        logger.info("rank: %s, saving consolidated checkpoint to %s",
                    rank,
                    weight_path,
                    local_main_process_only=False)
        save_file(cpu_state, weight_path)
        logger.info("rank: %s, consolidated checkpoint saved to %s",
                    rank,
                    weight_path,
                    local_main_process_only=False)

        # Save model config
        config_dict = transformer.hf_config
        if "dtype" in config_dict:
            del config_dict["dtype"]  # TODO
        config_path = os.path.join(save_dir, "config.json")
        # save dict as json
        with open(config_path, "w") as f:
            json.dump(config_dict, f, indent=4)
        logger.info("--> checkpoint saved at step %s to %s", step, weight_path)


def compute_density_for_timestep_sampling(
    weighting_scheme: str,
    batch_size: int,
    generator,
    logit_mean: Optional[float] = None,
    logit_std: Optional[float] = None,
    mode_scale: Optional[float] = None,
):
    """
    Compute the density for sampling the timesteps when doing SD3 training.

    Courtesy: This was contributed by Rafie Walker in https://github.com/huggingface/diffusers/pull/8528.

    SD3 paper reference: https://arxiv.org/abs/2403.03206v1.
    """
    if weighting_scheme == "logit_normal":
        # See 3.1 in the SD3 paper ($rf/lognorm(0.00,1.00)$).
        u = torch.normal(
            mean=logit_mean,
            std=logit_std,
            size=(batch_size, ),
            device="cpu",
            generator=generator,
        )
        u = torch.nn.functional.sigmoid(u)
    elif weighting_scheme == "mode":
        u = torch.rand(size=(batch_size, ), device="cpu", generator=generator)
        u = 1 - u - mode_scale * (torch.cos(math.pi * u / 2)**2 - 1 + u)
    else:
        u = torch.rand(size=(batch_size, ), device="cpu", generator=generator)
    return u


def get_sigmas(noise_scheduler,
               device,
               timesteps,
               n_dim=4,
               dtype=torch.float32) -> torch.Tensor:
    sigmas = noise_scheduler.sigmas.to(device=device, dtype=dtype)
    schedule_timesteps = noise_scheduler.timesteps.to(device)
    timesteps = timesteps.to(device)
    step_indices = [(schedule_timesteps == t).nonzero().item()
                    for t in timesteps]

    sigma = sigmas[step_indices].flatten()
    while len(sigma.shape) < n_dim:
        sigma = sigma.unsqueeze(-1)
    return sigma


def save_checkpoint(transformer, rank, output_dir, step) -> None:
    # Configure FSDP to save full state dict
    FSDP.set_state_dict_type(
        transformer,
        state_dict_type=StateDictType.FULL_STATE_DICT,
        state_dict_config=FullStateDictConfig(offload_to_cpu=True,
                                              rank0_only=True),
    )

    # Now get the state dict
    cpu_state = transformer.state_dict()

    # Save it (only on rank 0 since we used rank0_only=True)
    if rank <= 0:
        save_dir = os.path.join(output_dir, f"checkpoint-{step}")
        os.makedirs(save_dir, exist_ok=True)
        weight_path = os.path.join(save_dir, "diffusion_pytorch_model.pt")
        torch.save(cpu_state, weight_path)
        config_dict = transformer.hf_config
        if "dtype" in config_dict:
            del config_dict["dtype"]  # TODO
        config_path = os.path.join(save_dir, "config.json")
        # save dict as json
        with open(config_path, "w") as f:
            json.dump(config_dict, f, indent=4)
        logger.info("--> checkpoint saved at step %s to %s", step, weight_path)


def normalize_dit_input(model_type, latents, args=None) -> torch.Tensor:
    if model_type == "hunyuan_hf" or model_type == "hunyuan":
        return latents * 0.476986
    elif model_type == "wan":
        from fastvideo.v1.configs.models.vaes.wanvae import WanVAEConfig
        vae_config = WanVAEConfig()
        latents_mean = torch.tensor(vae_config.arch_config.latents_mean)
        latents_std = 1.0 / torch.tensor(vae_config.arch_config.latents_std)

        latents_mean = latents_mean.view(1, -1, 1, 1,
                                         1).to(device=latents.device)
        latents_std = latents_std.view(1, -1, 1, 1, 1).to(device=latents.device)
        latents = ((latents.float() - latents_mean) * latents_std).to(latents)
        return latents
    else:
        raise NotImplementedError(f"model_type {model_type} not supported")


def clip_grad_norm_while_handling_failing_dtensor_cases(
    parameters: Union[torch.Tensor, List[torch.Tensor]],
    max_norm: float,
    norm_type: float = 2.0,
    error_if_nonfinite: bool = False,
    foreach: Optional[bool] = None,
    pp_mesh: Optional[torch.distributed.device_mesh.DeviceMesh] = None,
) -> Optional[torch.Tensor]:
    global _HAS_ERRORED_CLIP_GRAD_NORM_WHILE_HANDLING_FAILING_DTENSOR_CASES

    if not _HAS_ERRORED_CLIP_GRAD_NORM_WHILE_HANDLING_FAILING_DTENSOR_CASES:
        try:
            return clip_grad_norm_(parameters, max_norm, norm_type,
                                   error_if_nonfinite, foreach, pp_mesh)
        except NotImplementedError as e:
            if "DTensor does not support cross-mesh operation" in str(e):
                # https://github.com/pytorch/pytorch/issues/134212
                logger.warning(
                    "DTensor does not support cross-mesh operation. If you haven't fully tensor-parallelized your "
                    "model, while combining other parallelisms such as FSDP, it could be the reason for this error. "
                    "Gradient clipping will be skipped and gradient norm will not be logged."
                )
        except Exception as e:
            logger.warning(
                "An error occurred while clipping gradients: %s. Gradient clipping will be skipped and gradient "
                "norm will not be logged.", e)
            _HAS_ERRORED_CLIP_GRAD_NORM_WHILE_HANDLING_FAILING_DTENSOR_CASES = True
    return None


# Copied from https://github.com/pytorch/torchtitan/blob/4a169701555ab9bd6ca3769f9650ae3386b84c6e/torchtitan/utils.py#L362
@torch.no_grad()
def clip_grad_norm_(
    parameters: Union[torch.Tensor, List[torch.Tensor]],
    max_norm: float,
    norm_type: float = 2.0,
    error_if_nonfinite: bool = False,
    foreach: Optional[bool] = None,
    pp_mesh: Optional[torch.distributed.device_mesh.DeviceMesh] = None,
) -> torch.Tensor:
    r"""
    Clip the gradient norm of parameters.

    Gradient norm clipping requires computing the gradient norm over the entire model.
    `torch.nn.utils.clip_grad_norm_` only computes gradient norm along DP/FSDP/TP dimensions.
    We need to manually reduce the gradient norm across PP stages.
    See https://github.com/pytorch/torchtitan/issues/596 for details.

    Args:
        parameters (`torch.Tensor` or `List[torch.Tensor]`):
            Tensors that will have gradients normalized.
        max_norm (`float`):
            Maximum norm of the gradients after clipping.
        norm_type (`float`, defaults to `2.0`):
            Type of p-norm to use. Can be `inf` for infinity norm.
        error_if_nonfinite (`bool`, defaults to `False`):
            If `True`, an error is thrown if the total norm of the gradients from `parameters` is `nan`, `inf`, or `-inf`.
        foreach (`bool`, defaults to `None`):
            Use the faster foreach-based implementation. If `None`, use the foreach implementation for CUDA and CPU native tensors
            and silently fall back to the slow implementation for other device types.
        pp_mesh (`torch.distributed.device_mesh.DeviceMesh`, defaults to `None`):
            Pipeline parallel device mesh. If not `None`, will reduce gradient norm across PP stages.

    Returns:
        `torch.Tensor`:
            Total norm of the gradients
    """
    grads = [p.grad for p in parameters if p.grad is not None]

    # TODO(aryan): Wait for next Pytorch release to use `torch.nn.utils.get_total_norm`
    # total_norm = torch.nn.utils.get_total_norm(grads, norm_type, error_if_nonfinite, foreach)
    total_norm = _get_total_norm(grads, norm_type, error_if_nonfinite, foreach)

    # If total_norm is a DTensor, the placements must be `torch.distributed._tensor.ops.math_ops._NormPartial`.
    # We can simply reduce the DTensor to get the total norm in this tensor's process group
    # and then convert it to a local tensor.
    # It has two purposes:
    #   1. to make sure the total norm is computed correctly when PP is used (see below)
    #   2. to return a reduced total_norm tensor whose .item() would return the correct value
    if isinstance(total_norm, torch.distributed.tensor.DTensor):
        # Will reach here if any non-PP parallelism is used.
        # If only using PP, total_norm will be a local tensor.
        total_norm = total_norm.full_tensor()

    if pp_mesh is not None:
        raise NotImplementedError("Pipeline parallel is not supported")
        if math.isinf(norm_type):
            dist.all_reduce(total_norm,
                            op=dist.ReduceOp.MAX,
                            group=pp_mesh.get_group())
        else:
            total_norm **= norm_type
            dist.all_reduce(total_norm,
                            op=dist.ReduceOp.SUM,
                            group=pp_mesh.get_group())
            total_norm **= 1.0 / norm_type

    _clip_grads_with_norm_(parameters, max_norm, total_norm, foreach)
    return total_norm


@torch.no_grad()
def _clip_grads_with_norm_(
    parameters: Union[torch.Tensor, List[torch.Tensor]],
    max_norm: float,
    total_norm: torch.Tensor,
    foreach: Optional[bool] = None,
) -> None:
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    grads = [p.grad for p in parameters if p.grad is not None]
    max_norm = float(max_norm)
    if len(grads) == 0:
        return
    grouped_grads: dict[Tuple[torch.device, torch.dtype],
                        Tuple[List[List[torch.Tensor]],
                              List[int]]] = (_group_tensors_by_device_and_dtype(
                                  [grads]))  # type: ignore[assignment]

    clip_coef = max_norm / (total_norm + 1e-6)

    # Note: multiplying by the clamped coef is redundant when the coef is clamped to 1, but doing so
    # avoids a `if clip_coef < 1:`


def load_checkpoint_new(transformer,
                        rank,
                        checkpoint_path,
                        optimizer=None,
                        dataloader=None,
                        scheduler=None) -> int:
    """
    Load checkpoint following finetrainer's distributed checkpoint approach.
    Returns the step number from which training should resume.
    """
    if not os.path.exists(checkpoint_path):
        logger.warning("Checkpoint path %s does not exist", checkpoint_path)
        return 0

    # Extract step number from checkpoint path
    step = int(os.path.basename(checkpoint_path).split('-')[-1])

    if rank <= 0:
        logger.info("Loading checkpoint from step %s", step)

    distcp_dir = os.path.join(checkpoint_path, "distributed_checkpoint")

    if not os.path.exists(distcp_dir):
        logger.warning("Distributed checkpoint directory %s does not exist",
                       distcp_dir)
        return 0

    states = {
        "model": ModelWrapper(transformer),
    }

    if optimizer is not None:
        states["optimizer"] = OptimizerWrapper(
            transformer,
            optimizer)  # Use wrapper for proper Stateful implementation

    if dataloader is not None:
        states["dataloader"] = dataloader

    if scheduler is not None:
        states["scheduler"] = SchedulerWrapper(scheduler)

    logger.info("rank: %s, loading distributed checkpoint from %s",
                rank,
                distcp_dir,
                local_main_process_only=False)

    begin_time = time.perf_counter()
    dist_cp.load(states, checkpoint_id=distcp_dir)
    end_time = time.perf_counter()

    logger.info("rank: %s, distributed checkpoint loaded in %.2f seconds",
                rank,
                end_time - begin_time,
                local_main_process_only=False)
    logger.info("--> checkpoint loaded from step %s", step)

    return step


class ModelWrapper(torch.distributed.checkpoint.stateful.Stateful):

    def __init__(self, model: torch.nn.Module) -> None:
        self.model = model

    def state_dict(self) -> Dict[str, Any]:
        return get_model_state_dict(self.model)

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        set_model_state_dict(
            self.model,
            model_state_dict=state_dict,
            options=StateDictOptions(strict=False),
        )


class OptimizerWrapper(torch.distributed.checkpoint.stateful.Stateful):

    def __init__(self, model: torch.nn.Module,
                 optimizer: torch.optim.Optimizer) -> None:
        self.model = model
        self.optimizer = optimizer

    def state_dict(self) -> Dict[str, Any]:
        return get_optimizer_state_dict(
            self.model,
            self.optimizer,
            options=StateDictOptions(flatten_optimizer_state_dict=True),
        )

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        set_optimizer_state_dict(
            self.model,
            self.optimizer,
            optim_state_dict=state_dict,
            options=StateDictOptions(flatten_optimizer_state_dict=True),
        )


class SchedulerWrapper(torch.distributed.checkpoint.stateful.Stateful):

    def __init__(self, scheduler) -> None:
        self.scheduler = scheduler

    def state_dict(self) -> Dict[str, Any]:
        return {"scheduler": self.scheduler.state_dict()}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self.scheduler.load_state_dict(state_dict["scheduler"])
