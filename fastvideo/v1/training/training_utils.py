import json
import math
import os
from typing import Optional

import torch
from torch.distributed.fsdp import FullStateDictConfig
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import StateDictType

from fastvideo.v1.logger import init_logger

logger = init_logger(__name__)


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
