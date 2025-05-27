import json
import os

import torch
from torch.distributed.fsdp import FullStateDictConfig
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import StateDictType

from fastvideo.v1.logger import init_logger

logger = init_logger(__name__)


def save_checkpoint(transformer, rank, output_dir, step):
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
    logger.info("--> checkpoint saved at step {step} to {weight_path}",
                step=step,
                weight_path=weight_path)
