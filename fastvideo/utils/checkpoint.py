# import 
import os
import json
import torch
from fastvideo.utils.logging import main_print
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, StateDictType, FullStateDictConfig
from safetensors.torch import save_file, load_file

def save_checkpoint(model, optimizer, rank, output_dir, step, discriminator=False):

    with FSDP.state_dict_type(
        model, StateDictType.FULL_STATE_DICT, FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    ):
        cpu_state = model.state_dict()
        optim_state = FSDP.optim_state_dict(
            model, 
            optimizer,
        )
    
    #todo move to get_state_dict
    save_dir = os.path.join(output_dir, f"checkpoint-{step}")
    os.makedirs(save_dir, exist_ok=True)
    # save using safetensors 
    if rank <= 0 and not discriminator:
        weight_path = os.path.join(save_dir, "diffusion_pytorch_model.safetensors")
        save_file(cpu_state, weight_path)
        config_dict = dict(model.config)
        config_path = os.path.join(save_dir, "config.json")
        # save dict as json
        with open(config_path, "w") as f:
            json.dump(config_dict, f, indent=4)
        optimizer_path = os.path.join(save_dir, "optimizer.pt")
        torch.save(optim_state, optimizer_path)
    else:
        weight_path = os.path.join(save_dir, "discriminator_pytorch_model.safetensors")
        save_file(cpu_state, weight_path)
        optimizer_path = os.path.join(save_dir, "discriminator_optimizer.pt")
        torch.save(optim_state, optimizer_path)
        
      



def save_lora_checkpoint(
    transformer, 
    optimizer,
    rank, 
    output_dir, 
    step
):
    main_print(f"--> saving LoRA checkpoint at step {step}")
    with FSDP.state_dict_type(
        transformer, 
        StateDictType.FULL_STATE_DICT, 
        FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    ):
        full_state_dict = transformer.state_dict()
        lora_state_dict = {
            k: v for k, v in full_state_dict.items() 
            if 'lora' in k.lower()  
        }
        lora_optim_state = FSDP.optim_state_dict(
            transformer, 
            optimizer,
        )
    if rank <= 0:
        save_dir = os.path.join(output_dir, f"lora-checkpoint-{step}")
        os.makedirs(save_dir, exist_ok=True)
        weight_path = os.path.join(save_dir, "lora_weights.safetensors")
        save_file(lora_state_dict, weight_path)
        optim_path = os.path.join(save_dir, "lora_optimizer.pt")
        torch.save(lora_optim_state, optim_path)
        lora_config = {
            'step': step,
            'lora_params': {
                'lora_rank': transformer.config.lora_rank, 
                'lora_alpha': transformer.config.lora_alpha,
                'target_modules': transformer.config.lora_target_modules
            }
        }
        config_path = os.path.join(save_dir, "lora_config.json")
        with open(config_path, "w") as f:
            json.dump(lora_config, f, indent=4)
    main_print(f"--> LoRA checkpoint saved at step {step}")

def resume_lora_training(
    transformer,
    checkpoint_dir,
    optimizer
):
    weight_path = os.path.join(checkpoint_dir, "lora_weights.safetensors")
    lora_weights = load_file(weight_path)
    config_path = os.path.join(checkpoint_dir, "lora_config.json")
    with open(config_path, "r") as f:
        config_dict = json.load(f)
    with FSDP.state_dict_type(
        transformer,
        StateDictType.FULL_STATE_DICT,
        FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    ):
        current_state = transformer.state_dict()
        current_state.update(lora_weights)
        transformer.load_state_dict(current_state, strict=False)
    optim_path = os.path.join(checkpoint_dir, "lora_optimizer.pt")
    optimizer_state_dict = torch.load(optim_path, weights_only=False)
    optim_state = FSDP.optim_state_dict_to_load(
            model=transformer,
            optim=optimizer,
            optim_state_dict=optimizer_state_dict
        )
    optimizer.load_state_dict(optim_state)
    step = config_dict['step']
    main_print(f"-->  Successfully resuming LoRA training from step {step}")
    return transformer, optimizer, step