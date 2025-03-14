import os
import torch
import torch.nn as nn
import argparse
import numpy as np
import math

from fastvideo.v1.logger import init_logger
from fastvideo.v1.distributed.parallel_state import (
    init_distributed_environment,
    initialize_model_parallel,
    get_sequence_model_parallel_rank,
    get_sequence_model_parallel_world_size,
    destroy_model_parallel,
    destroy_distributed_environment,
    cleanup_dist_env_and_memory
)
import json
from fastvideo.v1.models.dits.wanvideo import WanTransformer3DModel
from fastvideo.models.wan.modules.model import WanModel
from fastvideo.v1.loader.fsdp_load import load_fsdp_model
import glob
logger = init_logger(__name__)

def setup_args():
    parser = argparse.ArgumentParser(description='Distributed WanVideo Test')
    parser.add_argument('--sequence_model_parallel_size', type=int, default=1,
                        help='Degree of sequence model parallelism')
    return parser.parse_args()

def test_wan_distributed():
    args = setup_args()
    
    # Initialize distributed environment
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    
    logger.info(f"Initializing process: rank={rank}, local_rank={local_rank}, world_size={world_size}")
    
    # Initialize distributed environment
    init_distributed_environment(
        world_size=world_size,
        rank=rank,
        local_rank=local_rank
    )
    torch.cuda.set_device(f"cuda:{local_rank}")
    # Initialize tensor model parallel groups
    initialize_model_parallel(
        sequence_model_parallel_size=args.sequence_model_parallel_size
    )

    # Get tensor parallel info
    sp_rank = get_sequence_model_parallel_rank()
    sp_world_size = get_sequence_model_parallel_world_size()
    
    logger.info(f"Process rank {rank} initialized with SP rank {sp_rank} in SP world size {sp_world_size}")
    
    with open("/workspace/data/Wan2.1-I2V-14B-720P-Diffusers/transformer/config.json", "r") as f:
        config = json.load(f)
    # remove   "_class_name": "HunyuanVideoTransformer3DModel",   "_diffusers_version": "0.32.0.dev0",
    # TODO: write normalize config function
    config.pop("_class_name")
    config.pop("_diffusers_version")
    # load data/hunyuanvideo_community/transformer/*.safetensors
    weight_dir_list = glob.glob("/workspace/data/Wan2.1-I2V-14B-720P-Diffusers/transformer/*.safetensors")
    # to str
    weight_dir_list = [str(path) for path in weight_dir_list]
    model1 = load_fsdp_model(
        model_cls=WanTransformer3DModel,
        init_params=config,
        weight_dir_list=weight_dir_list,
        device=torch.device(f"cuda:{local_rank}"),
        cpu_offload=False
    ).to(torch.bfloat16)
    
    # successfully sharded the model (hunyuanvideo bf16 should take around 26GB in total)
    total_params = sum(p.numel() for p in model1.parameters())
    logger.info(f"Total parameters: {total_params / 1e9}B")
    
    # Calculate weight sum for model1 (converting to float64 to avoid overflow)
    weight_sum_model1 = sum(p.to(torch.float64).sum().item() for p in model1.parameters())
    # Also calculate mean for more stable comparison
    weight_mean_model1 = weight_sum_model1 / total_params
    
    with open("/workspace/data/Wan2.1-I2V-14B-720P/config.json", "r") as f:
        wan_config = json.load(f)
    wan_config.pop("_class_name")
    wan_config.pop("_diffusers_version")
    model2 = WanModel(
        **wan_config,
        parallel=world_size > 1,
    ).to(torch.bfloat16)
    
    from safetensors.torch import load_file
    file_paths = glob.glob(f"/workspace/data/Wan2.1-I2V-14B-720P/*.safetensors")
    state_dict = {}
    for file_path in file_paths:
        state_dict.update(load_file(file_path))
    model2.load_state_dict(state_dict, strict=True)
    model2.bfloat16()
    print("load state dict done")

    # Calculate weight sum for model2 (converting to float64 to avoid overflow)
    total_params_model2 = sum(p.numel() for p in model2.parameters())
    weight_sum_model2 = sum(p.to(torch.float64).sum().item() for p in model2.parameters())
    # Also calculate mean for more stable comparison
    weight_mean_model2 = weight_sum_model2 / total_params_model2
    logger.info(f"Model 2 weight sum: {weight_sum_model2}")
    logger.info(f"Model 2 weight mean: {weight_mean_model2}")
    
    # Set both models to eval mode
    model1.eval()
    model2.eval()
    
    # Move to GPU based on local rank (0 or 1 for 2 GPUs)
    device = torch.device(f"cuda:{local_rank}")
    model1 = model1.to(device)
    # model2 = model2.to(device)
    
    # Create identical inputs for both models
    batch_size = 1
    text_seq_len = 30
    
    # Video latents [B, C, T, H, W]
    hidden_states = torch.randn(batch_size, 16, 21, 138, 104, device=device, dtype=torch.bfloat16)
    y = torch.randn(batch_size, 20, 21, 138, 104, device=device, dtype=torch.bfloat16)
    clip_fea = torch.randn(batch_size, 257, 1280, device=device, dtype=torch.bfloat16)
    seq_len = 75348
   
    # Text embeddings [B, L, D] (including global token)
    encoder_hidden_states = torch.randn(batch_size, text_seq_len + 1, 4096, device=device, dtype=torch.bfloat16)
    
    # Timestep
    timestep = torch.tensor([500], device=device, dtype=torch.bfloat16)
    
    # Disable gradients for inference
    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            output1 = model1(
                hidden_states=hidden_states,
                y=y,
                encoder_hidden_states=encoder_hidden_states,
                encoder_hidden_states_image=clip_fea,
                timestep=timestep
            )
        model1 = model1.cpu()
        model2 = model2.to(device)
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            output2 = model2(
                x=[hidden_states[0]],
                context=[encoder_hidden_states[0]],
                t=timestep,
                seq_len=seq_len,
                clip_fea=clip_fea,
                y=[y[0]],
            )[0]

    # Check if outputs have the same shape
    assert output1.shape == output2.shape, f"Output shapes don't match: {output1.shape} vs {output2.shape}"

    # Compare weight sums and means
    logger.info(f"Model 1 weight sum: {weight_sum_model1}")
    logger.info(f"Model 2 weight sum: {weight_sum_model2}")
    weight_sum_diff = abs(weight_sum_model1 - weight_sum_model2)
    logger.info(f"Weight sum difference: {weight_sum_diff}")
    
    logger.info(f"Model 1 weight mean: {weight_mean_model1}")
    logger.info(f"Model 2 weight mean: {weight_mean_model2}")
    weight_mean_diff = abs(weight_mean_model1 - weight_mean_model2)
    logger.info(f"Weight mean difference: {weight_mean_diff}")
    
    # Check if outputs are similar (allowing for small numerical differences)
    max_diff = torch.max(torch.abs(output1 - output2))
    assert max_diff < 1e-2, f"Maximum difference between outputs: {max_diff.item()}"
    
    # mean diff
    mean_diff = torch.mean(torch.abs(output1 - output2))
    assert mean_diff < 1e-4, f"Mean difference between outputs: {mean_diff.item()}"
    
    # diff sum 
    diff_sum = torch.sum(torch.abs(output1 - output2))
    logger.info(f"Diff sum between outputs: {diff_sum.item()}")
    
    # sum
    sum_output1 = torch.sum(output1.float())
    sum_output2 = torch.sum(output2.float())
    logger.info(f"Rank {sp_rank} Sum of output1: {sum_output1.item()}")
    logger.info(f"Rank {sp_rank} Sum of output2: {sum_output2.item()}")
    
    # Clean up
    logger.info("Cleaning up distributed environment")
    destroy_model_parallel()
    destroy_distributed_environment()
    cleanup_dist_env_and_memory()
    
    logger.info("Test completed successfully")

if __name__ == "__main__":
    test_wan_distributed()