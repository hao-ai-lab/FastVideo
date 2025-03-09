import os
import torch
import torch.nn as nn
import argparse
import numpy as np

from fastvideo.logger import init_logger
from fastvideo.distributed.parallel_state import (
    init_distributed_environment,
    initialize_model_parallel,
    get_sequence_model_parallel_rank,
    get_sequence_model_parallel_world_size,
    destroy_model_parallel,
    destroy_distributed_environment,
    cleanup_dist_env_and_memory
)
import json
from fastvideo.models.dits.hunyuanvideo import HunyuanVideoDiT
from fastvideo.models.hunyuan.modules.models import HUNYUAN_VIDEO_CONFIG
from fastvideo.models.hunyuan.modules.models import HYVideoDiffusionTransformer
from fastvideo.loader.fsdp_load import load_fsdp_model
from fastvideo.models.hunyuan.inference import Inference
import glob
logger = init_logger(__name__)


def setup_args():
    parser = argparse.ArgumentParser(description='Distributed HunyuanVideo Test')
    parser.add_argument('--sequence_model_parallel_size', type=int, default=1,
                        help='Degree of sequence model parallelism')
    return parser.parse_args()

def test_hunyuanvideo_distributed():
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
    
    # load data/hunyuanvideo_community/transformer/config.json
    with open("/mbz/users/hao.zhang/peiyuan/FastVideo/data/hunyuanvideo_community/transformer/config.json", "r") as f:
        config = json.load(f)
    # remove   "_class_name": "HunyuanVideoTransformer3DModel",   "_diffusers_version": "0.32.0.dev0",
    # TODO: write normalize config function
    config.pop("_class_name")
    config.pop("_diffusers_version")
    # load data/hunyuanvideo_community/transformer/*.safetensors
    weight_dir_list = glob.glob("/mbz/users/hao.zhang/peiyuan/FastVideo/data/hunyuanvideo_community/transformer/*.safetensors")
    # to str
    weight_dir_list = [str(path) for path in weight_dir_list]
    model1 = load_fsdp_model(
        model_name="HunyuanVideoTransformer3DModel",
        init_params=config,
        weight_dir_list=weight_dir_list,
        device=torch.device(f"cuda:{local_rank}"),
        cpu_offload=False
    )
    
    # successfully sharded the model (hunyuanvideo bf16 should take around 26GB in total)
    total_params = sum(p.numel() for p in model1.parameters())
    logger.info(f"Total parameters: {total_params / 1e9}B")
    
    # Calculate weight sum for model1 (converting to float64 to avoid overflow)
    weight_sum_model1 = sum(p.to(torch.float64).sum().item() for p in model1.parameters())
    # Also calculate mean for more stable comparison
    weight_mean_model1 = weight_sum_model1 / total_params

    model2 = HYVideoDiffusionTransformer(
        in_channels=16,
        out_channels=16,
        **HUNYUAN_VIDEO_CONFIG["HYVideo-T/2-cfgdistill"],
        device=torch.device(f"cuda:{local_rank}"),
        dtype=torch.bfloat16
    )
    # data/hunyuan/hunyuan-video-t2v-720p/transformers/mp_rank_00_model_states.pt
    state_dict = torch.load("/mbz/users/hao.zhang/peiyuan/FastVideo/data/hunyuan/hunyuan-video-t2v-720p/transformers/mp_rank_00_model_states.pt", map_location=lambda storage, loc: storage)["module"]
    model2.load_state_dict(state_dict, strict=True)
    model2.to(torch.device(f"cuda:{local_rank}")).bfloat16()
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
    
    # Create random inputs for testing
    batch_size = 1
    seq_len = 3
    device = torch.device(f"cuda:{local_rank}")
    
    # Video latents [B, C, T, H, W]
    hidden_states = torch.randn(batch_size, 16, 8, 16, 16, device=device, dtype=torch.bfloat16)
    chunk_per_rank = hidden_states.shape[2] // sp_world_size
    hidden_states = hidden_states[:, :, sp_rank * chunk_per_rank:(sp_rank + 1) * chunk_per_rank]
   
    # Text embeddings [B, L, D] (including global token)
    encoder_hidden_states = torch.randn(batch_size, seq_len + 1, 4096, device=device, dtype=torch.bfloat16)
    
    # Timestep
    timestep = torch.tensor([500], device=device, dtype=torch.bfloat16)
    
    # Attention mask for text
    encoder_attention_mask = torch.ones(batch_size, seq_len, device=device, dtype=torch.bfloat16)
    guidance = torch.tensor([1.0], device=device, dtype=torch.bfloat16)
    
    # Disable gradients for inference
    with torch.no_grad():
        # Run inference on model1
        output1 = model1(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            timestep=timestep,
        )
        logger.info("Model 1 inference completed")
        
        # Run inference on model2
        output2, _ = model2(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            timestep=timestep,
            encoder_attention_mask=encoder_attention_mask,
        )
        logger.info("Model 2 inference completed")

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
    test_hunyuanvideo_distributed()