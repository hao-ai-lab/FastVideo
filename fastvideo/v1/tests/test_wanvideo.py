from itertools import chain
import os
import torch
import torch.nn as nn
import argparse
import numpy as np
import math

from fastvideo.v1.logger import init_logger
from fastvideo.v1.loader.fsdp_load import shard_model
from fastvideo.models.wan.distributed.fsdp import shard_model as shard_model_wan
from torch.distributed.device_mesh import init_device_mesh
from fastvideo.v1.utils.parallel_states import initialize_sequence_parallel_state
from fastvideo.v1.distributed.parallel_state import (
    init_distributed_environment,
    initialize_model_parallel,
    get_sequence_model_parallel_rank,
    get_sequence_model_parallel_world_size,
    destroy_model_parallel,
    destroy_distributed_environment,
    cleanup_dist_env_and_memory
)
from fastvideo.v1.models.dits.wanvideo import WanTransformer3DModel as WanVideoDiT
from fastvideo.models.wan.modules.model import WanModel
logger = init_logger(__name__)

def initialize_identical_weights(model1, model2, seed=42):
    """Initialize both models with identical weights using a fixed seed for reproducibility."""
    # Get all parameters from both models
    params1 = dict(model1.named_parameters())
    params2 = dict(model2.named_parameters())
    
    # Initialize each layer with identical values
    with torch.no_grad():
        # Initialize weights
        for name1, param1 in params1.items():
            if 'weight' in name1:
                # Set seed before each weight initialization
                torch.manual_seed(seed)
                nn.init.normal_(param1, mean=0.0, std=0.1)
                param1.data = param1.data.to(torch.bfloat16)
                
        for name2, param2 in params2.items():
            if 'weight' in name2:
                # Reset seed to get same initialization
                torch.manual_seed(seed)
                nn.init.normal_(param2, mean=0.0, std=0.1)
                param2.data = param2.data.to(torch.bfloat16)
                
        # Initialize biases
        for name1, param1 in params1.items():
            if 'bias' in name1:
                torch.manual_seed(seed)
                nn.init.normal_(param1, mean=0.0, std=0.1)
                param1.data = param1.data.to(torch.bfloat16)
                
        for name2, param2 in params2.items():
            if 'bias' in name2:
                torch.manual_seed(seed)
                nn.init.normal_(param2, mean=0.0, std=0.1)
                param2.data = param2.data.to(torch.bfloat16)

        for name1, param1 in params1.items():
            if 'scale_shift_table' in name1:
                torch.manual_seed(seed)
                nn.init.normal_(param1, mean=0.0, std=0.1)
                param1.data = param1.data.to(torch.bfloat16)
        
        for name2, param2 in params2.items():
            if 'modulation' in name2:
                torch.manual_seed(seed)
                nn.init.normal_(param2, mean=0.0, std=0.1)
                param2.data = param2.data.to(torch.bfloat16)
    
    logger.info("Both models initialized with identical weights in bfloat16")
    return model1, model2

def setup_args():
    parser = argparse.ArgumentParser(description='Distributed WanVideo Test')
    parser.add_argument('--sequence_model_parallel_size', type=int, default=1,
                        help='Degree of sequence model parallelism')
    parser.add_argument('--hidden-size', type=int, default=1536,
                        help='Hidden size for the model')
    parser.add_argument('--heads-num', type=int, default=12,
                        help='Number of attention heads')
    parser.add_argument('--num-layers', type=int, default=1,
                        help='Number of transformer layers')
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
    
    # Initialize tensor model parallel groups
    initialize_model_parallel(
        sequence_model_parallel_size=args.sequence_model_parallel_size
    )
    initialize_sequence_parallel_state(world_size)
    # Get tensor parallel info
    sp_rank = get_sequence_model_parallel_rank()
    sp_world_size = get_sequence_model_parallel_world_size()
    
    logger.info(f"Process rank {rank} initialized with SP rank {sp_rank} in SP world size {sp_world_size}")
    
    # Set fixed random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Small model parameters for testing
    hidden_size = args.hidden_size
    heads_num = args.heads_num
    num_layers = args.num_layers
    patch_size = (1, 2, 2)
    torch.cuda.set_device(f"cuda:{local_rank}")
    # Initialize the two model implementations
    model1 = WanVideoDiT(
        patch_size = patch_size,
        num_attention_heads = heads_num,
        attention_head_dim = hidden_size // heads_num,
        in_channels = 16,
        out_channels = 16,
        text_dim = 4096,
        freq_dim = 256,
        ffn_dim = 8960,
        num_layers = num_layers,
        cross_attn_norm = True,
        qk_norm = "rms_norm_across_heads",
        eps= 1e-6
    ).to(torch.bfloat16)
    
    model2 = WanModel(
        model_type='t2v',
        patch_size=patch_size,
        in_dim=16,
        dim=hidden_size,
        ffn_dim=8960,
        freq_dim=256,
        text_dim=4096,
        out_dim=16,
        num_heads=heads_num,
        num_layers=num_layers,
        qk_norm=True,
        cross_attn_norm=True,
        eps=1e-6,
        parallel=world_size > 1,
    ).to(torch.bfloat16)
    # for name, param in model1.named_parameters():
    #     print(name)
    # return
            
    # print("--------------------------------")
    # for name, param in model3.named_parameters():
    #     print(name)
    # import pdb; pdb.set_trace()
    # # Initialize with identical weights
    model1, model2 = initialize_identical_weights(model1, model2, seed=42)
    if world_size > 1:
        device_mesh = init_device_mesh(
            "cuda",
            mesh_shape=(sp_world_size,),
            mesh_dim_names=("dp", ),
        )
        model2 = shard_model_wan(model2, device_id=local_rank)
        shard_model(model1, cpu_offload=False, reshard_after_forward=True)
    for n, p in chain(model1.named_parameters(), model1.named_buffers()):
        if p.is_meta:
            raise RuntimeError(f"Unexpected param or buffer {n} on meta device.")
    for p in model1.parameters():
            p.requires_grad = False
    # Set both models to eval mode
    model1.eval()
    model2.eval()
    
    # Move to GPU based on local rank (0 or 1 for 2 GPUs)
    device = torch.device(f"cuda:{local_rank}")
    model1 = model1.to(device)
    model2 = model2.to(device)
    
    # Create identical inputs for both models
    batch_size = 1
    text_len = 30
    
    # Video latents [B, C, T, H, W]
    # hidden_states = torch.randn(batch_size, 4, 2, 16, 16, device=device, dtype=torch.bfloat16)
    # hidden_states = torch.randn(batch_size, 4, 8, 16, 16, device=device, dtype=torch.bfloat16)
    hidden_states = torch.randn(batch_size, 16, 21, 160, 90, device=device, dtype=torch.bfloat16)
    # hidden_states = hidden_states[:, :, sp_rank:sp_rank + 1]
   
    # Text embeddings [B, L, D] (including global token)
    encoder_hidden_states = torch.randn(batch_size, text_len + 1, 4096, device=device, dtype=torch.bfloat16)
    
    # Timestep
    timestep = torch.tensor([500], device=device, dtype=torch.bfloat16)

    seq_len = math.ceil((160 * 90) /
                            (2 * 2) *
                            21 / sp_world_size) * sp_world_size
    
    # Disable gradients for inference
    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            output1 = model1(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                timestep=timestep,
                seq_len=seq_len,
            )
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            output2 = model2(
                x=[hidden_states[0]],
                context=[encoder_hidden_states[0]],
                t=timestep,
                seq_len=seq_len,
            )[0]

    # Check if outputs have the same shape
    assert output1.shape == output2.shape, f"Output shapes don't match: {output1.shape} vs {output2.shape}"

    # Check if outputs are similar (allowing for small numerical differences)
    max_diff = torch.max(torch.abs(output1 - output2))
    logger.info(f"Maximum difference between outputs: {max_diff.item()}")
    # mean diff
    mean_diff = torch.mean(torch.abs(output1 - output2))
    logger.info(f"Mean difference between outputs: {mean_diff.item()}")
    # diff sum 
    diff_sum = torch.sum(torch.abs(output1 - output2))
    logger.info(f"Diff sum between outputs: {diff_sum.item()}")
    # sum
    sum_output1 = torch.sum(output1.float())
    sum_output2 = torch.sum(output2.float())
    logger.info(f"Sum of output1: {sum_output1.item()}")
    logger.info(f"Sum of output2: {sum_output2.item()}")
    # The outputs should be very close if not identical
    assert max_diff < 1e-3, f"Outputs differ significantly: max diff = {max_diff.item()}"  # Increased tolerance for bf16
    
    logger.info("Test passed! Both model implementations produce the same outputs.")
    
    # Clean up
    logger.info("Cleaning up distributed environment")
    destroy_model_parallel()
    destroy_distributed_environment()
    cleanup_dist_env_and_memory()
    
    logger.info("Test completed successfully")

if __name__ == "__main__":
    test_wan_distributed()