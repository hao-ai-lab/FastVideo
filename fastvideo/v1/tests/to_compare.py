from itertools import chain
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

from fastvideo.utils.parallel_states import initialize_sequence_parallel_state
from torch.distributed._composable.fsdp import CPUOffloadPolicy, fully_shard
from torch.distributed.device_mesh import init_device_mesh
from fastvideo.loader.fsdp_load import shard_model
from fastvideo.models.dits.hunyuanvideo import HunyuanVideoDiT
from fastvideo.models.hunyuan.modules.models import HYVideoDiffusionTransformer
from fastvideo.models.hunyuan_hf.modeling_hunyuan import HunyuanVideoTransformer3DModel
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
                nn.init.normal_(param1, mean=0.0, std=0.05)
                
        for name2, param2 in params2.items():
            if 'weight' in name2:
                # Reset seed to get same initialization
                torch.manual_seed(seed)
                nn.init.normal_(param2, mean=0.0, std=0.05)
                
        # Initialize biases
        for name1, param1 in params1.items():
            if 'bias' in name1:
                torch.manual_seed(seed)
                nn.init.normal_(param1, mean=0.0, std=0.05)
                param1.data = param1.data.to(torch.bfloat16)
                
        for name2, param2 in params2.items():
            if 'bias' in name2:
                torch.manual_seed(seed)
                nn.init.normal_(param2, mean=0.0, std=0.05)
                param2.data = param2.data.to(torch.bfloat16)
    
    logger.info("Both models initialized with identical weights in bfloat16")
    return model1, model2

def setup_args():
    parser = argparse.ArgumentParser(description='Distributed HunyuanVideo Test')
    parser.add_argument('--sequence_model_parallel_size', type=int, default=1,
                        help='Degree of sequence model parallelism')
    parser.add_argument('--hidden-size', type=int, default=128,
                        help='Hidden size for the model')
    parser.add_argument('--heads-num', type=int, default=4,
                        help='Number of attention heads')
    parser.add_argument('--double-blocks-depth', type=int, default=2,
                        help='Number of double stream blocks')
    parser.add_argument('--single-blocks-depth', type=int, default=2,
                        help='Number of single stream blocks')
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
    mm_double_blocks_depth = args.double_blocks_depth
    mm_single_blocks_depth = args.single_blocks_depth
    patch_size = [1, 2, 2]
    torch.cuda.set_device(f"cuda:{local_rank}")
    # Initialize the two model implementations
    model1 = HunyuanVideoDiT(
        patch_size=2,
        patch_size_t=1,
        in_channels=4,
        out_channels=4,
        attention_head_dim=hidden_size//heads_num,
        num_attention_heads=heads_num,
        num_layers=mm_double_blocks_depth,
        num_single_layers=mm_single_blocks_depth,
        rope_axes_dim=[8, 16, 8],  # sum = hidden_size // heads_num = 32
        dtype=torch.bfloat16
    ).to(torch.bfloat16)
    model2 = HYVideoDiffusionTransformer(
        patch_size=patch_size,
        in_channels=4,
        hidden_size=hidden_size,
        heads_num=heads_num,
        mm_double_blocks_depth=mm_double_blocks_depth,
        mm_single_blocks_depth=mm_single_blocks_depth,
        rope_dim_list=[8, 16, 8],  # sum = hidden_size // heads_num = 32
        dtype=torch.bfloat16
    ).to(torch.bfloat16)

            
    # print("--------------------------------")
    # for name, param in model3.named_parameters():
    #     print(name)
    # import pdb; pdb.set_trace()
    # # Initialize with identical weights
    model1, model2 = initialize_identical_weights(model1, model2, seed=42)
    device_mesh = init_device_mesh(
        "cuda",
        mesh_shape=(sp_world_size,),
        mesh_dim_names=("dp", ),
    )
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
    seq_len = 3
    
    # Video latents [B, C, T, H, W]
    hidden_states = torch.randn(batch_size, 4, 8, 16, 16, device=device, dtype=torch.bfloat16)
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
        output1 = model1(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            timestep=timestep,
        )
        print("--------------------------------")
        output2, _ = model2(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            timestep=timestep,
            encoder_attention_mask=encoder_attention_mask,
        )

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
    logger.info(f"Rank {sp_rank} Sum of output1: {sum_output1.item()}")
    logger.info(f"Rank {sp_rank} Sum of output2: {sum_output2.item()}")
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
    test_hunyuanvideo_distributed()