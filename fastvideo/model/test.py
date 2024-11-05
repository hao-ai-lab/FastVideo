import torch
from fastvideo.model.pipeline_mochi import MochiPipeline
import torch.distributed as dist

from diffusers.utils import export_to_video
from fastvideo.utils.parallel_states import initialize_sequence_parallel_state, nccl_info
import argparse
import os
from diffusers.models.transformers.transformer_mochi import MochiTransformerBlock
from fastvideo.model.mochi_monkey_patches import hf_mochi_add_sp_monkey_patch
from diffusers import MochiTransformer3DModel


import sys
import pdb

class ForkedPdb(pdb.Pdb):
    """
    PDB Subclass for debugging multi-processed code
    Suggested in: https://stackoverflow.com/questions/4716533/how-to-attach-debugger-to-a-python-subproccess
    """
    def interaction(self, *args, **kwargs):
        _stdin = sys.stdin
        try:
            sys.stdin = open('/dev/stdin')
            pdb.Pdb.interaction(self, *args, **kwargs)
        finally:
            sys.stdin = _stdin
def assert_all_close_list(input_list):
    for i in range(len(input_list) - 1):
        assert torch.allclose(input_list[i], input_list[i + 1]), f"input_list[{i}]: {input_list[i]}, input_list[{i+1}]: {input_list[i+1]}"
            
weight_dtype = torch.float32
def initialize_distributed():
    local_rank = int(os.getenv('RANK', 0))
    world_size = int(os.getenv('WORLD_SIZE', 1))
    print('world_size', world_size)
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=local_rank)
    return world_size

def main_print(content):
    if int(os.getenv('RANK', 0)) <= 0:
        print(content)
        
@torch.inference_mode
def test_single_block(batch_size, device, seed):
    # set manual seed
    torch.manual_seed(seed)
    device = torch.cuda.current_device()
    block =  MochiTransformerBlock(
                    dim=768,
                    num_attention_heads=12,
                    attention_head_dim=64,
                    pooled_projection_dim=256,
                    qk_norm="rms_norm",
                    activation_fn="swiglu",
                    context_pre_only=False,
                ).to(device)
    hidden_states = torch.randn(1, 16, 768).to(device).repeat(batch_size, 1, 1)
    encoder_hidden_states = torch.randn(1, 4, 256).to(device).repeat(batch_size, 1, 1)
    temb = torch.randn(1, 768).to(device).repeat(batch_size, 1)
    # shard hiddent_states according to world_size
    local_seq_length = hidden_states.shape[1] // nccl_info.sp_size
    hidden_states = hidden_states.narrow(1, nccl_info.global_rank * local_seq_length, local_seq_length)
    main_print(hidden_states.shape)
    hidden_states, encoder_hidden_states = block(
        hidden_states=hidden_states,
        encoder_hidden_states=encoder_hidden_states,
        temb=temb,
    )
    mean = hidden_states[0].mean()
    torch.distributed.all_reduce(mean, op=torch.distributed.ReduceOp.SUM)
    mean = mean / nccl_info.sp_size
    return mean

@torch.inference_mode
def test_DiT(batch_size, transformer, seed):
    generator = torch.Generator(torch.cuda.current_device()).manual_seed(seed)
    device = torch.cuda.current_device()
    latent = torch.randn((1, 12, 8, 12, 8), device=device, dtype=weight_dtype, generator=generator).repeat(batch_size, 1, 1, 1, 1)
    prompt_embeds = torch.randn((1, 20, 4096), device=device, dtype=weight_dtype, generator=generator).repeat(batch_size, 1, 1)
    prompt_attention_mask = torch.ones((1, 20), device=device, dtype=weight_dtype).repeat(batch_size, 1)
    timestep = 0
    timestep = torch.tensor(timestep, device=device, dtype=weight_dtype).unsqueeze(0).repeat(batch_size)
    local_seq_length = latent.shape[2] // nccl_info.sp_size
    latent = latent.narrow(2, nccl_info.global_rank * local_seq_length, local_seq_length)  
    # main_print(latent.shape)
    hidden_states = transformer(
        hidden_states=latent,
        encoder_hidden_states=prompt_embeds,
        encoder_attention_mask=prompt_attention_mask,
        timestep=timestep,
        return_dict=False,
    )[0]
    def calculate_mean(states):
        mean = states.mean()
        torch.distributed.all_reduce(mean, op=torch.distributed.ReduceOp.SUM)
        mean = mean / int(os.getenv('WORLD_SIZE', 1))
        return mean
    mean1 =  calculate_mean(hidden_states[0])
    main_print(hidden_states.shape)
    if hidden_states.shape[0] > 1:
        mean2 = calculate_mean(hidden_states[1])
        return mean1, mean2
    return mean1
    
    
if __name__ == "__main__":
    world_size = initialize_distributed()
    device = torch.cuda.current_device()
    
        
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test_single_block", action="store_true")
    args = parser.parse_args()
    seed = args.seed
    
    if args.test_single_block:
        pass
        single_no_patch_bs_1 = test_single_block(1)
        single_no_patch_bs_2 = test_single_block(2)
        # check all close 
        assert torch.allclose(single_no_patch_bs_1, single_no_patch_bs_2)
        hf_mochi_add_sp_monkey_patch()
        single_patch_bs_1 = test_single_block(1)
        single_patch_bs_2 = test_single_block(2)
        assert torch.allclose(single_patch_bs_1, single_patch_bs_2)
        assert torch.allclose(single_no_patch_bs_1, single_patch_bs_2)
        initialize_sequence_parallel_state(world_size)
        
        sp_patch_bs_1 = test_single_block(1)
        sp_patch_bs_2 = test_single_block(2)
        
        assert torch.allclose(sp_patch_bs_1, sp_patch_bs_2)
        assert torch.allclose(single_no_patch_bs_1, sp_patch_bs_2)
      
    else:
        transformer = MochiTransformer3DModel.from_pretrained("data/mochi/transformer", torch_dtype=weight_dtype).to(device)
        single_no_patch_bs_1 = test_DiT(1, transformer, seed)
        single_no_patch_bs_2_a, single_no_patch_bs_2_b = test_DiT(2, transformer, seed)
        
        hf_mochi_add_sp_monkey_patch()
        
        single_patch_bs_1 = test_DiT(1, transformer, seed)
        single_patch_bs_2_a, single_patch_bs_2_b = test_DiT(2, transformer, seed)
        
        initialize_sequence_parallel_state(world_size)
        
        sp_patch_bs_1 = test_DiT(1, transformer, seed)
        sp_patch_bs_2_a, sp_patch_bs_2_b = test_DiT(2, transformer, seed)
        
        assert_all_close_list([single_no_patch_bs_1, single_no_patch_bs_2_a, single_no_patch_bs_2_b, single_patch_bs_1, single_patch_bs_2_a, single_patch_bs_2_b, sp_patch_bs_1, sp_patch_bs_2_a, sp_patch_bs_2_b])
        
        main_print(sp_patch_bs_1)
        
