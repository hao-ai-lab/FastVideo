import torch
from fastvideo.model.pipeline_mochi import MochiPipeline
import torch.distributed as dist

from diffusers.utils import export_to_video
from fastvideo.utils.parallel_states import initialize_sequence_parallel_state, nccl_info
import argparse
import os
from fastvideo.model.modeling_mochi import MochiTransformer3DModel
import json
from typing import Optional
from safetensors.torch import save_file, load_file
from peft import set_peft_model_state_dict, inject_adapter_in_model, load_peft_weights
from peft import LoraConfig
import sys
import pdb
import copy
from typing import Dict

def initialize_distributed():
    local_rank = int(os.getenv('RANK', 0))
    world_size = int(os.getenv('WORLD_SIZE', 1))
    print('world_size', world_size)
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=local_rank)
    initialize_sequence_parallel_state(world_size)

def merge_lora_weights(
    base_model: torch.nn.Module,
    lora_weights: Dict[str, torch.Tensor],
    lora_config: LoraConfig,
    num_layers: Optional[int] = None
) -> torch.nn.Module:
    """
    Manually merge LoRA weights into transformer blocks.
    
    Args:
        base_model: The base model to merge weights into
        lora_weights: Dictionary containing LoRA weights
        lora_config: LoRA configuration
        num_layers: Number of transformer layers
        
    Returns:
        Modified model with merged weights
    """
    # Create a deep copy to avoid modifying the original model
    merged_model = copy.deepcopy(base_model)
    if num_layers is None:
        num_layers = len(merged_model.transformer_blocks)
    print(f"Processing {num_layers} transformer layers")
    # Calculate scaling factor
    scaling = lora_config.lora_alpha / lora_config.r
    
    def merge_component(
        base_weight: torch.Tensor,
        lora_a: torch.Tensor,
        lora_b: torch.Tensor
    ) -> torch.Tensor:
        """Merge LoRA weights for one component."""
        device = base_weight.device
        lora_a = lora_a.to(device)
        lora_b = lora_b.to(device)
        
        # Compute LoRA contribution
        lora_contribution = (lora_b @ lora_a) * scaling
        
        if lora_contribution.shape != base_weight.shape:
            raise ValueError(
                f"Shape mismatch: base={base_weight.shape}, "
                f"lora={lora_contribution.shape}"
            )
            
        return base_weight + lora_contribution

    # Iterate through all transformer layers
    for layer_idx in range(num_layers):
        # Get the transformer layer
        transformer_layer = merged_model.transformer_blocks[layer_idx].attn1
        
        # Merge weights for each target module
        for target_module in lora_config.target_modules:
            # Get base weights based on target module
            if target_module == "to_out.0":
                base_weight = transformer_layer.to_out[0].weight
                lora_a_key = f"transformer_blocks.{layer_idx}.attn1.to_out.0.lora_A.default.weight"
                lora_b_key = f"transformer_blocks.{layer_idx}.attn1.to_out.0.lora_B.default.weight"
            else:
                base_weight = getattr(transformer_layer, target_module).weight
                lora_a_key = f"transformer_blocks.{layer_idx}.attn1.{target_module}.lora_A.default.weight"
                lora_b_key = f"transformer_blocks.{layer_idx}.attn1.{target_module}.lora_B.default.weight"
            #try:
            lora_a = lora_weights[lora_a_key]
            lora_b = lora_weights[lora_b_key]
            #except KeyError as e:
            #    raise KeyError(f"Missing LoRA weight: {e}")
            
            # Merge weights
            merged_weight = merge_component(base_weight, lora_a, lora_b)
            
            # Update the model weights
            if target_module == "to_out.0":
                transformer_layer.to_out[0].weight.data.copy_(merged_weight)
            else:
                getattr(transformer_layer, target_module).weight.data.copy_(merged_weight)
            merged_model.transformer_blocks[layer_idx].attn1 = transformer_layer
            #print(f"Merged weights for layer {layer_idx}, module {target_module}")
    
    return merged_model

def load_lora_checkpoint(
    transformer: MochiTransformer3DModel,
    optimizer,
    output_dir: str,
    step: Optional[int] = None,
):
    """
    Load LoRA weights and optimizer states before FSDP training.
    If step is not specified, loads the latest checkpoint.
    
    Args:
        transformer: The transformer model (before FSDP wrapping)
        optimizer: The optimizer used for training
        output_dir: Directory containing checkpoint folders
        step: Optional specific step to load. If None, loads latest checkpoint
    Returns:
        transformer: The updated transformer model with loaded LoRA weights
        step: The step number of the loaded checkpoint
    """
    # Find the checkpoint to load
    if step is None:
        checkpoints = [d for d in os.listdir(output_dir) 
                      if d.startswith("lora-checkpoint-")]
        if not checkpoints:
            print("No checkpoints found in directory")
            return transformer, 0
        steps = [int(d.split("-")[-1]) for d in checkpoints]
        step = max(steps)
        print(f"Loading latest checkpoint from step {step}")
    else:
        print(f"Loading specified checkpoint from step {step}")
    
    checkpoint_dir = os.path.join(output_dir, f"lora-checkpoint-{step}")
    
    # Load and set the LoRA config
    config_path = os.path.join(checkpoint_dir, "lora_config.json")
    with open(config_path, 'r') as f:
        lora_config = json.load(f)
        
    # Set config attributes
    for key, value in lora_config['lora_params'].items():
        setattr(transformer.config, f"lora_{key}", value)
    
    # Load weights
    weight_path = os.path.join(checkpoint_dir, "lora_weights.safetensors")
    lora_state_dict = load_file(weight_path)

    lora_config = LoraConfig(
        r=128,
        lora_alpha=256,
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
    )

    transformer = merge_lora_weights(transformer, lora_state_dict, lora_config)

    print(f"--> Successfully loaded LoRA checkpoint from step {step}")
    return transformer

def main(args):
    initialize_distributed()
    print(nccl_info.sp_size)
    device = torch.cuda.current_device()
    generator = torch.Generator(device).manual_seed(args.seed)
    weight_dtype = torch.bfloat16
    
    if args.transformer_path is not None:
        transformer = MochiTransformer3DModel.from_pretrained(args.transformer_path, torch_dtype=torch.bfloat16)
        
    else:
        transformer = MochiTransformer3DModel.from_pretrained(args.model_path, subfolder = 'transformer/', torch_dtype=torch.bfloat16)
    if args.lora_path is not None:
        # Load and merge LoRA weights
        transformer = load_lora_checkpoint(
            transformer=transformer,
            optimizer=None,  # No optimizer needed for inference
            output_dir=args.lora_path,
            step=args.lora_step if hasattr(args, 'lora_step') else None
        )
        print(f"Loaded and merged LoRA weights from {args.lora_path}")
    pipe = MochiPipeline.from_pretrained(args.model_path, transformer = transformer, torch_dtype=torch.bfloat16)
    
    pipe.enable_vae_tiling()
    pipe.to(device)
    #pipe.enable_model_cpu_offload()
    # Generate videos from the input prompt

    if args.prompt_embed_path is not None:
        prompt_embeds = torch.load(args.prompt_embed_path, map_location="cpu", weights_only=True).to(device).unsqueeze(0)
        encoder_attention_mask = torch.load(args.encoder_attention_mask_path, map_location="cpu", weights_only=True).to(device).unsqueeze(0)
        prompts = None
    else:
        prompts = args.prompts
        prompt_embeds = None
        encoder_attention_mask = None
    videos = pipe(
        prompt=prompts,
        prompt_embeds=prompt_embeds,
        prompt_attention_mask=encoder_attention_mask,
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        generator=generator,
    ).frames

    if nccl_info.global_rank <= 0:
        if prompts is not None:
            for video, prompt in zip(videos, prompts):
                suffix = prompt.split(".")[0]
                export_to_video(video, args.output_path + f"_{suffix}.mp4", fps=30)
        else:
            export_to_video(videos[0], args.output_path + ".mp4", fps=30)

if __name__ == "__main__":
    # arg parse 
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompts", nargs='+', default=[])
    parser.add_argument("--num_frames", type=int, default=163)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=848)
    parser.add_argument("--num_inference_steps", type=int, default=64)
    parser.add_argument("--guidance_scale", type=float, default=4.5)
    parser.add_argument("--model_path", type=str, default="data/mochi")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_path", type=str, default="./outputs.mp4")
    parser.add_argument("--transformer_path", type=str, default=None)
    parser.add_argument("--prompt_embed_path", type=str, default=None)
    parser.add_argument("--encoder_attention_mask_path", type=str, default=None)
    parser.add_argument('--lora_path', type=str, default=None, help='Path to the directory containing LoRA checkpoints')
    parser.add_argument('--lora_step', type=int, default=None, help='Specific LoRA checkpoint step to load. If not provided, loads latest')
    args = parser.parse_args()
    main(args)
