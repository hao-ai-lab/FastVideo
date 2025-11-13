#!/usr/bin/env python3
"""
Step 1: Run original LongCat with LoRA and save intermediate activations.
Run this with: conda activate longcat_shao
"""

import torch
import numpy as np
import os
import sys
from pathlib import Path

sys.path.insert(0, "/mnt/fast-disks/hao_lab/shao/LongCat-Video")

# Create output directory
output_dir = Path("outputs/debug_lora_comparison")
output_dir.mkdir(parents=True, exist_ok=True)

def setup_environment():
    """Setup distributed environment for single GPU."""
    os.environ['CUDA_VISIBLE_DEVICES'] = '7'
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = str(29500 + np.random.randint(0, 1000))
    os.environ['RANK'] = '0'
    os.environ['WORLD_SIZE'] = '1'
    os.environ['LOCAL_RANK'] = '0'

def run_original_with_lora():
    """Run original LongCat with LoRA and save activations."""
    setup_environment()
    
    print("="*100)
    print("STEP 1: ORIGINAL LONGCAT WITH LORA")
    print("="*100)
    
    # ========================================================================
    # 1. LOAD ORIGINAL MODEL WITH LORA
    # ========================================================================
    print("\n[1] LOADING ORIGINAL MODEL WITH LORA")
    print("="*100)
    
    from transformers import AutoTokenizer, UMT5EncoderModel
    from longcat_video.pipeline_longcat_video import LongCatVideoPipeline
    from longcat_video.modules.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler
    from longcat_video.modules.autoencoder_kl_wan import AutoencoderKLWan
    from longcat_video.modules.longcat_video_dit import LongCatVideoTransformer3DModel
    
    checkpoint_dir = "/mnt/fast-disks/hao_lab/shao/LongCat-Video/weights/LongCat-Video"
    lora_path_orig = "/mnt/fast-disks/hao_lab/shao/LongCat-Video/weights/LongCat-Video/lora/cfg_step_lora.safetensors"
    
    print(f"  Loading from: {checkpoint_dir}")
    print(f"  LoRA from: {lora_path_orig}")
    
    # Load components
    print("  Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir, subfolder="tokenizer", torch_dtype=torch.bfloat16)
    print("  Loading text encoder...")
    text_encoder = UMT5EncoderModel.from_pretrained(checkpoint_dir, subfolder="text_encoder", torch_dtype=torch.bfloat16)
    print("  Loading VAE...")
    vae = AutoencoderKLWan.from_pretrained(checkpoint_dir, subfolder="vae", torch_dtype=torch.bfloat16)
    print("  Loading scheduler...")
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(checkpoint_dir, subfolder="scheduler", torch_dtype=torch.bfloat16)
    print("  Loading DiT...")
    dit = LongCatVideoTransformer3DModel.from_pretrained(
        checkpoint_dir, 
        subfolder="dit", 
        cp_split_hw=[1, 1],  # Disable context parallelism
        torch_dtype=torch.bfloat16
    )
    
    # Load and enable LoRA
    print("  Loading LoRA weights...")
    dit.load_lora(lora_path_orig, 'cfg_step_lora')
    dit.enable_loras(['cfg_step_lora'])
    print("  ✓ LoRA enabled")
    
    # Create pipeline
    orig_pipe = LongCatVideoPipeline(
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        vae=vae,
        scheduler=scheduler,
        dit=dit,
    )
    orig_pipe.to("cuda")
    
    orig_model = orig_pipe.dit
    orig_model.eval()
    print("  ✓ Original pipeline with LoRA loaded")
    print(f"  Model type: {type(orig_model).__name__}")
    
    # ========================================================================
    # 2. CREATE TEST INPUTS AND SAVE THEM
    # ========================================================================
    print("\n[2] CREATING TEST INPUTS")
    print("="*100)
    
    torch.manual_seed(42)
    
    batch_size = 1
    num_frames = 9
    height = 30
    width = 52
    latent_channels = 16
    
    # Input latents
    latents = torch.randn(
        batch_size, latent_channels, num_frames, height, width,
        dtype=torch.bfloat16, device="cuda"
    ) * 0.18215
    
    # Timestep - use a timestep relevant for distilled model
    timestep = torch.tensor([250.0], device="cuda")
    
    # Text embeddings - original expects [B, 1, N, C]
    text_emb = torch.randn(batch_size, 1, 256, 4096, dtype=torch.bfloat16, device="cuda")
    
    print(f"  Latents: {latents.shape}")
    print(f"  Timestep: {timestep.item()}")
    print(f"  Text: {text_emb.shape}")
    
    # Save inputs for native run
    inputs_path = output_dir / "test_inputs.pt"
    torch.save({
        'latents': latents.cpu(),
        'timestep': timestep.cpu(),
        'text_emb_orig_format': text_emb.cpu(),  # [B, 1, N, C] format
    }, inputs_path)
    print(f"  ✓ Saved test inputs to: {inputs_path}")
    
    # ========================================================================
    # 3. HOOK TO CAPTURE INTERMEDIATE OUTPUTS
    # ========================================================================
    print("\n[3] SETTING UP HOOKS TO CAPTURE INTERMEDIATE OUTPUTS")
    print("="*100)
    
    orig_activations = {}
    
    def make_hook(name, storage):
        def hook(module, input, output):
            if isinstance(output, tuple):
                output = output[0]
            storage[name] = output.detach().cpu()  # Move to CPU to save memory
        return hook
    
    # Register hooks on original model
    orig_hooks = []
    
    # Patch embedding
    orig_hooks.append(orig_model.x_embedder.register_forward_hook(
        make_hook("patch_embed", orig_activations)))
    
    # Timestep embedding
    orig_hooks.append(orig_model.t_embedder.register_forward_hook(
        make_hook("time_embed", orig_activations)))
    
    # Caption embedding
    orig_hooks.append(orig_model.y_embedder.register_forward_hook(
        make_hook("caption_embed", orig_activations)))
    
    # Transformer blocks (sample every 6th block + first and last)
    sample_blocks = [0, 6, 12, 18, 24, 30, 36, 42, 47]
    for i in sample_blocks:
        orig_hooks.append(orig_model.blocks[i].register_forward_hook(
            make_hook(f"block_{i}", orig_activations)))
        
        # Self-attention within block
        orig_hooks.append(orig_model.blocks[i].attn.register_forward_hook(
            make_hook(f"block_{i}_self_attn", orig_activations)))
        
        # Cross-attention within block
        orig_hooks.append(orig_model.blocks[i].cross_attn.register_forward_hook(
            make_hook(f"block_{i}_cross_attn", orig_activations)))
        
        # FFN within block
        orig_hooks.append(orig_model.blocks[i].ffn.register_forward_hook(
            make_hook(f"block_{i}_ffn", orig_activations)))
    
    # Final layer
    orig_hooks.append(orig_model.final_layer.register_forward_hook(
        make_hook("final_layer", orig_activations)))
    
    print(f"  ✓ Registered {len(orig_hooks)} hooks on original model")
    
    # ========================================================================
    # 4. RUN FORWARD PASS
    # ========================================================================
    print("\n[4] RUNNING FORWARD PASS")
    print("="*100)
    
    with torch.no_grad():
        print("  Running original model forward pass...")
        orig_output = orig_model(
            hidden_states=latents,
            timestep=timestep,
            encoder_hidden_states=text_emb,
        )
        print(f"  ✓ Original output: {orig_output.shape}")
        
        # Save output
        orig_activations["final_output"] = orig_output.cpu()
    
    # Clean up hooks
    for hook in orig_hooks:
        hook.remove()
    
    # ========================================================================
    # 5. SAVE ACTIVATIONS
    # ========================================================================
    print("\n[5] SAVING ACTIVATIONS")
    print("="*100)
    
    activations_path = output_dir / "original_activations.pt"
    
    # Convert any tuples to first element
    cleaned_activations = {}
    for name, value in orig_activations.items():
        if isinstance(value, tuple):
            cleaned_activations[name] = value[0]
        else:
            cleaned_activations[name] = value
    
    torch.save(cleaned_activations, activations_path)
    print(f"  ✓ Saved {len(cleaned_activations)} activations to: {activations_path}")
    
    # Print summary
    print("\n[6] SUMMARY")
    print("="*100)
    print(f"  Captured {len(cleaned_activations)} layer outputs")
    print(f"  Sample blocks: {sample_blocks}")
    print(f"  Files saved:")
    print(f"    - {inputs_path}")
    print(f"    - {activations_path}")
    
    print("\n" + "="*100)
    print("✅ Step 1 complete! Run step 2 with fastvideo_shao environment.")
    print("="*100)

if __name__ == "__main__":
    run_original_with_lora()

