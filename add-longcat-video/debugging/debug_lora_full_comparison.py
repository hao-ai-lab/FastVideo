#!/usr/bin/env python3
"""
Complete LoRA debugging: Compare original and native models both with and without LoRA.

This script needs to be run in TWO stages:
Stage 1: longcat_shao environment - captures original model activations
Stage 2: fastvideo_shao environment - captures native model activations and analyzes

Usage:
    # Stage 1 (longcat_shao env)
    conda activate longcat_shao
    python debug_lora_full_comparison.py --stage 1
    
    # Stage 2 (fastvideo_shao env)  
    conda activate fastvideo_shao
    python debug_lora_full_comparison.py --stage 2
"""

import argparse
import torch
import numpy as np
import os
import sys
from pathlib import Path
from typing import Dict

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

def save_tensor_dict(tensors: Dict[str, torch.Tensor], filename: str):
    """Save tensors to file, moving to CPU if needed."""
    cpu_tensors = {}
    for name, tensor in tensors.items():
        if isinstance(tensor, tuple):
            cpu_tensors[name] = tensor[0].cpu() if tensor[0].is_cuda else tensor[0]
        else:
            cpu_tensors[name] = tensor.cpu() if tensor.is_cuda else tensor
    torch.save(cpu_tensors, output_dir / filename)
    print(f"  ✓ Saved {len(cpu_tensors)} tensors to: {filename}")

def make_hook(name, storage):
    """Create a forward hook that captures outputs."""
    def hook(module, input, output):
        if isinstance(output, tuple):
            output = output[0]
        storage[name] = output.detach()
    return hook

def register_hooks(model, storage, sample_blocks, model_type="original"):
    """Register forward hooks on model layers."""
    hooks = []
    
    if model_type == "original":
        # Original model naming
        hooks.append(model.x_embedder.register_forward_hook(make_hook("patch_embed", storage)))
        hooks.append(model.t_embedder.register_forward_hook(make_hook("time_embed", storage)))
        hooks.append(model.y_embedder.register_forward_hook(make_hook("caption_embed", storage)))
        
        for i in sample_blocks:
            hooks.append(model.blocks[i].register_forward_hook(make_hook(f"block_{i}", storage)))
            hooks.append(model.blocks[i].attn.register_forward_hook(make_hook(f"block_{i}_self_attn", storage)))
            hooks.append(model.blocks[i].cross_attn.register_forward_hook(make_hook(f"block_{i}_cross_attn", storage)))
            hooks.append(model.blocks[i].ffn.register_forward_hook(make_hook(f"block_{i}_ffn", storage)))
        
        hooks.append(model.final_layer.register_forward_hook(make_hook("final_layer", storage)))
    else:
        # Native model naming
        hooks.append(model.patch_embed.register_forward_hook(make_hook("patch_embed", storage)))
        hooks.append(model.time_embedder.register_forward_hook(make_hook("time_embed", storage)))
        hooks.append(model.caption_embedder.register_forward_hook(make_hook("caption_embed", storage)))
        
        for i in sample_blocks:
            hooks.append(model.blocks[i].register_forward_hook(make_hook(f"block_{i}", storage)))
            hooks.append(model.blocks[i].self_attn.register_forward_hook(make_hook(f"block_{i}_self_attn", storage)))
            hooks.append(model.blocks[i].cross_attn.register_forward_hook(make_hook(f"block_{i}_cross_attn", storage)))
            hooks.append(model.blocks[i].ffn.register_forward_hook(make_hook(f"block_{i}_ffn", storage)))
        
        hooks.append(model.final_layer.register_forward_hook(make_hook("final_layer", storage)))
    
    return hooks

# ============================================================================
# STAGE 1: ORIGINAL MODEL (longcat_shao environment)
# ============================================================================

def stage1_original():
    """Run original LongCat model with and without LoRA."""
    sys.path.insert(0, "/mnt/fast-disks/hao_lab/shao/LongCat-Video")
    setup_environment()
    
    print("="*100)
    print("STAGE 1: ORIGINAL LONGCAT MODEL")
    print("="*100)
    
    from transformers import AutoTokenizer, UMT5EncoderModel
    from longcat_video.pipeline_longcat_video import LongCatVideoPipeline
    from longcat_video.modules.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler
    from longcat_video.modules.autoencoder_kl_wan import AutoencoderKLWan
    from longcat_video.modules.longcat_video_dit import LongCatVideoTransformer3DModel
    
    checkpoint_dir = "/mnt/fast-disks/hao_lab/shao/LongCat-Video/weights/LongCat-Video"
    lora_path = "/mnt/fast-disks/hao_lab/shao/LongCat-Video/weights/LongCat-Video/lora/cfg_step_lora.safetensors"
    
    # Create test inputs ONCE
    print("\n[1] CREATING TEST INPUTS")
    print("="*100)
    
    torch.manual_seed(42)
    batch_size = 1
    num_frames = 9
    height = 30
    width = 52
    latent_channels = 16
    
    latents = torch.randn(batch_size, latent_channels, num_frames, height, width,
                         dtype=torch.bfloat16, device="cuda") * 0.18215
    timestep = torch.tensor([250.0], device="cuda")
    text_emb = torch.randn(batch_size, 1, 256, 4096, dtype=torch.bfloat16, device="cuda")
    
    print(f"  Latents: {latents.shape}")
    print(f"  Timestep: {timestep.item()}")
    print(f"  Text: {text_emb.shape}")
    
    # Save inputs for stage 2
    torch.save({
        'latents': latents.cpu(),
        'timestep': timestep.cpu(),
        'text_emb_orig_format': text_emb.cpu(),
    }, output_dir / "test_inputs.pt")
    print(f"  ✓ Saved test inputs")
    
    sample_blocks = [0, 6, 12, 18, 24, 30, 36, 42, 47]
    
    # ========================================================================
    # RUN 1: Original WITHOUT LoRA
    # ========================================================================
    print("\n[2] ORIGINAL MODEL WITHOUT LORA")
    print("="*100)
    
    print("  Loading model components...")
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir, subfolder="tokenizer", torch_dtype=torch.bfloat16)
    text_encoder = UMT5EncoderModel.from_pretrained(checkpoint_dir, subfolder="text_encoder", torch_dtype=torch.bfloat16)
    vae = AutoencoderKLWan.from_pretrained(checkpoint_dir, subfolder="vae", torch_dtype=torch.bfloat16)
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(checkpoint_dir, subfolder="scheduler", torch_dtype=torch.bfloat16)
    dit_no_lora = LongCatVideoTransformer3DModel.from_pretrained(
        checkpoint_dir, subfolder="dit", cp_split_hw=[1, 1], torch_dtype=torch.bfloat16
    )
    
    pipe_no_lora = LongCatVideoPipeline(
        tokenizer=tokenizer, text_encoder=text_encoder, vae=vae,
        scheduler=scheduler, dit=dit_no_lora
    )
    pipe_no_lora.to("cuda")
    dit_no_lora.eval()
    
    print("  ✓ Model loaded WITHOUT LoRA")
    
    # Register hooks and run
    activations_orig_no_lora = {}
    hooks = register_hooks(dit_no_lora, activations_orig_no_lora, sample_blocks, "original")
    
    print("  Running forward pass...")
    with torch.no_grad():
        output = dit_no_lora(
            hidden_states=latents,
            timestep=timestep,
            encoder_hidden_states=text_emb,
        )
        activations_orig_no_lora["final_output"] = output
    
    for hook in hooks:
        hook.remove()
    
    print(f"  ✓ Captured {len(activations_orig_no_lora)} activations")
    save_tensor_dict(activations_orig_no_lora, "original_no_lora.pt")
    
    # Clean up
    del pipe_no_lora, dit_no_lora
    torch.cuda.empty_cache()
    
    # ========================================================================
    # RUN 2: Original WITH LoRA
    # ========================================================================
    print("\n[3] ORIGINAL MODEL WITH LORA")
    print("="*100)
    
    print("  Loading model with LoRA...")
    dit_with_lora = LongCatVideoTransformer3DModel.from_pretrained(
        checkpoint_dir, subfolder="dit", cp_split_hw=[1, 1], torch_dtype=torch.bfloat16
    )
    
    # Load and enable LoRA
    dit_with_lora.load_lora(lora_path, 'cfg_step_lora')
    dit_with_lora.enable_loras(['cfg_step_lora'])
    
    pipe_with_lora = LongCatVideoPipeline(
        tokenizer=tokenizer, text_encoder=text_encoder, vae=vae,
        scheduler=scheduler, dit=dit_with_lora
    )
    pipe_with_lora.to("cuda")
    dit_with_lora.eval()
    
    print("  ✓ Model loaded WITH LoRA")
    
    # Register hooks and run
    activations_orig_with_lora = {}
    hooks = register_hooks(dit_with_lora, activations_orig_with_lora, sample_blocks, "original")
    
    print("  Running forward pass...")
    with torch.no_grad():
        output = dit_with_lora(
            hidden_states=latents,
            timestep=timestep,
            encoder_hidden_states=text_emb,
        )
        activations_orig_with_lora["final_output"] = output
    
    for hook in hooks:
        hook.remove()
    
    print(f"  ✓ Captured {len(activations_orig_with_lora)} activations")
    save_tensor_dict(activations_orig_with_lora, "original_with_lora.pt")
    
    print("\n" + "="*100)
    print("✅ STAGE 1 COMPLETE! Now run stage 2 with fastvideo_shao environment.")
    print("="*100)

# ============================================================================
# STAGE 2: NATIVE MODEL (fastvideo_shao environment)
# ============================================================================

def run_generator_with_hooks(generator, latents, timestep, text_emb_native, sample_blocks, activations_dict):
    """
    Run a forward pass through the generator's model with hooks to capture activations.
    This accesses the model through the executor's worker, just like actual inference.
    """
    from fastvideo.forward_context import set_forward_context
    from fastvideo.pipelines.pipeline_batch_info import ForwardBatch
    
    # Access the model through the executor's worker
    # For single GPU, there's one worker and we can access it
    worker = generator.executor.workers[0]
    model = worker.pipeline.modules["transformer"]
    model.eval()
    
    print(f"  Model type: {type(model).__name__}")
    print(f"  Model device: {next(model.parameters()).device}")
    
    # Register hooks
    hooks = register_hooks(model, activations_dict, sample_blocks, "native")
    print(f"  Registered {len(hooks)} hooks")
    
    # Run forward pass
    print("  Running forward pass...")
    dummy_batch = ForwardBatch(data_type="t2v")
    with torch.no_grad():
        with set_forward_context(current_timestep=0, attn_metadata=None, forward_batch=dummy_batch):
            output = model(
                hidden_states=latents,
                encoder_hidden_states=text_emb_native,
                timestep=timestep,
            )
            activations_dict["final_output"] = output
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
    
    return output

def stage2_native():
    """Run native FastVideo model with and without LoRA using VideoGenerator."""
    setup_environment()
    
    print("="*100)
    print("STAGE 2: NATIVE FASTVIDEO MODEL (using VideoGenerator)")
    print("="*100)
    
    from fastvideo import VideoGenerator
    
    # Load test inputs
    print("\n[1] LOADING TEST INPUTS")
    print("="*100)
    
    inputs = torch.load(output_dir / "test_inputs.pt")
    latents = inputs['latents'].to("cuda").to(torch.bfloat16)
    timestep = inputs['timestep'].to("cuda")
    text_emb_orig = inputs['text_emb_orig_format'].to("cuda").to(torch.bfloat16)
    text_emb_native = text_emb_orig.squeeze(1)  # Native expects [B, N, C]
    
    print(f"  ✓ Loaded test inputs")
    print(f"  Latents: {latents.shape}")
    print(f"  Timestep: {timestep.item()}")
    print(f"  Text: {text_emb_native.shape}")
    
    model_path = "weights/longcat-native"
    lora_path = f"{model_path}/lora/distilled"
    
    sample_blocks = [0, 6, 12, 18, 24, 30, 36, 42, 47]
    
    # ========================================================================
    # RUN 1: Native WITHOUT LoRA (like test_longcat_lora_inference.py standard mode)
    # ========================================================================
    print("\n[2] NATIVE MODEL WITHOUT LORA")
    print("="*100)
    
    print("  Loading via VideoGenerator (no LoRA)...")
    generator_no_lora = VideoGenerator.from_pretrained(
        model_path,
        num_gpus=1,
        use_fsdp_inference=False,
        dit_cpu_offload=False,
    )
    
    print("  ✓ Generator loaded WITHOUT LoRA")
    
    # Run with hooks
    activations_native_no_lora = {}
    output = run_generator_with_hooks(
        generator_no_lora, latents, timestep, text_emb_native, 
        sample_blocks, activations_native_no_lora
    )
    
    print(f"  ✓ Captured {len(activations_native_no_lora)} activations")
    save_tensor_dict(activations_native_no_lora, "native_no_lora.pt")
    
    # Clean up
    generator_no_lora.shutdown()
    del generator_no_lora
    torch.cuda.empty_cache()
    
    # ========================================================================
    # RUN 2: Native WITH LoRA (like test_longcat_lora_inference.py distilled mode)
    # ========================================================================
    print("\n[3] NATIVE MODEL WITH LORA")
    print("="*100)
    
    print("  Loading via VideoGenerator (with LoRA)...")
    print(f"  Model path: {model_path}")
    print(f"  LoRA path: {lora_path}")
    
    generator_with_lora = VideoGenerator.from_pretrained(
        model_path,
        lora_path=lora_path,
        lora_nickname="distilled",
        num_gpus=1,
        use_fsdp_inference=False,
        dit_cpu_offload=False,
    )
    
    print("  ✓ Generator loaded WITH LoRA")
    
    # Run with hooks
    activations_native_with_lora = {}
    output = run_generator_with_hooks(
        generator_with_lora, latents, timestep, text_emb_native,
        sample_blocks, activations_native_with_lora
    )
    
    print(f"  ✓ Captured {len(activations_native_with_lora)} activations")
    save_tensor_dict(activations_native_with_lora, "native_with_lora.pt")
    
    # Clean up
    generator_with_lora.shutdown()
    del generator_with_lora
    torch.cuda.empty_cache()
    
    # ========================================================================
    # ANALYSIS
    # ========================================================================
    print("\n" + "="*100)
    print("ANALYSIS")
    print("="*100)
    
    # Load all activations
    orig_no_lora = torch.load(output_dir / "original_no_lora.pt")
    orig_with_lora = torch.load(output_dir / "original_with_lora.pt")
    native_no_lora_loaded = torch.load(output_dir / "native_no_lora.pt")
    native_with_lora_loaded = torch.load(output_dir / "native_with_lora.pt")
    
    def compute_mse(a, b):
        return torch.mean((a.float() - b.float()) ** 2).item()
    
    print("\n[1] Comparing BASE models (no LoRA):")
    print("-" * 80)
    base_mse = compute_mse(orig_no_lora["final_output"], native_no_lora_loaded["final_output"])
    print(f"  Final output MSE: {base_mse:.6e}")
    if base_mse < 1e-4:
        print("  ✅ Base models match!")
    else:
        print("  ❌ Base models differ significantly")
    
    print("\n[2] Comparing models WITH LoRA:")
    print("-" * 80)
    lora_mse = compute_mse(orig_with_lora["final_output"], native_with_lora_loaded["final_output"])
    print(f"  Final output MSE: {lora_mse:.6e}")
    if lora_mse < 1e-4:
        print("  ✅ LoRA models match!")
    else:
        print("  ❌ LoRA models differ significantly")
    
    print("\n[3] Effect of LoRA on original model:")
    print("-" * 80)
    orig_lora_effect = compute_mse(orig_no_lora["final_output"], orig_with_lora["final_output"])
    print(f"  Original (no LoRA vs with LoRA) MSE: {orig_lora_effect:.6e}")
    print(f"  LoRA changes the output significantly: {orig_lora_effect > 1e-4}")
    
    print("\n[4] Effect of LoRA on native model:")
    print("-" * 80)
    native_lora_effect = compute_mse(native_no_lora_loaded["final_output"], native_with_lora_loaded["final_output"])
    print(f"  Native (no LoRA vs with LoRA) MSE: {native_lora_effect:.6e}")
    print(f"  LoRA changes the output significantly: {native_lora_effect > 1e-4}")
    
    print("\n" + "="*100)
    print("✅ STAGE 2 COMPLETE!")
    print("="*100)
    print(f"\nResults saved in: {output_dir}/")
    print("  - original_no_lora.pt")
    print("  - original_with_lora.pt")
    print("  - native_no_lora.pt")
    print("  - native_with_lora.pt")

def main():
    parser = argparse.ArgumentParser(description="Full LoRA comparison")
    parser.add_argument("--stage", type=int, required=True, choices=[1, 2],
                       help="Stage 1: original model (longcat_shao), Stage 2: native model (fastvideo_shao)")
    args = parser.parse_args()
    
    if args.stage == 1:
        stage1_original()
    else:
        stage2_native()

if __name__ == "__main__":
    main()

