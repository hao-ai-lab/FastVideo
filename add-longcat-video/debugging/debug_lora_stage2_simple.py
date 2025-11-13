#!/usr/bin/env python3
"""
Stage 2 simplified: Load models directly and manually apply LoRA.
"""

import torch
import numpy as np
import os
from pathlib import Path

# Create output directory
output_dir = Path("outputs/debug_lora_comparison")

def setup_environment():
    """Setup distributed environment for single GPU."""
    os.environ['CUDA_VISIBLE_DEVICES'] = '7'
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = str(29500 + np.random.randint(0, 1000))
    os.environ['RANK'] = '0'
    os.environ['WORLD_SIZE'] = '1'
    os.environ['LOCAL_RANK'] = '0'

def make_hook(name, storage):
    """Create a forward hook that captures outputs."""
    def hook(module, input, output):
        if isinstance(output, tuple):
            output = output[0]
        storage[name] = output.detach().cpu()
    return hook

def register_hooks(model, storage, sample_blocks):
    """Register forward hooks on model layers."""
    hooks = []
    
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

def manually_merge_lora(model, lora_state_dict, lora_scale=1.0):
    """Manually merge LoRA weights into model by modifying linear layer weights."""
    print(f"  Manually merging LoRA with scale={lora_scale}...")
    merged_count = 0
    
    for name, module in model.named_modules():
        # Check if module has weights (Linear, ReplicatedLinear, etc.)
        if not hasattr(module, 'weight'):
            continue
        
        # Look for LoRA weights for this layer (no .weight suffix in keys)
        lora_A_key = f"{name}.lora_A"
        lora_B_key = f"{name}.lora_B"
        
        if lora_A_key in lora_state_dict and lora_B_key in lora_state_dict:
            lora_A = lora_state_dict[lora_A_key].to(module.weight.device).to(module.weight.dtype)
            lora_B = lora_state_dict[lora_B_key].to(module.weight.device).to(module.weight.dtype)
            
            # Compute LoRA delta: lora_B @ lora_A
            # A is (rank, in_features), B is (out_features, rank)
            # Delta should be (out_features, in_features)
            with torch.no_grad():
                delta = lora_scale * (lora_B @ lora_A)
                module.weight.data += delta
            
            merged_count += 1
            if merged_count <= 5:  # Print first few for verification
                print(f"    Merged: {name} (A: {lora_A.shape}, B: {lora_B.shape}, delta: {delta.shape})")
    
    print(f"  ✓ Merged LoRA into {merged_count} layers")
    return merged_count

def main():
    setup_environment()
    
    print("="*100)
    print("STAGE 2: NATIVE FASTVIDEO MODEL (simplified)")
    print("="*100)
    
    from fastvideo.models.dits.longcat import LongCatTransformer3DModel
    from fastvideo.configs.models.dits.longcat import LongCatVideoConfig
    from fastvideo.distributed.parallel_state import init_distributed_environment, initialize_model_parallel
    from fastvideo.forward_context import set_forward_context
    from fastvideo.pipelines.pipeline_batch_info import ForwardBatch
    from safetensors.torch import load_file
    import json
    
    init_distributed_environment()
    initialize_model_parallel()
    
    # Load test inputs
    print("\n[1] LOADING TEST INPUTS")
    print("="*100)
    
    inputs = torch.load(output_dir / "test_inputs.pt")
    latents = inputs['latents'].to("cuda").to(torch.bfloat16)
    timestep = inputs['timestep'].to("cuda")
    text_emb_native = inputs['text_emb_orig_format'].squeeze(1).to("cuda").to(torch.bfloat16)
    
    print(f"  ✓ Loaded test inputs")
    print(f"  Latents: {latents.shape}")
    print(f"  Timestep: {timestep.item()}")
    print(f"  Text: {text_emb_native.shape}")
    
    model_path = "weights/longcat-native/transformer"
    lora_path = "weights/longcat-native/lora/distilled/cfg_step_lora.safetensors"
    
    sample_blocks = [0, 6, 12, 18, 24, 30, 36, 42, 47]
    
    # Load model config and base weights
    with open(f"{model_path}/config.json") as f:
        config_dict = json.load(f)
    model_config = LongCatVideoConfig()
    base_state_dict = load_file(f"{model_path}/model.safetensors")
    
    # ========================================================================
    # RUN 1: Native WITHOUT LoRA
    # ========================================================================
    print("\n[2] NATIVE MODEL WITHOUT LORA")
    print("="*100)
    
    print("  Loading base model...")
    model_no_lora = LongCatTransformer3DModel(config=model_config, hf_config=config_dict)
    model_no_lora.load_state_dict(base_state_dict, strict=False)
    model_no_lora = model_no_lora.to("cuda").to(torch.bfloat16).eval()
    print("  ✓ Model loaded WITHOUT LoRA")
    
    # Register hooks and run
    activations_native_no_lora = {}
    hooks = register_hooks(model_no_lora, activations_native_no_lora, sample_blocks)
    print(f"  Registered {len(hooks)} hooks")
    
    print("  Running forward pass...")
    dummy_batch = ForwardBatch(data_type="t2v")
    with torch.no_grad():
        with set_forward_context(current_timestep=0, attn_metadata=None, forward_batch=dummy_batch):
            output = model_no_lora(
                hidden_states=latents,
                encoder_hidden_states=text_emb_native,
                timestep=timestep,
            )
            activations_native_no_lora["final_output"] = output.cpu()
    
    for hook in hooks:
        hook.remove()
    
    print(f"  ✓ Captured {len(activations_native_no_lora)} activations")
    torch.save(activations_native_no_lora, output_dir / "native_no_lora.pt")
    print(f"  ✓ Saved to: native_no_lora.pt")
    
    # Clean up
    del model_no_lora
    torch.cuda.empty_cache()
    
    # ========================================================================
    # RUN 2: Native WITH LoRA (manually merged)
    # ========================================================================
    print("\n[3] NATIVE MODEL WITH LORA (manually merged)")
    print("="*100)
    
    print("  Loading base model...")
    model_with_lora = LongCatTransformer3DModel(config=model_config, hf_config=config_dict)
    model_with_lora.load_state_dict(base_state_dict, strict=False)
    model_with_lora = model_with_lora.to("cuda").to(torch.bfloat16)
    
    # Load and merge LoRA
    print(f"  Loading LoRA from: {lora_path}")
    lora_state_dict = load_file(lora_path)
    print(f"  LoRA has {len(lora_state_dict)} keys")
    
    # IMPORTANT: Original LoRA uses alpha_scale=0.5 (alpha=64, rank=128, so 64/128=0.5)
    # The converted LoRA is missing this scale information!
    merged_count = manually_merge_lora(model_with_lora, lora_state_dict, lora_scale=0.5)
    
    model_with_lora.eval()
    print("  ✓ Model loaded WITH LoRA")
    
    # Register hooks and run
    activations_native_with_lora = {}
    hooks = register_hooks(model_with_lora, activations_native_with_lora, sample_blocks)
    print(f"  Registered {len(hooks)} hooks")
    
    print("  Running forward pass...")
    with torch.no_grad():
        with set_forward_context(current_timestep=0, attn_metadata=None, forward_batch=dummy_batch):
            output = model_with_lora(
                hidden_states=latents,
                encoder_hidden_states=text_emb_native,
                timestep=timestep,
            )
            activations_native_with_lora["final_output"] = output.cpu()
    
    for hook in hooks:
        hook.remove()
    
    print(f"  ✓ Captured {len(activations_native_with_lora)} activations")
    torch.save(activations_native_with_lora, output_dir / "native_with_lora.pt")
    print(f"  ✓ Saved to: native_with_lora.pt")
    
    # Clean up
    del model_with_lora
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
    native_no_lora_loaded = activations_native_no_lora  # Already in memory
    native_with_lora_loaded = activations_native_with_lora  # Already in memory
    
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

if __name__ == "__main__":
    main()

