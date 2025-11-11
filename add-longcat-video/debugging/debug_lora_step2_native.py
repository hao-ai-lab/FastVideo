#!/usr/bin/env python3
"""
Step 2: Run FastVideo native with LoRA and save intermediate activations.
Run this with: conda activate fastvideo_shao
"""

import torch
import numpy as np
import os
import sys
from pathlib import Path

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

def run_native_with_lora():
    """Run native FastVideo with LoRA and save activations."""
    setup_environment()
    
    print("="*100)
    print("STEP 2: FASTVIDEO NATIVE WITH LORA")
    print("="*100)
    
    # ========================================================================
    # 1. LOAD TEST INPUTS
    # ========================================================================
    print("\n[1] LOADING TEST INPUTS")
    print("="*100)
    
    inputs_path = output_dir / "test_inputs.pt"
    if not inputs_path.exists():
        raise FileNotFoundError(
            f"Test inputs not found at {inputs_path}. "
            "Please run debug_lora_step1_original.py first!"
        )
    
    inputs = torch.load(inputs_path)
    latents = inputs['latents'].to("cuda").to(torch.bfloat16)
    timestep = inputs['timestep'].to("cuda")
    text_emb_orig_format = inputs['text_emb_orig_format'].to("cuda").to(torch.bfloat16)
    
    # Native expects [B, N, C] (squeeze the 1 dimension)
    text_emb_native = text_emb_orig_format.squeeze(1)
    
    print(f"  ✓ Loaded test inputs")
    print(f"  Latents: {latents.shape}")
    print(f"  Timestep: {timestep.item()}")
    print(f"  Text (native format): {text_emb_native.shape}")
    
    # ========================================================================
    # 2. LOAD NATIVE MODEL WITH LORA
    # ========================================================================
    print("\n[2] LOADING NATIVE MODEL WITH LORA")
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
    
    model_path_native = "weights/longcat-native/transformer"
    lora_path_native = "weights/longcat-native/lora/distilled/adapter_model.safetensors"
    
    print(f"  Loading from: {model_path_native}")
    print(f"  LoRA from: {lora_path_native}")
    
    # Load model config
    with open(f"{model_path_native}/config.json") as f:
        config_dict = json.load(f)
    
    model_config = LongCatVideoConfig()
    
    # Load base model
    print("  Loading base model...")
    native_model = LongCatTransformer3DModel(config=model_config, hf_config=config_dict)
    state_dict = load_file(f"{model_path_native}/model.safetensors")
    native_model.load_state_dict(state_dict, strict=False)
    native_model = native_model.to("cuda").to(torch.bfloat16)
    print("  ✓ Base model loaded")
    
    # NOTE: The native model uses the pipeline infrastructure for LoRA
    # For this comparison, we need to use the VideoGenerator to properly load LoRA
    # But we'll access the model directly after LoRA has been set up
    
    # Temporarily - let's first check if base models match without LoRA
    print("  ⚠️  WARNING: Comparing BASE models WITHOUT LoRA for now")
    print("  This will help us verify if the base weights match correctly")
    
    native_model.eval()
    print("  ✓ Native base model loaded (NO LoRA)")
    print(f"  Model type: {type(native_model).__name__}")
    
    # ========================================================================
    # 3. HOOK TO CAPTURE INTERMEDIATE OUTPUTS
    # ========================================================================
    print("\n[3] SETTING UP HOOKS TO CAPTURE INTERMEDIATE OUTPUTS")
    print("="*100)
    
    native_activations = {}
    
    def make_hook(name, storage):
        def hook(module, input, output):
            if isinstance(output, tuple):
                output = output[0]
            storage[name] = output.detach().cpu()  # Move to CPU to save memory
        return hook
    
    # Register hooks on native model
    native_hooks = []
    
    # Patch embedding
    native_hooks.append(native_model.patch_embed.register_forward_hook(
        make_hook("patch_embed", native_activations)))
    
    # Timestep embedding
    native_hooks.append(native_model.time_embedder.register_forward_hook(
        make_hook("time_embed", native_activations)))
    
    # Caption embedding
    native_hooks.append(native_model.caption_embedder.register_forward_hook(
        make_hook("caption_embed", native_activations)))
    
    # Transformer blocks (same indices as original)
    sample_blocks = [0, 6, 12, 18, 24, 30, 36, 42, 47]
    for i in sample_blocks:
        native_hooks.append(native_model.blocks[i].register_forward_hook(
            make_hook(f"block_{i}", native_activations)))
        
        # Self-attention within block
        native_hooks.append(native_model.blocks[i].self_attn.register_forward_hook(
            make_hook(f"block_{i}_self_attn", native_activations)))
        
        # Cross-attention within block
        native_hooks.append(native_model.blocks[i].cross_attn.register_forward_hook(
            make_hook(f"block_{i}_cross_attn", native_activations)))
        
        # FFN within block
        native_hooks.append(native_model.blocks[i].ffn.register_forward_hook(
            make_hook(f"block_{i}_ffn", native_activations)))
    
    # Final layer
    native_hooks.append(native_model.final_layer.register_forward_hook(
        make_hook("final_layer", native_activations)))
    
    print(f"  ✓ Registered {len(native_hooks)} hooks on native model")
    
    # ========================================================================
    # 4. RUN FORWARD PASS
    # ========================================================================
    print("\n[4] RUNNING FORWARD PASS")
    print("="*100)
    
    from fastvideo.forward_context import set_forward_context
    from fastvideo.pipelines.pipeline_batch_info import ForwardBatch
    
    with torch.no_grad():
        print("  Running native model forward pass...")
        dummy_batch = ForwardBatch(data_type="t2v")
        with set_forward_context(
            current_timestep=0,
            attn_metadata=None,
            forward_batch=dummy_batch,
        ):
            native_output = native_model(
                hidden_states=latents,
                encoder_hidden_states=text_emb_native,
                timestep=timestep,
            )
        print(f"  ✓ Native output: {native_output.shape}")
        
        # Save output
        native_activations["final_output"] = native_output.cpu()
    
    # Clean up hooks
    for hook in native_hooks:
        hook.remove()
    
    # ========================================================================
    # 5. SAVE ACTIVATIONS
    # ========================================================================
    print("\n[5] SAVING ACTIVATIONS")
    print("="*100)
    
    activations_path = output_dir / "native_activations.pt"
    
    # Convert any tuples to first element
    cleaned_activations = {}
    for name, value in native_activations.items():
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
    print(f"  File saved: {activations_path}")
    
    print("\n" + "="*100)
    print("✅ Step 2 complete! Run step 3 to analyze the results.")
    print("="*100)

if __name__ == "__main__":
    run_native_with_lora()

