"""
Layer-by-layer comparison between original LongCat and native implementation.

Runs models separately to avoid OOM, saves intermediate outputs to disk.
"""

import torch
import numpy as np
import os
import sys
from pathlib import Path
import pickle

sys.path.insert(0, "/mnt/fast-disks/hao_lab/shao/LongCat-Video")

# Create output directory
output_dir = Path("outputs/debug_layers")
output_dir.mkdir(parents=True, exist_ok=True)

def setup_environment():
    """Setup distributed environment for single GPU."""
    os.environ['CUDA_VISIBLE_DEVICES'] = '7'
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = str(29500 + np.random.randint(0, 1000))
    os.environ['RANK'] = '0'
    os.environ['WORLD_SIZE'] = '1'
    os.environ['LOCAL_RANK'] = '0'

def run_original_model():
    """Run original model and save intermediate outputs."""
    print("="*100)
    print("RUNNING ORIGINAL LONGCAT MODEL")
    print("="*100)
    
    from longcat_video.modules.longcat_video_dit import LongCatVideoTransformer3DModel as OriginalModel
    
    dit_path = "/mnt/fast-disks/hao_lab/shao/LongCat-Video/weights/LongCat-Video/dit"
    print("\n[1] Loading original model...")
    orig_model = OriginalModel.from_pretrained(
        dit_path,
        torch_dtype=torch.bfloat16,
        cp_split_hw=[1, 1],  # Single GPU inference
    )
    orig_model = orig_model.to("cuda").eval()
    print("  ✓ Loaded")
    
    # Create test inputs
    print("\n[2] Creating test inputs...")
    torch.manual_seed(42)
    
    latents = torch.randn(1, 16, 9, 30, 52, dtype=torch.bfloat16, device="cuda") * 0.18215
    timestep = torch.tensor([500.0], device="cuda")
    text_emb = torch.randn(1, 1, 256, 4096, dtype=torch.bfloat16, device="cuda")
    
    print(f"  Latents: {latents.shape}")
    print(f"  Timestep: {timestep.item()}")
    print(f"  Text: {text_emb.shape}")
    
    # Save inputs
    torch.save({
        'latents': latents.cpu(),
        'timestep': timestep.cpu(),
        'text_emb': text_emb.cpu(),
    }, output_dir / "inputs.pt")
    
    # Setup hooks
    print("\n[3] Setting up hooks...")
    activations = {}
    
    def make_hook(name):
        def hook(module, input, output):
            if isinstance(output, tuple):
                output = output[0]
            activations[name] = output.detach().cpu()
            print(f"  Captured: {name} {output.shape}")
        return hook
    
    hooks = []
    hooks.append(orig_model.x_embedder.register_forward_hook(make_hook("patch_embed")))
    hooks.append(orig_model.t_embedder.register_forward_hook(make_hook("time_embed")))
    hooks.append(orig_model.y_embedder.register_forward_hook(make_hook("caption_embed")))
    
    for i in [0, 11, 23, 35, 47]:
        hooks.append(orig_model.blocks[i].register_forward_hook(make_hook(f"block_{i}")))
        hooks.append(orig_model.blocks[i].attn.register_forward_hook(make_hook(f"block_{i}_self_attn")))
        hooks.append(orig_model.blocks[i].cross_attn.register_forward_hook(make_hook(f"block_{i}_cross_attn")))
        hooks.append(orig_model.blocks[i].ffn.register_forward_hook(make_hook(f"block_{i}_ffn")))
    
    hooks.append(orig_model.final_layer.register_forward_hook(make_hook("final_layer")))
    
    # Run forward pass
    print("\n[4] Running forward pass...")
    with torch.no_grad():
        output = orig_model(
            hidden_states=latents,
            timestep=timestep,
            encoder_hidden_states=text_emb,
        )
    print(f"  ✓ Output: {output.shape}")
    activations['output'] = output.cpu()
    
    # Save activations
    print("\n[5] Saving activations...")
    torch.save(activations, output_dir / "orig_activations.pt")
    print(f"  ✓ Saved {len(activations)} activations")
    
    # Clean up
    for hook in hooks:
        hook.remove()
    del orig_model
    torch.cuda.empty_cache()
    
    print("\n✅ Original model complete!")
    return activations

def run_native_model():
    """Run native model and save intermediate outputs."""
    print("\n" + "="*100)
    print("RUNNING NATIVE LONGCAT MODEL")
    print("="*100)
    
    from fastvideo.models.dits.longcat import LongCatTransformer3DModel
    from fastvideo.configs.models.dits.longcat import LongCatVideoConfig
    from fastvideo.distributed.parallel_state import init_distributed_environment, initialize_model_parallel
    from fastvideo.forward_context import set_forward_context
    from fastvideo.pipelines.pipeline_batch_info import ForwardBatch
    from safetensors.torch import load_file
    import json
    
    setup_environment()
    
    print("\n[1] Initializing distributed environment...")
    init_distributed_environment()
    initialize_model_parallel()
    
    print("\n[2] Loading native model...")
    transformer_path = "weights/longcat-native/transformer"
    with open(f"{transformer_path}/config.json") as f:
        config_dict = json.load(f)
    
    model_config = LongCatVideoConfig()
    native_model = LongCatTransformer3DModel(config=model_config, hf_config=config_dict)
    
    state_dict = load_file(f"{transformer_path}/model.safetensors")
    native_model.load_state_dict(state_dict, strict=False)
    native_model = native_model.to("cuda").to(torch.bfloat16).eval()
    print("  ✓ Loaded")
    
    # Load test inputs
    print("\n[3] Loading test inputs...")
    inputs = torch.load(output_dir / "inputs.pt")
    latents = inputs['latents'].to("cuda").to(torch.bfloat16)
    timestep = inputs['timestep'].to("cuda")
    text_emb = inputs['text_emb'].squeeze(1).to("cuda").to(torch.bfloat16)  # Remove dim for native
    
    print(f"  Latents: {latents.shape}")
    print(f"  Timestep: {timestep.item()}")
    print(f"  Text: {text_emb.shape}")
    
    # Setup hooks
    print("\n[4] Setting up hooks...")
    activations = {}
    
    def make_hook(name):
        def hook(module, input, output):
            if isinstance(output, tuple):
                output = output[0]
            activations[name] = output.detach().cpu()
            print(f"  Captured: {name} {output.shape}")
        return hook
    
    hooks = []
    hooks.append(native_model.patch_embed.register_forward_hook(make_hook("patch_embed")))
    hooks.append(native_model.time_embedder.register_forward_hook(make_hook("time_embed")))
    hooks.append(native_model.caption_embedder.register_forward_hook(make_hook("caption_embed")))
    
    for i in [0, 11, 23, 35, 47]:
        hooks.append(native_model.blocks[i].register_forward_hook(make_hook(f"block_{i}")))
        hooks.append(native_model.blocks[i].self_attn.register_forward_hook(make_hook(f"block_{i}_self_attn")))
        hooks.append(native_model.blocks[i].cross_attn.register_forward_hook(make_hook(f"block_{i}_cross_attn")))
        hooks.append(native_model.blocks[i].ffn.register_forward_hook(make_hook(f"block_{i}_ffn")))
    
    hooks.append(native_model.final_layer.register_forward_hook(make_hook("final_layer")))
    
    # Run forward pass
    print("\n[5] Running forward pass...")
    dummy_batch = ForwardBatch(data_type="t2v")
    with torch.no_grad():
        with set_forward_context(
            current_timestep=0,
            attn_metadata=None,
            forward_batch=dummy_batch,
        ):
            output = native_model(
                hidden_states=latents,
                encoder_hidden_states=text_emb,
                timestep=timestep,
            )
    print(f"  ✓ Output: {output.shape}")
    activations['output'] = output.cpu()
    
    # Save activations
    print("\n[6] Saving activations...")
    torch.save(activations, output_dir / "native_activations.pt")
    print(f"  ✓ Saved {len(activations)} activations")
    
    # Clean up
    for hook in hooks:
        hook.remove()
    del native_model
    torch.cuda.empty_cache()
    
    print("\n✅ Native model complete!")
    return activations

def compare_activations():
    """Compare saved activations."""
    print("\n" + "="*100)
    print("COMPARING ACTIVATIONS")
    print("="*100)
    
    print("\n[1] Loading activations...")
    orig_act = torch.load(output_dir / "orig_activations.pt")
    native_act = torch.load(output_dir / "native_activations.pt")
    print(f"  Original: {len(orig_act)} activations")
    print(f"  Native: {len(native_act)} activations")
    
    def compare(name, orig, native):
        orig_f = orig.float()
        native_f = native.float()
        
        diff = torch.abs(orig_f - native_f)
        rel_diff = diff / (orig_f.abs() + 1e-8)
        
        print(f"\n{name}")
        print("-"*80)
        print(f"  Shape: {orig.shape}")
        print(f"  Original: mean={orig_f.mean():.6f}, std={orig_f.std():.6f}, range=[{orig_f.min():.6f}, {orig_f.max():.6f}]")
        print(f"  Native:   mean={native_f.mean():.6f}, std={native_f.std():.6f}, range=[{native_f.min():.6f}, {native_f.max():.6f}]")
        print(f"  Abs diff: max={diff.max():.6e}, mean={diff.mean():.6e}")
        print(f"  Rel diff: max={rel_diff.max():.6e}, mean={rel_diff.mean():.6e}")
        
        if diff.max() < 1e-4:
            print("  ✅ VERY SIMILAR")
            return 0
        elif diff.max() < 1e-2:
            print("  ✅ Similar")
            return 1
        elif diff.max() < 0.1:
            print("  ⚠️  Moderate difference")
            return 2
        else:
            print("  ❌ SIGNIFICANT DIFFERENCE")
            return 3
        
        return diff.max().item()
    
    print("\n[2] Comparing layers...")
    
    results = {}
    
    # Compare embeddings
    print("\n--- EMBEDDINGS ---")
    results['patch_embed'] = compare("Patch Embedding", orig_act['patch_embed'], native_act['patch_embed'])
    results['time_embed'] = compare("Timestep Embedding", orig_act['time_embed'], native_act['time_embed'])
    
    # Caption embeddings - handle tuple return
    print("\n--- CAPTION EMBEDDING ---")
    if isinstance(orig_act['caption_embed'], tuple):
        print(f"Original caption (tuple): {len(orig_act['caption_embed'])} elements")
        for i, elem in enumerate(orig_act['caption_embed']):
            print(f"  Element {i}: {elem.shape if hasattr(elem, 'shape') else type(elem)}")
    else:
        print(f"Original caption: {orig_act['caption_embed'].shape}")
    
    if isinstance(native_act['caption_embed'], tuple):
        print(f"Native caption (tuple): {len(native_act['caption_embed'])} elements")
        for i, elem in enumerate(native_act['caption_embed']):
            print(f"  Element {i}: {elem.shape if hasattr(elem, 'shape') else type(elem)}")
    else:
        print(f"Native caption: {native_act['caption_embed'].shape}")
    
    # Compare blocks
    print("\n--- TRANSFORMER BLOCKS ---")
    for i in [0, 11, 23, 35, 47]:
        print(f"\n=== Block {i} ===")
        results[f'block_{i}_self_attn'] = compare(f"Block {i} Self-Attention", 
                                                   orig_act[f'block_{i}_self_attn'], 
                                                   native_act[f'block_{i}_self_attn'])
        results[f'block_{i}_cross_attn'] = compare(f"Block {i} Cross-Attention",
                                                    orig_act[f'block_{i}_cross_attn'],
                                                    native_act[f'block_{i}_cross_attn'])
        results[f'block_{i}_ffn'] = compare(f"Block {i} FFN",
                                             orig_act[f'block_{i}_ffn'],
                                             native_act[f'block_{i}_ffn'])
        results[f'block_{i}'] = compare(f"Block {i} Output",
                                         orig_act[f'block_{i}'],
                                         native_act[f'block_{i}'])
    
    # Compare final layer and output
    print("\n--- FINAL LAYER ---")
    results['final_layer'] = compare("Final Layer", orig_act['final_layer'], native_act['final_layer'])
    results['output'] = compare("Final Output", orig_act['output'], native_act['output'])
    
    # Summary
    print("\n" + "="*100)
    print("SUMMARY")
    print("="*100)
    
    # Find first divergence
    check_order = ['patch_embed', 'time_embed']
    for i in [0, 11, 23, 35, 47]:
        check_order.extend([f'block_{i}_self_attn', f'block_{i}_cross_attn', f'block_{i}_ffn', f'block_{i}'])
    check_order.extend(['final_layer', 'output'])
    
    first_divergence = None
    for name in check_order:
        if name in results and results[name] >= 2:  # Moderate or significant
            first_divergence = name
            print(f"\n❌ First significant divergence at: {name}")
            break
    
    if first_divergence is None:
        print("\n✅ No significant divergence found!")
    
    print("\n✅ Comparison complete!")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        mode = sys.argv[1]
    else:
        mode = "all"
    
    if mode in ["all", "original"]:
        run_original_model()
    
    if mode in ["all", "native"]:
        run_native_model()
    
    if mode in ["all", "compare"]:
        compare_activations()

