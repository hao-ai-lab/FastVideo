import os
import json
import argparse
from safetensors import safe_open


def validate_components(model_path):
    """Validate all model components exist."""
    print("=" * 60)
    print("VALIDATING COMPONENTS")
    print("=" * 60)
    
    components = {
        "tokenizer": ["special_tokens_map.json", "tokenizer_config.json"],
        "text_encoder": ["config.json", "model.safetensors.index.json"],
        "vae": ["config.json", "diffusion_pytorch_model.safetensors"],
        "scheduler": ["scheduler_config.json"],
        "transformer": ["config.json", "diffusion_pytorch_model.safetensors.index.json"]
    }
    
    all_valid = True
    for component, required_files in components.items():
        component_path = os.path.join(model_path, component)
        print(f"\n{component}:")
        
        if not os.path.exists(component_path):
            print("  ✗ Directory not found")
            all_valid = False
            continue
        
        for req_file in required_files:
            file_path = os.path.join(component_path, req_file)
            exists = os.path.exists(file_path)
            symbol = "✓" if exists else "✗"
            print(f"  {symbol} {req_file}")
            if not exists:
                all_valid = False
    
    return all_valid


def validate_dit_weights(dit_path):
    """Validate DiT weights structure."""
    print("\n" + "=" * 60)
    print("VALIDATING DiT WEIGHTS")
    print("=" * 60)
    
    # Load config
    config_path = os.path.join(dit_path, "config.json")
    with open(config_path) as f:
        config = json.load(f)
    
    hidden_size = config["hidden_size"]
    depth = config["depth"]
    num_heads = config["num_heads"]
    
    print("\nArchitecture:")
    print(f"  - hidden_size: {hidden_size}")
    print(f"  - depth: {depth}")
    print(f"  - num_heads: {num_heads}")
    
    # Load weight index
    index_path = os.path.join(dit_path, "diffusion_pytorch_model.safetensors.index.json")
    with open(index_path) as f:
        index = json.load(f)
    
    weight_map = index["weight_map"]
    all_keys = list(weight_map.keys())
    
    print("\nWeight statistics:")
    print(f"  - Total keys: {len(all_keys)}")
    print(f"  - Total size: {index['metadata']['total_size'] / 1e9:.2f} GB")
    
    # Check structure
    embedder_keys = [k for k in all_keys if 'embedder' in k]
    block_keys = [k for k in all_keys if k.startswith('blocks.')]
    final_keys = [k for k in all_keys if k.startswith('final_layer.')]
    
    print("\nKey distribution:")
    print(f"  - Embedder layers: {len(embedder_keys)}")
    print(f"  - Transformer blocks: {len(block_keys)}")
    print(f"  - Final layer: {len(final_keys)}")
    
    # Verify all blocks present
    block_nums = set()
    for key in block_keys:
        if key.startswith('blocks.'):
            block_num = int(key.split('.')[1])
            block_nums.add(block_num)
    
    expected_blocks = set(range(depth))
    missing_blocks = expected_blocks - block_nums
    
    if missing_blocks:
        print(f"\n✗ Missing blocks: {sorted(missing_blocks)}")
        return False
    else:
        print(f"\n✓ All {depth} blocks present (0-{depth-1})")
    
    # Sample weights
    first_shard = os.path.join(dit_path, "diffusion_pytorch_model-00001-of-00006.safetensors")
    print("\nSampling weights from first shard:")
    
    with safe_open(first_shard, framework="pt", device="cpu") as f:
        sample_keys = [k for k in f.keys() if k in all_keys][:5]
        for key in sample_keys:
            tensor = f.get_tensor(key)
            print(f"  - {key}")
            print(f"    Shape: {tuple(tensor.shape)}, Dtype: {tensor.dtype}")
    
    return True


def validate_shapes(dit_path):
    """Validate weight shapes match expected architecture."""
    print("\n" + "=" * 60)
    print("VALIDATING WEIGHT SHAPES")
    print("=" * 60)
    
    # Load config
    config_path = os.path.join(dit_path, "config.json")
    with open(config_path) as f:
        config = json.load(f)
    
    hidden_size = config["hidden_size"]
    num_heads = config["num_heads"]
    head_dim = hidden_size // num_heads
    adaln_dim = config.get("adaln_tembed_dim", 512)
    mlp_ratio = config.get("mlp_ratio", 4)
    
    # Calculate FFN hidden_dim using SwiGLU formula from blocks.py
    # hidden_dim = int(2 * (hidden_size * mlp_ratio) / 3)
    # rounded to multiple_of=256
    multiple_of = 256
    ffn_hidden = int(2 * hidden_size * mlp_ratio / 3)
    ffn_hidden = multiple_of * ((ffn_hidden + multiple_of - 1) // multiple_of)
    
    # Expected shapes
    expected = {
        "x_embedder.proj.weight": (hidden_size, 16, 1, 2, 2),
        "x_embedder.proj.bias": (hidden_size,),
        "t_embedder.mlp.0.weight": (adaln_dim, 256),
        "t_embedder.mlp.2.weight": (adaln_dim, adaln_dim),
        "y_embedder.y_proj.0.weight": (hidden_size, 4096),
        "blocks.0.attn.qkv.weight": (3 * hidden_size, hidden_size),
        "blocks.0.attn.q_norm.weight": (head_dim,),
        "blocks.0.attn.proj.weight": (hidden_size, hidden_size),
        "blocks.0.cross_attn.q_linear.weight": (hidden_size, hidden_size),
        "blocks.0.cross_attn.kv_linear.weight": (2 * hidden_size, hidden_size),
        "blocks.0.ffn.w1.weight": (ffn_hidden, hidden_size),
        "blocks.0.adaLN_modulation.1.weight": (6 * hidden_size, adaln_dim),
        "final_layer.linear.weight": (64, hidden_size),
    }
    
    # Load and check
    first_shard = os.path.join(dit_path, "diffusion_pytorch_model-00001-of-00006.safetensors")
    
    all_valid = True
    with safe_open(first_shard, framework="pt", device="cpu") as f:
        for key, expected_shape in expected.items():
            if key in f.keys():
                tensor = f.get_tensor(key)
                actual_shape = tuple(tensor.shape)
                
                if actual_shape == expected_shape:
                    print(f"✓ {key}: {actual_shape}")
                else:
                    print(f"✗ {key}: expected {expected_shape}, got {actual_shape}")
                    all_valid = False
    
    return all_valid


def validate_model_index(model_path):
    """Validate model_index.json exists and is correct."""
    print("\n" + "=" * 60)
    print("VALIDATING MODEL INDEX")
    print("=" * 60)
    
    model_index_path = os.path.join(model_path, "model_index.json")
    
    if not os.path.exists(model_index_path):
        print("✗ model_index.json not found")
        return False
    
    with open(model_index_path) as f:
        index = json.load(f)
    
    required_keys = ["_class_name", "workload_type", "tokenizer", "text_encoder", 
                     "vae", "scheduler", "transformer"]
    
    all_valid = True
    for key in required_keys:
        if key in index:
            print(f"✓ {key}: {index[key]}")
        else:
            print(f"✗ {key}: missing")
            all_valid = False
    
    return all_valid


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate LongCat weights for FastVideo")
    parser.add_argument("--model-path", type=str, required=True,
                        help="Path to LongCat model directory")
    parser.add_argument("--check-shapes", action="store_true",
                        help="Also validate weight shapes (slower)")
    
    args = parser.parse_args()
    
    print(f"\nValidating: {args.model_path}\n")
    
    # Run validations
    components_valid = validate_components(args.model_path)
    model_index_valid = validate_model_index(args.model_path)
    dit_valid = validate_dit_weights(os.path.join(args.model_path, "transformer"))
    
    if args.check_shapes:
        shapes_valid = validate_shapes(os.path.join(args.model_path, "transformer"))
    else:
        shapes_valid = True
        print("\nSkipping shape validation (use --check-shapes to enable)")
    
    # Summary
    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)
    
    all_valid = components_valid and model_index_valid and dit_valid and shapes_valid
    
    if all_valid:
        print("✓ All validations passed!")
        print("✓ Model ready for FastVideo")
    else:
        print("✗ Some validations failed")
        print("✗ Please check errors above")
    
    print("=" * 60)

