import torch
import os
from safetensors import safe_open
from collections import OrderedDict
import numpy as np
from typing import Dict, Any, Tuple

def load_pytorch_model(model_path: str) -> Dict[str, torch.Tensor]:
    """
    Load a PyTorch model from .pt file
    
    Args:
        model_path: Path to the .pt model file
        
    Returns:
        Dictionary containing model state dict
    """
    print(f"Loading PyTorch model from: {model_path}")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    # Load the model checkpoint
    checkpoint = torch.load(model_path, map_location='cpu')
    
    # Handle different checkpoint formats
    if isinstance(checkpoint, dict):
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint
    
    print(f"Loaded PyTorch model with {len(state_dict)} parameters")
    return state_dict

def load_safetensors_model(model_path: str) -> Dict[str, torch.Tensor]:
    """
    Load a model from safetensors file or directory
    
    Args:
        model_path: Path to the .safetensors file or directory containing safetensors files
        
    Returns:
        Dictionary containing model tensors
    """
    print(f"Loading safetensors model from: {model_path}")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model path not found: {model_path}")
    
    state_dict = {}
    
    if os.path.isdir(model_path):
        # Load all safetensors files in the directory
        safetensors_files = []
        for file in os.listdir(model_path):
            if file.endswith('.safetensors'):
                safetensors_files.append(os.path.join(model_path, file))
        
        print(f"Found {len(safetensors_files)} safetensors files: {safetensors_files}")
        
        for file_path in safetensors_files:
            print(f"Loading {file_path}...")
            with safe_open(file_path, framework="pt", device="cpu") as f:
                for key in f.keys():
                    if key in state_dict:
                        print(f"Warning: Duplicate key '{key}' found in {file_path}")
                    state_dict[key] = f.get_tensor(key)
    else:
        # Load single safetensors file
        with safe_open(model_path, framework="pt", device="cpu") as f:
            for key in f.keys():
                state_dict[key] = f.get_tensor(key)
    
    print(f"Loaded safetensors model with {len(state_dict)} parameters")
    return state_dict

def compare_models(model1: Dict[str, torch.Tensor], 
                  model2: Dict[str, torch.Tensor],
                  model1_name: str = "Model 1",
                  model2_name: str = "Model 2") -> Dict[str, Any]:
    """
    Compare two models and return detailed comparison results
    
    Args:
        model1: First model state dict
        model2: Second model state dict
        model1_name: Name of first model for reporting
        model2_name: Name of second model for reporting
        
    Returns:
        Dictionary containing comparison results
    """
    print(f"\nComparing {model1_name} vs {model2_name}")
    print("=" * 50)
    
    # Get all unique keys
    all_keys = set(model1.keys()) | set(model2.keys())
    model1_keys = set(model1.keys())
    model2_keys = set(model2.keys())
    
    # Keys only in each model
    only_in_model1 = model1_keys - model2_keys
    only_in_model2 = model2_keys - model1_keys
    common_keys = model1_keys & model2_keys
    
    print(f"Total parameters in {model1_name}: {len(model1_keys)}")
    print(f"Total parameters in {model2_name}: {len(model2_keys)}")
    print(f"Common parameters: {len(common_keys)}")
    print(f"Only in {model1_name}: {len(only_in_model1)}")
    print(f"Only in {model2_name}: {len(only_in_model2)}")
    
    # Show some examples of unique keys
    if only_in_model1:
        print(f"\nSample keys only in {model1_name}:")
        for key in list(only_in_model1)[:5]:
            print(f"  - {key}")
        if len(only_in_model1) > 5:
            print(f"  ... and {len(only_in_model1) - 5} more")
    
    if only_in_model2:
        print(f"\nSample keys only in {model2_name}:")
        for key in list(only_in_model2)[:5]:
            print(f"  - {key}")
        if len(only_in_model2) > 5:
            print(f"  ... and {len(only_in_model2) - 5} more")
    
    # Compare common parameters
    if common_keys:
        print(f"\nComparing {len(common_keys)} common parameters...")
        
        differences = []
        shape_mismatches = []
        exact_matches = 0
        
        for key in common_keys:
            tensor1 = model1[key]
            tensor2 = model2[key]
            
            # Check shape compatibility
            if tensor1.shape != tensor2.shape:
                shape_mismatches.append((key, tensor1.shape, tensor2.shape))
                continue
            
            # Check if tensors are exactly equal
            if torch.equal(tensor1, tensor2):
                exact_matches += 1
                continue
            
            # Calculate differences
            diff = torch.abs(tensor1 - tensor2)
            max_diff = torch.max(diff).item()
            mean_diff = torch.mean(diff).item()
            std_diff = torch.std(diff).item()
            
            differences.append({
                'key': key,
                'shape': tensor1.shape,
                'max_diff': max_diff,
                'mean_diff': mean_diff,
                'std_diff': std_diff,
                'relative_max_diff': max_diff / (torch.max(torch.abs(tensor1)).item() + 1e-8)
            })
        
        print(f"Exact matches: {exact_matches}/{len(common_keys)} ({exact_matches/len(common_keys)*100:.2f}%)")
        print(f"Shape mismatches: {len(shape_mismatches)}")
        print(f"Parameters with differences: {len(differences)}")
        
        if shape_mismatches:
            print(f"\nShape mismatches:")
            for key, shape1, shape2 in shape_mismatches[:5]:
                print(f"  {key}: {shape1} vs {shape2}")
            if len(shape_mismatches) > 5:
                print(f"  ... and {len(shape_mismatches) - 5} more")
        
        if differences:
            # Sort by max difference
            differences.sort(key=lambda x: x['max_diff'], reverse=True)
            
            print(f"\nTop 10 parameters with largest differences:")
            for i, diff_info in enumerate(differences[:10]):
                print(f"  {i+1}. {diff_info['key']} (shape: {diff_info['shape']})")
                print(f"     Max diff: {diff_info['max_diff']:.6f}")
                print(f"     Mean diff: {diff_info['mean_diff']:.6f}")
                print(f"     Relative max diff: {diff_info['relative_max_diff']:.6f}")
            
            # Summary statistics
            max_diffs = [d['max_diff'] for d in differences]
            mean_diffs = [d['mean_diff'] for d in differences]
            relative_diffs = [d['relative_max_diff'] for d in differences]
            
            print(f"\nDifference statistics:")
            print(f"  Max difference range: {min(max_diffs):.6f} to {max(max_diffs):.6f}")
            print(f"  Mean difference range: {min(mean_diffs):.6f} to {max(mean_diffs):.6f}")
            print(f"  Relative max difference range: {min(relative_diffs):.6f} to {max(relative_diffs):.6f}")
    
    return {
        'model1_keys': model1_keys,
        'model2_keys': model2_keys,
        'common_keys': common_keys,
        'only_in_model1': only_in_model1,
        'only_in_model2': only_in_model2,
        'differences': differences if 'differences' in locals() else [],
        'shape_mismatches': shape_mismatches if 'shape_mismatches' in locals() else [],
        'exact_matches': exact_matches if 'exact_matches' in locals() else 0
    }

def main():
    """Main function to load and compare the two models"""
    
    # Model paths
    pytorch_model_path = "/mnt/sharefs/users/hao.zhang/DMD/wan_bidirectional_dmd_from_scratch/2025-06-20-08-17-06.607828_seed1024/checkpoint_model_004800/model.pt"
    safetensors_model_path = "../Wan-AI/Wan2.1-T2V-1.3B-Diffusers/transformer"
    
    print("Model Weight Comparison Tool")
    print("=" * 50)
    
    try:
        # Load both models
        pytorch_model = load_pytorch_model(pytorch_model_path)
        safetensors_model = load_safetensors_model(safetensors_model_path)
        generator_model = pytorch_model['generator']
        critic_model = pytorch_model['critic']
        
        # Debug: Add interactive breakpoint to inspect models
        print("\nModels loaded successfully!")
        print(f"PyTorch model keys: {list(pytorch_model.keys())[:10]}...")
        print(f"Safetensors model keys: {list(safetensors_model.keys())[:10]}...")
        
        # Uncomment the next line to enter interactive debug mode
        # from IPython import embed; embed()
        from IPython import embed; embed()
        
        # Compare the models
        comparison_results = compare_models(
            pytorch_model, 
            safetensors_model,
            "DMD PyTorch Model", 
            "Wan2.1 Safetensors Model"
        )
        
        # Save comparison results
        output_file = "model_comparison_results.txt"
        with open(output_file, 'w') as f:
            f.write("Model Weight Comparison Results\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"DMD PyTorch Model: {pytorch_model_path}\n")
            f.write(f"Wan2.1 Safetensors Model: {safetensors_model_path}\n\n")
            
            f.write(f"Total parameters in DMD model: {len(comparison_results['model1_keys'])}\n")
            f.write(f"Total parameters in Wan2.1 model: {len(comparison_results['model2_keys'])}\n")
            f.write(f"Common parameters: {len(comparison_results['common_keys'])}\n")
            f.write(f"Only in DMD model: {len(comparison_results['only_in_model1'])}\n")
            f.write(f"Only in Wan2.1 model: {len(comparison_results['only_in_model2'])}\n")
            f.write(f"Exact matches: {comparison_results['exact_matches']}\n")
            f.write(f"Parameters with differences: {len(comparison_results['differences'])}\n")
            f.write(f"Shape mismatches: {len(comparison_results['shape_mismatches'])}\n\n")
            
            if comparison_results['differences']:
                f.write("Top 20 parameters with largest differences:\n")
                for i, diff_info in enumerate(comparison_results['differences'][:20]):
                    f.write(f"{i+1}. {diff_info['key']} (shape: {diff_info['shape']})\n")
                    f.write(f"   Max diff: {diff_info['max_diff']:.6f}\n")
                    f.write(f"   Mean diff: {diff_info['mean_diff']:.6f}\n")
                    f.write(f"   Relative max diff: {diff_info['relative_max_diff']:.6f}\n\n")
        
        print(f"\nComparison results saved to: {output_file}")
        
    except Exception as e:
        print(f"Error during model comparison: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
