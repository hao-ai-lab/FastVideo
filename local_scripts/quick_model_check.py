import torch
import os
from safetensors import safe_open

def quick_model_check():
    """Quick check of both models to understand their structure"""
    
    # Model paths
    pytorch_model_path = "/mnt/sharefs/users/hao.zhang/DMD/wan_bidirectional_dmd_from_scratch/2025-06-20-08-17-06.607828_seed1024/checkpoint_model_004800/model.pt"
    safetensors_dir = "../Wan-AI/Wan2.1-T2V-1.3B-Diffusers/transformer"
    
    print("Quick Model Structure Check")
    print("=" * 40)
    
    # Check PyTorch model
    print(f"\n1. PyTorch Model: {pytorch_model_path}")
    if os.path.exists(pytorch_model_path):
        try:
            checkpoint = torch.load(pytorch_model_path, map_location='cpu')
            print(f"   File size: {os.path.getsize(pytorch_model_path) / (1024**3):.2f} GB")
            print(f"   Checkpoint type: {type(checkpoint)}")
            
            if isinstance(checkpoint, dict):
                print(f"   Checkpoint keys: {list(checkpoint.keys())}")
                if 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                elif 'model' in checkpoint:
                    state_dict = checkpoint['model']
                else:
                    state_dict = checkpoint
            else:
                state_dict = checkpoint
            
            print(f"   Parameters: {len(state_dict)}")
            total_params = sum(tensor.numel() for tensor in state_dict.values())
            print(f"   Total params: {total_params:,}")
            
            # Show first few keys
            print(f"   Sample keys:")
            for i, key in enumerate(list(state_dict.keys())[:5]):
                tensor = state_dict[key]
                print(f"     {i+1}. {key}: {tensor.shape}, {tensor.dtype}")
                
        except Exception as e:
            print(f"   Error: {e}")
    else:
        print(f"   File not found")
    
    # Check safetensors directory
    print(f"\n2. Safetensors Directory: {safetensors_dir}")
    if os.path.exists(safetensors_dir):
        safetensors_files = []
        for file in os.listdir(safetensors_dir):
            if file.endswith('.safetensors'):
                file_path = os.path.join(safetensors_dir, file)
                safetensors_files.append((file, file_path))
        
        print(f"   Found {len(safetensors_files)} safetensors files:")
        for file_name, file_path in safetensors_files:
            file_size = os.path.getsize(file_path) / (1024**3)
            print(f"     - {file_name}: {file_size:.2f} GB")
            
            # Quick load to check structure
            try:
                with safe_open(file_path, framework="pt", device="cpu") as f:
                    keys = list(f.keys())
                    print(f"       Parameters: {len(keys)}")
                    if keys:
                        # Get first tensor to check shape
                        first_key = keys[0]
                        first_tensor = f.get_tensor(first_key)
                        print(f"       Sample: {first_key}: {first_tensor.shape}, {first_tensor.dtype}")
            except Exception as e:
                print(f"       Error loading: {e}")
    else:
        print(f"   Directory not found")

if __name__ == "__main__":
    quick_model_check() 