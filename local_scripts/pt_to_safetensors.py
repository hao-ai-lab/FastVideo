#!/usr/bin/env python3
"""
Convert a PyTorch checkpoint (.pt / .pth) to safetensors format.

Usage:
    python pt_to_safetensors.py path/to/model.pt            # 输出同名 .safetensors
    python pt_to_safetensors.py model.pt -o model.safetensors
"""

import argparse
import torch
from safetensors.torch import save_file

def extract_state_dict(obj):
    """
    Try to get a pure tensor dict from various checkpoint formats.
    """
    if isinstance(obj, dict):
        # case 1: already a state_dict
        if all(isinstance(v, torch.Tensor) for v in obj.values()):
            return obj
        # case 2: wrapped dict
        if "state_dict" in obj:
            return obj["state_dict"]
        if "generator" in obj:
            return obj["generator"]
    # case 3: whole nn.Module
    if hasattr(obj, "state_dict"):
        return obj.state_dict()
    raise ValueError("Unsupported checkpoint format: cannot find tensors.")

def convert(pt_path: str, safe_path: str):
    print(f"Loading {pt_path}...")
    ckpt = torch.load(pt_path, map_location="cpu")
    state_dict = extract_state_dict(ckpt)

    tensors = {k: v.contiguous() for k, v in state_dict.items()}

    print(f"Saving safetensors to {safe_path}...")
    save_file(tensors, safe_path)
    print("✅ Done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert .pt/.pth to .safetensors")
    parser.add_argument("--input_pt", default="/mnt/sharefs/users/hao.zhang/DMD/wan_bidirectional_dmd_from_scratch/2025-06-20-08-17-06.607828_seed1024/checkpoint_model_004800/model.pt", help="Path to input .pt/.pth checkpoint")
    parser.add_argument("-o", "--output", default="./dmd.safetensors", help="Output .safetensors path")
    args = parser.parse_args()

    output_path = args.output or args.input_pt.rsplit(".", 1)[0] + ".safetensors"
    convert(args.input_pt, output_path)