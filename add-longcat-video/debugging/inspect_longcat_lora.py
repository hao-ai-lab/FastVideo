#!/usr/bin/env python3
"""
Inspect LongCat LoRA files to understand naming conventions.
"""

import sys
from pathlib import Path
from safetensors.torch import load_file

def inspect_lora_file(lora_path: str):
    """Inspect a LoRA safetensors file."""
    print(f"\n{'='*80}")
    print(f"Inspecting: {lora_path}")
    print(f"{'='*80}\n")
    
    if not Path(lora_path).exists():
        print(f"❌ File not found: {lora_path}")
        return
    
    try:
        lora_dict = load_file(lora_path)
    except Exception as e:
        print(f"❌ Error loading file: {e}")
        return
    
    print(f"Total keys: {len(lora_dict)}")
    print(f"\n{'─'*80}")
    print("Key patterns:")
    print(f"{'─'*80}\n")
    
    # Group keys by pattern
    patterns = {}
    for key in sorted(lora_dict.keys()):
        # Extract pattern (remove numbers and weights)
        pattern = key
        # Replace block numbers with X
        import re
        pattern = re.sub(r'blocks\.(\d+)', 'blocks.X', pattern)
        pattern = re.sub(r'___lorahyphen___(\d+)___lorahyphen___', '___lorahyphen___X___lorahyphen___', pattern)
        
        if pattern not in patterns:
            patterns[pattern] = []
        patterns[pattern].append(key)
    
    # Print unique patterns
    print(f"Unique patterns: {len(patterns)}\n")
    for pattern in sorted(patterns.keys())[:30]:  # Show first 30 patterns
        example_key = patterns[pattern][0]
        shape = lora_dict[example_key].shape
        count = len(patterns[pattern])
        print(f"  [{count:3d}x] {pattern}")
        print(f"         Example: {example_key}")
        print(f"         Shape: {shape}")
        print()
    
    # Print some actual keys for detailed inspection
    print(f"\n{'─'*80}")
    print("Sample actual keys (first 20):")
    print(f"{'─'*80}\n")
    for key in sorted(lora_dict.keys())[:20]:
        shape = lora_dict[key].shape
        print(f"  {key}")
        print(f"    Shape: {shape}")


def main():
    # Default paths (adjust based on your setup)
    lora_dir = Path("/mnt/fast-disks/hao_lab/shao/LongCat-Video/weights/LongCat-Video/lora")
    
    cfg_step_lora = lora_dir / "cfg_step_lora.safetensors"
    refinement_lora = lora_dir / "refinement_lora.safetensors"
    
    # Allow command line override
    if len(sys.argv) > 1:
        cfg_step_lora = Path(sys.argv[1])
    if len(sys.argv) > 2:
        refinement_lora = Path(sys.argv[2])
    
    # Inspect both files
    inspect_lora_file(str(cfg_step_lora))
    inspect_lora_file(str(refinement_lora))
    
    print(f"\n{'='*80}")
    print("Summary")
    print(f"{'='*80}\n")
    print("Next steps:")
    print("1. Identify the naming pattern (e.g., 'lora___lorahyphen___module___lorahyphen___name')")
    print("2. Update lora_param_names_mapping in LongCatVideoArchConfig")
    print("3. Create regex patterns to map LoRA keys to FastVideo layer names")
    print()


if __name__ == "__main__":
    main()





"""
Inspect LongCat LoRA files to understand naming conventions.
"""

import sys
from pathlib import Path
from safetensors.torch import load_file

def inspect_lora_file(lora_path: str):
    """Inspect a LoRA safetensors file."""
    print(f"\n{'='*80}")
    print(f"Inspecting: {lora_path}")
    print(f"{'='*80}\n")
    
    if not Path(lora_path).exists():
        print(f"❌ File not found: {lora_path}")
        return
    
    try:
        lora_dict = load_file(lora_path)
    except Exception as e:
        print(f"❌ Error loading file: {e}")
        return
    
    print(f"Total keys: {len(lora_dict)}")
    print(f"\n{'─'*80}")
    print("Key patterns:")
    print(f"{'─'*80}\n")
    
    # Group keys by pattern
    patterns = {}
    for key in sorted(lora_dict.keys()):
        # Extract pattern (remove numbers and weights)
        pattern = key
        # Replace block numbers with X
        import re
        pattern = re.sub(r'blocks\.(\d+)', 'blocks.X', pattern)
        pattern = re.sub(r'___lorahyphen___(\d+)___lorahyphen___', '___lorahyphen___X___lorahyphen___', pattern)
        
        if pattern not in patterns:
            patterns[pattern] = []
        patterns[pattern].append(key)
    
    # Print unique patterns
    print(f"Unique patterns: {len(patterns)}\n")
    for pattern in sorted(patterns.keys())[:30]:  # Show first 30 patterns
        example_key = patterns[pattern][0]
        shape = lora_dict[example_key].shape
        count = len(patterns[pattern])
        print(f"  [{count:3d}x] {pattern}")
        print(f"         Example: {example_key}")
        print(f"         Shape: {shape}")
        print()
    
    # Print some actual keys for detailed inspection
    print(f"\n{'─'*80}")
    print("Sample actual keys (first 20):")
    print(f"{'─'*80}\n")
    for key in sorted(lora_dict.keys())[:20]:
        shape = lora_dict[key].shape
        print(f"  {key}")
        print(f"    Shape: {shape}")


def main():
    # Default paths (adjust based on your setup)
    lora_dir = Path("/mnt/fast-disks/hao_lab/shao/LongCat-Video/weights/LongCat-Video/lora")
    
    cfg_step_lora = lora_dir / "cfg_step_lora.safetensors"
    refinement_lora = lora_dir / "refinement_lora.safetensors"
    
    # Allow command line override
    if len(sys.argv) > 1:
        cfg_step_lora = Path(sys.argv[1])
    if len(sys.argv) > 2:
        refinement_lora = Path(sys.argv[2])
    
    # Inspect both files
    inspect_lora_file(str(cfg_step_lora))
    inspect_lora_file(str(refinement_lora))
    
    print(f"\n{'='*80}")
    print("Summary")
    print(f"{'='*80}\n")
    print("Next steps:")
    print("1. Identify the naming pattern (e.g., 'lora___lorahyphen___module___lorahyphen___name')")
    print("2. Update lora_param_names_mapping in LongCatVideoArchConfig")
    print("3. Create regex patterns to map LoRA keys to FastVideo layer names")
    print()


if __name__ == "__main__":
    main()








