import os
import json
import shutil
import argparse
from pathlib import Path


def create_model_index():
    """Create model_index.json for FastVideo.
    
    Note: The first element in each component array is the library source
    ('diffusers' or 'transformers'), not the directory name.
    """
    return {
        "_class_name": "LongCatPipeline",
        "_diffusers_version": "0.32.0",
        "workload_type": "video-generation",
        "tokenizer": ["transformers", "AutoTokenizer"],
        "text_encoder": ["transformers", "UMT5EncoderModel"],
        "vae": ["diffusers", "AutoencoderKLWan"],
        "scheduler": ["diffusers", "FlowMatchEulerDiscreteScheduler"],
        "transformer": ["diffusers", "LongCatVideoTransformer3DModel"]
    }


def prepare_longcat_weights(source_path, output_path):
    """
    Organize LongCat weights for FastVideo.
    
    Args:
        source_path: Path to LongCat-Video/weights/LongCat-Video/
        output_path: Path to output directory
    """
    source = Path(source_path)
    output = Path(output_path)
    
    if not source.exists():
        raise ValueError(f"Source path does not exist: {source}")
    
    print(f"Source: {source}")
    print(f"Output: {output}")
    
    # Create output directory
    output.mkdir(parents=True, exist_ok=True)
    
    # Copy components - map source to target names
    components = {
        "tokenizer": "tokenizer",
        "text_encoder": "text_encoder",
        "vae": "vae",
        "scheduler": "scheduler",
        "dit": "transformer"  # Source is 'dit', target must be 'transformer'
    }
    
    for src_name, dst_name in components.items():
        src = source / src_name
        dst = output / dst_name
        
        if not src.exists():
            print(f"WARNING: {src_name} not found in source, skipping...")
            continue
            
        print(f"Copying {src_name} -> {dst_name}...")
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
    
    # Copy LoRA weights if present
    lora_src = source / "lora"
    if lora_src.exists():
        lora_dst = output / "lora"
        print("Copying lora...")
        if lora_dst.exists():
            shutil.rmtree(lora_dst)
        shutil.copytree(lora_src, lora_dst)
    
    # Create model_index.json
    model_index_path = output / "model_index.json"
    with open(model_index_path, 'w') as f:
        json.dump(create_model_index(), f, indent=2)
    
    print(f"\n✓ Model prepared at: {output}")
    print(f"✓ Created: {model_index_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare LongCat weights for FastVideo")
    parser.add_argument("--source", type=str, required=True,
                        help="Path to LongCat-Video/weights/LongCat-Video/")
    parser.add_argument("--output", type=str, required=True,
                        help="Output directory path")
    
    args = parser.parse_args()
    prepare_longcat_weights(args.source, args.output)

