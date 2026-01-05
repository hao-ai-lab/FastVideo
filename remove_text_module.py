
import argparse
import os
import torch
import json
from safetensors.torch import load_file, save_file
from tqdm import tqdm

def main(args):
    input_dir = args.input_dir
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # List all safetensors files
    files = [f for f in os.listdir(input_dir) if f.endswith(".safetensors")]
    
    for filename in tqdm(files, desc="Processing files"):
        file_path = os.path.join(input_dir, filename)
        state_dict = load_file(file_path)
        new_state_dict = {}
        
        modified = False
        for k, v in state_dict.items():
            # Remove keys related to text_embedder
            if "condition_embedder.text_embedder" in k:
                print(f"Removing key: {k} from {filename}")
                modified = True
                continue
            new_state_dict[k] = v
        
        if modified or not args.only_save_modified:
            output_path = os.path.join(output_dir, filename)
            save_file(new_state_dict, output_path)
            print(f"Saved processed checkpoint to {output_path}")
        else:
            print(f"No changes in {filename}, skipping save (if only_save_modified=True)")

    # Process config.json
    config_path = os.path.join(input_dir, "config.json")
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)

        if "arch_config" in config: config["arch_config"]["text_dim"] = 0
        if "text_dim" in config: config["text_dim"] = 0
            
        output_config_path = os.path.join(output_dir, "config.json")
        with open(output_config_path, 'w') as f:
            json.dump(config, f, indent=2)
        print(f"Updated config saved to {output_config_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True, help="Input directory containing safetensors files")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory to save modified checkpoints")
    parser.add_argument("--only_save_modified", action="store_true", help="Only save files that were modified")
    args = parser.parse_args()
    main(args)
