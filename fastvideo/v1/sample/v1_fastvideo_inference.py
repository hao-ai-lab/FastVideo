import os


import imageio
import numpy as np
import torch
import torchvision
from einops import rearrange
import sys
# Fix the import path
from fastvideo.v1.inference_engine import InferenceEngine
from fastvideo.v1.inference_args import InferenceArgs
from fastvideo.v1.inference_args import prepare_inference_args
from fastvideo.v1.distributed import init_distributed_environment, initialize_model_parallel
from fastvideo.v1.logger import logger
from fastvideo.v1.models.loader.fsdp_load import get_param_names_mapping
from safetensors.torch import safe_open, save_file
import shutil

def initialize_distributed_and_parallelism(inference_args: InferenceArgs):
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    torch.cuda.set_device(local_rank)
    init_distributed_environment(
        world_size=world_size,
        rank=rank,
        local_rank=local_rank
    )
    device_str = f"cuda:{local_rank}"
    inference_args.device_str = device_str
    inference_args.device = torch.device(device_str)
    initialize_model_parallel(
        sequence_model_parallel_size=inference_args.sp_size,
        tensor_model_parallel_size=inference_args.tp_size,
    )
    
def convert_and_save_lora_weights(input_folder, output_folder, lora_param_mapping_fn):

    # Check if the input folder exists
    if not os.path.exists(input_folder):
        raise ValueError(f"Input folder {input_folder} not found")
    
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Process safetensors file
    safetensors_file = os.path.join(input_folder, "pytorch_lora_weights.safetensors")
    if os.path.exists(safetensors_file):
        # Load all tensors from the safetensors file
        tensors = {}
        with safe_open(safetensors_file, framework="pt", device="cpu") as f:
            tensor_names = f.keys()
            print(f"Found {len(tensor_names)} tensors in {safetensors_file}")
            
            for name in tensor_names:
                tensors[name] = f.get_tensor(name)
        
        # Extract base names (without LoRA suffix) and LoRA suffixes
        base_names_dict = {}
        lora_suffixes = {}
        
        for name in tensor_names:
            if name.endswith(".lora_A.weight") or name.endswith(".lora_B.weight"):
                for suffix in [".lora_A.weight", ".lora_B.weight"]:
                    if name.endswith(suffix):
                        base_name = name[:-len(suffix)] + ".weight"
                        base_names_dict[name] = base_name
                        lora_suffixes[name] = suffix
                        break
            else:
                base_names_dict[name] = name
                lora_suffixes[name] = ""
        
        # Apply the mapping function to get all mappings at once
        name_mappings = {}
        for full_name, base_name in base_names_dict.items():
            try:
                # First remove "transformer." prefix if it exists
                has_transformer_prefix = False
                processed_base_name = base_name
                if base_name.startswith("transformer."):
                    processed_base_name = base_name[12:]  # Remove "transformer." (12 characters)
                    has_transformer_prefix = True
                
                # Apply the mapping function to the processed name
                target_base_name, merge_index, total_splitted_params = lora_param_mapping_fn(processed_base_name)
                
                # Add back the "transformer." prefix if it was removed
                if target_base_name is not None and has_transformer_prefix:
                    target_base_name = "transformer." + target_base_name
                
                if target_base_name is not None:
                    name_mappings[full_name] = (target_base_name, merge_index, total_splitted_params)
            except Exception as e:
                print(f"Error mapping parameter {base_name}: {e}")
        
        # Create the converted weights using the mappings
        converted_weights = {}
        for name, tensor in tensors.items():
            suffix = lora_suffixes.get(name, "")
            
            if name in name_mappings:
                target_base_name, merge_index, _ = name_mappings[name]
                
                # Reconstruct full parameter name with LoRA suffix
                if target_base_name.endswith('.weight'):
                    target_base_name_without_suffix = target_base_name[:-7]  # 移除 .weight
                    new_key = target_base_name_without_suffix + suffix
                else:
                    new_key = target_base_name + suffix
                
                if merge_index is not None:
                    new_key += f"_{merge_index}"
                
                converted_weights[new_key] = tensor
            else:
                # Keep original name if no mapping was found
                converted_weights[name] = tensor
        
        print(f"Converted {len(converted_weights)} parameters")
        
        # Save the converted weights
        output_safetensors_file = os.path.join(output_folder, "pytorch_lora_weights.safetensors")
        save_file(converted_weights, output_safetensors_file)
        print(f"Converted weights saved to {output_safetensors_file}")
    else:
        print(f"Warning: No safetensors file found in {input_folder}")
    
    # Copy other files (like config and optimizer state) without modification
    for filename in os.listdir(input_folder):
        if filename != "pytorch_lora_weights.safetensors":
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)
            
            if os.path.isfile(input_path):
                shutil.copy2(input_path, output_path)
                print(f"Copied {filename} to output folder")
    
    print(f"LoRA conversion complete: {input_folder} → {output_folder}")

def main(inference_args: InferenceArgs):
    initialize_distributed_and_parallelism(inference_args)
    engine = InferenceEngine.create_engine(
        inference_args,
    )
    
    if inference_args.prompt_path is not None:
        with open(inference_args.prompt_path) as f:
            prompts = [line.strip() for line in f.readlines()]
    else:
        prompts = [inference_args.prompt]
    # from IPython import embed; embed()
    
    # # convert lora format
    # input_path = "data/Hunyuan-Black-Myth-Wukong-lora-weight/"
    # output_path = "data/Hunyuan-Black-Myth-Wukong-lora-weight_converted"
    # # from IPython import embed; embed()
    # lora_param_mapping_fn = get_param_names_mapping(engine.pipeline.transformer._param_names_mapping)
    # converted_weights = convert_and_save_lora_weights(
    #     input_path,
    #     output_path,
    #     lora_param_mapping_fn,
    # )
    
    # Process each prompt
    for prompt in prompts:
        # lora_checkpoint = "data/Hunyuan-Black-Myth-Wukong-lora-weight_converted/"
        # if lora_checkpoint:
        #     import json
        #     print(f"Loading LoRA weights from lora_checkpoint: {lora_checkpoint}")
        #     config_path = os.path.join(lora_checkpoint, "lora_config.json")
        #     with open(config_path, "r") as f:
        #         lora_config_dict = json.load(f)
        #     rank = lora_config_dict["lora_params"]["lora_rank"]
        #     lora_alpha = lora_config_dict["lora_params"]["lora_alpha"]
        #     lora_scaling = lora_alpha / rank
        #     engine.pipeline.load_lora_weights(lora_checkpoint, adapter_name="default")
        #     from IPython import embed; embed()
        #     engine.pipeline.set_adapters(["default"], [lora_scaling])
        #     print(f"Successfully Loaded LoRA weights from {lora_checkpoint}")
        outputs = engine.run(
            prompt=prompt,
            inference_args=inference_args,
        )
        # img_attn_qkv, img_attn_proj, linear1, linear2
        # Process outputs
        videos = rearrange(outputs["samples"], "b c t h w -> t b c h w")
        frames = []
        for x in videos:
            x = torchvision.utils.make_grid(x, nrow=6)
            x = x.transpose(0, 1).transpose(1, 2).squeeze(-1)
            frames.append((x * 255).numpy().astype(np.uint8))
            
        # Save video
        os.makedirs(os.path.dirname(inference_args.output_path), exist_ok=True)
        imageio.mimsave(
            os.path.join(inference_args.output_path, f"{prompt[:100]}.mp4"), 
            frames, 
            fps=inference_args.fps
        )


if __name__ == "__main__":
    inference_args = prepare_inference_args(sys.argv[1:])
    main(inference_args)