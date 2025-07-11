# convert_and_save_transformer.py

import pathlib
from typing import Union, Optional
from safetensors.torch import load_file, save_file
from huggingface_hub import snapshot_download

import argparse
import pathlib
from typing import Any, Dict, Tuple

import torch
from accelerate import init_empty_weights
from huggingface_hub import hf_hub_download, snapshot_download
from safetensors.torch import load_file
from transformers import AutoProcessor, AutoTokenizer, CLIPVisionModelWithProjection, UMT5EncoderModel

from diffusers import (
    AutoencoderKLWan,
    UniPCMultistepScheduler,
    WanImageToVideoPipeline,
    WanPipeline,
    WanTransformer3DModel,
)


TRANSFORMER_KEYS_RENAME_DICT = {
    "time_embedding.0": "condition_embedder.time_embedder.linear_1",
    "time_embedding.2": "condition_embedder.time_embedder.linear_2",
    "text_embedding.0": "condition_embedder.text_embedder.linear_1",
    "text_embedding.2": "condition_embedder.text_embedder.linear_2",
    "time_projection.1": "condition_embedder.time_proj",
    "head.modulation": "scale_shift_table",
    "head.head": "proj_out",
    "modulation": "scale_shift_table",
    "ffn.0": "ffn.net.0.proj",
    "ffn.2": "ffn.net.2",
    # Hack to swap the layer names
    # The original model calls the norms in following order: norm1, norm3, norm2
    # We convert it to: norm1, norm2, norm3
    "norm2": "norm__placeholder",
    "norm3": "norm2",
    "norm__placeholder": "norm3",
    # For the I2V model
    "img_emb.proj.0": "condition_embedder.image_embedder.norm1",
    "img_emb.proj.1": "condition_embedder.image_embedder.ff.net.0.proj",
    "img_emb.proj.3": "condition_embedder.image_embedder.ff.net.2",
    "img_emb.proj.4": "condition_embedder.image_embedder.norm2",
    # for the FLF2V model
    "img_emb.emb_pos": "condition_embedder.image_embedder.pos_embed",
    # Add attention component mappings
    "self_attn.q": "attn1.to_q",
    "self_attn.k": "attn1.to_k",
    "self_attn.v": "attn1.to_v",
    "self_attn.o": "attn1.to_out.0",
    "self_attn.norm_q": "attn1.norm_q",
    "self_attn.norm_k": "attn1.norm_k",
    "cross_attn.q": "attn2.to_q",
    "cross_attn.k": "attn2.to_k",
    "cross_attn.v": "attn2.to_v",
    "cross_attn.o": "attn2.to_out.0",
    "cross_attn.norm_q": "attn2.norm_q",
    "cross_attn.norm_k": "attn2.norm_k",
    "attn2.to_k_img": "attn2.add_k_proj",
    "attn2.to_v_img": "attn2.add_v_proj",
    "attn2.norm_k_img": "attn2.norm_added_k",
}

TRANSFORMER_SPECIAL_KEYS_REMAP = {}
VACE_TRANSFORMER_SPECIAL_KEYS_REMAP = {}


def update_state_dict_(state_dict: Dict[str, Any], old_key: str, new_key: str) -> Dict[str, Any]:
    state_dict[new_key] = state_dict.pop(old_key)


def load_sharded_safetensors(dir: pathlib.Path):
    file_paths = list(dir.glob("diffusion_pytorch_model*.safetensors"))
    state_dict = {}
    for path in file_paths:
        state_dict.update(load_file(path))
    return state_dict


def get_transformer_config(model_type: str) -> Tuple[Dict[str, Any], ...]:
    if model_type == "Wan-T2V-1.3B":
        config = {
            "model_id": "StevenZhang/Wan2.1-T2V-1.3B-Diff",
            "diffusers_config": {
                "added_kv_proj_dim": None,
                "attention_head_dim": 128,
                "cross_attn_norm": True,
                "eps": 1e-06,
                "ffn_dim": 8960,
                "freq_dim": 256,
                "in_channels": 16,
                "num_attention_heads": 12,
                "num_layers": 30,
                "out_channels": 16,
                "patch_size": [1, 2, 2],
                "qk_norm": "rms_norm_across_heads",
                "text_dim": 4096,
            },
        }
        RENAME_DICT = TRANSFORMER_KEYS_RENAME_DICT
        SPECIAL_KEYS_REMAP = TRANSFORMER_SPECIAL_KEYS_REMAP
    elif model_type == "Wan-T2V-14B":
        config = {
            "model_id": "StevenZhang/Wan2.1-T2V-14B-Diff",
            "diffusers_config": {
                "added_kv_proj_dim": None,
                "attention_head_dim": 128,
                "cross_attn_norm": True,
                "eps": 1e-06,
                "ffn_dim": 13824,
                "freq_dim": 256,
                "in_channels": 16,
                "num_attention_heads": 40,
                "num_layers": 40,
                "out_channels": 16,
                "patch_size": [1, 2, 2],
                "qk_norm": "rms_norm_across_heads",
                "text_dim": 4096,
            },
        }
        RENAME_DICT = TRANSFORMER_KEYS_RENAME_DICT
        SPECIAL_KEYS_REMAP = TRANSFORMER_SPECIAL_KEYS_REMAP
    elif model_type == "Wan-I2V-14B-480p":
        config = {
            "model_id": "StevenZhang/Wan2.1-I2V-14B-480P-Diff",
            "diffusers_config": {
                "image_dim": 1280,
                "added_kv_proj_dim": 5120,
                "attention_head_dim": 128,
                "cross_attn_norm": True,
                "eps": 1e-06,
                "ffn_dim": 13824,
                "freq_dim": 256,
                "in_channels": 36,
                "num_attention_heads": 40,
                "num_layers": 40,
                "out_channels": 16,
                "patch_size": [1, 2, 2],
                "qk_norm": "rms_norm_across_heads",
                "text_dim": 4096,
            },
        }
        RENAME_DICT = TRANSFORMER_KEYS_RENAME_DICT
        SPECIAL_KEYS_REMAP = TRANSFORMER_SPECIAL_KEYS_REMAP
    elif model_type == "Wan-I2V-14B-720p":
        config = {
            "model_id": "StevenZhang/Wan2.1-I2V-14B-720P-Diff",
            "diffusers_config": {
                "image_dim": 1280,
                "added_kv_proj_dim": 5120,
                "attention_head_dim": 128,
                "cross_attn_norm": True,
                "eps": 1e-06,
                "ffn_dim": 13824,
                "freq_dim": 256,
                "in_channels": 36,
                "num_attention_heads": 40,
                "num_layers": 40,
                "out_channels": 16,
                "patch_size": [1, 2, 2],
                "qk_norm": "rms_norm_across_heads",
                "text_dim": 4096,
            },
        }
        RENAME_DICT = TRANSFORMER_KEYS_RENAME_DICT
        SPECIAL_KEYS_REMAP = TRANSFORMER_SPECIAL_KEYS_REMAP
    elif model_type == "Wan-FLF2V-14B-720P":
        config = {
            "model_id": "ypyp/Wan2.1-FLF2V-14B-720P",  # This is just a placeholder
            "diffusers_config": {
                "image_dim": 1280,
                "added_kv_proj_dim": 5120,
                "attention_head_dim": 128,
                "cross_attn_norm": True,
                "eps": 1e-06,
                "ffn_dim": 13824,
                "freq_dim": 256,
                "in_channels": 36,
                "num_attention_heads": 40,
                "num_layers": 40,
                "out_channels": 16,
                "patch_size": [1, 2, 2],
                "qk_norm": "rms_norm_across_heads",
                "text_dim": 4096,
                "rope_max_seq_len": 1024,
                "pos_embed_seq_len": 257 * 2,
            },
        }
        RENAME_DICT = TRANSFORMER_KEYS_RENAME_DICT
        SPECIAL_KEYS_REMAP = TRANSFORMER_SPECIAL_KEYS_REMAP
    elif model_type == "Wan-VACE-1.3B":
        config = {
            "model_id": "Wan-AI/Wan2.1-VACE-1.3B",
            "diffusers_config": {
                "added_kv_proj_dim": None,
                "attention_head_dim": 128,
                "cross_attn_norm": True,
                "eps": 1e-06,
                "ffn_dim": 8960,
                "freq_dim": 256,
                "in_channels": 16,
                "num_attention_heads": 12,
                "num_layers": 30,
                "out_channels": 16,
                "patch_size": [1, 2, 2],
                "qk_norm": "rms_norm_across_heads",
                "text_dim": 4096,
                "vace_layers": [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28],
                "vace_in_channels": 96,
            },
        }
        RENAME_DICT = VACE_TRANSFORMER_KEYS_RENAME_DICT
        SPECIAL_KEYS_REMAP = VACE_TRANSFORMER_SPECIAL_KEYS_REMAP
    elif model_type == "Wan-VACE-14B":
        config = {
            "model_id": "Wan-AI/Wan2.1-VACE-14B",
            "diffusers_config": {
                "added_kv_proj_dim": None,
                "attention_head_dim": 128,
                "cross_attn_norm": True,
                "eps": 1e-06,
                "ffn_dim": 13824,
                "freq_dim": 256,
                "in_channels": 16,
                "num_attention_heads": 40,
                "num_layers": 40,
                "out_channels": 16,
                "patch_size": [1, 2, 2],
                "qk_norm": "rms_norm_across_heads",
                "text_dim": 4096,
                "vace_layers": [0, 5, 10, 15, 20, 25, 30, 35],
                "vace_in_channels": 96,
            },
        }
        RENAME_DICT = VACE_TRANSFORMER_KEYS_RENAME_DICT
        SPECIAL_KEYS_REMAP = VACE_TRANSFORMER_SPECIAL_KEYS_REMAP
    return config, RENAME_DICT, SPECIAL_KEYS_REMAP

def convert_transformer(
    model_type: str,
    safetensor_path: Optional[Union[str, pathlib.Path]] = None,
    output_path: Optional[Union[str, pathlib.Path]] = None,
):
    config, RENAME_DICT, SPECIAL_KEYS_REMAP = get_transformer_config(model_type)
    diffusers_config = config["diffusers_config"]

    # Load weights from HF or local
    if safetensor_path is None:
        model_dir = pathlib.Path(snapshot_download(config["model_id"], repo_type="model"))
        original_state_dict = load_sharded_safetensors(model_dir)
    else:
        safetensor_path = pathlib.Path(safetensor_path).expanduser().resolve()
        if safetensor_path.is_dir():
            original_state_dict = load_sharded_safetensors(safetensor_path)
        elif safetensor_path.is_file():
            original_state_dict = load_file(str(safetensor_path))
        else:
            raise FileNotFoundError(f"{safetensor_path} is neither a file nor a directory.")

    # Build model skeleton
    with init_empty_weights():
        transformer = (
            WanTransformer3DModel.from_config(diffusers_config)
            if "VACE" not in model_type
            else WanVACETransformer3DModel.from_config(diffusers_config)
        )
    # rectified 
    for key in list(original_state_dict.keys()):
        if key.startswith("model."):
            new_key = key.replace("model.", "")
            original_state_dict[new_key] = original_state_dict.pop(key)
            
    # Rename keys
    for key in list(original_state_dict.keys()):
        key = key.replace("model.", "")
        new_key = key
        for old, new in RENAME_DICT.items():
            new_key = new_key.replace(old, new)
        update_state_dict_(original_state_dict, key, new_key)

    # Special key mapping
    for key in list(original_state_dict.keys()):
        for special_key, handler in SPECIAL_KEYS_REMAP.items():
            if special_key in key:
                handler(key, original_state_dict)

    # Load weights
    transformer.load_state_dict(original_state_dict, strict=True, assign=True)

    # Optionally save
    if output_path is not None:
        output_path = pathlib.Path(output_path).expanduser().resolve()
        state_to_save = {k: v.cpu() for k, v in transformer.state_dict().items()}

        if output_path.suffix == ".safetensors":
            output_path.parent.mkdir(parents=True, exist_ok=True)
            save_file(state_to_save, str(output_path))
        else:
            raise ValueError("Only saving to a single .safetensors file is supported.")

    return transformer


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, default="Wan-T2V-1.3B", help="Model type string")
    parser.add_argument("--safetensor_path", type=str, default="data/Wan2.1-T2V-1.3B/diffusion_pytorch_model.safetensors", help="Local safetensors path or directory")
    parser.add_argument("--output_path", type=str, default="data/dmd_wan/diffusion_pytorch_model.safetensors", help="Optional path to save converted weights")
    args = parser.parse_args()

    convert_transformer(
        model_type=args.model_type,
        safetensor_path=args.safetensor_path,
        output_path=args.output_path,
    )
