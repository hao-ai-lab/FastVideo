import argparse
import json
import re
from pathlib import Path

from safetensors.torch import load_file, save_file


MAPPING = {
    r"^patch_embedding\.(.*)$": r"patch_embedding.proj.\1",
    r"^patch_embedding_wancamctrl\.(.*)$": r"patch_embedding_wancamctrl.proj.\1",
    r"^c2ws_hidden_states_layer1\.(.*)$": r"c2ws_mlp.fc_in.\1",
    r"^c2ws_hidden_states_layer2\.(.*)$": r"c2ws_mlp.fc_out.\1",
    r"^text_embedding\.0\.(.*)$": r"condition_embedder.text_embedder.fc_in.\1",
    r"^text_embedding\.2\.(.*)$": r"condition_embedder.text_embedder.fc_out.\1",
    r"^time_embedding\.0\.(.*)$": r"condition_embedder.time_embedder.mlp.fc_in.\1",
    r"^time_embedding\.2\.(.*)$": r"condition_embedder.time_embedder.mlp.fc_out.\1",
    r"^time_projection\.1\.(.*)$": r"condition_embedder.time_modulation.linear.\1",
    r"^blocks\.(\d+)\.self_attn\.q\.(.*)$": r"blocks.\1.to_q.\2",
    r"^blocks\.(\d+)\.self_attn\.k\.(.*)$": r"blocks.\1.to_k.\2",
    r"^blocks\.(\d+)\.self_attn\.v\.(.*)$": r"blocks.\1.to_v.\2",
    r"^blocks\.(\d+)\.self_attn\.o\.(.*)$": r"blocks.\1.to_out.\2",
    r"^blocks\.(\d+)\.self_attn\.norm_q\.(.*)$": r"blocks.\1.norm_q.\2",
    r"^blocks\.(\d+)\.self_attn\.norm_k\.(.*)$": r"blocks.\1.norm_k.\2",
    r"^blocks\.(\d+)\.cross_attn\.q\.(.*)$": r"blocks.\1.attn2.to_q.\2",
    r"^blocks\.(\d+)\.cross_attn\.k\.(.*)$": r"blocks.\1.attn2.to_k.\2",
    r"^blocks\.(\d+)\.cross_attn\.v\.(.*)$": r"blocks.\1.attn2.to_v.\2",
    r"^blocks\.(\d+)\.cross_attn\.o\.(.*)$": r"blocks.\1.attn2.to_out.\2",
    r"^blocks\.(\d+)\.cross_attn\.norm_q\.(.*)$": r"blocks.\1.attn2.norm_q.\2",
    r"^blocks\.(\d+)\.cross_attn\.norm_k\.(.*)$": r"blocks.\1.attn2.norm_k.\2",
    r"^blocks\.(\d+)\.ffn\.0\.(.*)$": r"blocks.\1.ffn.fc_in.\2",
    r"^blocks\.(\d+)\.ffn\.2\.(.*)$": r"blocks.\1.ffn.fc_out.\2",
    r"^blocks\.(\d+)\.modulation$": r"blocks.\1.scale_shift_table",
    r"^blocks\.(\d+)\.norm3\.(.*)$": r"blocks.\1.self_attn_residual_norm.norm.\2",
    r"^blocks\.(\d+)\.cam_injector_layer1\.(.*)$": r"blocks.\1.cam_conditioner.cam_injector.fc_in.\2",
    r"^blocks\.(\d+)\.cam_injector_layer2\.(.*)$": r"blocks.\1.cam_conditioner.cam_injector.fc_out.\2",
    r"^blocks\.(\d+)\.cam_scale_layer\.(.*)$": r"blocks.\1.cam_conditioner.cam_scale_layer.\2",
    r"^blocks\.(\d+)\.cam_shift_layer\.(.*)$": r"blocks.\1.cam_conditioner.cam_shift_layer.\2",
    r"^head\.head\.(.*)$": r"proj_out.\1",
    r"^head\.modulation$": r"scale_shift_table",
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    return parser.parse_args()


def rename_key(key: str):
    for pattern, replacement in MAPPING.items():
        if re.match(pattern, key):
            return re.sub(pattern, replacement, key)
    return None


def convert_one_file(input_file: Path, output_file: Path):
    state_dict = load_file(str(input_file))
    out = {}
    key_map = {}
    for key, tensor in state_dict.items():
        new_key = rename_key(key)
        if new_key is None:
            raise ValueError(f"unmapped key: {key}")
        out[new_key] = tensor
        key_map[key] = new_key
    output_file.parent.mkdir(parents=True, exist_ok=True)
    save_file(out, str(output_file))
    return key_map


def convert_index(input_index: Path, output_index: Path):
    with input_index.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if "weight_map" not in data or not isinstance(data["weight_map"], dict):
        raise ValueError("invalid index json: missing weight_map")
    new_weight_map = {}
    for key, shard in data["weight_map"].items():
        new_key = rename_key(key)
        if new_key is None:
            raise ValueError(f"unmapped key in index: {key}")
        new_weight_map[new_key] = shard
    data["weight_map"] = new_weight_map
    output_index.parent.mkdir(parents=True, exist_ok=True)
    with output_index.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
        f.write("\n")


def main():
    args = parse_args()
    input_path = Path(args.input_path)
    output_path = Path(args.output_path)

    if input_path.is_dir():
        if output_path.suffix:
            raise ValueError(
                "--output_path must be a directory when --input_path is a directory"
            )
        index_name = "diffusion_pytorch_model.safetensors.index.json"
        input_index = input_path / index_name
        if not input_index.exists():
            raise ValueError(f"missing index file: {input_index}")
        with input_index.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if "weight_map" not in data or not isinstance(data["weight_map"], dict):
            raise ValueError("invalid index json: missing weight_map")
        shard_names = sorted(set(data["weight_map"].values()))
        for shard_name in shard_names:
            in_shard = input_path / shard_name
            out_shard = output_path / shard_name
            if not in_shard.exists():
                raise ValueError(f"missing shard file: {in_shard}")
            if in_shard.suffix != ".safetensors":
                raise ValueError(f"non-safetensors shard in index: {in_shard}")
            convert_one_file(in_shard, out_shard)
        convert_index(input_index, output_path / index_name)
        return

    if input_path.suffix != ".safetensors":
        raise ValueError(
            "--input_path must be a .safetensors file or a directory with index json"
        )
    if output_path.suffix != ".safetensors":
        raise ValueError("--output_path must be a .safetensors file")
    convert_one_file(input_path, output_path)


if __name__ == "__main__":
    main()
