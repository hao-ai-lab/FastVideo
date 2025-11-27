import json
import shutil
from pathlib import Path
from typing import Dict, Any

import torch
from safetensors.torch import save_file, load_file
import re
from collections import OrderedDict

from transformers import CLIPImageProcessor, CLIPVisionConfig, CLIPVisionModel

SOURCE_DIR = "/workspace/Matrix-Game-2.0"
OUTPUT_DIR = "/workspace/Matrix-Game-2.0-Diffusers"
# MODEL_VARIANT = "base_model"
MODEL_VARIANT = "base_distilled_model"

_param_names_mapping = {
    r"^head\.modulation": r"scale_shift_table",
    r"^head\.head\.(.*)$": r"proj_out.\1",
    r"^img_emb\.proj\.0\.(.*)$": r"condition_embedder.image_embedder.norm1.\1",
    r"^img_emb\.proj\.1\.(.*)$": r"condition_embedder.image_embedder.ff.net.0.proj.\1",
    r"^img_emb\.proj\.3\.(.*)$": r"condition_embedder.image_embedder.ff.net.2.\1",
    r"^img_emb\.proj\.4\.(.*)$": r"condition_embedder.image_embedder.norm2.\1",
    r"^time_embedding\.0\.(.*)$": r"condition_embedder.time_embedder.linear_1.\1",
    r"^time_embedding\.2\.(.*)$": r"condition_embedder.time_embedder.linear_2.\1",
    r"^time_projection\.1\.(.*)$": r"condition_embedder.time_proj.\1",
    r"^blocks\.(\d+)\.self_attn\.q\.(.*)$": r"blocks.\1.attn1.to_q.\2",
    r"^blocks\.(\d+)\.self_attn\.k\.(.*)$": r"blocks.\1.attn1.to_k.\2",
    r"^blocks\.(\d+)\.self_attn\.v\.(.*)$": r"blocks.\1.attn1.to_v.\2",
    r"^blocks\.(\d+)\.self_attn\.o\.(.*)$": r"blocks.\1.attn1.to_out.0.\2",
    r"^blocks\.(\d+)\.self_attn\.norm_q\.(.*)$": r"blocks.\1.attn1.norm_q.\2",
    r"^blocks\.(\d+)\.self_attn\.norm_k\.(.*)$": r"blocks.\1.attn1.norm_k.\2",
    r"^blocks\.(\d+)\.cross_attn\.q\.(.*)$": r"blocks.\1.attn2.to_q.\2",
    r"^blocks\.(\d+)\.cross_attn\.k\.(.*)$": r"blocks.\1.attn2.to_k.\2",
    r"^blocks\.(\d+)\.cross_attn\.v\.(.*)$": r"blocks.\1.attn2.to_v.\2",
    r"^blocks\.(\d+)\.cross_attn\.o\.(.*)$": r"blocks.\1.attn2.to_out.0.\2",
    r"^blocks\.(\d+)\.cross_attn\.norm_q\.(.*)$": r"blocks.\1.attn2.norm_q.\2",
    r"^blocks\.(\d+)\.cross_attn\.norm_k\.(.*)$": r"blocks.\1.attn2.norm_k.\2",
    r"^blocks\.(\d+)\.ffn\.0\.(.*)$": r"blocks.\1.ffn.net.0.proj.\2",
    r"^blocks\.(\d+)\.ffn\.2\.(.*)$": r"blocks.\1.ffn.net.2.\2",
    r"^blocks\.(\d+)\.modulation": r"blocks.\1.scale_shift_table",
    r"^blocks\.(\d+)\.norm3\.(.*)$": r"blocks.\1.norm2.\2",
}

MODEL_VARIANTS = {
    "base_model": {
        "config_file": "base_config.json",
        "weights_file": "diffusion_pytorch_model.safetensors",
        "description": "base model"
    },
    "base_distilled_model": {
        "config_file": "config.json",
        "weights_file": "base_distill.safetensors",
        "description": "base distilled model"
    },
    "gta_distilled_model": {
        "config_file": "config.json",
        "weights_file": "gta_keyboard2dim.safetensors",
        "description": "GTA distilled model"
    },
    "templerun_distilled_model": {
        "config_file": "config.json",
        "weights_file": "templerun_7dim_onlykey.safetensors",
        "description": "Temple Run distilled model"
    }
}


def create_model_index_json(action_config: Dict[str, Any]) -> Dict[str, Any]:
    return {
        # "_class_name": "MatrixGamePipeline",
        "_class_name": "MatrixCausalGameDMDPipeline",
        "_diffusers_version": "0.33.1",
        "scheduler": ["diffusers", "SelfForcingFlowMatchScheduler"],
        "transformer": ["diffusers", "MatrixGameWanModel"],
        "vae": ["diffusers", "AutoencoderKLWan"],
        "image_encoder": ["transformers", "CLIPVisionModel"],
        "image_processor": ["transformers", "CLIPImageProcessor"]
    }


def create_transformer_config(source_config: Dict[str, Any]) -> Dict[str, Any]:
    dim = source_config["dim"]
    num_heads = source_config["num_heads"]

    transformer_config = {
        # "_class_name": "MatrixGameWanModel",
        "_class_name": "CausalMatrixGameWanModel",
        "_diffusers_version": source_config.get("_diffusers_version", "0.33.1"),
        "hidden_size": dim,
        "num_attention_heads": num_heads,
        "attention_head_dim": dim // num_heads,
        "in_channels": source_config.get("in_dim", 36),
        "out_channels": source_config.get("out_dim", 16),
        "num_layers": source_config["num_layers"],
        "ffn_dim": source_config["ffn_dim"],
        "freq_dim": source_config.get("freq_dim", 256),
        "eps": source_config.get("eps", 1e-06),
        "qk_norm": source_config.get("qk_norm", "rms_norm_across_heads"),
        "patch_size": source_config["action_config"].get("patch_size", [1, 2, 2]) if "action_config" in source_config else [1, 2, 2],
        "action_config": source_config.get("action_config"),
        "image_dim": 1280,
        "text_dim": 0,
    }

    # if transformer_config["action_config"] and transformer_config["action_config"].get("keyboard_dim_in") == 4:
        # transformer_config["action_config"]["keyboard_dim_in"] = 6

    return {k: v for k, v in transformer_config.items() if v is not None}


def create_vae_config() -> Dict[str, Any]:
    return {
        "_class_name": "AutoencoderKLWan",
        "_diffusers_version": "0.33.1",
        "attn_scales": [],
        "base_dim": 96,
        "dim_mult": [1, 2, 4, 4],
        "dropout": 0.0,
        "latents_mean": [
            -0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508,
            0.4134, -0.0715, 0.5517, -0.3632, -0.1922, -0.9497, 0.2503, -0.2921
        ],
        "latents_std": [
            2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743,
            3.2687, 2.1526, 2.8652, 1.5579, 1.6382, 1.1253, 2.8251, 1.916
        ],
        "num_res_blocks": 2,
        "temperal_downsample": [False, True, True],
        "z_dim": 16
    }


def create_scheduler_config() -> Dict[str, Any]:
    return {
        "_class_name": "SelfForcingFlowMatchScheduler",
        "_diffusers_version": "0.33.1",
        "num_train_timesteps": 1000,
        "num_inference_steps": 1000,
        "shift": 5.0,
        "sigma_max": 1.0,
        "sigma_min": 0.0,
        "inverse_timesteps": False,
        "extra_one_step": True,
        "reverse_sigmas": False,
        "training": True
    }


def convert_vae_param_name(key: str) -> str:
    if key.startswith("conv1."):
        return key.replace("conv1.", "quant_conv.")
    if key.startswith("conv2."):
        return key.replace("conv2.", "post_quant_conv.")

    if key.startswith("encoder."):
        if key.startswith("encoder.conv1."):
            key = key.replace("encoder.conv1.", "encoder.conv_in.")
        elif key.startswith("encoder.head."):
            if ".0.gamma" in key:
                key = key.replace("encoder.head.0.gamma", "encoder.norm_out.gamma")
            elif ".2." in key:
                key = key.replace("encoder.head.2.", "encoder.conv_out.")
        elif key.startswith("encoder.downsamples."):
            key = key.replace("encoder.downsamples.", "encoder.down_blocks.")
        elif key.startswith("encoder.middle."):
            key = key.replace("encoder.middle.", "encoder.mid_block.")
            if ".0.residual." in key:
                key = key.replace(".0.residual.", ".resnets.0.")
            elif ".2.residual." in key:
                key = key.replace(".2.residual.", ".resnets.1.")
            if ".1." in key:
                key = key.replace("mid_block.1.", "mid_block.attentions.0.")
            if ".resnets.0.0.gamma" in key or ".resnets.1.0.gamma" in key:
                key = key.replace(".0.gamma", ".norm1.gamma")
            elif ".resnets.0.2." in key or ".resnets.1.2." in key:
                key = key.replace(".2.", ".conv1.")
            elif ".resnets.0.3.gamma" in key or ".resnets.1.3.gamma" in key:
                key = key.replace(".3.gamma", ".norm2.gamma")
            elif ".resnets.0.6." in key or ".resnets.1.6." in key:
                key = key.replace(".6.", ".conv2.")
            if ".shortcut." in key:
                key = key.replace(".shortcut.", ".conv_shortcut.")

    if key.startswith("decoder."):
        if key.startswith("decoder.conv1."):
            key = key.replace("decoder.conv1.", "decoder.conv_in.")
        elif key.startswith("decoder.head."):
            if ".0.gamma" in key:
                key = key.replace("decoder.head.0.gamma", "decoder.norm_out.gamma")
            elif ".2." in key:
                key = key.replace("decoder.head.2.", "decoder.conv_out.")
        elif key.startswith("decoder.upsamples."):
            parts = key.split(".")
            idx = int(parts[2])
            suffix = ".".join(parts[3:])

            if "residual.0.gamma" in suffix:
                suffix = suffix.replace("residual.0.gamma", "norm1.gamma")
            elif "residual.2." in suffix:
                suffix = suffix.replace("residual.2.", "conv1.")
            elif "residual.3.gamma" in suffix:
                suffix = suffix.replace("residual.3.gamma", "norm2.gamma")
            elif "residual.6." in suffix:
                suffix = suffix.replace("residual.6.", "conv2.")
            if "shortcut." in suffix:
                suffix = suffix.replace("shortcut.", "conv_shortcut.")

            if idx < 12:
                stage = idx // 4
                offset = idx % 4
                if offset < 3:
                    new_key = f"decoder.up_blocks.{stage}.resnets.{offset}.{suffix}"
                else:
                    new_key = f"decoder.up_blocks.{stage}.upsamplers.0.{suffix}"
            else:
                stage = 3
                offset = idx - 12
                new_key = f"decoder.up_blocks.{stage}.resnets.{offset}.{suffix}"

            return new_key
        elif key.startswith("decoder.middle."):
            key = key.replace("decoder.middle.", "decoder.mid_block.")
            if ".0.residual." in key:
                key = key.replace(".0.residual.", ".resnets.0.")
            elif ".2.residual." in key:
                key = key.replace(".2.residual.", ".resnets.1.")
            if ".1." in key:
                key = key.replace("mid_block.1.", "mid_block.attentions.0.")
            if ".resnets.0.0.gamma" in key or ".resnets.1.0.gamma" in key:
                key = key.replace(".0.gamma", ".norm1.gamma")
            elif ".resnets.0.2." in key or ".resnets.1.2." in key:
                key = key.replace(".2.", ".conv1.")
            elif ".resnets.0.3.gamma" in key or ".resnets.1.3.gamma" in key:
                key = key.replace(".3.gamma", ".norm2.gamma")
            elif ".resnets.0.6." in key or ".resnets.1.6." in key:
                key = key.replace(".6.", ".conv2.")
            if ".shortcut." in key:
                key = key.replace(".shortcut.", ".conv_shortcut.")

    if ".residual.0.gamma" in key:
        key = key.replace(".residual.0.gamma", ".norm1.gamma")
    elif ".residual.2." in key:
        key = key.replace(".residual.2.", ".conv1.")
    elif ".residual.3.gamma" in key:
        key = key.replace(".residual.3.gamma", ".norm2.gamma")
    elif ".residual.6." in key:
        key = key.replace(".residual.6.", ".conv2.")
    if ".shortcut." in key:
        key = key.replace(".shortcut.", ".conv_shortcut.")

    return key


def convert_vae_weights(source_vae_path: str, output_path: str):
    state_dict = torch.load(source_vae_path, map_location="cpu")
    if isinstance(state_dict, dict) and "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]

    converted_state_dict = OrderedDict()
    for orig_key, value in state_dict.items():
        new_key = convert_vae_param_name(orig_key)
        converted_state_dict[new_key] = value

    save_file(converted_state_dict, output_path)


def convert_openclip_to_hf_clip(openclip_state: Dict[str, Any],
                                hf_model: CLIPVisionModel) -> Dict[str, torch.Tensor]:
    hidden_size = hf_model.config.hidden_size
    new_state: Dict[str, torch.Tensor] = {}

    def _tensor(name: str) -> torch.Tensor:
        tensor = openclip_state[name]
        if isinstance(tensor, torch.Tensor):
            return tensor
        return torch.tensor(tensor)

    new_state["vision_model.embeddings.class_embedding"] = _tensor(
        "visual.cls_embedding").squeeze(0).squeeze(0)
    new_state["vision_model.embeddings.patch_embedding.weight"] = _tensor(
        "visual.patch_embedding.weight")
    new_state["vision_model.embeddings.position_embedding.weight"] = (
        _tensor("visual.pos_embedding").squeeze(0))

    new_state["vision_model.pre_layrnorm.weight"] = _tensor(
        "visual.pre_norm.weight")
    new_state["vision_model.pre_layrnorm.bias"] = _tensor(
        "visual.pre_norm.bias")
    new_state["vision_model.post_layernorm.weight"] = _tensor(
        "visual.post_norm.weight")
    new_state["vision_model.post_layernorm.bias"] = _tensor(
        "visual.post_norm.bias")

    for idx in range(hf_model.config.num_hidden_layers):
        src_prefix = f"visual.transformer.{idx}"
        dst_prefix = f"vision_model.encoder.layers.{idx}"

        qkv_weight = _tensor(f"{src_prefix}.attn.to_qkv.weight")
        qkv_bias = _tensor(f"{src_prefix}.attn.to_qkv.bias")
        q_w, k_w, v_w = torch.split(qkv_weight, hidden_size, dim=0)
        q_b, k_b, v_b = torch.split(qkv_bias, hidden_size, dim=0)
        new_state[f"{dst_prefix}.self_attn.q_proj.weight"] = q_w
        new_state[f"{dst_prefix}.self_attn.q_proj.bias"] = q_b
        new_state[f"{dst_prefix}.self_attn.k_proj.weight"] = k_w
        new_state[f"{dst_prefix}.self_attn.k_proj.bias"] = k_b
        new_state[f"{dst_prefix}.self_attn.v_proj.weight"] = v_w
        new_state[f"{dst_prefix}.self_attn.v_proj.bias"] = v_b
        new_state[f"{dst_prefix}.self_attn.out_proj.weight"] = _tensor(
            f"{src_prefix}.attn.proj.weight")
        new_state[f"{dst_prefix}.self_attn.out_proj.bias"] = _tensor(
            f"{src_prefix}.attn.proj.bias")

        new_state[f"{dst_prefix}.layer_norm1.weight"] = _tensor(
            f"{src_prefix}.norm1.weight")
        new_state[f"{dst_prefix}.layer_norm1.bias"] = _tensor(
            f"{src_prefix}.norm1.bias")
        new_state[f"{dst_prefix}.layer_norm2.weight"] = _tensor(
            f"{src_prefix}.norm2.weight")
        new_state[f"{dst_prefix}.layer_norm2.bias"] = _tensor(
            f"{src_prefix}.norm2.bias")

        new_state[f"{dst_prefix}.mlp.fc1.weight"] = _tensor(
            f"{src_prefix}.mlp.0.weight")
        new_state[f"{dst_prefix}.mlp.fc1.bias"] = _tensor(
            f"{src_prefix}.mlp.0.bias")
        new_state[f"{dst_prefix}.mlp.fc2.weight"] = _tensor(
            f"{src_prefix}.mlp.2.weight")
        new_state[f"{dst_prefix}.mlp.fc2.bias"] = _tensor(
            f"{src_prefix}.mlp.2.bias")

    return new_state


def setup_image_encoder(source_dir: Path, output_dir: Path, overwrite: bool = True):
    image_encoder_dir = output_dir / "image_encoder"
    image_processor_dir = output_dir / "image_processor"

    if overwrite:
        shutil.rmtree(image_encoder_dir, ignore_errors=True)
        shutil.rmtree(image_processor_dir, ignore_errors=True)

    if image_encoder_dir.exists() and image_processor_dir.exists():
        return

    openclip_ckpt = source_dir / "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"
    if not openclip_ckpt.exists():
        raise FileNotFoundError(f"OpenCLIP checkpoint not found: {openclip_ckpt}")

    openclip_state = torch.load(openclip_ckpt, map_location="cpu")
    if isinstance(openclip_state, dict) and "state_dict" in openclip_state:
        openclip_state = openclip_state["state_dict"]

    model_id = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
    config = CLIPVisionConfig.from_pretrained(model_id)
    image_encoder = CLIPVisionModel(config)
    converted_state = convert_openclip_to_hf_clip(openclip_state, image_encoder)
    image_encoder.load_state_dict(converted_state, strict=True)
    image_encoder.save_pretrained(image_encoder_dir)

    config_path = image_encoder_dir / "config.json"
    with open(config_path, 'r') as f:
        saved_config = json.load(f)

    allowed_fields = {
        "architectures", "hidden_size", "intermediate_size", "projection_dim",
        "num_hidden_layers", "num_attention_heads", "num_channels", "image_size",
        "patch_size", "hidden_act", "layer_norm_eps", "dropout", "attention_dropout",
        "initializer_range", "initializer_factor", "output_hidden_states",
        "use_return_dict", "_name_or_path", "transformers_version", "model_type",
        "torch_dtype"
    }

    cleaned_config = {k: v for k, v in saved_config.items() if k in allowed_fields}

    with open(config_path, 'w') as f:
        json.dump(cleaned_config, f, indent=2)

    image_processor = CLIPImageProcessor.from_pretrained(model_id)
    image_processor.save_pretrained(image_processor_dir)


def convert_checkpoint(
    model_variant: str,
    source_dir: str,
    output_dir: str,
    use_symlink: bool = True
):
    source_path = Path(source_dir)
    output_path = Path(output_dir) / model_variant

    if model_variant not in MODEL_VARIANTS:
        raise ValueError(
            f"Unknown model variant: {model_variant}. "
            f"Choose from: {list(MODEL_VARIANTS.keys())}"
        )

    variant_info = MODEL_VARIANTS[model_variant]
    model_source_dir = source_path / model_variant
    config_path = model_source_dir / variant_info["config_file"]
    weights_path = model_source_dir / variant_info["weights_file"]
    vae_source_path = source_path / "Wan2.1_VAE.pth"

    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    if not weights_path.exists():
        raise FileNotFoundError(f"Weights not found: {weights_path}")
    if not vae_source_path.exists():
        raise FileNotFoundError(f"VAE not found: {vae_source_path}")

    with open(config_path) as f:
        source_config = json.load(f)

    output_path.mkdir(parents=True, exist_ok=True)

    model_index = create_model_index_json(source_config.get("action_config"))
    with open(output_path / "model_index.json", "w") as f:
        json.dump(model_index, f, indent=2)

    transformer_dir = output_path / "transformer"
    transformer_dir.mkdir(parents=True, exist_ok=True)
    transformer_config = create_transformer_config(source_config)
    with open(transformer_dir / "config.json", "w") as f:
        json.dump(transformer_config, f, indent=2)

    state_dict = load_file(weights_path)
    new_state_dict = OrderedDict()

    for raw_key, v in state_dict.items():
        if raw_key.startswith("model."):
            key = raw_key[6:]
        else:
            key = raw_key

        new_key = key
        for pattern, replacement in _param_names_mapping.items():
            if re.match(pattern, key):
                new_key = re.sub(pattern, replacement, key)
                break

        new_state_dict[new_key] = v

    save_file(new_state_dict, transformer_dir / "diffusion_pytorch_model.safetensors")

    vae_dir = output_path / "vae"
    vae_dir.mkdir(parents=True, exist_ok=True)
    vae_config = create_vae_config()
    with open(vae_dir / "config.json", "w") as f:
        json.dump(vae_config, f, indent=2)
    convert_vae_weights(str(vae_source_path), str(vae_dir / "diffusion_pytorch_model.safetensors"))

    setup_image_encoder(source_path, output_path)

    scheduler_dir = output_path / "scheduler"
    scheduler_dir.mkdir(parents=True, exist_ok=True)
    scheduler_config = create_scheduler_config()
    with open(scheduler_dir / "scheduler_config.json", "w") as f:
        json.dump(scheduler_config, f, indent=2)

    return output_path


def main():
    output_path = convert_checkpoint(
        model_variant=MODEL_VARIANT,
        source_dir=SOURCE_DIR,
        output_dir=OUTPUT_DIR,
        use_symlink=True
    )

    print(f"output saved to: {output_path}")


if __name__ == "__main__":
    main()
