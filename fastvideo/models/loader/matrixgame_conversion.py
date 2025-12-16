import json
import os
import re
import shutil
from collections import OrderedDict
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file, save_file

from fastvideo.logger import init_logger

logger = init_logger(__name__)


def _get_cache_dir() -> str:
    """Get the cache directory for MatrixGame models."""
    from huggingface_hub.constants import HF_HUB_CACHE
    return str(Path(HF_HUB_CACHE).parent)

MODEL_VARIANTS: dict[str, dict[str, str]] = {
    "base_model": {
        "config_file": "base_config.json",
        "weights_file": "diffusion_pytorch_model.safetensors",
    },
    "base_distilled_model": {
        "config_file": "config.json",
        "weights_file": "base_distill.safetensors",
    },
    "gta_distilled_model": {
        "config_file": "config.json",
        "weights_file": "gta_keyboard2dim.safetensors",
    },
    "templerun_distilled_model": {
        "config_file": "config.json",
        "weights_file": "templerun_7dim_onlykey.safetensors",
    },
}

_TRANSFORMER_PARAM_MAPPING = {
    r"^patch_embedding\.weight$": r"patch_embedding.proj.weight",
    r"^patch_embedding\.bias$": r"patch_embedding.proj.bias",
    r"^head\.head\.(.*)$": r"proj_out.\1",
    r"^head\.modulation$": r"scale_shift_table",
    r"^img_emb\.proj\.0\.(.*)$": r"condition_embedder.image_embedder.norm1.\1",
    r"^img_emb\.proj\.1\.(.*)$": r"condition_embedder.image_embedder.ff.fc_in.\1",
    r"^img_emb\.proj\.3\.(.*)$": r"condition_embedder.image_embedder.ff.fc_out.\1",
    r"^img_emb\.proj\.4\.(.*)$": r"condition_embedder.image_embedder.norm2.\1",
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
}


def is_matrixgame_original_checkpoint(model_path: str) -> bool:
    """Check if the given path points to an original MatrixGame checkpoint."""
    matrixgame_patterns = ["Matrix-Game-2", "Skywork--Matrix-Game", "matrixgame"]
    if not any(p.lower() in model_path.lower() for p in matrixgame_patterns):
        return False

    if not os.path.isdir(model_path):
        return True

    if os.path.exists(os.path.join(model_path, "model_index.json")):
        return False

    return any(
        os.path.isdir(os.path.join(model_path, variant))
        for variant in MODEL_VARIANTS
    )


def _get_converted_model_path(
    model_variant: str = "base_distilled_model",
    cache_dir: str | None = None,
) -> str:
    """Get the path where converted model should be stored."""
    if cache_dir is None:
        cache_dir = _get_cache_dir()
    return os.path.join(cache_dir, "Matrix-Game-2.0-Diffusers", model_variant)


def _is_conversion_needed(converted_path: str) -> bool:
    """Check if conversion is needed."""
    if not os.path.isdir(converted_path):
        return True
    required = ["transformer", "vae", "scheduler"]
    if not all(os.path.isdir(os.path.join(converted_path, d)) for d in required):
        return True
    weights = os.path.join(converted_path, "transformer", "diffusion_pytorch_model.safetensors")
    return not os.path.exists(weights)


def _download_matrixgame(
    repo_id: str,
    local_dir: str | None = None,
) -> str:
    """Download MatrixGame from HuggingFace Hub."""
    from huggingface_hub import snapshot_download

    if local_dir is None:
        local_dir = os.path.join(_get_cache_dir(), "Matrix-Game-2.0")

    if os.path.isdir(local_dir) and any(
        os.path.isdir(os.path.join(local_dir, v)) for v in MODEL_VARIANTS
    ):
        logger.info("MatrixGame already downloaded at %s", local_dir)
        return local_dir

    logger.info("Downloading MatrixGame from %s...", repo_id)
    local_path = snapshot_download(
        repo_id=repo_id,
        local_dir=local_dir,
        ignore_patterns=["*.md", "*.txt"],
    )
    logger.info("Downloaded MatrixGame to %s", local_path)
    return str(local_path)


def _create_model_index() -> dict[str, Any]:
    """Create model_index.json for diffusers format."""
    return {
        "_class_name": "MatrixGameCausalDMDPipeline",
        "_diffusers_version": "0.33.1",
        "scheduler": ["diffusers", "SelfForcingFlowMatchScheduler"],
        "transformer": ["diffusers", "MatrixGameWanModel"],
        "vae": ["diffusers", "AutoencoderKLWan"],
        "image_encoder": ["transformers", "CLIPVisionModel"],
        "image_processor": ["transformers", "CLIPImageProcessor"],
    }


def _create_transformer_config(source_config: dict[str, Any]) -> dict[str, Any]:
    """Create transformer config for diffusers format."""
    dim = source_config["dim"]
    num_heads = source_config["num_heads"]
    has_action = bool(source_config.get("action_config"))
    action_config = source_config.get("action_config", {})

    config = {
        "_class_name": "CausalMatrixGameWanModel",
        "_diffusers_version": "0.33.1",
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
        "patch_size": action_config.get("patch_size", [1, 2, 2]),
        "action_config": source_config.get("action_config"),
        "image_dim": 1280,
        "text_dim": 0,
        "local_attn_size": source_config.get("local_attn_size", 6 if has_action else -1),
        "sink_size": source_config.get("sink_size", 0),
        "text_len": source_config.get("text_len", 512),
    }
    return {k: v for k, v in config.items() if v is not None}


def _create_vae_config() -> dict[str, Any]:
    """Create VAE config for diffusers format."""
    return {
        "_class_name": "AutoencoderKLWan",
        "_diffusers_version": "0.33.1",
        "attn_scales": [],
        "base_dim": 96,
        "dim_mult": [1, 2, 4, 4],
        "dropout": 0.0,
        "latents_mean": [
            -0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508,
            0.4134, -0.0715, 0.5517, -0.3632, -0.1922, -0.9497, 0.2503, -0.2921,
        ],
        "latents_std": [
            2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743,
            3.2687, 2.1526, 2.8652, 1.5579, 1.6382, 1.1253, 2.8251, 1.916,
        ],
        "num_res_blocks": 2,
        "temperal_downsample": [False, True, True],
        "z_dim": 16,
    }


def _create_scheduler_config() -> dict[str, Any]:
    """Create scheduler config for diffusers format."""
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
        "training": True,
    }


def _convert_transformer_weights(source_path: Path) -> OrderedDict:
    """Convert transformer weights to diffusers format."""
    state_dict = load_file(source_path)
    new_state_dict = OrderedDict()

    for raw_key, value in state_dict.items():
        key = raw_key[6:] if raw_key.startswith("model.") else raw_key
        new_key = key
        for pattern, replacement in _TRANSFORMER_PARAM_MAPPING.items():
            if re.match(pattern, key):
                new_key = re.sub(pattern, replacement, key)
                break
        new_state_dict[new_key] = value

    return new_state_dict


def _convert_vae_key(key: str) -> str:
    """Convert a single VAE parameter key to diffusers format."""
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
                    return f"decoder.up_blocks.{stage}.resnets.{offset}.{suffix}"
                return f"decoder.up_blocks.{stage}.upsamplers.0.{suffix}"
            return f"decoder.up_blocks.3.resnets.{idx - 12}.{suffix}"
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


def _convert_vae_weights(source_path: str, output_path: str) -> None:
    """Convert VAE weights to diffusers format."""
    state_dict = torch.load(source_path, map_location="cpu", weights_only=False)
    if isinstance(state_dict, dict) and "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]

    converted = OrderedDict()
    for key, value in state_dict.items():
        converted[_convert_vae_key(key)] = value
    save_file(converted, output_path)


def _setup_image_encoder(source_dir: Path, output_dir: Path) -> None:
    """Set up image encoder from OpenCLIP checkpoint."""
    from transformers import CLIPImageProcessor, CLIPVisionConfig, CLIPVisionModel

    encoder_dir = output_dir / "image_encoder"
    processor_dir = output_dir / "image_processor"

    if encoder_dir.exists() and processor_dir.exists():
        return

    ckpt_path = source_dir / "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"OpenCLIP checkpoint not found: {ckpt_path}")

    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]

    model_id = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
    config = CLIPVisionConfig.from_pretrained(model_id)
    encoder = CLIPVisionModel(config)
    hidden = config.hidden_size

    def get(name: str) -> torch.Tensor:
        t = state[name]
        return t if isinstance(t, torch.Tensor) else torch.tensor(t)

    new_state: dict[str, torch.Tensor] = {
        "vision_model.embeddings.class_embedding": get("visual.cls_embedding").squeeze(0).squeeze(0),
        "vision_model.embeddings.patch_embedding.weight": get("visual.patch_embedding.weight"),
        "vision_model.embeddings.position_embedding.weight": get("visual.pos_embedding").squeeze(0),
        "vision_model.pre_layrnorm.weight": get("visual.pre_norm.weight"),
        "vision_model.pre_layrnorm.bias": get("visual.pre_norm.bias"),
        "vision_model.post_layernorm.weight": get("visual.post_norm.weight"),
        "vision_model.post_layernorm.bias": get("visual.post_norm.bias"),
    }

    for i in range(config.num_hidden_layers):
        src, dst = f"visual.transformer.{i}", f"vision_model.encoder.layers.{i}"
        qkv_w = get(f"{src}.attn.to_qkv.weight")
        qkv_b = get(f"{src}.attn.to_qkv.bias")
        q_w, k_w, v_w = torch.split(qkv_w, hidden, dim=0)
        q_b, k_b, v_b = torch.split(qkv_b, hidden, dim=0)

        new_state.update({
            f"{dst}.self_attn.q_proj.weight": q_w,
            f"{dst}.self_attn.q_proj.bias": q_b,
            f"{dst}.self_attn.k_proj.weight": k_w,
            f"{dst}.self_attn.k_proj.bias": k_b,
            f"{dst}.self_attn.v_proj.weight": v_w,
            f"{dst}.self_attn.v_proj.bias": v_b,
            f"{dst}.self_attn.out_proj.weight": get(f"{src}.attn.proj.weight"),
            f"{dst}.self_attn.out_proj.bias": get(f"{src}.attn.proj.bias"),
            f"{dst}.layer_norm1.weight": get(f"{src}.norm1.weight"),
            f"{dst}.layer_norm1.bias": get(f"{src}.norm1.bias"),
            f"{dst}.layer_norm2.weight": get(f"{src}.norm2.weight"),
            f"{dst}.layer_norm2.bias": get(f"{src}.norm2.bias"),
            f"{dst}.mlp.fc1.weight": get(f"{src}.mlp.0.weight"),
            f"{dst}.mlp.fc1.bias": get(f"{src}.mlp.0.bias"),
            f"{dst}.mlp.fc2.weight": get(f"{src}.mlp.2.weight"),
            f"{dst}.mlp.fc2.bias": get(f"{src}.mlp.2.bias"),
        })

    encoder.load_state_dict(new_state, strict=True)
    encoder.save_pretrained(encoder_dir)

    processor = CLIPImageProcessor.from_pretrained(model_id)
    processor.save_pretrained(processor_dir)


def convert_matrixgame_checkpoint(
    source_dir: str,
    output_dir: str,
    model_variant: str = "base_distilled_model",
) -> str:
    """Convert MatrixGame checkpoint to diffusers format."""
    source_path = Path(source_dir)
    output_path = Path(output_dir) / model_variant

    if model_variant not in MODEL_VARIANTS:
        raise ValueError(f"Unknown variant: {model_variant}. Options: {list(MODEL_VARIANTS.keys())}")

    variant = MODEL_VARIANTS[model_variant]
    variant_dir = source_path / model_variant
    config_file = variant_dir / variant["config_file"]
    weights_file = variant_dir / variant["weights_file"]
    vae_file = source_path / "Wan2.1_VAE.pth"

    for path, name in [(config_file, "Config"), (weights_file, "Weights"), (vae_file, "VAE")]:
        if not path.exists():
            raise FileNotFoundError(f"{name} not found: {path}")

    logger.info("Converting MatrixGame %s...", model_variant)

    with open(config_file) as f:
        source_config = json.load(f)

    output_path.mkdir(parents=True, exist_ok=True)

    with open(output_path / "model_index.json", "w") as f:
        json.dump(_create_model_index(), f, indent=2)

    transformer_dir = output_path / "transformer"
    transformer_dir.mkdir(exist_ok=True)
    with open(transformer_dir / "config.json", "w") as f:
        json.dump(_create_transformer_config(source_config), f, indent=2)
    weights = _convert_transformer_weights(weights_file)
    save_file(weights, transformer_dir / "diffusion_pytorch_model.safetensors")
    logger.info("Converted transformer weights")

    vae_dir = output_path / "vae"
    vae_dir.mkdir(exist_ok=True)
    with open(vae_dir / "config.json", "w") as f:
        json.dump(_create_vae_config(), f, indent=2)
    _convert_vae_weights(str(vae_file), str(vae_dir / "diffusion_pytorch_model.safetensors"))
    logger.info("Converted VAE weights")

    _setup_image_encoder(source_path, output_path)
    logger.info("Set up image encoder")

    scheduler_dir = output_path / "scheduler"
    scheduler_dir.mkdir(exist_ok=True)
    with open(scheduler_dir / "scheduler_config.json", "w") as f:
        json.dump(_create_scheduler_config(), f, indent=2)

    logger.info("Conversion complete: %s", output_path)
    return str(output_path)


def maybe_convert_matrixgame(
    model_path: str,
    model_variant: str = "base_distilled_model",
    cache_dir: str | None = None,
    delete_original: bool = False,
) -> str:
    """Automatically download and convert MatrixGame if needed."""
    if cache_dir is None:
        cache_dir = _get_cache_dir()

    if os.path.isdir(model_path) and os.path.exists(os.path.join(model_path, "model_index.json")):
        return model_path

    converted_path = _get_converted_model_path(model_variant, cache_dir)
    if not _is_conversion_needed(converted_path):
        logger.info("Using cached model: %s", converted_path)
        return converted_path

    if not os.path.isdir(model_path):
        source_dir = _download_matrixgame(model_path, os.path.join(cache_dir, "Matrix-Game-2.0"))
    else:
        source_dir = model_path

    converted_path = convert_matrixgame_checkpoint(
        source_dir=source_dir,
        output_dir=os.path.join(cache_dir, "Matrix-Game-2.0-Diffusers"),
        model_variant=model_variant,
    )

    if delete_original and source_dir != model_path:
        logger.info("Deleting original: %s", source_dir)
        shutil.rmtree(source_dir, ignore_errors=True)

    return converted_path
