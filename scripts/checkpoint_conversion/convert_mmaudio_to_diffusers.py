# SPDX-License-Identifier: Apache-2.0
"""Convert MMAudio ``large_44k_v2`` assets into a FastVideo component tree.

The converter is deliberately offline: every large source asset must already
exist locally. It splits the shared DFN5B OpenCLIP checkpoint into native
FastVideo text and vision encoders, preserves the exact MMAudio transformer,
Synchformer, VAE, and BigVGAN weights, and emits the standard
``model_index.json`` layout consumed by ``ComposedPipelineBase``.

Example::

    python scripts/checkpoint_conversion/convert_mmaudio_to_diffusers.py \
      --transformer-checkpoint ../MMAudio/weights/mmaudio_large_44k_v2.pth \
      --audio-vae-checkpoint ../MMAudio/ext_weights/v1-44.pth \
      --synchformer-checkpoint ../MMAudio/ext_weights/synchformer_state_dict.pth \
      --dfn5b-dir official_weights/mmaudio/DFN5B-CLIP-ViT-H-14-384 \
      --bigvgan-dir official_weights/mmaudio/bigvgan_v2_44khz_128band_512x \
      --output converted_weights/mmaudio/large_44k_v2
"""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import save_file


TRANSFORMER_CONFIG = {
    "_class_name": "MMAudioTransformer",
    "latent_dim": 40,
    "clip_dim": 1024,
    "sync_dim": 768,
    "text_dim": 1024,
    "hidden_dim": 896,
    "depth": 21,
    "fused_depth": 14,
    "num_heads": 14,
    "mlp_ratio": 4.0,
    "latent_seq_len": 345,
    "clip_seq_len": 64,
    "sync_seq_len": 192,
    "text_seq_len": 77,
    "v2": True,
}

TEXT_ENCODER_CONFIG = {
    "architectures": ["MMAudioDFNCLIPTextEncoder"],
    "vocab_size": 49408,
    "hidden_size": 1024,
    "intermediate_size": 4096,
    "projection_dim": 1024,
    "num_hidden_layers": 24,
    "num_attention_heads": 16,
    "max_position_embeddings": 77,
    "text_len": 77,
    "hidden_act": "quick_gelu",
    "layer_norm_eps": 1e-5,
    "pad_token_id": 0,
    "bos_token_id": 49406,
    "eos_token_id": 49407,
}

IMAGE_ENCODER_CONFIG = {
    "architectures": ["MMAudioDFNCLIPVisionEncoder"],
    "hidden_size": 1280,
    "intermediate_size": 5120,
    "projection_dim": 1024,
    "num_hidden_layers": 32,
    "num_attention_heads": 16,
    "num_channels": 3,
    "image_size": 378,
    "patch_size": 14,
    "hidden_act": "quick_gelu",
    "layer_norm_eps": 1e-5,
}

SYNCHFORMER_CONFIG = {
    "architectures": ["MMAudioSynchformerVisualEncoder"],
    "image_size": 224,
    "num_channels": 3,
    "segment_size": 16,
    "segment_stride": 8,
    "hidden_size": 768,
    "tokens_per_segment": 8,
}

MODEL_INDEX = {
    "_class_name": "MMAudioPipeline",
    "_diffusers_version": "0.36.0",
    "_fastvideo_model_family": "mmaudio",
    "_fastvideo_workload_types": ["V2A", "T2A"],
    "transformer": [
        "fastvideo.models.dits.mmaudio",
        "MMAudioTransformer",
    ],
    "text_encoder": [
        "fastvideo.models.encoders.mmaudio_clip",
        "MMAudioDFNCLIPTextEncoder",
    ],
    "tokenizer": ["transformers", "CLIPTokenizer"],
    "image_encoder": [
        "fastvideo.models.encoders.mmaudio_clip",
        "MMAudioDFNCLIPVisionEncoder",
    ],
    "image_encoder_2": [
        "fastvideo.models.encoders.mmaudio_synchformer",
        "MMAudioSynchformerVisualEncoder",
    ],
    "audio_vae": ["fastvideo.models.audio.mmaudio_vae", "MMAudioVAE"],
    "vocoder": ["fastvideo.models.audio.bigvgan", "BigVGANV2"],
    "scheduler": ["diffusers", "FlowMatchEulerDiscreteScheduler"],
}


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _write_component(output: Path, name: str, state: dict[str, torch.Tensor], config: dict[str, Any]) -> None:
    directory = output / name
    directory.mkdir(parents=True, exist_ok=True)
    contiguous = {key: tensor.detach().cpu().contiguous() for key, tensor in state.items()}
    save_file(contiguous, directory / "diffusion_pytorch_model.safetensors", metadata={"format": "pt"})
    _write_json(directory / "config.json", config)


def _load_torch_state(path: Path) -> dict[str, torch.Tensor]:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(value, dict):
        raise TypeError(f"Expected a state dict in {path}, got {type(value)}")
    for key in ("state_dict", "model", "generator"):
        nested = value.get(key)
        if isinstance(nested, dict) and nested:
            value = nested
            break
    if not all(isinstance(tensor, torch.Tensor) for tensor in value.values()):
        raise TypeError(f"Checkpoint {path} contains non-tensor state entries")
    return value


def map_open_clip_text_state(
    state: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    mapped: dict[str, torch.Tensor] = {
        "text_model.embeddings.token_embedding.weight": state["token_embedding.weight"],
        "text_model.embeddings.position_embedding.weight": state["positional_embedding"],
        "text_model.final_layer_norm.weight": state["ln_final.weight"],
        "text_model.final_layer_norm.bias": state["ln_final.bias"],
    }
    for name, tensor in state.items():
        if not name.startswith("transformer.resblocks."):
            continue
        target = name.replace("transformer.resblocks.", "text_model.encoder.layers.")
        target = target.replace(".ln_1.", ".layer_norm1.")
        target = target.replace(".ln_2.", ".layer_norm2.")
        target = target.replace(".attn.in_proj_", ".self_attn.qkv_proj.")
        target = target.replace(".attn.out_proj.", ".self_attn.out_proj.")
        target = target.replace(".mlp.c_fc.", ".mlp.fc1.")
        target = target.replace(".mlp.c_proj.", ".mlp.fc2.")
        mapped[target] = tensor
    return mapped


def map_open_clip_vision_state(
    state: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    mapped: dict[str, torch.Tensor] = {
        "vision_model.embeddings.class_embedding": state["visual.class_embedding"],
        "vision_model.embeddings.patch_embedding.weight": state["visual.conv1.weight"],
        "vision_model.embeddings.position_embedding.weight": state["visual.positional_embedding"],
        "vision_model.pre_layrnorm.weight": state["visual.ln_pre.weight"],
        "vision_model.pre_layrnorm.bias": state["visual.ln_pre.bias"],
        "vision_model.post_layernorm.weight": state["visual.ln_post.weight"],
        "vision_model.post_layernorm.bias": state["visual.ln_post.bias"],
        "visual_projection.weight": state["visual.proj"].t(),
    }
    for name, tensor in state.items():
        if not name.startswith("visual.transformer.resblocks."):
            continue
        target = name.replace("visual.transformer.resblocks.", "vision_model.encoder.layers.")
        target = target.replace(".ln_1.", ".layer_norm1.")
        target = target.replace(".ln_2.", ".layer_norm2.")
        target = target.replace(".attn.in_proj_", ".self_attn.qkv_proj.")
        target = target.replace(".attn.out_proj.", ".self_attn.out_proj.")
        target = target.replace(".mlp.c_fc.", ".mlp.fc1.")
        target = target.replace(".mlp.c_proj.", ".mlp.fc2.")
        mapped[target] = tensor
    return mapped


def write_open_clip_tokenizer(output: Path) -> None:
    """Write the bundled OpenAI CLIP BPE as an AutoTokenizer component.

    OpenCLIP pads its 77-token tensor with integer zero. ``CLIPTokenizer``
    cannot use vocabulary ID zero as a special pad token without changing how
    a literal exclamation mark is tokenized, so the MMAudio text stage zeros
    positions selected by ``attention_mask`` after tokenization.
    """
    from open_clip.tokenizer import bytes_to_unicode, default_bpe

    with gzip.open(default_bpe()) as bpe_file:
        merges_raw = bpe_file.read().decode("utf-8").split("\n")
    merges = merges_raw[1 : 49152 - 256 - 2 + 1]
    merge_pairs = [tuple(merge.split()) for merge in merges]
    vocab = list(bytes_to_unicode().values())
    vocab += [token + "</w>" for token in vocab]
    vocab += ["".join(pair) for pair in merge_pairs]
    vocab += ["<start_of_text>", "<end_of_text>"]
    encoder = {token: index for index, token in enumerate(vocab)}

    directory = output / "tokenizer"
    directory.mkdir(parents=True, exist_ok=True)
    _write_json(directory / "vocab.json", encoder)
    with (directory / "merges.txt").open("w", encoding="utf-8") as handle:
        handle.write("#version: 0.2\n")
        for first, second in merge_pairs:
            handle.write(f"{first} {second}\n")
    _write_json(
        directory / "tokenizer_config.json",
        {
            "tokenizer_class": "CLIPTokenizer",
            "model_max_length": 77,
            "bos_token": "<start_of_text>",
            "eos_token": "<end_of_text>",
            "unk_token": "<end_of_text>",
            "pad_token": "<end_of_text>",
            "do_lower_case": True,
        },
    )
    _write_json(
        directory / "special_tokens_map.json",
        {
            "bos_token": "<start_of_text>",
            "eos_token": "<end_of_text>",
            "unk_token": "<end_of_text>",
            "pad_token": "<end_of_text>",
        },
    )


def _load_dfn5b_state(directory: Path) -> dict[str, torch.Tensor]:
    if not directory.is_dir():
        raise FileNotFoundError(directory)
    from open_clip import create_model_from_pretrained

    model = create_model_from_pretrained(f"local-dir:{directory}",
                                         return_transform=False)
    state = {key: tensor.detach().cpu() for key, tensor in model.state_dict().items()}
    del model
    return state


def convert(args: argparse.Namespace) -> None:
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    transformer_state = _load_torch_state(args.transformer_checkpoint)
    # Official ``MMAudio.load_weights`` discards this derived buffer. Keeping
    # it would make a standard strict FastVideo component load fail.
    transformer_state.pop("t_embed.freqs", None)
    transformer_state.pop("latent_rot", None)
    transformer_state.pop("clip_rot", None)
    _write_component(output, "transformer", transformer_state, TRANSFORMER_CONFIG)

    vae_state = _load_torch_state(args.audio_vae_checkpoint)
    decoder_state = {
        key: tensor
        for key, tensor in vae_state.items()
        if key.startswith("decoder.") or key in {"data_mean", "data_std"}
    }
    if not decoder_state:
        raise ValueError("Audio VAE checkpoint did not contain decoder weights")
    _write_component(
        output,
        "audio_vae",
        decoder_state,
        {"_class_name": "MMAudioVAE", "mode": "44k", "need_encoder": False},
    )

    synchformer_state = _load_torch_state(args.synchformer_checkpoint)
    synchformer_visual_state = {
        name: tensor
        for name, tensor in synchformer_state.items()
        if name.startswith("vfeat_extractor.")
    }
    if not synchformer_visual_state:
        raise ValueError(
            "Synchformer checkpoint did not contain vfeat_extractor weights")
    _write_component(output, "image_encoder_2", synchformer_visual_state,
                     SYNCHFORMER_CONFIG)

    dfn_state = _load_dfn5b_state(args.dfn5b_dir)
    _write_component(output, "text_encoder", map_open_clip_text_state(dfn_state), TEXT_ENCODER_CONFIG)
    _write_component(output, "image_encoder", map_open_clip_vision_state(dfn_state), IMAGE_ENCODER_CONFIG)
    write_open_clip_tokenizer(output)

    bigvgan_config_path = args.bigvgan_dir / "config.json"
    if not bigvgan_config_path.is_file():
        raise FileNotFoundError(bigvgan_config_path)
    with bigvgan_config_path.open(encoding="utf-8") as handle:
        bigvgan_config = json.load(handle)
    bigvgan_config["_class_name"] = "BigVGANV2"
    bigvgan_config["weight_norm_removed"] = False
    bigvgan_state = _load_torch_state(args.bigvgan_dir / "bigvgan_generator.pt")
    _write_component(output, "vocoder", bigvgan_state, bigvgan_config)

    _write_json(
        output / "scheduler/scheduler_config.json",
        {
            "_class_name": "FlowMatchEulerDiscreteScheduler",
            "num_train_timesteps": 1000,
            "shift": 1.0,
            "invert_sigmas": True,
            "sigma_min": 0.0,
            "use_reference_discrete_timesteps": True,
        },
    )
    _write_json(output / "model_index.json", MODEL_INDEX)
    print(f"Converted MMAudio large_44k_v2 components to {output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--transformer-checkpoint", type=Path, required=True)
    parser.add_argument("--audio-vae-checkpoint", type=Path, required=True)
    parser.add_argument("--synchformer-checkpoint", type=Path, required=True)
    parser.add_argument("--dfn5b-dir", type=Path, required=True)
    parser.add_argument("--bigvgan-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    convert(parse_args())
