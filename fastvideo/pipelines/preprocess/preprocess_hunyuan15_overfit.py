# SPDX-License-Identifier: Apache-2.0
"""Preprocess HunyuanVideo 1.5 overfit data into parquet format."""

from __future__ import annotations

import gc
import glob
import json
import os
from typing import Any

import cv2
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from safetensors.torch import load_file as safetensors_load_file
from transformers import AutoTokenizer, Qwen2_5_VLTextModel, T5EncoderModel

from fastvideo.configs.models.vaes import Hunyuan15VAEConfig
from fastvideo.configs.pipelines.hunyuan15 import (
    Hunyuan15T2V480PConfig,
    byt5_postprocess_text,
    byt5_preprocess_text,
    qwen_postprocess_text,
    qwen_preprocess_text,
)
from fastvideo.dataset.dataloader.schema import pyarrow_schema_t2v
from fastvideo.models.vaes.hunyuan15vae import AutoencoderKLHunyuanVideo15
from fastvideo.utils import maybe_download_model

NUM_FRAMES = 81
MAX_HEIGHT = 480
MAX_WIDTH = 832
TRAIN_FPS = 16.0

DATA_DIR = "data/hunyuan15_overfit"
OUTPUT_DIR = "data/hunyuan15_overfit_preprocessed"
MODEL_REPO = "hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v"

MAX_SAMPLES: int | None = 1


def clear_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def tensor_to_record(
    tensor: torch.Tensor,
    prefix: str,
) -> dict[str, Any]:
    tensor = tensor.detach().contiguous().float().cpu()
    array = tensor.numpy()
    return {
        f"{prefix}_bytes": array.tobytes(),
        f"{prefix}_shape": list(array.shape),
        f"{prefix}_dtype": str(array.dtype),
    }


def load_caption_data() -> list[dict[str, Any]]:
    caption_path = os.path.join(DATA_DIR, "videos2caption.json")
    with open(caption_path, encoding="utf-8") as file:
        caption_data = json.load(file)

    if MAX_SAMPLES is not None:
        caption_data = caption_data[:MAX_SAMPLES]

    if not caption_data:
        raise ValueError(f"No caption records found in {caption_path}")

    return caption_data


def load_video(
    path: str,
    num_frames: int,
    height: int,
    width: int,
) -> torch.Tensor:
    cap = cv2.VideoCapture(path)
    frames: list[np.ndarray] = []

    while len(frames) < num_frames:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(
            frame,
            (width, height),
            interpolation=cv2.INTER_AREA,
        )
        frames.append(frame)

    cap.release()

    if not frames:
        raise ValueError(f"No frames could be decoded from {path}")

    while len(frames) < num_frames:
        frames.append(frames[-1].copy())

    video = np.stack(frames[:num_frames], axis=0)
    video = torch.from_numpy(video).float()
    video = video / 127.5 - 1.0
    return video.permute(3, 0, 1, 2).unsqueeze(0)


def apply_qwen_chat_template(
    tokenizer,
    prompt: str,
    max_length: int,
) -> dict[str, torch.Tensor]:
    messages = qwen_preprocess_text(prompt)

    encoded = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        max_length=max_length,
        truncation=True,
    )

    if isinstance(encoded, torch.Tensor):
        return {
            "input_ids": encoded,
            "attention_mask": torch.ones_like(encoded),
        }

    if "attention_mask" not in encoded:
        encoded["attention_mask"] = torch.ones_like(encoded["input_ids"])

    return {
        "input_ids": encoded["input_ids"],
        "attention_mask": encoded["attention_mask"],
    }


@torch.inference_mode()
def encode_qwen_caption(
    caption: str,
    tokenizer,
    encoder: Qwen2_5_VLTextModel,
    device: torch.device,
    max_length: int,
) -> torch.Tensor:
    tokens = apply_qwen_chat_template(tokenizer, caption, max_length)
    tokens = {key: value.to(device) for key, value in tokens.items()}

    outputs = encoder(
        input_ids=tokens["input_ids"],
        attention_mask=tokens["attention_mask"],
        output_hidden_states=True,
        return_dict=True,
    )

    embeddings, attention_mask = qwen_postprocess_text(
        outputs,
        tokens["attention_mask"],
    )

    embeddings = embeddings.squeeze(0)
    attention_mask = attention_mask.squeeze(0)
    valid_length = int(attention_mask.sum().item())

    return embeddings[:valid_length].float().cpu()


@torch.inference_mode()
def encode_byt5_caption(
    caption: str,
    tokenizer,
    encoder: T5EncoderModel,
    device: torch.device,
    max_length: int,
) -> torch.Tensor:
    glyph_prompt = byt5_preprocess_text(caption)
    hidden_size = int(encoder.config.d_model)

    if glyph_prompt is None:
        return torch.empty((0, hidden_size), dtype=torch.float32)

    tokens = tokenizer(
        glyph_prompt,
        add_special_tokens=True,
        padding=False,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    tokens = {key: value.to(device) for key, value in tokens.items()}

    outputs = encoder(
        input_ids=tokens["input_ids"],
        attention_mask=tokens.get("attention_mask"),
        return_dict=True,
    )

    embeddings = byt5_postprocess_text(outputs).squeeze(0)

    attention_mask = tokens.get("attention_mask")
    if attention_mask is not None:
        valid_length = int(attention_mask.squeeze(0).sum().item())
        embeddings = embeddings[:valid_length]

    return embeddings.float().cpu()


def load_hunyuan15_vae(
    model_path: str,
    device: torch.device,
) -> AutoencoderKLHunyuanVideo15:
    vae_path = os.path.join(model_path, "vae")
    config_path = os.path.join(vae_path, "config.json")

    with open(config_path, encoding="utf-8") as file:
        raw_config = json.load(file)

    raw_config.pop("_class_name", None)
    raw_config.pop("_diffusers_version", None)
    raw_config.pop("_name_or_path", None)

    vae_config = Hunyuan15VAEConfig()
    vae_config.update_model_arch(raw_config)
    vae_config.load_encoder = True
    vae_config.load_decoder = False

    vae = AutoencoderKLHunyuanVideo15(vae_config)

    weight_files = sorted(glob.glob(os.path.join(vae_path, "*.safetensors")))
    if not weight_files:
        raise FileNotFoundError(f"No VAE safetensors files found under {vae_path}")

    state_dict: dict[str, torch.Tensor] = {}
    for weight_file in weight_files:
        state_dict.update(safetensors_load_file(weight_file, device="cpu"))

    missing_keys, unexpected_keys = vae.load_state_dict(
        state_dict,
        strict=False,
    )

    encoder_missing = [key for key in missing_keys if not key.startswith("decoder.")]
    if encoder_missing:
        raise RuntimeError(f"Missing VAE encoder weights: {encoder_missing[:20]}")

    if unexpected_keys:
        print("Ignored checkpoint keys not used by the encoder-only VAE: "
              f"{len(unexpected_keys)}")

    del state_dict
    clear_cuda()

    vae = vae.to(
        device=device,
        dtype=torch.float16,
    ).eval()
    vae.requires_grad_(False)
    vae.use_tiling = False
    return vae


@torch.inference_mode()
def encode_video_latent(
    vae: AutoencoderKLHunyuanVideo15,
    video: torch.Tensor,
) -> torch.Tensor:
    encoded = vae.encode(video)

    # HunyuanVideo 1.5 VAE currently returns
    # DiagonalGaussianDistribution directly.
    if hasattr(encoded, "latent_dist"):
        latent_dist = encoded.latent_dist
    else:
        latent_dist = encoded

    if hasattr(latent_dist, "mode"):
        latent = latent_dist.mode()
    elif hasattr(latent_dist, "mean"):
        latent = latent_dist.mean
    else:
        raise TypeError("Unsupported VAE encode output type: "
                        f"{type(encoded).__name__}")

    return latent.squeeze(0).float().cpu()


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("This preprocessing script requires CUDA.")

    device = torch.device("cuda:0")
    model_path = maybe_download_model(MODEL_REPO)
    pipeline_config = Hunyuan15T2V480PConfig()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    caption_data = load_caption_data()
    captions = [item["cap"][0] for item in caption_data]

    records: list[dict[str, Any]] = [{
        "id": f"{index:06d}_{item['path']}",
        "file_name": item["path"],
        "caption": item["cap"][0],
        "media_type": "video",
        "width": MAX_WIDTH,
        "height": MAX_HEIGHT,
        "num_frames": NUM_FRAMES,
        "duration_sec": NUM_FRAMES / TRAIN_FPS,
        "fps": TRAIN_FPS,
    } for index, item in enumerate(caption_data)]

    print("Loading Qwen2.5-VL text encoder...")
    qwen_tokenizer = AutoTokenizer.from_pretrained(
        os.path.join(model_path, "tokenizer"),
        local_files_only=True,
    )
    qwen_encoder = Qwen2_5_VLTextModel.from_pretrained(
        os.path.join(model_path, "text_encoder"),
        torch_dtype=torch.bfloat16,
        local_files_only=True,
    ).to(device).eval()

    qwen_max_length = int(pipeline_config.text_encoder_max_lengths[0])

    for index, caption in enumerate(captions):
        embedding = encode_qwen_caption(
            caption,
            qwen_tokenizer,
            qwen_encoder,
            device,
            qwen_max_length,
        )
        records[index].update(tensor_to_record(embedding, "text_embedding"))
        print(f"[Qwen {index + 1}/{len(records)}] "
              f"{tuple(embedding.shape)}")

    del qwen_encoder, qwen_tokenizer
    clear_cuda()

    print("Loading ByT5/T5 text encoder...")
    byt5_tokenizer = AutoTokenizer.from_pretrained(
        os.path.join(model_path, "tokenizer_2"),
        local_files_only=True,
    )
    byt5_encoder = T5EncoderModel.from_pretrained(
        os.path.join(model_path, "text_encoder_2"),
        torch_dtype=torch.float32,
        local_files_only=True,
    ).to(device).eval()

    byt5_max_length = int(pipeline_config.text_encoder_max_lengths[1])

    for index, caption in enumerate(captions):
        embedding = encode_byt5_caption(
            caption,
            byt5_tokenizer,
            byt5_encoder,
            device,
            byt5_max_length,
        )
        records[index].update(tensor_to_record(embedding, "text_embedding_2"))
        print(f"[ByT5 {index + 1}/{len(records)}] "
              f"{tuple(embedding.shape)}")

    del byt5_encoder, byt5_tokenizer
    clear_cuda()

    print("Loading HunyuanVideo 1.5 VAE encoder...")
    vae = load_hunyuan15_vae(model_path, device)

    for index, item in enumerate(caption_data):
        video_path = os.path.join(
            DATA_DIR,
            "videos",
            item["path"],
        )
        video = load_video(
            video_path,
            NUM_FRAMES,
            MAX_HEIGHT,
            MAX_WIDTH,
        ).to(
            device=device,
            dtype=torch.float16,
        )

        latent = encode_video_latent(vae, video)
        records[index].update(tensor_to_record(latent, "vae_latent"))
        print(f"[VAE {index + 1}/{len(records)}] "
              f"{tuple(latent.shape)}")

        del video, latent
        clear_cuda()

    del vae
    clear_cuda()

    columns = {field.name: [record.get(field.name) for record in records] for field in pyarrow_schema_t2v}

    table = pa.Table.from_pydict(
        columns,
        schema=pyarrow_schema_t2v,
    )

    output_path = os.path.join(
        OUTPUT_DIR,
        "data_00000.parquet",
    )
    pq.write_table(
        table,
        output_path,
        compression="zstd",
    )

    validation_path = os.path.join(
        OUTPUT_DIR,
        "validation_prompts.json",
    )
    with open(
            validation_path,
            "w",
            encoding="utf-8",
    ) as file:
        json.dump(
            {"data": [{
                "caption": caption
            } for caption in captions]},
            file,
            ensure_ascii=False,
            indent=2,
        )

    print(f"Wrote {len(records)} records to {output_path}")
    print(f"Wrote validation prompts to {validation_path}")


if __name__ == "__main__":
    main()
