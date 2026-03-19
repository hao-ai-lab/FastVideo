# SPDX-License-Identifier: Apache-2.0
"""Quick preprocessing script for HunyuanVideo overfitting test.

Encodes a small set of videos with HunyuanVideo's VAE and dual
text encoders (LLaMA + CLIP), producing parquet training data.

Usage:
    python scripts/preprocess_hunyuan_overfit.py
"""

import json
import os
import sys

# Patch flash_attn import issue before anything else
import importlib
_orig_import = __builtins__.__import__ if hasattr(
    __builtins__, '__import__') else importlib.__import__

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import torchvision
from huggingface_hub import snapshot_download
from transformers import (
    AutoTokenizer,
    CLIPTextModel,
    LlamaModel,
)

from fastvideo.configs.pipelines.hunyuan import (
    llama_postprocess_text,
    llama_preprocess_text,
)
from fastvideo.dataset.dataloader.schema import (
    pyarrow_schema_t2v, )

DATA_DIR = "data/hunyuan_overfit"
OUTPUT_DIR = "data/hunyuan_overfit_preprocessed"
MODEL_PATH = "hunyuanvideo-community/HunyuanVideo"
MAX_HEIGHT = 480
MAX_WIDTH = 832
NUM_FRAMES = 77
TRAIN_FPS = 16

device = torch.device("cuda:0")


def load_and_resize_video(path, num_frames, fps, h, w):
    """Load video, sample frames, resize."""
    video, _, meta = torchvision.io.read_video(
        path, pts_unit="sec")
    src_fps = meta["video_fps"]

    # Sample at target fps
    if src_fps > 0 and src_fps != fps:
        step = max(1, round(src_fps / fps))
        video = video[::step]

    # Trim to num_frames
    video = video[:num_frames]

    # Resize: [T, H, W, C] -> [T, C, H, W]
    video = video.permute(0, 3, 1, 2).float()
    video = torch.nn.functional.interpolate(
        video, size=(h, w), mode="bilinear",
        align_corners=False)

    # Normalize to [-1, 1]
    video = video / 127.5 - 1.0

    # [T, C, H, W] -> [1, C, T, H, W]
    video = video.permute(1, 0, 2, 3).unsqueeze(0)
    return video


def load_vae(model_dir):
    """Load HunyuanVideo VAE using FastVideo's loader."""
    from fastvideo.models.vaes.hunyuanvae import (
        AutoencoderKLHunyuanVideo, )
    from fastvideo.configs.models.vaes.hunyuanvae import (
        HunyuanVAEConfig, )
    from safetensors.torch import load_file

    vae_dir = os.path.join(model_dir, "vae")
    config = HunyuanVAEConfig()
    config.load_encoder = True
    config.load_decoder = False
    config.use_tiling = True

    vae = AutoencoderKLHunyuanVideo(config)
    state_dict = load_file(
        os.path.join(
            vae_dir,
            "diffusion_pytorch_model.safetensors"))
    vae.load_state_dict(state_dict, strict=False)
    return vae


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load metadata
    with open(
        os.path.join(DATA_DIR, "videos2caption.json")
    ) as f:
        metadata = json.load(f)

    # Get local model path
    model_dir = snapshot_download(MODEL_PATH)
    print(f"Model dir: {model_dir}")

    # --- Load VAE ---
    print("Loading VAE...")
    vae = load_vae(model_dir)
    vae = vae.to(device, dtype=torch.float16).eval()

    # --- Load LLaMA ---
    print("Loading LLaMA text encoder...")
    llama_tok = AutoTokenizer.from_pretrained(
        os.path.join(model_dir, "tokenizer"))
    llama_enc = LlamaModel.from_pretrained(
        os.path.join(model_dir, "text_encoder"),
        torch_dtype=torch.float16)
    llama_enc = llama_enc.to(device).eval()

    # --- Load CLIP ---
    print("Loading CLIP text encoder...")
    clip_tok = AutoTokenizer.from_pretrained(
        os.path.join(model_dir, "tokenizer_2"))
    clip_enc = CLIPTextModel.from_pretrained(
        os.path.join(model_dir, "text_encoder_2"),
        torch_dtype=torch.float16)
    clip_enc = clip_enc.to(device).eval()

    records = []

    for entry in metadata:
        video_path = os.path.join(
            DATA_DIR, "videos", entry["path"])
        caption = entry["cap"][0]
        video_name = os.path.splitext(entry["path"])[0]
        print(
            f"Processing {video_name}: "
            f"{caption[:60]}...")

        # --- Encode video ---
        video = load_and_resize_video(
            video_path, NUM_FRAMES, TRAIN_FPS,
            MAX_HEIGHT, MAX_WIDTH)
        video = video.to(device, dtype=torch.float16)

        with torch.no_grad():
            latent = vae.encode(video).mean
        latent = latent.squeeze(0).cpu()  # [C, T, H, W]
        print(f"  VAE latent: {latent.shape}")

        # --- Encode text (LLaMA) ---
        llama_text = llama_preprocess_text(caption)
        llama_inputs = llama_tok(
            llama_text,
            return_tensors="pt",
            padding="max_length",
            max_length=256,
            truncation=True,
        ).to(device)

        with torch.no_grad():
            llama_out = llama_enc(
                **llama_inputs,
                output_hidden_states=True,
            )
        llama_embeds = llama_postprocess_text(llama_out)
        llama_embeds = llama_embeds.squeeze(0).float().cpu()
        print(f"  LLaMA embeds: {llama_embeds.shape}")

        # --- Encode text (CLIP pooled) ---
        clip_inputs = clip_tok(
            caption,
            return_tensors="pt",
            padding="max_length",
            max_length=77,
            truncation=True,
        ).to(device)

        with torch.no_grad():
            clip_out = clip_enc(**clip_inputs)
        clip_pooled = (
            clip_out.pooler_output.squeeze(0)
            .float().cpu())
        print(f"  CLIP pooled: {clip_pooled.shape}")

        # --- Combine text embeddings ---
        # HunyuanVideo forward:
        #   txt = encoder_hidden_states[:, 1:]
        #   pooled = encoder_hidden_states[:, 0, :768]
        llama_dim = llama_embeds.shape[-1]  # 4096
        pooled_row = torch.zeros(llama_dim)
        pooled_row[:clip_pooled.shape[-1]] = clip_pooled
        text_embedding = torch.cat(
            [pooled_row.unsqueeze(0), llama_embeds],
            dim=0,
        ).float()
        print(f"  Combined text: {text_embedding.shape}")

        # --- Build record ---
        vae_np = latent.float().numpy()
        text_np = text_embedding.numpy()

        records.append({
            "id": video_name,
            "vae_latent_bytes": vae_np.tobytes(),
            "vae_latent_shape": list(vae_np.shape),
            "vae_latent_dtype": str(vae_np.dtype),
            "text_embedding_bytes": text_np.tobytes(),
            "text_embedding_shape": list(text_np.shape),
            "text_embedding_dtype": str(text_np.dtype),
            "file_name": entry["path"],
            "caption": caption,
            "media_type": "video",
            "width": MAX_WIDTH,
            "height": MAX_HEIGHT,
            "num_frames": NUM_FRAMES,
            "duration_sec": float(
                entry.get("duration", 0)),
            "fps": float(TRAIN_FPS),
        })

    # --- Write parquet ---
    arrays = {col: [] for col in pyarrow_schema_t2v.names}
    for r in records:
        for col in pyarrow_schema_t2v.names:
            arrays[col].append(r[col])

    table = pa.table(arrays, schema=pyarrow_schema_t2v)
    out_path = os.path.join(
        OUTPUT_DIR, "data_chunk_0.parquet")
    pq.write_table(table, out_path)
    print(f"\nWrote {len(records)} samples to {out_path}")

    # --- Write validation prompts ---
    val_data = {
        "data": [{"caption": r["caption"]}
                 for r in records]
    }
    val_path = os.path.join(
        OUTPUT_DIR, "validation_prompts.json")
    with open(val_path, "w") as f:
        json.dump(val_data, f, indent=2)
    print(f"Wrote validation prompts to {val_path}")

    # Cleanup
    del vae, llama_enc, clip_enc
    torch.cuda.empty_cache()
    print("Done!")


if __name__ == "__main__":
    main()
