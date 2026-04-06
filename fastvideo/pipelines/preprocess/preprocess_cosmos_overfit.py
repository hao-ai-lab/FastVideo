# SPDX-License-Identifier: Apache-2.0
"""Preprocess Cosmos-Predict2 overfit data into parquet format.

Encodes videos with the Cosmos (Wan-style) VAE and captions with the
single T5 Large text encoder into the t2v parquet schema expected by
the training framework.

Usage:
    CUDA_VISIBLE_DEVICES=0 python fastvideo/pipelines/preprocess/preprocess_cosmos_overfit.py
"""

import json
import os

import cv2
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from fastvideo.configs.models.encoders import T5LargeConfig
from fastvideo.configs.models.encoders.base import BaseEncoderOutput
from fastvideo.configs.pipelines.cosmos import t5_large_postprocess_text
from fastvideo.dataset.dataloader.schema import pyarrow_schema_t2v
from fastvideo.utils import maybe_download_model

# --- Config ---
NUM_FRAMES = 93  # 4*23+1 for temporal compression ratio 4 -> 24 latent frames
MAX_HEIGHT = 480
MAX_WIDTH = 832
TRAIN_FPS = 16.0

DATA_DIR = "data/cosmos_overfit"
OUTPUT_DIR = "data/cosmos_overfit_preprocessed"
MODEL_REPO = "nvidia/Cosmos-Predict2-2B-Video2World"


def load_video(path: str, num_frames: int) -> torch.Tensor:
    """Load video as [1, C, T, H, W] in [-1, 1]."""
    cap = cv2.VideoCapture(path)
    frames: list[np.ndarray] = []
    while len(frames) < num_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    cap.release()

    if len(frames) < num_frames:
        # Repeat last frame to fill
        while len(frames) < num_frames:
            frames.append(frames[-1])

    frames = frames[:num_frames]
    video = np.stack(frames, axis=0)
    video = torch.from_numpy(video).float()
    video = video / 127.5 - 1.0  # [0,255] -> [-1,1]
    video = video.permute(3, 0, 1, 2).unsqueeze(0)  # [1,C,T,H,W]
    return video


def main() -> None:
    device = torch.device("cuda:0")
    model_path = maybe_download_model(MODEL_REPO)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load captions
    with open(os.path.join(DATA_DIR, "videos2caption.json")) as f:
        caption_data = json.load(f)

    # --- Load VAE ---
    # Cosmos-Predict2-2B ships a Wan-style VAE; vae/config.json declares
    # `_class_name: AutoencoderKLWan`, so diffusers can load it directly.
    print("Loading Cosmos VAE (AutoencoderKLWan)...")
    from diffusers import AutoencoderKLWan
    vae = AutoencoderKLWan.from_pretrained(
        model_path,
        subfolder="vae",
        torch_dtype=torch.float16,
    ).to(device).eval()
    print(f"VAE loaded ({sum(p.numel() for p in vae.parameters())/1e6:.0f}M)")

    # --- Load T5 Large text encoder ---
    print("Loading T5 Large text encoder...")
    from transformers import AutoTokenizer, T5EncoderModel

    t5_cfg = T5LargeConfig()
    tok_kwargs = dict(t5_cfg.tokenizer_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(
        os.path.join(model_path, "tokenizer"))
    text_encoder = T5EncoderModel.from_pretrained(
        os.path.join(model_path, "text_encoder"),
        torch_dtype=torch.bfloat16,
    ).to(device).eval()

    # --- Process each video ---
    records = []
    for idx, item in enumerate(caption_data):
        video_name = item["path"]
        record_id = f"{idx:04d}_{video_name}"
        caption = item["cap"][0]
        video_path = os.path.join(DATA_DIR, "videos", video_name)

        print(f"\nProcessing: {video_name}")
        print(f"  Caption: {caption[:80]}...")

        # Encode video
        video = load_video(video_path,
                           NUM_FRAMES).to(device=device, dtype=torch.float16)
        print(f"  Video shape: {video.shape}")

        with torch.no_grad():
            latent_dist = vae.encode(video).latent_dist
            # Cast to fp32 — dataloader hardcodes np.float32
            latent = latent_dist.mean.squeeze(0).float().cpu()
        print(f"  Latent shape: {latent.shape}")

        # Encode text with T5
        with torch.no_grad():
            inputs = tokenizer(caption, **tok_kwargs).to(device)
            outputs = text_encoder(**inputs)
            enc_out = BaseEncoderOutput(
                last_hidden_state=outputs.last_hidden_state,
                attention_mask=inputs["attention_mask"],
            )
            # [1, max_len, 1024], zeros beyond real length
            t5_embed = t5_large_postprocess_text(enc_out).squeeze(0)
            # Trim to real sequence length so dataloader's pad() builds
            # the correct attention mask.
            real_len = int(inputs["attention_mask"].sum().item())
            text_embedding = t5_embed[:real_len].float().cpu()  # [seq, 1024]
        print(f"  Text embedding shape: {text_embedding.shape}")

        record = {
            "id": record_id,
            "vae_latent_bytes": latent.numpy().tobytes(),
            "vae_latent_shape": list(latent.shape),
            "vae_latent_dtype": str(latent.dtype).replace("torch.", ""),
            "text_embedding_bytes": text_embedding.numpy().tobytes(),
            "text_embedding_shape": list(text_embedding.shape),
            "text_embedding_dtype": str(text_embedding.dtype).replace(
                "torch.", ""),
            "file_name": video_name,
            "caption": caption,
            "media_type": "video",
            "width": MAX_WIDTH,
            "height": MAX_HEIGHT,
            "num_frames": NUM_FRAMES,
            "duration_sec": NUM_FRAMES / TRAIN_FPS,
            "fps": TRAIN_FPS,
        }
        records.append(record)

    # Clean up encoders
    del text_encoder, tokenizer, vae

    # Write parquet
    table = pa.table(
        {k: [r[k] for r in records]
         for k in records[0]},
        schema=pyarrow_schema_t2v,
    )
    output_path = os.path.join(OUTPUT_DIR, "data_00000.parquet")
    pq.write_table(table, output_path)
    print(f"\nWrote {len(records)} records to {output_path}")

    # Write validation prompts for callback
    # Wrap in "data" key — ValidationDataset expects field="data"
    # Use "caption" field — ValidationDataset aliases it to "prompt"
    val_prompts = {
        "data": [{
            "caption": item["cap"][0]
        } for item in caption_data]
    }
    val_path = os.path.join(OUTPUT_DIR, "validation_prompts.json")
    with open(val_path, "w") as f:
        json.dump(val_prompts, f, indent=2)
    print(f"Wrote validation prompts to {val_path}")

    print("\nDone! Use data_path: " + OUTPUT_DIR + " in training config.")


if __name__ == "__main__":
    main()
