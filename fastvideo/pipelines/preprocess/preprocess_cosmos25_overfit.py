# SPDX-License-Identifier: Apache-2.0
"""Preprocess Cosmos 2.5 overfit data into parquet format.

Encodes videos with the Cosmos (Wan-style) VAE and captions with the
Reason1 (Qwen2.5-VL) text encoder into the t2v parquet schema.

Usage:
    CUDA_VISIBLE_DEVICES=0 python fastvideo/pipelines/preprocess/preprocess_cosmos25_overfit.py
"""

import json
import os

import cv2
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from fastvideo.dataset.dataloader.schema import pyarrow_schema_t2v
from fastvideo.utils import maybe_download_model

# --- Config ---
NUM_FRAMES = 93  # 4*23+1 → 24 latent frames
MAX_HEIGHT = 480
MAX_WIDTH = 832
TRAIN_FPS = 16.0

DATA_DIR = "data/cosmos_overfit"
OUTPUT_DIR = "data/cosmos25_overfit_preprocessed"
MODEL_REPO = "KyleShao/Cosmos-Predict2.5-2B-Diffusers"
# The VAE is architecturally identical to Cosmos Predict2;
# use the Predict2 model for VAE since its weights are in
# standard diffusers format.
VAE_REPO = "nvidia/Cosmos-Predict2-2B-Video2World"


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

    # --- Load VAE (Wan-style, same arch for Cosmos 2 and 2.5) ---
    print("Loading Cosmos VAE (AutoencoderKLWan)...")
    vae_path = maybe_download_model(VAE_REPO)
    from diffusers import AutoencoderKLWan
    vae = AutoencoderKLWan.from_pretrained(
        vae_path,
        subfolder="vae",
        torch_dtype=torch.float16,
    ).to(device).eval()
    print(f"VAE loaded "
          f"({sum(p.numel() for p in vae.parameters())/1e6:.0f}M)")

    # --- Load Reason1 (Qwen2.5-VL) text encoder ---
    print("Loading Reason1 text encoder...")
    from fastvideo.configs.pipelines.cosmos2_5 import (
        Cosmos25Config, )
    from fastvideo.models.encoders.reason1 import (
        Reason1TextEncoder, )
    pipeline_cfg = Cosmos25Config()
    text_enc_cfg = pipeline_cfg.text_encoder_configs[0]
    text_enc_path = os.path.join(model_path, "text_encoder")

    # Instantiate Reason1TextEncoder with config and checkpoint
    text_encoder = Reason1TextEncoder(
        text_enc_cfg,
        checkpoint_path=text_enc_path,
    )
    # Load weights from safetensors into the meta-device model.
    # Materialize empty tensors in bf16 on the target device,
    # then overwrite with checkpoint weights.
    text_encoder = text_encoder.to_empty(device=device)
    text_encoder = text_encoder.to(torch.bfloat16)
    import glob
    from safetensors.torch import load_file
    sd: dict[str, torch.Tensor] = {}
    for sf in sorted(glob.glob(os.path.join(text_enc_path, "*.safetensors"))):
        sd.update(load_file(sf, device=str(device)))
    sd = {k: v.to(torch.bfloat16) for k, v in sd.items()}
    text_encoder.load_state_dict(sd, strict=False, assign=True)
    del sd
    torch.cuda.empty_cache()
    text_encoder = text_encoder.eval()
    print("Reason1 text encoder loaded")

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
        video = load_video(video_path, NUM_FRAMES).to(device=device, dtype=torch.float16)
        print(f"  Video shape: {video.shape}")

        with torch.no_grad():
            latent_dist = vae.encode(video).latent_dist
            latent = latent_dist.mean.squeeze(0).float().cpu()
        print(f"  Latent shape: {latent.shape}")

        # Encode text with Reason1 (Qwen2.5-VL)
        with torch.no_grad():
            text_embedding = text_encoder.compute_text_embeddings(
                [caption],
                device=device,
            )
            text_embedding = text_embedding.squeeze(0).float().cpu()
        print(f"  Text embedding shape: {text_embedding.shape}")

        record = {
            "id": record_id,
            "vae_latent_bytes": latent.numpy().tobytes(),
            "vae_latent_shape": list(latent.shape),
            "vae_latent_dtype": str(latent.dtype).replace("torch.", ""),
            "text_embedding_bytes": (text_embedding.numpy().tobytes()),
            "text_embedding_shape": list(text_embedding.shape),
            "text_embedding_dtype": str(text_embedding.dtype).replace("torch.", ""),
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

    # Clean up
    del text_encoder, vae
    torch.cuda.empty_cache()

    # Write parquet
    table = pa.table(
        {k: [r[k] for r in records]
         for k in records[0]},
        schema=pyarrow_schema_t2v,
    )
    output_path = os.path.join(OUTPUT_DIR, "data_00000.parquet")
    pq.write_table(table, output_path)
    print(f"\nWrote {len(records)} records to {output_path}")

    # Write T2W validation prompts (no image_path for T2W)
    val_prompts = {
        "data": [{
            "caption": item["cap"][0],
        } for item in caption_data],
    }
    val_path = os.path.join(OUTPUT_DIR, "validation_prompts.json")
    with open(val_path, "w") as f:
        json.dump(val_prompts, f, indent=2)
    print(f"Wrote validation prompts to {val_path}")

    print("\nDone! Use data_path: " + OUTPUT_DIR + " in training config.")


if __name__ == "__main__":
    main()
