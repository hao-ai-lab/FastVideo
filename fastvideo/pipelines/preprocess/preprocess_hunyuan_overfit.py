# SPDX-License-Identifier: Apache-2.0
"""Preprocess HunyuanVideo overfit data into parquet format.

Encodes videos with HunyuanVideo VAE and captions with dual text
encoders (LLaMA + CLIP) into the t2v parquet schema expected by
the training framework.

Usage:
    CUDA_VISIBLE_DEVICES=0 python scripts/preprocess_hunyuan_overfit.py
"""

import glob
import json
import os

import cv2
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from safetensors.torch import load_file as safetensors_load_file

from fastvideo.configs.models.vaes import HunyuanVAEConfig
from fastvideo.configs.pipelines.hunyuan import (
    clip_preprocess_text,
    clip_postprocess_text,
    llama_preprocess_text,
    llama_postprocess_text,
)
from fastvideo.dataset.dataloader.schema import pyarrow_schema_t2v
from fastvideo.models.vaes.hunyuanvae import AutoencoderKLHunyuanVideo
from fastvideo.utils import maybe_download_model

# --- Config ---
NUM_FRAMES = 77  # 4k+1 for temporal compression ratio 4
MAX_HEIGHT = 480
MAX_WIDTH = 832
TRAIN_FPS = 16.0

DATA_DIR = "data/hunyuan_overfit"
OUTPUT_DIR = "data/hunyuan_overfit_preprocessed"


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
    model_path = maybe_download_model("hunyuanvideo-community/HunyuanVideo")
    vae_path = os.path.join(model_path, "vae")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load captions
    with open(os.path.join(DATA_DIR, "videos2caption.json")) as f:
        caption_data = json.load(f)

    # --- Load VAE ---
    print("Loading HunyuanVideo VAE...")
    vae_config = HunyuanVAEConfig()
    vae_config.load_encoder = True
    vae_config.load_decoder = False
    vae = AutoencoderKLHunyuanVideo(vae_config)
    sf_files = glob.glob(os.path.join(vae_path, "*.safetensors"))
    loaded = {}
    for sf_file in sf_files:
        loaded.update(safetensors_load_file(sf_file))
    vae.load_state_dict(loaded, strict=False)
    vae = vae.to(device=device, dtype=torch.float16).eval()
    vae.use_parallel_tiling = False
    print(f"VAE loaded ({sum(p.numel() for p in vae.parameters())/1e6:.0f}M)")

    # --- Load text encoders ---
    print("Loading LLaMA text encoder...")
    from transformers import AutoTokenizer, CLIPTextModel, LlamaModel
    from fastvideo.configs.pipelines.hunyuan import (LlamaConfig, CLIPTextConfig)

    llama_cfg = LlamaConfig()
    llama_tok = AutoTokenizer.from_pretrained(os.path.join(model_path, "tokenizer"))
    llama_enc = LlamaModel.from_pretrained(
        os.path.join(model_path, "text_encoder"),
        torch_dtype=torch.float16,
    ).to(device).eval()
    llama_tok_kwargs = dict(llama_cfg.tokenizer_kwargs)

    print("Loading CLIP text encoder...")
    clip_cfg = CLIPTextConfig()
    clip_tok = AutoTokenizer.from_pretrained(os.path.join(model_path, "tokenizer_2"))
    clip_enc = CLIPTextModel.from_pretrained(
        os.path.join(model_path, "text_encoder_2"),
        torch_dtype=torch.float16,
    ).to(device).eval()
    clip_tok_kwargs = dict(clip_cfg.tokenizer_kwargs)

    # --- Process each video ---
    records = []
    for item in caption_data:
        video_name = item["path"]
        caption = item["cap"][0]
        video_path = os.path.join(DATA_DIR, "videos", video_name)

        print(f"\nProcessing: {video_name}")
        print(f"  Caption: {caption[:80]}...")

        # Encode video
        video = load_video(video_path, NUM_FRAMES).to(device=device, dtype=torch.float16)
        print(f"  Video shape: {video.shape}")

        with torch.no_grad():
            latent_dist = vae.encode(video)
            # Cast to fp32 — dataloader hardcodes np.float32
            latent = latent_dist.mean.squeeze(0).float().cpu()
        print(f"  Latent shape: {latent.shape}")

        # Encode text with LLaMA
        llama_text = llama_preprocess_text(caption)
        with torch.no_grad():
            llama_inputs = llama_tok(llama_text, **llama_tok_kwargs).to(device)
            llama_out = llama_enc(**llama_inputs, output_hidden_states=True)
            llama_embeds = llama_postprocess_text(llama_out).squeeze(0)  # [seq, dim]

        # Encode text with CLIP
        clip_text = clip_preprocess_text(caption)
        with torch.no_grad():
            clip_inputs = clip_tok(clip_text, **clip_tok_kwargs).to(device)
            clip_out = clip_enc(**clip_inputs)
            clip_pooled = clip_postprocess_text(clip_out).squeeze(0)  # [dim]

        # Combine: [pooled_clip_row, llama_embeds]
        llama_dim = llama_embeds.shape[-1]
        pooled_row = torch.zeros(llama_dim, device=device, dtype=torch.float16)
        pooled_row[:clip_pooled.shape[-1]] = clip_pooled
        text_embedding = torch.cat(
            [pooled_row.unsqueeze(0), llama_embeds],
            dim=0,
        ).float().cpu()  # [seq+1, dim]
        print(f"  Text embedding shape: {text_embedding.shape}")

        record = {
            "id": video_name,
            "vae_latent_bytes": latent.numpy().tobytes(),
            "vae_latent_shape": list(latent.shape),
            "vae_latent_dtype": str(latent.dtype).replace("torch.", ""),
            "text_embedding_bytes": text_embedding.numpy().tobytes(),
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

    # Clean up encoders
    del llama_enc, llama_tok, clip_enc, clip_tok, vae

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
    val_prompts = {"data": [{"caption": item["cap"][0]} for item in caption_data]}
    val_path = os.path.join(OUTPUT_DIR, "validation_prompts.json")
    with open(val_path, "w") as f:
        json.dump(val_prompts, f, indent=2)
    print(f"Wrote validation prompts to {val_path}")

    print("\nDone! Use data_path: " + OUTPUT_DIR + " in training config.")


if __name__ == "__main__":
    main()
