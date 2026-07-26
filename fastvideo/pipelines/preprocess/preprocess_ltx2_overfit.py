# SPDX-License-Identifier: Apache-2.0
"""Preprocess LTX-2 overfit data into parquet format.

Encodes videos with the LTX-2 causal video VAE and captions with the
Gemma text encoder (feature extractor + embedding connector) into the
t2v parquet schema expected by the training framework.

The stored text embeddings are POST-connector: the connector replaces
pad positions with learnable registers and returns an all-valid mask,
so the parquet collate's ones/zeros mask stays semantically correct
and training needs no text encoder at all. Captions are encoded via
the encoder's forward() (the exact inference path), which handles both
LTX-2.0 (shared 3840-d features) and LTX-2.3 (separate 4096-d video /
2048-d audio feature extractors).

Videos are resampled to TRAIN_FPS and the preprocessed clip is also
saved as an mp4 next to the parquet so overfit tests can use it as
the SSIM reference.

Usage:
    CUDA_VISIBLE_DEVICES=0 python fastvideo/pipelines/preprocess/preprocess_ltx2_overfit.py
"""

import json
import os
import shutil
from typing import Any

import av
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from fastvideo.dataset.dataloader.schema import pyarrow_schema_t2v
from fastvideo.utils import maybe_download_model, verify_model_config_and_directory

# --- Config ---
NUM_FRAMES = 81  # 8k+1 for temporal compression ratio 8
MAX_HEIGHT = 480  # divisible by 32 (spatial compression)
MAX_WIDTH = 832
TRAIN_FPS = 24.0  # matches the LTX-2 preset fps used at validation

DATA_DIR = os.environ.get("LTX2_OVERFIT_DATA_DIR", "data/cats")
CAPTION_JSON = os.environ.get("LTX2_OVERFIT_CAPTION_JSON", "videos2caption_1_sample.json")
VIDEO_SUBDIR = os.environ.get("LTX2_OVERFIT_VIDEO_SUBDIR", "video")
OUTPUT_DIR = os.environ.get("LTX2_OVERFIT_OUTPUT_DIR", "data/ltx2_overfit_preprocessed")
MODEL_REPO = os.environ.get("LTX2_OVERFIT_MODEL", "FastVideo/LTX2-Distilled-Diffusers")
# The train dataloader samples with drop_last=True across data-parallel
# groups, so the dataset must hold at least num_sp_groups * batch_size
# rows or every rank gets zero batches. Replicate the overfit sample so
# a 4-GPU FSDP run still sees one batch per rank.
NUM_COPIES = int(os.environ.get("LTX2_OVERFIT_NUM_COPIES", "4"))


def _init_single_process_distributed() -> None:
    """FastVideo component loaders expect an initialized distributed env."""
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29511")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("LOCAL_RANK", "0")
    from fastvideo.distributed import (
        maybe_init_distributed_environment_and_model_parallel, )
    maybe_init_distributed_environment_and_model_parallel(1, 1)


def load_video(path: str, num_frames: int, target_fps: float, height: int,
               width: int) -> tuple[torch.Tensor, np.ndarray]:
    """Load a video as [1, C, T, H, W] in [-1, 1], resampled to target_fps.

    Also returns the uint8 RGB frames [T, H, W, C] for reference-video export.
    """
    with av.open(path) as container:
        if not container.streams.video:
            raise RuntimeError(f"No video stream found in {path}")
        stream = container.streams.video[0]
        native_fps = float(stream.average_rate or target_fps)
        decoded = [torch.from_numpy(frame.to_ndarray(format="rgb24")) for frame in container.decode(video=0)]
    if not decoded:
        raise RuntimeError(f"Could not read any frames from {path}")
    raw = torch.stack(decoded).permute(0, 3, 1, 2)  # [T, C, H, W] uint8
    step = native_fps / target_fps
    wanted = [min(int(round(i * step)), raw.shape[0] - 1) for i in range(num_frames)]

    frames = raw[wanted].float()  # [T, C, H, W] in [0, 255]
    src_h, src_w = frames.shape[2], frames.shape[3]
    scale = max(height / src_h, width / src_w)
    new_h, new_w = int(round(src_h * scale)), int(round(src_w * scale))
    frames = torch.nn.functional.interpolate(frames, size=(new_h, new_w), mode="bilinear", antialias=True)
    top = (new_h - height) // 2
    left = (new_w - width) // 2
    frames = frames[:, :, top:top + height, left:left + width]

    frames_np = (frames.permute(0, 2, 3, 1).clamp(0, 255).round().to(torch.uint8).numpy())
    video = frames / 127.5 - 1.0  # [0,255] -> [-1,1]
    video = video.permute(1, 0, 2, 3).unsqueeze(0)  # [1,C,T,H,W]
    return video, frames_np


def main() -> None:
    _init_single_process_distributed()

    from fastvideo.fastvideo_args import FastVideoArgs
    from fastvideo.models.loader.component_loader import (
        PipelineComponentLoader, )
    from fastvideo.pipelines.basic.ltx2.pipeline_configs import LTX2T2VConfig

    device = torch.device("cuda:0")
    model_path = maybe_download_model(MODEL_REPO)
    model_index = verify_model_config_and_directory(model_path)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    # The map-style dataset caches parquet file metadata; a stale cache
    # next to a regenerated parquet can crash or serve old rows.
    shutil.rmtree(os.path.join(OUTPUT_DIR, "map_style_cache"), ignore_errors=True)

    with open(os.path.join(DATA_DIR, CAPTION_JSON)) as f:
        caption_data = json.load(f)

    pipeline_config = LTX2T2VConfig()
    fastvideo_args = FastVideoArgs(
        model_path=model_path,
        pipeline_config=pipeline_config,
        num_gpus=1,
        tp_size=1,
        sp_size=1,
        hsdp_shard_dim=1,
        vae_cpu_offload=False,
        text_encoder_cpu_offload=False,
    )

    def load_component(name: str) -> Any:
        transformers_or_diffusers, _ = model_index[name]
        return PipelineComponentLoader.load_module(
            module_name=name,
            component_model_path=os.path.join(model_path, name),
            transformers_or_diffusers=transformers_or_diffusers,
            fastvideo_args=fastvideo_args,
        )

    print("Loading LTX-2 VAE...")
    vae = load_component("vae")
    vae_dtype = next(vae.parameters()).dtype
    print(f"VAE loaded ({sum(p.numel() for p in vae.parameters())/1e6:.0f}M, {vae_dtype})")

    print("Loading Gemma text encoder + tokenizer...")
    text_encoder = load_component("text_encoder")
    tokenizer = load_component("tokenizer")
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    encoder_config = pipeline_config.text_encoder_configs[0]
    tokenizer_kwargs = dict(encoder_config.tokenizer_kwargs)
    if "max_length" not in tokenizer_kwargs:
        tokenizer_kwargs["max_length"] = encoder_config.arch_config.text_len
    preprocess_text = pipeline_config.preprocess_text_funcs[0]

    # --- Process each video ---
    records = []
    for idx, item in enumerate(caption_data):
        video_name = item["path"]
        record_id = f"{idx:04d}_{video_name}"
        caption = item["cap"][0] if isinstance(item["cap"], list) else item["cap"]
        video_path = os.path.join(DATA_DIR, VIDEO_SUBDIR, video_name)

        print(f"\nProcessing: {video_name}")
        print(f"  Caption: {caption[:80]}...")

        video, frames_np = load_video(video_path, NUM_FRAMES, TRAIN_FPS, MAX_HEIGHT, MAX_WIDTH)
        video = video.to(device=device, dtype=vae_dtype)
        print(f"  Video shape: {video.shape}")

        with torch.no_grad():
            # LTX-2 encode() returns a deterministic distribution whose
            # mean is already per-channel normalized; store as-is.
            latent = vae.encode(video).mean.squeeze(0).float().cpu()
        print(f"  Latent shape: {latent.shape}")

        with torch.no_grad():
            # Encode through forward() — the inference text path. The
            # two-step preprocess_text_embeddings + run_connectors route
            # breaks on LTX-2.3, whose separate audio feature extractor
            # is narrower than the video one.
            text_inputs = tokenizer([preprocess_text(caption)], **tokenizer_kwargs)
            encoder_out = text_encoder(
                input_ids=text_inputs["input_ids"].to(device),
                attention_mask=text_inputs["attention_mask"].to(device),
            )
            text_embedding = encoder_out.last_hidden_state.squeeze(0).float().cpu()
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

        # Save the preprocessed clip so overfit tests can compare
        # validation output against the memorization target.
        import imageio
        ref_path = os.path.join(OUTPUT_DIR, f"training_sample_{idx}.mp4")
        with imageio.get_writer(ref_path, fps=TRAIN_FPS) as writer:
            for frame in frames_np:
                writer.append_data(frame)
        print(f"  Wrote reference clip to {ref_path}")

    # Clean up
    del text_encoder, tokenizer, vae
    torch.cuda.empty_cache()

    # Write parquet (replicated NUM_COPIES times; see comment at top)
    replicated = []
    for copy_idx in range(max(1, NUM_COPIES)):
        for r in records:
            row = dict(r)
            row["id"] = f"{r['id']}_copy{copy_idx}"
            replicated.append(row)
    table = pa.table(
        {k: [r[k] for r in replicated]
         for k in replicated[0]},
        schema=pyarrow_schema_t2v,
    )
    output_path = os.path.join(OUTPUT_DIR, "data_00000.parquet")
    pq.write_table(table, output_path)
    print(f"\nWrote {len(replicated)} records "
          f"({len(records)} unique x {max(1, NUM_COPIES)} copies) to {output_path}")

    # Write validation prompts for the validation callback
    val_prompts = {
        "data": [{
            "caption": (item["cap"][0] if isinstance(item["cap"], list) else item["cap"]),
        } for item in caption_data],
    }
    val_path = os.path.join(OUTPUT_DIR, "validation_prompts.json")
    with open(val_path, "w") as f:
        json.dump(val_prompts, f, indent=2)
    print(f"Wrote validation prompts to {val_path}")

    print("\nDone! Use data_path: " + OUTPUT_DIR + " in training config.")


if __name__ == "__main__":
    main()
