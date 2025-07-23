# SPDX-License-Identifier: Apache-2.0
# schema.py
"""
Unified data schema and format for saving and loading image/video data after
preprocessing.

It uses apache arrow in-memory format that can be consumed by modern data
frameworks that can handle parquet or lance file.
"""

import pyarrow as pa

pyarrow_schema_i2v = pa.schema([
    pa.field("id", pa.string()),
    # --- Image/Video VAE latents ---
    # Tensors are stored as raw bytes with shape and dtype info for loading
    pa.field("vae_latent_bytes", pa.binary()),
    # e.g., [C, T, H, W] or [C, H, W]
    pa.field("vae_latent_shape", pa.list_(pa.int64())),
    # e.g., 'float32'
    pa.field("vae_latent_dtype", pa.string()),
    # --- Text encoder output tensor ---
    # Tensors are stored as raw bytes with shape and dtype info for loading
    pa.field("text_embedding_bytes", pa.binary()),
    # e.g., [SeqLen, Dim]
    pa.field("text_embedding_shape", pa.list_(pa.int64())),
    # e.g., 'bfloat16' or 'float32'
    pa.field("text_embedding_dtype", pa.string()),
    #I2V
    pa.field("clip_feature_bytes", pa.binary()),
    pa.field("clip_feature_shape", pa.list_(pa.int64())),
    pa.field("clip_feature_dtype", pa.string()),
    pa.field("first_frame_latent_bytes", pa.binary()),
    pa.field("first_frame_latent_shape", pa.list_(pa.int64())),
    pa.field("first_frame_latent_dtype", pa.string()),
    # I2V Validation
    pa.field("pil_image_bytes", pa.binary()),
    pa.field("pil_image_shape", pa.list_(pa.int64())),
    pa.field("pil_image_dtype", pa.string()),
    # --- Metadata ---
    pa.field("file_name", pa.string()),
    pa.field("caption", pa.string()),
    pa.field("media_type", pa.string()),  # 'image' or 'video'
    pa.field("width", pa.int64()),
    pa.field("height", pa.int64()),
    # -- Video-specific (can be null/default for images) ---
    # Number of frames processed (e.g., 1 for image, N for video)
    pa.field("num_frames", pa.int64()),
    pa.field("duration_sec", pa.float64()),
    pa.field("fps", pa.float64()),
])

pyarrow_schema_t2v = pa.schema([
    pa.field("id", pa.string()),
    # --- Image/Video VAE latents ---
    # Tensors are stored as raw bytes with shape and dtype info for loading
    pa.field("vae_latent_bytes", pa.binary()),
    # e.g., [C, T, H, W] or [C, H, W]
    pa.field("vae_latent_shape", pa.list_(pa.int64())),
    # e.g., 'float32'
    pa.field("vae_latent_dtype", pa.string()),
    # --- Text encoder output tensor ---
    # Tensors are stored as raw bytes with shape and dtype info for loading
    pa.field("text_embedding_bytes", pa.binary()),
    # e.g., [SeqLen, Dim]
    pa.field("text_embedding_shape", pa.list_(pa.int64())),
    # e.g., 'bfloat16' or 'float32'
    pa.field("text_embedding_dtype", pa.string()),
    # --- Metadata ---
    pa.field("file_name", pa.string()),
    pa.field("caption", pa.string()),
    pa.field("media_type", pa.string()),  # 'image' or 'video'
    pa.field("width", pa.int64()),
    pa.field("height", pa.int64()),
    # -- Video-specific (can be null/default for images) ---
    # Number of frames processed (e.g., 1 for image, N for video)
    pa.field("num_frames", pa.int64()),
    pa.field("duration_sec", pa.float64()),
    pa.field("fps", pa.float64()),
])
