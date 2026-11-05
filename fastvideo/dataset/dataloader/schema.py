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

# HunyuanVideo 1.5 conditions on two text streams (Qwen2.5-VL plus a ByT5 glyph
# stream), so its rows carry a second text-embedding triplet. The paired
# text_attention_mask_2 is synthesised by collate_rows_from_parquet_schema from
# the padding, exactly as text_attention_mask is for the primary stream.
pyarrow_schema_t2v_dual_text = pa.schema(
    list(pyarrow_schema_t2v) + [
        pa.field("text_embedding_2_bytes", pa.binary()),
        pa.field("text_embedding_2_shape", pa.list_(pa.int64())),
        pa.field("text_embedding_2_dtype", pa.string()),
    ])

# One text-to-video-and-audio (T2VA) row owns synchronized video, audio, and text tensors so
# collate_rows_from_parquet_schema cannot pair targets from different samples.
# Each tensor uses the bytes/shape/dtype triplet that the collator discovers.
pyarrow_schema_t2va = pa.schema([
    pa.field("id", pa.string()),
    # --- Video VAE latent tensor ---
    pa.field("vae_latent_bytes", pa.binary()),
    pa.field("vae_latent_shape", pa.list_(pa.int64())),
    pa.field("vae_latent_dtype", pa.string()),
    # --- Stereo audio VAE latent tensor ---
    pa.field("audio_latent_bytes", pa.binary()),
    pa.field("audio_latent_shape", pa.list_(pa.int64())),
    pa.field("audio_latent_dtype", pa.string()),
    # --- Text encoder output tensor ---
    pa.field("text_embedding_bytes", pa.binary()),
    pa.field("text_embedding_shape", pa.list_(pa.int64())),
    pa.field("text_embedding_dtype", pa.string()),
    # --- Paired audio-video metadata ---
    pa.field("file_name", pa.string()),
    pa.field("caption", pa.string()),
    pa.field("media_type", pa.string()),
    pa.field("width", pa.int64()),
    pa.field("height", pa.int64()),
    pa.field("num_frames", pa.int64()),
    pa.field("duration_sec", pa.float64()),
    pa.field("fps", pa.float64()),
    pa.field("audio_sample_rate", pa.int64()),
])

# Ref2VA training consumes the same paired target tensors as T2VA plus the
# inference-equivalent, ordered reference presentation. Reference visual
# latents are stored after channel-major patchification as clean [rows, 96]
# tensors; audio latents use [rows, 32]. Keeping them precomputed avoids
# loading either VAE or Qwen3-VL in every training worker. The ordered list
# metadata is sufficient to rebuild exactly the same RoPE/layout contract as
# build_ref2va_packed_sequence. `reference_noise_seed` makes the 0.999 visual
# noise augmentation stable when a carried DMD trajectory rebuilds its batch.
pyarrow_schema_ref2va = pa.schema([
    pa.field("id", pa.string()),
    # --- Target video/audio latents (may be empty for simulated rollout) ---
    pa.field("vae_latent_bytes", pa.binary()),
    pa.field("vae_latent_shape", pa.list_(pa.int64())),
    pa.field("vae_latent_dtype", pa.string()),
    pa.field("audio_latent_bytes", pa.binary()),
    pa.field("audio_latent_shape", pa.list_(pa.int64())),
    pa.field("audio_latent_dtype", pa.string()),
    # --- Qwen3-VL Ref2VA presentation ---
    pa.field("text_embedding_bytes", pa.binary()),
    pa.field("text_embedding_shape", pa.list_(pa.int64())),
    pa.field("text_embedding_dtype", pa.string()),
    # Tags are stored as float32 values 0/1 for compatibility with existing
    # packed-training rows and cast to long after model-side validation.
    pa.field("text_token_tags_bytes", pa.binary()),
    pa.field("text_token_tags_shape", pa.list_(pa.int64())),
    pa.field("text_token_tags_dtype", pa.string()),
    # --- Clean, normalized reference latent rows ---
    pa.field("reference_video_latent_rows_bytes", pa.binary()),
    pa.field("reference_video_latent_rows_shape", pa.list_(pa.int64())),
    pa.field("reference_video_latent_rows_dtype", pa.string()),
    pa.field("reference_audio_latent_rows_bytes", pa.binary()),
    pa.field("reference_audio_latent_rows_shape", pa.list_(pa.int64())),
    pa.field("reference_audio_latent_rows_dtype", pa.string()),
    # --- Ordered reference geometry ---
    pa.field("reference_media_types", pa.list_(pa.string())),
    pa.field("reference_has_audio", pa.list_(pa.bool_())),
    pa.field("reference_num_latent_frames", pa.list_(pa.int64())),
    pa.field("reference_latent_heights", pa.list_(pa.int64())),
    pa.field("reference_latent_widths", pa.list_(pa.int64())),
    pa.field("reference_num_audio_latents", pa.list_(pa.int64())),
    pa.field("reference_noise_seed", pa.int64()),
    # --- Paired audio-video metadata ---
    pa.field("file_name", pa.string()),
    pa.field("caption", pa.string()),
    pa.field("media_type", pa.string()),
    pa.field("width", pa.int64()),
    pa.field("height", pa.int64()),
    pa.field("num_frames", pa.int64()),
    pa.field("duration_sec", pa.float64()),
    pa.field("fps", pa.float64()),
    pa.field("audio_sample_rate", pa.int64()),
])

pyarrow_schema_ode_trajectory_text_only = pa.schema([
    pa.field("id", pa.string()),
    # --- Text encoder output tensor ---
    # Tensors are stored as raw bytes with shape and dtype info for loading
    pa.field("text_embedding_bytes", pa.binary()),
    # e.g., [SeqLen, Dim]
    pa.field("text_embedding_shape", pa.list_(pa.int64())),
    # e.g., 'bfloat16' or 'float32'
    pa.field("text_embedding_dtype", pa.string()),
    # --- ODE Trajectory ---
    pa.field("trajectory_latents_bytes", pa.binary()),
    pa.field("trajectory_latents_shape", pa.list_(pa.int64())),
    pa.field("trajectory_latents_dtype", pa.string()),
    pa.field("trajectory_timesteps_bytes", pa.binary()),
    pa.field("trajectory_timesteps_shape", pa.list_(pa.int64())),
    pa.field("trajectory_timesteps_dtype", pa.string()),
    # --- Metadata ---
    pa.field("file_name", pa.string()),
    pa.field("caption", pa.string()),
    pa.field("media_type", pa.string()),  # Always 'text' for text-only
])

pyarrow_schema_text_only = pa.schema([
    pa.field("id", pa.string()),
    # --- Text encoder output tensor ---
    # Tensors are stored as raw bytes with shape and dtype info for loading
    pa.field("text_embedding_bytes", pa.binary()),
    # e.g., [SeqLen, Dim]
    pa.field("text_embedding_shape", pa.list_(pa.int64())),
    # e.g., 'bfloat16' or 'float32'
    pa.field("text_embedding_dtype", pa.string()),
    # --- Metadata ---
    pa.field("caption", pa.string()),
])

pyarrow_schema_matrixgame2 = pa.schema([
    pa.field("id", pa.string()),
    # --- Image/Video VAE latents ---
    # Tensors are stored as raw bytes with shape and dtype info for loading
    pa.field("vae_latent_bytes", pa.binary()),
    # e.g., [C, T, H, W] or [C, H, W]
    pa.field("vae_latent_shape", pa.list_(pa.int64())),
    # e.g., 'float32'
    pa.field("vae_latent_dtype", pa.string()),
    #I2V
    pa.field("clip_feature_bytes", pa.binary()),
    pa.field("clip_feature_shape", pa.list_(pa.int64())),
    pa.field("clip_feature_dtype", pa.string()),
    pa.field("first_frame_latent_bytes", pa.binary()),
    pa.field("first_frame_latent_shape", pa.list_(pa.int64())),
    pa.field("first_frame_latent_dtype", pa.string()),
    # --- Action ---
    pa.field("mouse_cond_bytes", pa.binary()),
    pa.field("mouse_cond_shape", pa.list_(pa.int64())),  # [T, 2]
    pa.field("mouse_cond_dtype", pa.string()),
    pa.field("keyboard_cond_bytes", pa.binary()),
    pa.field("keyboard_cond_shape", pa.list_(pa.int64())),  # [T, 4]
    pa.field("keyboard_cond_dtype", pa.string()),
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

pyarrow_schema_matrixgame2_ode_trajectory = pa.schema([
    pa.field("id", pa.string()),
    # --- Action ---
    pa.field("mouse_cond_bytes", pa.binary()),
    pa.field("mouse_cond_shape", pa.list_(pa.int64())),  # [T, 2]
    pa.field("mouse_cond_dtype", pa.string()),
    pa.field("keyboard_cond_bytes", pa.binary()),
    pa.field("keyboard_cond_shape", pa.list_(pa.int64())),  # [T, 4]
    pa.field("keyboard_cond_dtype", pa.string()),
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
    # --- ODE Trajectory ---
    pa.field("trajectory_latents_bytes", pa.binary()),
    pa.field("trajectory_latents_shape", pa.list_(pa.int64())),
    pa.field("trajectory_latents_dtype", pa.string()),
    pa.field("trajectory_timesteps_bytes", pa.binary()),
    pa.field("trajectory_timesteps_shape", pa.list_(pa.int64())),
    pa.field("trajectory_timesteps_dtype", pa.string()),
    # --- Metadata ---
    pa.field("file_name", pa.string()),
    pa.field("caption", pa.string()),
    pa.field("media_type", pa.string()),  # 'image' or 'video'
])
