# generate_dummy_data.py
"""
Generates a dummy Lance dataset conforming to the schema defined in schema.py.
Run this script before using the lance_video_dataset.py example.
"""

import lance
import pyarrow as pa
import numpy as np
import os
import random
import time
import argparse
import logging

from schema import schema

DEFAULT_VAE_LATENT_CHANNELS = 16
DEFAULT_VAE_LATENT_HEIGHT = 90
DEFAULT_VAE_LATENT_WIDTH = 160
DEFAULT_T5_EMBED_DIM = 4096  # Example T5 XXL dimension
DEFAULT_MAX_TEXT_SEQ_LEN = 512  # Example Wan-2.1 dimension

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def generate_dummy_lance_dataset(
    output_path: str,
    num_items: int,
    max_t_frames: int,
    overwrite: bool = False,
):
    """Generates and writes a dummy Lance dataset."""
    mode = "overwrite" if overwrite else "create"
    if not overwrite and os.path.exists(output_path):
        try:
            ds = lance.dataset(output_path)
            logger.warning(
                f"Dataset path {output_path} already exists with {len(ds)} rows. "
                f"Skipping generation. Use --overwrite to replace."
            )
            return
        except Exception as e:
            logger.warning(
                f"Could not read existing dataset at {output_path}: {e}. Will attempt to {mode}."
            )
            # Fall through to potentially overwrite/create if reading failed but overwrite is False

    logger.info(
        f"Generating {num_items} dummy data items for {output_path} (mode: {mode})..."
    )
    records = []
    start_time = time.time()

    for i in range(num_items):
        # Simulate variable temporal length for latents
        t_frames = random.randint(max_t_frames // 2, max_t_frames)
        if t_frames < 1:
            t_frames = 1

        # Simulate variable *actual* text sequence length before padding
        actual_seq_len = random.randint(
            DEFAULT_MAX_TEXT_SEQ_LEN // 4, DEFAULT_MAX_TEXT_SEQ_LEN
        )
        padded_seq_len = DEFAULT_MAX_TEXT_SEQ_LEN  # Assume fixed padding length

        # --- Create dummy tensors ---
        # VAE Latents (C, T, H, W) - Use float16 for space? Or float32? Match model.
        latent_np = np.random.randn(
            DEFAULT_VAE_LATENT_CHANNELS,
            t_frames,
            DEFAULT_VAE_LATENT_HEIGHT,
            DEFAULT_VAE_LATENT_WIDTH,
        ).astype(np.float32)  # Or float16

        # T5 Embeddings (PaddedSeqLen, Dim) - Often float16 or bfloat16
        embedding_np = np.random.randn(
            padded_seq_len, DEFAULT_T5_EMBED_DIM
        ).astype(np.float16)

        # Attention Mask (PaddedSeqLen) - 1 for actual tokens, 0 for padding
        mask_np = np.zeros(
            padded_seq_len, dtype=np.uint8
        )  # Store bools as uint8
        mask_np[:actual_seq_len] = 1

        # --- Create record dictionary ---
        records.append(
            {
                "id": f"item_{i:06d}",
                # VAE Latent Data
                "vae_latent_bytes": latent_np.tobytes(),
                "vae_latent_shape": list(latent_np.shape),
                "vae_latent_dtype": str(latent_np.dtype),
                # Text Embedding Data
                "text_embedding_bytes": embedding_np.tobytes(),
                "text_embedding_shape": list(embedding_np.shape),
                "text_embedding_dtype": str(embedding_np.dtype),
                # Text Mask Data
                "text_attention_mask_bytes": mask_np.tobytes(),
                "text_attention_mask_shape": list(mask_np.shape),
                "text_attention_mask_dtype": "bool",  # Logical type is bool
                # Dummy Metadata (can be None if schema allows nullable)
                "file_name": f"video_{i:06d}.mp4",
                "caption": f"Dummy caption for video {i}",
                "media_type": "video",
                "width": DEFAULT_VAE_LATENT_WIDTH * 8,  # Guess original size
                "height": DEFAULT_VAE_LATENT_HEIGHT * 8,
                "num_frames": t_frames,
                "duration_sec": t_frames / 15.0,  # Dummy fps
                "fps": 15.0,
            }
        )
        if (i + 1) % (num_items // 10 if num_items >= 10 else 1) == 0:
            logger.info(f"Generated {i+1}/{num_items} records...")

    # Create PyArrow Table and write to Lance dataset
    try:
        logger.info("Converting records to PyArrow Table...")
        table = pa.Table.from_pylist(records, schema=schema)
        logger.info(f"Writing Lance dataset to {output_path} (mode: {mode})...")
        lance.write_dataset(
            table, output_path, mode=mode, max_rows_per_group=1024 * 10
        )
        end_time = time.time()
        logger.info(
            f"Successfully generated dummy dataset with {len(records)} items "
            f"at {output_path} in {end_time - start_time:.2f} seconds."
        )
    except Exception as e:
        logger.error(f"Error writing dummy Lance dataset: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate Dummy Lance Video Dataset"
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="./dummy_lance_video_data.lance",
        help="Path to save the generated Lance dataset.",
    )
    parser.add_argument(
        "--num-items",
        type=int,
        default=1000,
        help="Number of dummy data items to generate.",
    )
    parser.add_argument(
        "--max-t-frames",
        type=int,
        default=64,  # Max latent time frames
        help="Maximum number of temporal frames for VAE latents.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite the dataset if it already exists.",
    )
    args = parser.parse_args()

    generate_dummy_lance_dataset(
        output_path=args.output_path,
        num_items=args.num_items,
        max_t_frames=args.max_t_frames,
        overwrite=args.overwrite,
    )
