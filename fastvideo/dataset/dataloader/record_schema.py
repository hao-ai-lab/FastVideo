from typing import Any

import numpy as np

from fastvideo.pipelines.pipeline_batch_info import PreprocessBatch


def basic_t2v_record_creator(batch: PreprocessBatch) -> list[dict[str, Any]]:
    """Create a record for the Parquet dataset from PreprocessBatch."""
    # For batch processing, we need to handle the case where some fields might be single values
    # or lists depending on the batch size

    assert isinstance(batch.prompt, list)
    assert isinstance(batch.width, list)
    assert isinstance(batch.height, list)
    assert isinstance(batch.fps, list)
    assert isinstance(batch.num_frames, list)

    records = []
    for idx, video_name in enumerate(batch.video_file_name):
        width = batch.width[idx] if batch.width is not None else 0
        height = batch.height[idx] if batch.height is not None else 0

        # Get FPS - single value in PreprocessBatch
        fps_val = float(batch.fps[idx]) if batch.fps is not None else 0.0

        # For duration, we need to calculate it or use a default since it's not in PreprocessBatch
        # duration = num_frames / fps if available
        duration_val = 0.0
        if batch.num_frames[idx] and batch.fps[idx] and batch.fps[idx] > 0:
            duration_val = float(batch.num_frames[idx]) / float(batch.fps[idx])

        record = {
            "id":
            video_name,
            "vae_latent_bytes":
            batch.latents[idx].tobytes(),
            "vae_latent_shape":
            list(batch.latents[idx].shape),
            "vae_latent_dtype":
            str(batch.latents[idx].dtype),
            "text_embedding_bytes":
            batch.prompt_embeds[idx].tobytes(),
            "text_embedding_shape":
            list(batch.prompt_embeds[idx].shape),
            "text_embedding_dtype":
            str(batch.prompt_embeds[idx].dtype),
            "file_name":
            video_name,
            "caption":
            batch.prompt[idx],
            "media_type":
            "video",
            "width":
            int(width),
            "height":
            int(height),
            "num_frames":
            batch.latents[idx].shape[1]
            if len(batch.latents[idx].shape) > 1 else 0,
            "duration_sec":
            duration_val,
            "fps":
            fps_val,
        }
        records.append(record)

    return records


def i2v_record_creator(batch: PreprocessBatch) -> list[dict[str, Any]]:
    """Create a record for the Parquet dataset with CLIP features."""
    records = basic_t2v_record_creator(batch)

    assert len(
        batch.image_embeds) == 1, "image embedding should be a single tensor"
    image_embeds = batch.image_embeds[0]
    image_latent = batch.image_latent
    pil_image = batch.pil_image

    for idx, record in enumerate(records):
        if image_embeds is not None:
            record.update({
                "clip_feature_bytes": image_embeds[idx].tobytes(),
                "clip_feature_shape": list(image_embeds[idx].shape),
                "clip_feature_dtype": str(image_embeds[idx].dtype),
            })
        else:
            record.update({
                "clip_feature_bytes": b"",
                "clip_feature_shape": [],
                "clip_feature_dtype": "",
            })

        if image_latent is not None:
            record.update({
                "first_frame_latent_bytes":
                image_latent[idx].tobytes(),
                "first_frame_latent_shape":
                list(image_latent[idx].shape),
                "first_frame_latent_dtype":
                str(image_latent[idx].dtype),
            })
        else:
            record.update({
                "first_frame_latent_bytes": b"",
                "first_frame_latent_shape": [],
                "first_frame_latent_dtype": "",
            })

        if pil_image is not None:
            record.update({
                "pil_image_bytes": pil_image[idx].tobytes(),
                "pil_image_shape": list(pil_image[idx].shape),
                "pil_image_dtype": str(pil_image[idx].dtype),
            })
        else:
            record.update({
                "pil_image_bytes": b"",
                "pil_image_shape": [],
                "pil_image_dtype": "",
            })

    return records


def ode_text_only_record_creator(
        video_name: str, text_embedding: np.ndarray, caption: str,
        trajectory_latents: np.ndarray,
        trajectory_timesteps: np.ndarray) -> dict[str, Any]:
    """Create a text-only ODE trajectory record matching pyarrow_schema_ode_trajectory_text_only.

    Args:
        video_name: Base name/id for the sample (without extension).
        text_embedding: Text encoder output array [SeqLen, Dim].
        caption: Original text prompt.
        trajectory_latents: Collected trajectory latents array.
        trajectory_timesteps: Collected timesteps array.

    Returns:
        dict suitable for records_to_table(…, pyarrow_schema_ode_trajectory_text_only)
    """
    assert trajectory_latents is not None, "trajectory_latents is required"
    assert trajectory_timesteps is not None, "trajectory_timesteps is required"

    record = {
        "id": f"text_{video_name}",
        "text_embedding_bytes": text_embedding.tobytes(),
        "text_embedding_shape": list(text_embedding.shape),
        "text_embedding_dtype": str(text_embedding.dtype),
        "file_name": video_name,
        "caption": caption,
        "media_type": "text",
    }

    record.update({
        "trajectory_latents_bytes": trajectory_latents.tobytes(),
        "trajectory_latents_shape": list(trajectory_latents.shape),
        "trajectory_latents_dtype": str(trajectory_latents.dtype),
    })

    record.update({
        "trajectory_timesteps_bytes": trajectory_timesteps.tobytes(),
        "trajectory_timesteps_shape": list(trajectory_timesteps.shape),
        "trajectory_timesteps_dtype": str(trajectory_timesteps.dtype),
    })

    return record


def text_only_record_creator(text_name: str, text_embedding: np.ndarray,
                             caption: str) -> dict[str, Any]:
    """Create a text-only record matching pyarrow_schema_text_only.

    Args:
        text_name: Base id/name for the text sample.
        text_embedding: Text encoder output array [SeqLen, Dim].
        caption: Original text prompt.

    Returns:
        dict suitable for records_to_table(…, pyarrow_schema_text_only)
    """
    record = {
        "id": f"text_{text_name}",
        "text_embedding_bytes": text_embedding.tobytes(),
        "text_embedding_shape": list(text_embedding.shape),
        "text_embedding_dtype": str(text_embedding.dtype),
        "caption": caption,
    }
    return record
