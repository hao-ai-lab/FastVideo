from typing import Any

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
