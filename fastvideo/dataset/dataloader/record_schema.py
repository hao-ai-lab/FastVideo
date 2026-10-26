from collections.abc import Sequence
from numbers import Integral
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
            "id": video_name,
            "vae_latent_bytes": batch.latents[idx].tobytes(),
            "vae_latent_shape": list(batch.latents[idx].shape),
            "vae_latent_dtype": str(batch.latents[idx].dtype),
            "text_embedding_bytes": batch.prompt_embeds[idx].tobytes(),
            "text_embedding_shape": list(batch.prompt_embeds[idx].shape),
            "text_embedding_dtype": str(batch.prompt_embeds[idx].dtype),
            "file_name": video_name,
            "caption": batch.prompt[idx],
            "media_type": "video",
            "width": int(width),
            "height": int(height),
            "num_frames": batch.latents[idx].shape[1] if len(batch.latents[idx].shape) > 1 else 0,
            "duration_sec": duration_val,
            "fps": fps_val,
        }
        records.append(record)

    return records


def i2v_record_creator(batch: PreprocessBatch) -> list[dict[str, Any]]:
    """Create a record for the Parquet dataset with CLIP features."""
    records = basic_t2v_record_creator(batch)

    if len(batch.image_embeds) > 0:
        assert len(batch.image_embeds) == 1, "image embedding should be a single tensor"
        image_embeds = batch.image_embeds[0]
    else:
        image_embeds = None
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
                "first_frame_latent_bytes": image_latent[idx].tobytes(),
                "first_frame_latent_shape": list(image_latent[idx].shape),
                "first_frame_latent_dtype": str(image_latent[idx].dtype),
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


def _float32_tensor_fields(name: str, value: np.ndarray | None) -> dict[str, Any]:
    """Serialize one tensor for the generic parquet collator.

    The reader honors each dtype label. This writer stores float32 consistently
    so reference rows, target latents and embeddings share one portable format.
    """
    if value is None:
        return {
            f"{name}_bytes": b"",
            f"{name}_shape": [],
            f"{name}_dtype": "float32",
        }
    array = np.asarray(value)
    if array.dtype.kind not in "fiu":
        raise ValueError(f"{name} must be a numeric array, got dtype {array.dtype}")
    array = np.ascontiguousarray(array, dtype=np.float32)
    return {
        f"{name}_bytes": array.tobytes(),
        f"{name}_shape": list(array.shape),
        f"{name}_dtype": "float32",
    }


def ref2va_record_creator(
    sample_id: str,
    caption: str,
    *,
    text_embedding: np.ndarray,
    text_token_tags: np.ndarray,
    reference_video_latent_rows: np.ndarray,
    reference_audio_latent_rows: np.ndarray | None,
    reference_media_types: Sequence[str],
    reference_has_audio: Sequence[bool],
    reference_num_latent_frames: Sequence[int],
    reference_latent_heights: Sequence[int],
    reference_latent_widths: Sequence[int],
    reference_num_audio_latents: Sequence[int],
    reference_noise_seed: int,
    vae_latent: np.ndarray | None = None,
    audio_latent: np.ndarray | None = None,
    file_name: str | None = None,
    width: int = 0,
    height: int = 0,
    num_frames: int = 0,
    fps: float = 24.0,
    audio_sample_rate: int = 32_000,
) -> dict[str, Any]:
    """Serialize an already-precomputed MiniMax-H3 Ref2VA training row.

    This is the writable counterpart of ``pyarrow_schema_ref2va``. It does
    not decode raw reference media or run Qwen3-VL/the audio-video VAEs; the
    caller must supply the inference-equivalent presentation and latent rows.
    Target latents may be omitted for a simulated rollout.
    """

    def require_shape(name: str, value: np.ndarray, ndim: int, feature_dim: int | None = None) -> np.ndarray:
        array = np.asarray(value)
        if array.ndim != ndim:
            raise ValueError(f"{name} must have {ndim} dimensions, got shape {array.shape}")
        if feature_dim is not None and array.shape[-1] != feature_dim:
            raise ValueError(f"{name} must end in feature dimension {feature_dim}, got shape {array.shape}")
        return array

    embedding = require_shape("text_embedding", text_embedding, 2, 5120)
    tags = require_shape("text_token_tags", text_token_tags, 1)
    if tags.shape[0] != embedding.shape[0]:
        raise ValueError("text_token_tags length must equal the text embedding sequence length")
    if not bool(np.all((tags == 0) | (tags == 1))):
        raise ValueError("text_token_tags may contain only text (1) and vision (0) tags")

    video_rows = require_shape("reference_video_latent_rows", reference_video_latent_rows, 2, 96)
    if reference_audio_latent_rows is None:
        audio_rows = np.empty((0, 32), dtype=np.float32)
    else:
        audio_rows = require_shape("reference_audio_latent_rows", reference_audio_latent_rows, 2, 32)

    metadata = {
        "reference_media_types": list(reference_media_types),
        "reference_has_audio": list(reference_has_audio),
        "reference_num_latent_frames": list(reference_num_latent_frames),
        "reference_latent_heights": list(reference_latent_heights),
        "reference_latent_widths": list(reference_latent_widths),
        "reference_num_audio_latents": list(reference_num_audio_latents),
    }
    reference_count = len(metadata["reference_media_types"])
    if reference_count == 0:
        raise ValueError("A Ref2VA record requires at least one ordered reference")
    mismatched = {name: len(values) for name, values in metadata.items() if len(values) != reference_count}
    if mismatched:
        raise ValueError("Ref2VA ordered metadata lists must have equal lengths; "
                         f"expected {reference_count}, got {mismatched}")
    if not any(media_type in ("image", "video") for media_type in metadata["reference_media_types"]):
        raise ValueError("A Ref2VA record requires at least one visual reference")

    expected_video_rows = 0
    expected_audio_rows = 0
    for index in range(reference_count):
        media_type = metadata["reference_media_types"][index]
        if media_type not in ("image", "video", "audio"):
            raise ValueError(f"Unsupported Ref2VA media type at index {index}: {media_type!r}")
        has_audio = metadata["reference_has_audio"][index]
        if not isinstance(has_audio, bool):
            raise ValueError(f"reference_has_audio[{index}] must be bool, got {has_audio!r}")

        dimensions = []
        for name in (
                "reference_num_latent_frames",
                "reference_latent_heights",
                "reference_latent_widths",
                "reference_num_audio_latents",
        ):
            value = metadata[name][index]
            if isinstance(value, bool) or not isinstance(value, Integral) or int(value) < 0:
                raise ValueError(f"{name}[{index}] must be a non-negative integer, got {value!r}")
            dimensions.append(int(value))
        latent_frames, latent_height, latent_width, audio_latents = dimensions

        if media_type == "audio":
            if (latent_frames, latent_height, latent_width) != (0, 0, 0):
                raise ValueError("Audio-only references must store zero visual latent geometry")
        else:
            if min(latent_frames, latent_height, latent_width) <= 0:
                raise ValueError(f"Visual reference {index} must have positive latent geometry")
            if latent_height % 2 or latent_width % 2:
                raise ValueError(f"Reference {index} latent geometry must be divisible by patch size (1, 2, 2)")
            if media_type == "image" and latent_frames != 1:
                raise ValueError(f"Reference image {index} must contain one latent frame")
            expected_video_rows += latent_frames * latent_height * latent_width // 4

        if media_type == "image" and has_audio:
            raise ValueError("An image reference cannot carry audio latents")
        if media_type == "audio" and not has_audio:
            raise ValueError("Every audio reference must set reference_has_audio=true")
        if has_audio != (audio_latents > 0):
            raise ValueError(f"Reference {index} has_audio={has_audio} but num_audio_latents={audio_latents}")
        expected_audio_rows += 2 * audio_latents

    if video_rows.shape[0] != expected_video_rows:
        raise ValueError("reference_video_latent_rows has the wrong row count: "
                         f"expected {expected_video_rows}, got {video_rows.shape[0]}")
    if audio_rows.shape[0] != expected_audio_rows:
        raise ValueError("reference_audio_latent_rows has the wrong row count: "
                         f"expected {expected_audio_rows}, got {audio_rows.shape[0]}")
    if isinstance(reference_noise_seed, bool) or not isinstance(reference_noise_seed, Integral) or int(
            reference_noise_seed) < 0:
        raise ValueError(f"reference_noise_seed must be a non-negative integer, got {reference_noise_seed!r}")

    if vae_latent is not None:
        target_video = require_shape("vae_latent", vae_latent, 4)
        if target_video.shape[0] != 24:
            raise ValueError(f"vae_latent must have shape [24, T, H, W], got {target_video.shape}")
    if audio_latent is not None:
        target_audio = require_shape("audio_latent", audio_latent, 3)
        if tuple(target_audio.shape[:2]) != (2, 32):
            raise ValueError(f"audio_latent must have shape [2, 32, T], got {target_audio.shape}")
    if (vae_latent is None) != (audio_latent is None):
        raise ValueError("vae_latent and audio_latent must either both be present or both be omitted")

    duration = float(num_frames) / float(fps) if num_frames > 0 and fps > 0 else 0.0
    record: dict[str, Any] = {
        "id": sample_id,
        "file_name": file_name or sample_id,
        "caption": caption,
        "media_type": "video",
        "width": int(width),
        "height": int(height),
        "num_frames": int(num_frames),
        "duration_sec": duration,
        "fps": float(fps),
        "audio_sample_rate": int(audio_sample_rate),
        **metadata,
        "reference_noise_seed": int(reference_noise_seed),
    }
    for name, value in (
        ("vae_latent", vae_latent),
        ("audio_latent", audio_latent),
        ("text_embedding", embedding),
        ("text_token_tags", tags),
        ("reference_video_latent_rows", video_rows),
        ("reference_audio_latent_rows", audio_rows),
    ):
        record.update(_float32_tensor_fields(name, value))
    return record


def ode_text_only_record_creator(video_name: str, text_embedding: np.ndarray, caption: str,
                                 trajectory_latents: np.ndarray, trajectory_timesteps: np.ndarray) -> dict[str, Any]:
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


def text_only_record_creator(text_name: str, text_embedding: np.ndarray, caption: str) -> dict[str, Any]:
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


def matrixgame2_ode_record_creator(video_name: str,
                                   clip_feature: np.ndarray,
                                   first_frame_latent: np.ndarray,
                                   trajectory_latents: np.ndarray,
                                   trajectory_timesteps: np.ndarray,
                                   pil_image: np.ndarray | None = None,
                                   keyboard_cond: np.ndarray | None = None,
                                   mouse_cond: np.ndarray | None = None,
                                   caption: str = "") -> dict[str, Any]:
    """Create a ODE trajectory record matching pyarrow_schema_matrixgame2_ode_trajectory.

    Args:
        video_name: Base name/id for the sample.
        clip_feature: CLIP image features.
        first_frame_latent: First frame VAE latent.
        trajectory_latents: Collected trajectory latents array.
        trajectory_timesteps: Collected timesteps array.
        pil_image: Optional PIL image array for validation.
        keyboard_cond: Optional keyboard action array.
        mouse_cond: Optional mouse action array.
        caption: Optional caption.

    Returns:
        dict suitable for records_to_table(…, pyarrow_schema_matrixgame2_ode_trajectory)
    """
    assert trajectory_latents is not None, "trajectory_latents is required"
    assert trajectory_timesteps is not None, "trajectory_timesteps is required"
    assert clip_feature is not None, "clip_feature is required"
    assert first_frame_latent is not None, "first_frame_latent is required"

    record = {
        "id": video_name,
        "file_name": video_name,
        "caption": caption,
        "media_type": "video",
    }

    # I2V features
    record.update({
        "clip_feature_bytes": clip_feature.tobytes(),
        "clip_feature_shape": list(clip_feature.shape),
        "clip_feature_dtype": str(clip_feature.dtype),
    })

    record.update({
        "first_frame_latent_bytes": first_frame_latent.tobytes(),
        "first_frame_latent_shape": list(first_frame_latent.shape),
        "first_frame_latent_dtype": str(first_frame_latent.dtype),
    })

    # Optional PIL Image
    if pil_image is not None:
        record.update({
            "pil_image_bytes": pil_image.tobytes(),
            "pil_image_shape": list(pil_image.shape),
            "pil_image_dtype": str(pil_image.dtype),
        })
    else:
        record.update({
            "pil_image_bytes": b"",
            "pil_image_shape": [],
            "pil_image_dtype": "",
        })

    # Actions
    if keyboard_cond is not None:
        record.update({
            "keyboard_cond_bytes": keyboard_cond.tobytes(),
            "keyboard_cond_shape": list(keyboard_cond.shape),
            "keyboard_cond_dtype": str(keyboard_cond.dtype),
        })
    else:
        record.update({
            "keyboard_cond_bytes": b"",
            "keyboard_cond_shape": [],
            "keyboard_cond_dtype": "",
        })

    if mouse_cond is not None:
        record.update({
            "mouse_cond_bytes": mouse_cond.tobytes(),
            "mouse_cond_shape": list(mouse_cond.shape),
            "mouse_cond_dtype": str(mouse_cond.dtype),
        })
    else:
        record.update({
            "mouse_cond_bytes": b"",
            "mouse_cond_shape": [],
            "mouse_cond_dtype": "",
        })

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
