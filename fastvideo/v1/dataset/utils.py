from typing import Any, Dict, List

import numpy as np
import torch

from fastvideo.v1.logger import init_logger

logger = init_logger(__name__)


def pad(t: torch.Tensor, padding_length: int) -> torch.Tensor:
    """
    Pad or crop an embedding [L, D] to exactly padding_length tokens.
    Return:
    - [L, D] tensor in pinned CPU memory
    - [L] attention mask in pinned CPU memory
    """
    L, D = t.shape
    if padding_length > L:  # pad
        pad = torch.zeros(padding_length - L, D, dtype=t.dtype, device=t.device)
        return torch.cat([t, pad], 0), torch.cat(
            [torch.ones(L), torch.zeros(padding_length - L)], 0)
    else:  # crop
        return t[:padding_length], torch.ones(padding_length)


def get_torch_tensors_from_row_dict(row_dict, keys) -> Dict[str, Any]:
    """
    Get the latents and prompts from a row dictionary.
    """
    return_dict = {}
    for key in keys:
        shape, bytes = None, None
        if isinstance(key, tuple):
            for k in key:
                try:
                    shape = row_dict[f"{k}_shape"]
                    bytes = row_dict[f"{k}_bytes"]
                except KeyError:
                    continue
            key = key[0]
            if shape is None or bytes is None:
                raise ValueError(f"Key {key} not found in row_dict")
        else:
            try:
                shape = row_dict[f"{key}_shape"]
                bytes = row_dict[f"{key}_bytes"]
            except KeyError:
                continue

        # TODO (peiyuan): read precision
        if len(bytes) == 0:
            return_dict[key] = torch.zeros(0, dtype=torch.bfloat16)
        else:
            data = np.frombuffer(bytes, dtype=np.float32).reshape(shape).copy()
            data = torch.from_numpy(data)
            if len(data.shape) == 3:
                B, L, D = data.shape
                assert B == 1, "Batch size must be 1"
                data = data.squeeze(0)
            return_dict[key] = data
    return return_dict


def collate_latents_embs_masks(
    batch_to_process, text_padding_length, keys
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[str], Dict[str, Any],
           List[Dict[str, Any]]]:
    # Initialize tensors to hold padded embeddings and masks
    all_latents = []
    all_embs = []
    all_masks = []
    all_clip_features = []
    all_first_frame_latents = []
    all_pil_images = []
    all_infos = []
    caption_text = []
    # Process each row individually
    for i, row in enumerate(batch_to_process):
        # Get info from row
        info_keys = [
            "caption", "file_name", "media_type", "width", "height",
            "num_frames", "duration_sec", "fps"
        ]
        info = {}
        for key in info_keys:
            if key in row:
                info[key] = row[key]
            else:
                info[key] = ""
        info["prompt"] = info["caption"]

        # Get tensors from row
        data = get_torch_tensors_from_row_dict(row, keys)
        latents, emb = data["vae_latent"], data["text_embedding"]
        clip_feature = data.get("clip_feature", None)
        first_frame_latent = data.get("first_frame_latent", None)
        pil_image = data.get("pil_image", None)

        padded_emb, mask = pad(emb, text_padding_length)
        # Store in batch tensors
        all_latents.append(latents)
        all_embs.append(padded_emb)
        all_masks.append(mask)
        all_clip_features.append(clip_feature)
        all_first_frame_latents.append(first_frame_latent)
        all_pil_images.append(pil_image)
        all_infos.append(info)
        # TODO(py): remove this once we fix preprocess
        try:
            caption_text.append(row["prompt"])
        except KeyError:
            caption_text.append(row["caption"])

    # Pin memory for faster transfer to GPU
    all_latents = torch.stack(all_latents)
    all_embs = torch.stack(all_embs)
    all_masks = torch.stack(all_masks)
    all_extra_latents = {
        "clip_feature": torch.stack(all_clip_features),
        "first_frame_latent": torch.stack(all_first_frame_latents),
        "pil_image": all_pil_images,
    }

    return all_latents, all_embs, all_masks, caption_text, all_extra_latents, all_infos


def collate_rows_from_parquet_schema(rows, parquet_schema,
                                     text_padding_length) -> Dict[str, Any]:
    """
    Collate rows from parquet files based on the provided schema.
    Dynamically processes tensor fields based on schema and returns batched data.
    
    Args:
        rows: List of row dictionaries from parquet files
        parquet_schema: PyArrow schema defining the structure of the data
    
    Returns:
        Dict containing batched tensors and metadata
    """
    if not rows:
        return {}

    # Initialize containers for different data types
    batch_data = {}

    # Get tensor and metadata field names from schema (fields ending with '_bytes')
    tensor_fields = []
    metadata_fields = []
    for field in parquet_schema.names:
        if field.endswith('_bytes'):
            shape_field = field.replace('_bytes', '_shape')
            dtype_field = field.replace('_bytes', '_dtype')
            tensor_name = field.replace('_bytes', '')
            tensor_fields.append(tensor_name)
            assert shape_field in parquet_schema.names, f"Shape field {shape_field} not found in schema for field {field}. Currently we only support *_bytes fields for tensors."
            assert dtype_field in parquet_schema.names, f"Dtype field {dtype_field} not found in schema for field {field}. Currently we only support *_bytes fields for tensors."
        elif not field.endswith('_shape') and not field.endswith('_dtype'):
            # Only add actual metadata fields, not the shape/dtype helper fields
            metadata_fields.append(field)

    # Process each tensor field efficiently
    for tensor_name in tensor_fields:
        tensor_list = []

        for row in rows:
            # Get tensor data from row using the existing helper function pattern
            shape_key = f"{tensor_name}_shape"
            bytes_key = f"{tensor_name}_bytes"

            if shape_key in row and bytes_key in row:
                # logger.info("row: %s", row)
                # logger.info("shape_key: %s", shape_key)
                # logger.info("bytes_key: %s", bytes_key)
                shape = row[shape_key]
                bytes_data = row[bytes_key]

                if len(bytes_data) == 0:
                    tensor = torch.zeros(0, dtype=torch.bfloat16)
                else:
                    # Convert bytes to tensor using float32 as default
                    # logger.info("len(bytes_data): %s", len(bytes_data))
                    # logger.info("shape: %s", shape)

                    data = np.frombuffer(
                        bytes_data, dtype=np.float32).reshape(shape).copy()
                    tensor = torch.from_numpy(data)

                tensor_list.append(tensor)
            else:
                # Handle missing tensor data
                tensor_list.append(torch.zeros(0, dtype=torch.bfloat16))

        # Stack tensors with special handling for text embeddings
        if tensor_list:
            if tensor_name == 'text_embedding':
                # Handle text embeddings with padding
                padded_tensors = []
                attention_masks = []

                for tensor in tensor_list:
                    if tensor.numel() > 0:
                        padded_tensor, mask = pad(tensor, text_padding_length)
                        padded_tensors.append(padded_tensor)
                        attention_masks.append(mask)
                    else:
                        # Handle empty embeddings - assume default embedding dimension
                        padded_tensors.append(
                            torch.zeros(text_padding_length,
                                        768,
                                        dtype=torch.bfloat16))
                        attention_masks.append(torch.zeros(text_padding_length))

                batch_data[tensor_name] = torch.stack(padded_tensors)
                batch_data['text_attention_mask'] = torch.stack(attention_masks)
            else:
                # Stack other tensors directly, handling None values
                valid_tensors = [
                    t for t in tensor_list if t is not None and t.numel() > 0
                ]
                if valid_tensors:
                    batch_data[tensor_name] = torch.stack(valid_tensors)
                elif tensor_list:  # All tensors are empty but exist
                    batch_data[tensor_name] = torch.stack(tensor_list)

    # Process metadata fields efficiently into info_list
    info_list = []
    for row in rows:
        info = {}
        for field in metadata_fields:
            info[field] = row.get(field, "")

        # Add prompt field for backward compatibility
        info["prompt"] = info.get("caption", "")
        info_list.append(info)

    batch_data['info_list'] = info_list

    # Add caption_text for backward compatibility
    if info_list and 'caption' in info_list[0]:
        batch_data['caption_text'] = [info['caption'] for info in info_list]

    return batch_data
