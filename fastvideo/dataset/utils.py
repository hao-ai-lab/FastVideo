import random
from typing import Any, cast

import numpy as np
import torch


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


def get_torch_tensors_from_row_dict(row_dict,
                                    keys,
                                    cfg_rate,
                                    rng=None) -> dict[str, Any]:
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
            shape = row_dict[f"{key}_shape"]
            bytes = row_dict[f"{key}_bytes"]

        # TODO (peiyuan): read precision
        if key == 'text_embedding' and (rng.random()
                                        if rng else random.random()) < cfg_rate:
            data = np.zeros((512, 4096), dtype=np.float32)
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
        batch_to_process,
        text_padding_length,
        keys,
        cfg_rate=0.0,
        rng=None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[str]]:
    # Initialize tensors to hold padded embeddings and masks
    all_latents = []
    all_embs = []
    all_masks = []
    caption_text = []
    # Process each row individually
    for i, row in enumerate(batch_to_process):
        # Get tensors from row
        data = get_torch_tensors_from_row_dict(row, keys, cfg_rate, rng)
        latents, emb = data["vae_latent"], data["text_embedding"]

        padded_emb, mask = pad(emb, text_padding_length)
        # Store in batch tensors
        all_latents.append(latents)
        all_embs.append(padded_emb)
        all_masks.append(mask)
        # TODO(py): remove this once we fix preprocess
        try:
            caption_text.append(row["prompt"])
        except KeyError:
            caption_text.append(row["caption"])

    # Pin memory for faster transfer to GPU
    all_latents = torch.stack(all_latents)
    all_embs = torch.stack(all_embs)
    all_masks = torch.stack(all_masks)

    return all_latents, all_embs, all_masks, caption_text


def collate_rows_from_parquet_schema(rows,
                                     parquet_schema,
                                     text_padding_length,
                                     cfg_rate=0.0,
                                     rng=None) -> dict[str, Any]:
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
        return cast(dict[str, Any], {})

    # Initialize containers for different data types
    batch_data: dict[str, Any] = {}

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

    # Process each tensor field
    for tensor_name in tensor_fields:
        tensor_list = []

        for row in rows:
            # Get tensor data from row using the existing helper function pattern
            shape_key = f"{tensor_name}_shape"
            bytes_key = f"{tensor_name}_bytes"

            if shape_key in row and bytes_key in row:
                shape = row[shape_key]
                bytes_data = row[bytes_key]

                if len(bytes_data) == 0:
                    tensor = torch.zeros(0, dtype=torch.bfloat16)
                else:
                    # Convert bytes to tensor using float32 as default
                    if tensor_name == 'text_embedding' and (rng.random(
                    ) if rng else random.random()) < cfg_rate:
                        data = np.zeros((512, 4096), dtype=np.float32)
                    else:
                        data = np.frombuffer(
                            bytes_data, dtype=np.float32).reshape(shape).copy()
                    tensor = torch.from_numpy(data)
                    # if len(data.shape) == 3:
                    #     B, L, D = tensor.shape
                    #     assert B == 1, "Batch size must be 1"
                    #     tensor = tensor.squeeze(0)

                tensor_list.append(tensor)
            else:
                # Handle missing tensor data
                tensor_list.append(torch.zeros(0, dtype=torch.bfloat16))

        # Stack tensors with special handling for text embeddings
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
            # Stack all tensors to preserve batch consistency
            # Don't filter out None or empty tensors as this breaks batch sizing
            try:
                batch_data[tensor_name] = torch.stack(tensor_list)
            except ValueError as e:
                shapes = [
                    t.shape
                    if t is not None and hasattr(t, 'shape') else 'None/Invalid'
                    for t in tensor_list
                ]
                raise ValueError(
                    f"Failed to stack tensors for field '{tensor_name}'. "
                    f"Tensor shapes: {shapes}. "
                    f"All tensors in a batch must have compatible shapes. "
                    f"Original error: {e}") from e

    # Process metadata fields into info_list
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
