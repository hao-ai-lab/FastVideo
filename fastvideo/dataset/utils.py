import random
from typing import Any, cast

import numpy as np
import torch


def pad(t: torch.Tensor, padding_length: int) -> tuple[torch.Tensor, torch.Tensor]:
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
            data = np.zeros(shape, dtype=np.float32)
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
                                     rng=None,
                                     seed=0) -> dict[str, Any]:
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

    # Precompute one CFG dropout decision per row (not per tensor field) so
    # that text_embedding (Qwen) and text_embedding_2 (ByT5) always drop
    # together for the same sample.
    cfg_drop_flags = []
    for row in rows:
        drop = False
        if cfg_rate > 0:
            sample_idx = row.get("_sample_index")
            if sample_idx is not None:
                # Deterministic per-sample CFG dropout using sample index
                # (resume-safe).
                drop = random.Random(seed ^ int(sample_idx)).random() < cfg_rate
            else:
                drop = (rng.random() if rng else random.random()) < cfg_rate
        cfg_drop_flags.append(drop)

    # Process each tensor field
    for tensor_name in tensor_fields:
        tensor_list = []

        for row_idx, row in enumerate(rows):
            # Get tensor data from row using the existing helper function pattern
            shape_key = f"{tensor_name}_shape"
            bytes_key = f"{tensor_name}_bytes"

            if (
                shape_key in row
                and bytes_key in row
                and row[shape_key] is not None
                and row[bytes_key] is not None
            ):
                shape = row[shape_key]
                bytes_data = row[bytes_key]

                if len(bytes_data) == 0:
                    # Preserve the full shape (e.g. [0, D]) so downstream
                    # padding knows the embedding dimension even when there
                    # are zero tokens (e.g. ByT5 embedding with no glyph text).
                    # zeros, not empty: a truncated row can declare a non-empty
                    # shape with no payload, and uninitialized memory would be
                    # stacked silently instead of failing.
                    tensor = torch.zeros(tuple(shape), dtype=torch.float32)
                else:
                    drop = (
                        tensor_name in ("text_embedding", "text_embedding_2")
                        and cfg_drop_flags[row_idx])
                    if drop:
                        data = np.zeros(shape, dtype=np.float32)
                    else:
                        data = np.frombuffer(
                            bytes_data,
                            dtype=np.float32,
                        ).reshape(shape).copy()
                    tensor = torch.from_numpy(data)
                    # if len(data.shape) == 3:
                    #     B, L, D = tensor.shape
                    #     assert B == 1, "Batch size must be 1"
                    #     tensor = tensor.squeeze(0)

                tensor_list.append(tensor)
            else:
                # Handle missing tensor data
                if tensor_name == "text_embedding_2":
                    tensor_list.append(None)
                else:
                    tensor_list.append(torch.zeros(0, dtype=torch.bfloat16))

        # Stack tensors with special handling for text embeddings
        if tensor_name in (
            "text_embedding",
            "text_embedding_2",
        ):
            # Handle text embeddings with padding

            valid_tensors = [t for t in tensor_list if t is not None]

            if len(valid_tensors) == 0:
                if tensor_name == "text_embedding_2":
                    # Legacy Wan/Cosmos data does not contain the secondary
                    # (ByT5) text embedding at all - skip this field entirely.
                    continue
                raise ValueError("text_embedding is missing from all rows.")

            reference_tensor = valid_tensors[0]
            if reference_tensor.ndim == 3:
                if reference_tensor.shape[0] != 1:
                    raise ValueError(
                        f"Expected '{tensor_name}' batch dimension "
                        f"to be 1, got {tuple(reference_tensor.shape)}"
                    )
                reference_tensor = reference_tensor.squeeze(0)
            if reference_tensor.ndim != 2:
                raise ValueError(
                    f"Expected '{tensor_name}' shape [L, D], "
                    f"got {tuple(reference_tensor.shape)}"
                )
            embedding_dim = reference_tensor.shape[-1]
            embedding_dtype = reference_tensor.dtype

            padded_tensors = []
            attention_masks = []
            for tensor in tensor_list:
                if tensor is None:
                    if tensor_name == "text_embedding":
                        raise ValueError(
                            "text_embedding is missing from one or more rows.")
                    # No ByT5 embedding field for this row (legacy row mixed
                    # into a Hunyuan batch) - treat as zero tokens.
                    tensor = torch.empty((0, embedding_dim), dtype=embedding_dtype)

                if tensor.ndim == 3:
                    if tensor.shape[0] != 1:
                        raise ValueError(
                            f"Expected '{tensor_name}' batch dimension "
                            f"to be 1, got {tuple(tensor.shape)}"
                        )
                    tensor = tensor.squeeze(0)

                if tensor.ndim != 2:
                    raise ValueError(
                        f"Expected '{tensor_name}' shape [L, D], "
                        f"got {tuple(tensor.shape)}"
                    )

                if tensor.shape[-1] != embedding_dim:
                    raise ValueError(
                        f"Inconsistent hidden dimension for "
                        f"'{tensor_name}': expected {embedding_dim}, "
                        f"got {tensor.shape[-1]}"
                    )

                padded_tensor, mask = pad(tensor, text_padding_length)
                padded_tensors.append(padded_tensor)
                attention_masks.append(mask)

            batch_data[tensor_name] = torch.stack(padded_tensors)
            if tensor_name == "text_embedding":
                batch_data["text_attention_mask"] = torch.stack(attention_masks)
            else:
                batch_data["text_attention_mask_2"] = torch.stack(attention_masks)
        else:
            # Stack all tensors to preserve batch consistency
            # Don't filter out None or empty tensors as this breaks batch sizing
            try:
                batch_data[tensor_name] = torch.stack(tensor_list)
            except (ValueError, RuntimeError) as e:
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

_NUMPY_DTYPES: dict[str, np.dtype] = {
    "float16": np.dtype(np.float16),
    "float32": np.dtype(np.float32),
    "float64": np.dtype(np.float64),
    "int32": np.dtype(np.int32),
    "int64": np.dtype(np.int64),
}


def _decode_tensor_from_parquet(
    row: dict[str, Any],
    tensor_name: str,
) -> torch.Tensor | None:
    shape_key = f"{tensor_name}_shape"
    bytes_key = f"{tensor_name}_bytes"
    dtype_key = f"{tensor_name}_dtype"

    shape = row.get(shape_key)
    bytes_data = row.get(bytes_key)
    dtype_name = row.get(dtype_key)

    # Old datasets do not have text_embedding_2.
    if shape is None or bytes_data is None:
        return None

    shape = tuple(int(x) for x in shape)

    # bfloat16 cannot be decoded directly through NumPy.
    if dtype_name == "bfloat16":
        if len(bytes_data) == 0:
            return torch.empty(shape, dtype=torch.bfloat16)

        tensor = torch.frombuffer(
            bytearray(bytes_data),
            dtype=torch.bfloat16,
        )
        return tensor.reshape(shape).clone()

    np_dtype = _NUMPY_DTYPES.get(dtype_name)
    if np_dtype is None:
        raise ValueError(
            f"Unsupported dtype '{dtype_name}' for tensor "
            f"'{tensor_name}'"
        )

    if len(bytes_data) == 0:
        return torch.empty(shape, dtype=torch.from_numpy(
            np.empty(0, dtype=np_dtype)
        ).dtype)

    array = np.frombuffer(
        bytes_data,
        dtype=np_dtype,
    ).reshape(shape).copy()

    return torch.from_numpy(array)
