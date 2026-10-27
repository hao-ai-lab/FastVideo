import functools
import math
from numbers import Integral
import random
from typing import Any, cast

import numpy as np
import torch

_PYARROW_BINARY_MAX_VALUE_BYTES = (1 << 31) - 1

_DTYPE_ALIASES = {
    "float": "float32",
    "double": "float64",
    "half": "float16",
    "long": "int64",
    "int": "int32",
}


@functools.lru_cache(maxsize=None)
def _normalize_tensor_dtype(dtype_value: Any) -> str:
    """Normalize dtype strings written by NumPy or PyTorch record creators."""
    dtype_name = str(dtype_value).strip().lower()
    for prefix in ("torch.", "numpy.", "np."):
        if dtype_name.startswith(prefix):
            dtype_name = dtype_name.removeprefix(prefix)
    return _DTYPE_ALIASES.get(dtype_name, dtype_name)


def _decode_tensor_bytes(
    bytes_data: bytes,
    shape: list[int] | tuple[int, ...],
    dtype_value: Any,
    *,
    zero: bool = False,
) -> torch.Tensor:
    """Decode one tensor using the dtype persisted beside its byte buffer.

    ``zero=True`` still validates the byte length but returns a correctly typed
    zero tensor instead of copying the payload, for CFG-dropped embeddings.
    """
    dtype_name = _normalize_tensor_dtype(dtype_value)
    is_bfloat16 = dtype_name == "bfloat16"
    if is_bfloat16:
        # NumPy has no portable bfloat16 dtype. Read the raw 16-bit storage,
        # then reinterpret it as torch.bfloat16 without changing the bits.
        numpy_dtype = np.dtype(np.uint16)
    else:
        try:
            numpy_dtype = np.dtype(dtype_name)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Unsupported serialized tensor dtype: {dtype_value!r}") from exc

    expected_nbytes = math.prod(shape) * numpy_dtype.itemsize
    if len(bytes_data) != expected_nbytes:
        raise ValueError(
            "Serialized tensor byte length does not match its shape and dtype: "
            f"shape={tuple(shape)}, dtype={dtype_name}, expected={expected_nbytes}, actual={len(bytes_data)}")

    if zero:
        tensor = torch.from_numpy(np.zeros(shape, dtype=numpy_dtype))
    else:
        array = np.frombuffer(bytes_data, dtype=numpy_dtype).reshape(shape).copy()
        tensor = torch.from_numpy(array)
    if is_bfloat16:
        tensor = tensor.view(torch.bfloat16)
    return tensor


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
        return torch.cat([t, pad], 0), torch.cat([torch.ones(L), torch.zeros(padding_length - L)], 0)
    else:  # crop
        return t[:padding_length], torch.ones(padding_length)


def get_torch_tensors_from_row_dict(row_dict, keys, cfg_rate, rng=None) -> dict[str, Any]:
    """
    Get the latents and prompts from a row dictionary.
    """
    return_dict = {}
    for key in keys:
        if isinstance(key, tuple):
            output_key = key[0]
            serialized_key = None
            for k in key:
                if f"{k}_shape" in row_dict and f"{k}_bytes" in row_dict:
                    serialized_key = k
            if serialized_key is None:
                raise ValueError(f"Key {output_key} not found in row_dict")
        else:
            output_key = serialized_key = key

        shape = row_dict[f"{serialized_key}_shape"]
        bytes_data = row_dict[f"{serialized_key}_bytes"]
        dtype_value = row_dict.get(f"{serialized_key}_dtype", "float32")
        drop = output_key == 'text_embedding' and (rng.random() if rng else random.random()) < cfg_rate
        data = _decode_tensor_bytes(bytes_data, shape, dtype_value, zero=drop)

        if len(data.shape) == 3:
            B, L, D = data.shape
            assert B == 1, "Batch size must be 1"
            data = data.squeeze(0)
        return_dict[output_key] = data
    return return_dict


def collate_latents_embs_masks(batch_to_process,
                               text_padding_length,
                               keys,
                               cfg_rate=0.0,
                               rng=None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[str]]:
    """Collate latents and text embeddings for the iterable-style loader.

    The batch is built from the explicit ``keys`` list, so this path is
    primary-encoder-only: the secondary ByT5 embedding (``text_embedding_2``)
    and its mask are not read even when the parquet rows contain them.
    """
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


# Text streams are padded to a fixed length and paired with the attention mask
# the trainer trims against. The second entry is HunyuanVideo 1.5's ByT5 glyph
# stream, which rides alongside the primary Qwen one.
_TEXT_STREAM_MASKS = {
    "text_embedding": "text_attention_mask",
    "text_embedding_2": "text_attention_mask_2",
}


def collate_rows_from_parquet_schema(rows,
                                     parquet_schema,
                                     text_padding_length: int | None,
                                     cfg_rate=0.0,
                                     rng=None,
                                     seed=0,
                                     max_presentation_tokens: int | None = None) -> dict[str, Any]:
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

    # Ref2VA's Qwen multimodal presentation is variable-length and must not
    # inherit the fixed text-encoder crop used by T2V/T2VA. In dynamic mode,
    # inspect stored shapes before materializing *any* tensor bytes. The cap is
    # a rejection-only allocation guard; successful batches pad only to their
    # longest stored presentation.
    if text_padding_length is None:
        if (isinstance(max_presentation_tokens, bool) or not isinstance(max_presentation_tokens, Integral)
                or int(max_presentation_tokens) <= 0):
            raise ValueError("Dynamic text collation requires max_presentation_tokens to be a positive integer")
        presentation_lengths: list[int] = []
        for row_index, row in enumerate(rows):
            embedding_shape = row.get("text_embedding_shape")
            if (not isinstance(embedding_shape, list | tuple) or len(embedding_shape) != 2
                    or any(isinstance(dim, bool) or not isinstance(dim, Integral) or int(dim) <= 0
                           for dim in embedding_shape)):
                raise ValueError("Dynamic text collation requires text_embedding_shape=[length, feature_dim] "
                                 f"with positive integers; row {row_index} has {embedding_shape!r}")
            presentation_length = int(embedding_shape[0])
            presentation_width = int(embedding_shape[1])
            dtype_name = _normalize_tensor_dtype(row.get("text_embedding_dtype", "float32"))
            itemsize = 2 if dtype_name == "bfloat16" else np.dtype(dtype_name).itemsize
            expected_embedding_bytes = presentation_length * presentation_width * itemsize
            if expected_embedding_bytes > _PYARROW_BINARY_MAX_VALUE_BYTES:
                sample_id = row.get("id", row_index)
                raise ValueError(
                    f"Ref2VA presentation {sample_id!r} shape {list(embedding_shape)!r} requires "
                    f"{expected_embedding_bytes} bytes, exceeding the pa.binary per-value envelope of "
                    f"{_PYARROW_BINARY_MAX_VALUE_BYTES} bytes; use chunked or external presentation storage")
            if presentation_length > int(max_presentation_tokens):
                sample_id = row.get("id", row_index)
                raise ValueError(f"Ref2VA presentation {sample_id!r} stores {presentation_length} tokens, exceeding "
                                 f"max_presentation_tokens={int(max_presentation_tokens)}; the presentation will not "
                                 "be truncated")

            tags_shape = row.get("text_token_tags_shape")
            if tags_shape is not None:
                if (not isinstance(tags_shape, list | tuple) or len(tags_shape) != 1
                        or isinstance(tags_shape[0], bool) or not isinstance(tags_shape[0], Integral)
                        or int(tags_shape[0]) != presentation_length):
                    raise ValueError("text_token_tags_shape must equal the stored text_embedding sequence length; "
                                     f"row {row_index} has {tags_shape!r} and {embedding_shape!r}")
            presentation_lengths.append(presentation_length)
        effective_text_padding_length = max(presentation_lengths)
    else:
        # Preserve the fixed-length crop/pad contract for existing T2V/T2VA
        # callers, including their configured text encoder length.
        effective_text_padding_length = text_padding_length

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

    # One CFG dropout decision per row, shared by all of its text streams, so
    # a sample's Qwen and ByT5 embeddings always drop together. With a
    # ``_sample_index`` it is a pure function of seed ^ index (resume-safe);
    # without one it is a single ``rng`` draw per row, taken the first time a
    # stream of that row needs it, so single-stream batches consume ``rng``
    # exactly as before while a second stream can no longer draw its own.
    row_drops: dict[int, bool] = {}

    def _cfg_drop(row_idx: int, row: dict[str, Any]) -> bool:
        if row_idx not in row_drops:
            sample_idx = row.get("_sample_index")
            if sample_idx is not None:
                row_drops[row_idx] = (random.Random(seed ^ int(sample_idx)).random() < cfg_rate)
            else:
                row_drops[row_idx] = ((rng.random() if rng else random.random()) < cfg_rate)
        return row_drops[row_idx]

    # Process each tensor field
    for tensor_name in tensor_fields:
        tensor_list = []
        shape_key = f"{tensor_name}_shape"
        bytes_key = f"{tensor_name}_bytes"
        dtype_key = f"{tensor_name}_dtype"

        # A secondary text stream that this parquet predates: leave the key out
        # so the trainer's own zero-token fallback (and its warning) applies,
        # rather than a placeholder of the wrong width.
        if (tensor_name == "text_embedding_2" and not any(row.get(bytes_key) is not None for row in rows)):
            continue

        # The primary stream has no such fallback: a row without it would be
        # collated as zeros and train unconditioned without a word, so name
        # the rows instead.
        if tensor_name == "text_embedding":
            missing = [i for i, row in enumerate(rows) if row.get(bytes_key) is None or row.get(shape_key) is None]
            if missing:
                where = ("every row" if len(missing) == len(rows) else f"rows {missing} of {len(rows)}")
                raise ValueError(f"text_embedding is missing from {where} in this batch; "
                                 "the schema declares it, so each parquet row must carry "
                                 "text_embedding_bytes and text_embedding_shape.")

        for row_idx, row in enumerate(rows):
            # Get tensor data from row using the existing helper function pattern
            if shape_key in row and bytes_key in row:
                shape = row[shape_key]
                bytes_data = row[bytes_key]

                if bytes_data is None or len(bytes_data) == 0:
                    # Keep the declared shape: a zero-token stream (e.g. a
                    # caption with no glyph text) still carries its width.
                    if shape:
                        tensor = torch.zeros(*shape, dtype=torch.bfloat16)
                    else:
                        tensor = torch.zeros(0, dtype=torch.bfloat16)
                else:
                    # Per-row CFG dropout, shared across text streams.
                    drop = (tensor_name in _TEXT_STREAM_MASKS and cfg_rate > 0 and _cfg_drop(row_idx, row))
                    tensor = _decode_tensor_bytes(
                        bytes_data,
                        shape,
                        row.get(dtype_key, "float32"),
                        zero=drop,
                    )
                    # if len(data.shape) == 3:
                    #     B, L, D = tensor.shape
                    #     assert B == 1, "Batch size must be 1"
                    #     tensor = tensor.squeeze(0)

                tensor_list.append(tensor)
            else:
                # Handle missing tensor data
                tensor_list.append(torch.zeros(0, dtype=torch.bfloat16))

        # Stack tensors with special handling for text embeddings
        if tensor_name in _TEXT_STREAM_MASKS:
            # Handle text embeddings with padding
            padded_tensors = []
            attention_masks = []
            # Empty entries have to stack against the real ones, so follow
            # their dtype instead of the bfloat16 placeholder default.
            empty_dtype = next((t.dtype for t in tensor_list if t.numel() > 0), torch.bfloat16)
            # ...and their width: a row whose stream is missing entirely (a
            # legacy shard mixed into the same batch) carries only a 1-D
            # placeholder, and must not inject a 768-wide stub that cannot
            # stack against the real rows.
            empty_width = next((int(t.shape[1]) for t in tensor_list if t.dim() > 1), 768)

            for tensor in tensor_list:
                if tensor.numel() > 0:
                    padded_tensor, mask = pad(tensor, effective_text_padding_length)
                    padded_tensors.append(padded_tensor)
                    attention_masks.append(mask)
                else:
                    # Handle empty embeddings - assume default embedding dimension
                    width = (int(tensor.shape[1]) if tensor.dim() > 1 else empty_width)
                    padded_tensors.append(torch.zeros(effective_text_padding_length, width, dtype=empty_dtype))
                    attention_masks.append(torch.zeros(effective_text_padding_length))

            batch_data[tensor_name] = torch.stack(padded_tensors)
            batch_data[_TEXT_STREAM_MASKS[tensor_name]] = torch.stack(attention_masks)
        elif tensor_name == 'text_token_tags':
            # Ref2VA token tags describe the same variable-length Qwen3-VL
            # presentation as text_embedding. Keep both sequence axes aligned;
            # zero-valued padding is ignored through text_attention_mask.
            padded_tags = []
            for tensor, row in zip(tensor_list, rows, strict=True):
                if tensor.ndim != 1:
                    raise ValueError(f"text_token_tags must have shape [length], got {tuple(tensor.shape)}")
                embedding_shape = row.get("text_embedding_shape")
                if (not isinstance(embedding_shape, list | tuple) or len(embedding_shape) != 2
                        or tensor.shape[0] != embedding_shape[0]):
                    raise ValueError("text_token_tags length must equal the unpadded text_embedding sequence length")
                tensor = tensor[:effective_text_padding_length]
                if tensor.shape[0] < effective_text_padding_length:
                    tensor = torch.cat((tensor, torch.zeros(effective_text_padding_length - tensor.shape[0],
                                                            dtype=tensor.dtype,
                                                            device=tensor.device)))
                padded_tags.append(tensor)
            batch_data[tensor_name] = torch.stack(padded_tags)
        else:
            # Stack all tensors to preserve batch consistency
            # Don't filter out None or empty tensors as this breaks batch sizing
            try:
                batch_data[tensor_name] = torch.stack(tensor_list)
            except (ValueError, RuntimeError) as e:
                shapes = [t.shape if t is not None and hasattr(t, 'shape') else 'None/Invalid' for t in tensor_list]
                raise ValueError(f"Failed to stack tensors for field '{tensor_name}'. "
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
