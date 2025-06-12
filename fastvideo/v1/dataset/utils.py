from typing import Any, Dict, List

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
            shape = row_dict[f"{key}_shape"]
            bytes = row_dict[f"{key}_bytes"]

        # TODO (peiyuan): read precision
        data = np.frombuffer(bytes, dtype=np.float32).reshape(shape).copy()
        data = torch.from_numpy(data)
        if len(data.shape) == 3:
            B, L, D = data.shape
            assert B == 1, "Batch size must be 1"
            data = data.squeeze(0)
        return_dict[key] = data
    return return_dict


def collate_latents_embs_masks(
        batch_to_process, text_padding_length,
        keys) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[str]]:
    # Initialize tensors to hold padded embeddings and masks
    all_latents = []
    all_embs = []
    all_masks = []
    caption_text = []
    # Process each row individually
    for i, row in enumerate(batch_to_process):
        # Get tensors from row
        data = get_torch_tensors_from_row_dict(row, keys)
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
