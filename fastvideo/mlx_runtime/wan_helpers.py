# SPDX-License-Identifier: Apache-2.0
"""Wan recipe constants, geometry, prompt encoding, and rotary embeddings."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from fastvideo.mlx_runtime.memory import cleanup_torch_mps
from fastvideo.mlx_runtime.prompt_cache import (
    fingerprint_digest,
    load_prompt_cache,
    save_prompt_cache,
    text_encoder_fingerprint,
)
from fastvideo.mlx_runtime.refine import RefinePlan, plan_refine_resolutions

WAN_DMD_STEPS = (1000, 757, 522)
WAN_TEMPORAL_COMPRESSION = 4
WAN21_SPATIAL_COMPRESSION = 8
WAN22_SPATIAL_COMPRESSION = 16
WAN21_CHANNELS = 16
WAN22_CHANNELS = 48


def plan_wan_generation(*, height: int, width: int, num_frames: int, wan22: bool = False) -> RefinePlan:
    """Check both VAE compression and DiT patch alignment before loading weights."""
    return plan_refine_resolutions(
        height=height,
        width=width,
        num_frames=num_frames,
        vae_spatial_compression=WAN22_SPATIAL_COMPRESSION if wan22 else WAN21_SPATIAL_COMPRESSION,
        vae_temporal_compression=WAN_TEMPORAL_COMPRESSION,
        enabled=False,
    )


def resolve_wan_torch_device(device_arg: str):
    """Pick the torch device for UMT5 text encoding."""
    import torch

    if device_arg == "auto":
        return torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    return torch.device(device_arg)


def resolve_wan_torch_dtype(dtype_arg: str):
    import torch

    dtypes = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}
    if dtype_arg not in dtypes:
        raise ValueError(f"Unsupported text-encoder dtype {dtype_arg!r}; expected one of {sorted(dtypes)}.")
    return dtypes[dtype_arg]


def encode_wan_prompt(
    *,
    model_root: Path,
    prompt: str,
    max_sequence_length: int,
    device_arg: str = "auto",
    dtype_arg: str = "bf16",
    cache_dir: Path | None = None,
):
    """Encode UMT5 with the family's recipe precision, optionally reusing disk cache."""
    import torch

    dtype = resolve_wan_torch_dtype(dtype_arg)
    device = resolve_wan_torch_device(device_arg)
    fingerprint = {
        "recipe": "wan-umt5-v1",
        "prompt": prompt,
        "text_encoder": text_encoder_fingerprint(model_root) if cache_dir is not None else None,
        "max_sequence_length": max_sequence_length,
        "dtype": dtype_arg,
        "device": str(device),
    }
    cache_path = cache_dir / f"{fingerprint_digest(fingerprint)}.npy" if cache_dir is not None else None
    cached = load_prompt_cache(cache_path, fingerprint)
    expected_dtype = np.dtype(np.float16 if dtype_arg == "fp16" else np.float32)
    if (cached is not None and cached.ndim == 3 and cached.shape[:2] == (1, max_sequence_length)
            and cached.dtype == expected_dtype and np.isfinite(cached).all()):
        return torch.from_numpy(cached).contiguous()

    from transformers import AutoTokenizer, UMT5EncoderModel

    tokenizer = AutoTokenizer.from_pretrained(model_root / "tokenizer", local_files_only=True)
    text_encoder = UMT5EncoderModel.from_pretrained(
        model_root / "text_encoder",
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        local_files_only=True,
    ).to(device)
    text_encoder.eval()
    text_inputs = tokenizer(
        [prompt],
        padding="max_length",
        max_length=max_sequence_length,
        truncation=True,
        add_special_tokens=True,
        return_attention_mask=True,
        return_tensors="pt",
    )
    input_ids = text_inputs.input_ids.to(device)
    attention_mask = text_inputs.attention_mask.to(device)
    valid_lengths = attention_mask.gt(0).sum(dim=1).long()
    with torch.no_grad():
        hidden_states = text_encoder(input_ids, attention_mask).last_hidden_state
    hidden_states = hidden_states.to(dtype=dtype)
    trimmed = [row[:length] for row, length in zip(hidden_states, valid_lengths, strict=False)]
    padded = torch.stack(
        [torch.cat([row, row.new_zeros(max_sequence_length - row.size(0), row.size(1))]) for row in trimmed],
        dim=0,
    )
    # NumPy has no bf16; fp32 preserves every bf16 value exactly.
    if padded.dtype == torch.bfloat16:
        padded = padded.float()
    padded = padded.cpu().contiguous()
    del text_encoder, tokenizer, text_inputs, input_ids, attention_mask, valid_lengths, hidden_states, trimmed
    cleanup_torch_mps()
    save_prompt_cache(cache_path, padded.numpy(), fingerprint)
    return padded


def make_wan_rotary_embeddings(config: dict[str, Any], *, latent_frames: int, latent_height: int, latent_width: int):
    """Build the rotary tables, with readable errors for incomplete DiT configs."""
    required = {"num_attention_heads", "attention_head_dim", "patch_size"}
    missing = required - config.keys()
    if missing:
        raise ValueError("Wan DiT config is missing: " + ", ".join(sorted(missing)))
    try:
        num_heads = int(config["num_attention_heads"])
        head_dim = int(config["attention_head_dim"])
        patch_size = tuple(int(value) for value in config["patch_size"])
    except (TypeError, ValueError) as error:
        raise ValueError("Wan DiT attention dimensions and patch_size must contain integers.") from error
    if num_heads <= 0 or head_dim <= 0 or len(patch_size) != 3 or any(value <= 0 for value in patch_size):
        raise ValueError("Wan DiT config requires positive attention dimensions and three positive patch_size values.")
    dimensions = (latent_frames, latent_height, latent_width)
    if any(value <= 0 or value % patch for value, patch in zip(dimensions, patch_size, strict=True)):
        raise ValueError(f"Wan latent dimensions {dimensions} must align with patch_size={patch_size}.")

    import mlx.core as mx
    import torch

    from fastvideo.layers.rotary_embedding import get_rotary_pos_embed

    post_patch = tuple(value // patch for value, patch in zip(dimensions, patch_size, strict=True))
    rope_dim_list = [head_dim - 4 * (head_dim // 6), 2 * (head_dim // 6), 2 * (head_dim // 6)]
    freqs_cos, freqs_sin = get_rotary_pos_embed(
        post_patch,
        num_heads * head_dim,
        num_heads,
        rope_dim_list,
        dtype=torch.float32,
        rope_theta=10000,
    )
    return mx.array(freqs_cos.numpy()).astype(mx.float32), mx.array(freqs_sin.numpy()).astype(mx.float32)
