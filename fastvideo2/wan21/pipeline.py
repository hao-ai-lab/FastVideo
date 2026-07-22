"""Wan2.1 T2V pipeline: text_encode -> denoise (loop) -> vae_decode.

The two component stages here own the family's encode/decode conventions
(UMT5 mask-trim + zero-pad; Wan VAE mean/std denormalization). The denoise
stage binds the card's declared loop. Slot edges are declared and enforced —
an undeclared read is an error, not a convention.
"""
from __future__ import annotations

from typing import Any, Mapping

from fastvideo2.pipeline import ComponentStage, LoopStage, Pipeline

_TEXT_MAX_LEN = 512


def _encode_one(tokenizer: Any, encoder: Any, text: str) -> Any:
    """UMT5 encoding with the Wan convention: pad to 512, then zero every
    position past the true sequence length (padding positions carry encoder
    outputs otherwise)."""
    import torch
    batch = tokenizer([text], padding="max_length", max_length=_TEXT_MAX_LEN,
                      truncation=True, add_special_tokens=True,
                      return_attention_mask=True, return_tensors="pt")
    device = next(encoder.parameters()).device
    ids = batch.input_ids.to(device)
    mask = batch.attention_mask.to(device)
    with torch.no_grad():
        embeds = encoder(ids, mask).last_hidden_state  # [1, 512, 4096]
    seq_len = int(mask[0].sum())
    embeds[:, seq_len:] = 0
    return embeds


def _text_encode(instance: Any, inputs: Mapping[str, Any], request: Any) -> dict:
    tok = instance.component("tokenizer")
    enc = instance.component("text_encoder")
    return {
        "text_embeds": _encode_one(tok, enc, inputs["prompt"]),
        "neg_text_embeds": _encode_one(tok, enc, inputs["negative_prompt"] or ""),
    }


def _vae_decode(instance: Any, inputs: Mapping[str, Any], request: Any) -> dict:
    """Wan VAE decode: denormalize with the config's per-channel latent
    mean/std, decode in the VAE's dtype, return uint8 frames [T, H, W, C]."""
    import torch
    vae = instance.component("vae")
    z = inputs["denoise_out"]["latents"].to(torch.float32)
    mean = torch.tensor(vae.config.latents_mean, device=z.device,
                        dtype=z.dtype).view(1, -1, 1, 1, 1)
    std = torch.tensor(vae.config.latents_std, device=z.device,
                       dtype=z.dtype).view(1, -1, 1, 1, 1)
    z = z * std + mean
    with torch.no_grad():
        video = vae.decode(z.to(next(vae.parameters()).dtype), return_dict=False)[0]
    video = (video / 2 + 0.5).clamp(0, 1)             # [1, 3, T, H, W] in [0, 1]
    frames = (video[0].permute(1, 2, 3, 0) * 255).round().to(torch.uint8)
    return {"video": frames.cpu().numpy()}


def build_wan_t2v_pipeline() -> Pipeline:
    return Pipeline(
        pipeline_id="wan21.t2v",
        inputs=("prompt", "negative_prompt"),
        stages=(
            ComponentStage("text_encode", fn=_text_encode,
                           reads=("prompt", "negative_prompt"),
                           writes=("text_embeds", "neg_text_embeds")),
            LoopStage("denoise", loop_id="denoise",
                      reads=("text_embeds", "neg_text_embeds"),
                      writes=("denoise_out",)),
            ComponentStage("vae_decode", fn=_vae_decode,
                           reads=("denoise_out",), writes=("video",)),
        ),
        outputs={"video": "video", "latents": "denoise_out"},
    ).validate()
