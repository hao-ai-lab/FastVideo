#!/usr/bin/env python3
"""Precompute the UMT5 embedding of an empty string.

This is the canonical Wan/T5 null-text convention: feed "" to the text encoder
and use the resulting non-zero embedding as the ∅ in classifier-free guidance.
Contrast with `torch.zeros_like(text_embedding)`, which is a different tensor
distribution that the base model was never asked to handle.

Saves a dict:
  {"embedding": [seq_len, dim], "attention_mask": [seq_len]}
so the val callback can just index into these when it needs a null text.

Usage:
  python data_pipeline/precompute_null_text.py \
    --model-dir /mnt/lustre/vlm-s4duan/exports/synth_stage2_paperLR_ckpt400 \
    --output /mnt/lustre/vlm-s4duan/exports/null_text_umt5.pt
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model-dir", required=True,
                   help="Diffusers export dir containing text_encoder/ and tokenizer/")
    p.add_argument("--output", required=True)
    p.add_argument("--text-len", type=int, default=256,
                   help="Pad/truncate to this length (matches training text_padding_length)")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    from transformers import AutoTokenizer
    try:
        from transformers import UMT5EncoderModel  # type: ignore
    except ImportError:
        UMT5EncoderModel = None  # type: ignore
    from transformers import AutoModel

    model_dir = Path(args.model_dir)
    tok = AutoTokenizer.from_pretrained(model_dir / "tokenizer")
    if UMT5EncoderModel is not None:
        try:
            enc = UMT5EncoderModel.from_pretrained(model_dir / "text_encoder", torch_dtype=torch.bfloat16)
        except Exception:
            enc = AutoModel.from_pretrained(model_dir / "text_encoder", torch_dtype=torch.bfloat16)
    else:
        enc = AutoModel.from_pretrained(model_dir / "text_encoder", torch_dtype=torch.bfloat16)
    enc = enc.to(args.device).eval()

    tokens = tok(
        [""],
        max_length=args.text_len,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
        return_attention_mask=True,
    )
    input_ids = tokens["input_ids"].to(args.device)
    attn_mask = tokens["attention_mask"].to(args.device)
    print(f"tokenized '' -> {input_ids.shape}, non-pad tokens = {int(attn_mask.sum().item())}")
    print(f"  first 5 token ids: {input_ids[0, :5].tolist()}")

    with torch.no_grad():
        out = enc(input_ids=input_ids, attention_mask=attn_mask)
    # take last_hidden_state; strip the batch dim
    hs = out.last_hidden_state[0].float().cpu()  # [seq_len, dim]
    am = attn_mask[0].cpu()
    print(f"embedding shape={list(hs.shape)}  norm={hs.norm().item():.3f}  mean={hs.mean().item():.5f}")
    print(f"attn_mask shape={list(am.shape)}  sum={int(am.sum().item())}")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"embedding": hs, "attention_mask": am, "text_len": args.text_len}, args.output)
    print(f"[null-text] saved -> {args.output}")


if __name__ == "__main__":
    main()
