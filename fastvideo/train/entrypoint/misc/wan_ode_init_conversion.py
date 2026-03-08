# SPDX-License-Identifier: Apache-2.0
"""Convert Self-Forcing ode_init.pt to HuggingFace diffusers format.

The official ode_init.pt from
https://huggingface.co/gdhe17/Self-Forcing/resolve/main/checkpoints/ode_init.pt
stores weights under ``{"generator": {<original_wan_keys>}}``.

This script converts those keys to diffusers
``WanTransformer3DModel`` format, verifies them against a reference
model, and saves a complete diffusers-compatible model directory
(transformer + scheduler + vae + text_encoder + tokenizer).

Usage:
    python -m fastvideo.train.entrypoint.misc.wan_ode_init_conversion \
        --input /path/to/ode_init.pt \
        --output /path/to/WanOdeInit \
        --base-model Wan-AI/Wan2.1-T2V-1.3B-Diffusers
"""

from __future__ import annotations

import argparse
import os
import shutil

import torch


# -- Key conversion -----------------------------------------------

def _convert_key(key: str) -> str:
    """Map a single original-Wan key to diffusers key."""
    k = key
    if k.startswith("model."):
        k = k[len("model."):]

    # Top-level modules
    k = k.replace(
        "head.modulation", "scale_shift_table"
    )
    k = k.replace("head.head.", "proj_out.")
    k = k.replace(
        "time_embedding.0.",
        "condition_embedder.time_embedder.linear_1.",
    )
    k = k.replace(
        "time_embedding.2.",
        "condition_embedder.time_embedder.linear_2.",
    )
    k = k.replace(
        "time_projection.1.",
        "condition_embedder.time_proj.",
    )
    k = k.replace(
        "text_embedding.0.",
        "condition_embedder.text_embedder.linear_1.",
    )
    k = k.replace(
        "text_embedding.2.",
        "condition_embedder.text_embedder.linear_2.",
    )

    # Block-level: order matters — rename attn sub-modules
    # before renaming q/k/v/o so prefixes don't collide.
    k = k.replace(".modulation", ".scale_shift_table")
    k = k.replace(".self_attn.", ".attn1.")
    k = k.replace(".cross_attn.", ".attn2.")
    k = k.replace(".norm3.", ".norm2.")

    # Attention projections
    k = k.replace(".attn1.q.", ".attn1.to_q.")
    k = k.replace(".attn1.k.", ".attn1.to_k.")
    k = k.replace(".attn1.v.", ".attn1.to_v.")
    k = k.replace(".attn1.o.", ".attn1.to_out.0.")
    k = k.replace(".attn2.q.", ".attn2.to_q.")
    k = k.replace(".attn2.k.", ".attn2.to_k.")
    k = k.replace(".attn2.v.", ".attn2.to_v.")
    k = k.replace(".attn2.o.", ".attn2.to_out.0.")

    # FFN
    k = k.replace(".ffn.0.", ".ffn.net.0.proj.")
    k = k.replace(".ffn.2.", ".ffn.net.2.")

    return k


def convert_state_dict(
    orig_sd: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Convert an entire original-Wan state dict."""
    return {_convert_key(k): v for k, v in orig_sd.items()}


# -- Main ---------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert ode_init.pt to diffusers format",
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to ode_init.pt",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output directory for diffusers model",
    )
    parser.add_argument(
        "--base-model",
        default="Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
        help="HF repo or local path for base model "
        "(provides config, VAE, tokenizer, etc.)",
    )
    parser.add_argument(
        "--skip-verify",
        action="store_true",
        help="Skip strict load verification",
    )
    args = parser.parse_args()

    # 1. Load checkpoint
    print(f"Loading {args.input} ...")
    ckpt = torch.load(
        args.input, map_location="cpu", weights_only=False
    )
    if "generator" in ckpt:
        orig_sd = ckpt["generator"]
    elif isinstance(ckpt, dict) and any(
        k.startswith("model.") for k in ckpt
    ):
        orig_sd = ckpt
    else:
        raise ValueError(
            "Cannot find weights in checkpoint. "
            "Expected key 'generator' or keys starting "
            "with 'model.'."
        )
    print(f"  Found {len(orig_sd)} weight tensors")

    # 2. Convert keys
    print("Converting keys ...")
    new_sd = convert_state_dict(orig_sd)

    # 3. Verify against reference model
    if not args.skip_verify:
        from diffusers import WanTransformer3DModel

        print(
            f"Loading reference model from "
            f"{args.base_model} for verification ..."
        )
        ref_model = WanTransformer3DModel.from_pretrained(
            args.base_model,
            subfolder="transformer",
            torch_dtype=torch.bfloat16,
        )
        ref_keys = set(ref_model.state_dict().keys())
        new_keys = set(new_sd.keys())
        missing = sorted(ref_keys - new_keys)
        extra = sorted(new_keys - ref_keys)
        if missing:
            print(f"  WARNING: {len(missing)} missing keys:")
            for k in missing[:10]:
                print(f"    {k}")
            if len(missing) > 10:
                print(f"    ... and {len(missing) - 10} more")
        if extra:
            print(f"  WARNING: {len(extra)} extra keys:")
            for k in extra[:10]:
                print(f"    {k}")
            if len(extra) > 10:
                print(f"    ... and {len(extra) - 10} more")
        if missing or extra:
            raise RuntimeError(
                "Key mismatch — conversion mapping needs "
                "updating. Use --skip-verify to bypass."
            )

        ref_model.load_state_dict(new_sd, strict=True)
        print("  Strict load OK — all keys match!")
    else:
        ref_model = None
        print("  Skipping verification")

    # 4. Save transformer
    output_dir = os.path.abspath(args.output)
    os.makedirs(output_dir, exist_ok=True)
    transformer_dir = os.path.join(output_dir, "transformer")

    if ref_model is not None:
        print(f"Saving transformer to {transformer_dir} ...")
        ref_model.save_pretrained(transformer_dir)
    else:
        from safetensors.torch import save_file

        os.makedirs(transformer_dir, exist_ok=True)
        print(
            f"Saving transformer weights to "
            f"{transformer_dir} ..."
        )
        save_file(new_sd, os.path.join(
            transformer_dir, "model.safetensors"
        ))

    # 5. Copy non-transformer files from base model
    from huggingface_hub import snapshot_download

    print(f"Downloading base model files from "
          f"{args.base_model} ...")
    base_path = snapshot_download(
        args.base_model,
        allow_patterns=[
            "model_index.json",
            "scheduler/*",
            "tokenizer/*",
            "vae/*",
            "text_encoder/*",
        ],
    )
    for item in sorted(os.listdir(base_path)):
        if item == "transformer":
            continue
        src = os.path.join(base_path, item)
        dst = os.path.join(output_dir, item)
        if os.path.isfile(src):
            shutil.copy2(src, dst)
            print(f"  Copied {item}")
        elif os.path.isdir(src):
            if os.path.exists(dst):
                shutil.rmtree(dst)
            shutil.copytree(src, dst)
            print(f"  Copied {item}/")

    print(f"\nDone! Diffusers model saved to {output_dir}")


if __name__ == "__main__":
    main()
