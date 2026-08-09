# SPDX-License-Identifier: Apache-2.0
"""Golden-gate: SD3.5-medium MMDiT joint transformer block 0.

Spec notes: arch-config defaults (configs/models/dits/sd3.py:11-23) match the
HF transformer/config.json of stabilityai/stable-diffusion-3.5-medium
field-for-field, so no explicit geometry pins. Layer 0 is dual-attention
(layers 0-12 carry extra attn2.* params; 23 is context_pre_only and returns
(None, hidden) — stay below 23). Checkpoint is SINGLE-FILE (no index.json);
the harness single-file fallback handles it, and param names match diffusers
bit-for-bit (param_names_mapping is empty, sd3.py:878) so renames are empty.
Forward returns a (encoder_hidden_states, hidden_states) TUPLE (sd3.py:868);
postprocess cats the two streams along dim=1. Inputs are the three (B, *, 1536)
tensors only — no RoPE, no mask; sincos pos embed is added before the blocks
(sd3.py:1024). SSIM test pins TORCH_SDPA (test_sd35_similarity.py:93).
"""

from __future__ import annotations

import torch

from fastvideo.tests.golden_gate._harness import GateSpec, distributed_runtime, run_gate

__all__ = ["distributed_runtime"]

DIM = 1536  # 24 heads x 64, sd3.py:896
N_IMG = 256  # 256x256 SSIM run: 32x32 latent, patch 2
N_TXT = 154


def _build_block(layer: int) -> torch.nn.Module:
    # Mirrors SD3Transformer2DModel.__init__ (sd3.py:923-933).
    from fastvideo.configs.models.dits.sd3 import SD3Transformer2DArchConfig
    from fastvideo.models.dits.sd3 import (SD3JointTransformerBlock, SD3Transformer2DModel)

    arch = SD3Transformer2DArchConfig()
    return SD3JointTransformerBlock(
        dim=arch.num_attention_heads * arch.attention_head_dim,
        num_attention_heads=arch.num_attention_heads,
        attention_head_dim=arch.attention_head_dim,
        context_pre_only=layer == arch.num_layers - 1,
        qk_norm=arch.qk_norm,
        use_dual_attention=layer in tuple(arch.dual_attention_layers),
        supported_attention_backends=SD3Transformer2DModel._supported_attention_backends,
    )


def _make_inputs(device: torch.device, seed: int) -> dict:
    # Block forward signature: sd3.py:771-777. dim=1536 everywhere: encoder
    # states enter AFTER context_embedder (4096->1536), temb is the (B, 1536)
    # time_text_embed output; AdaLN chunks temb along dim=1 so it stays 2-D.
    generator = torch.Generator(device="cpu").manual_seed(seed)

    hidden = torch.randn(1, N_IMG, DIM, generator=generator, dtype=torch.float32)
    encoder = torch.randn(1, N_TXT, DIM, generator=generator, dtype=torch.float32)
    temb = torch.randn(1, DIM, generator=generator, dtype=torch.float32)

    return {
        "hidden_states": hidden.to(device=device, dtype=torch.bfloat16),
        "encoder_hidden_states": encoder.to(device=device, dtype=torch.bfloat16),
        "temb": temb.to(device=device, dtype=torch.bfloat16),
    }


SPEC = GateSpec(
    name="sd35_medium",
    repo_id="stabilityai/stable-diffusion-3.5-medium",
    build_block=_build_block,
    make_inputs=_make_inputs,
    prefix_template="transformer_blocks.{N}.",
    attention_backend="TORCH_SDPA",
    # tuple (encoder_hidden_states, hidden_states) -> one tensor, both (B, S, 1536)
    postprocess=lambda out: torch.cat([out[0], out[1]], dim=1),
)


def test_sd35_golden_gate(distributed_runtime) -> None:
    run_gate(SPEC)
