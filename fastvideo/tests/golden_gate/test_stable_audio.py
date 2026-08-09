# SPDX-License-Identifier: Apache-2.0
"""Golden-gate: Stable Audio Open 1.0 ContinuousTransformer block 0.

Spec notes: geometry from StableAudioArchConfig defaults (embed_dim=1536,
24 heads -> dim_heads=64, cond_token_dim=768, project_cond_tokens=False so
dim_context=768, qk_norm=None for the 1.0 base). RoPE is built outside the
block exactly as ContinuousTransformer.forward does — RotaryEmbedding(32)
.forward_from_seq_len(seq) gives (freqs [seq, 32*2->partial rot_dim=32 of
head_dim=64], 1.0) applied inside Attention (stable_audio.py:179-194), not
via LocalAttention freqs_cis. Checkpoint is SINGLE-FILE (no index.json);
zero renames — block-relative keys match FastVideo module names exactly.
Backend pinned to TORCH_SDPA to match the SSIM lane. The pipeline runs the
DiT in fp16 but the harness casts blocks to bf16; the golden is minted and
compared in bf16 (fingerprint consistency, not pipeline parity).
"""

from __future__ import annotations

import torch

from fastvideo.tests.golden_gate._harness import GateSpec, distributed_runtime, run_gate

__all__ = ["distributed_runtime"]

EMBED_DIM = 1536  # StableAudioArchConfig.embed_dim
COND_DIM = 768  # StableAudioArchConfig.cond_token_dim (project_cond_tokens=False)


def _build_block(layer: int) -> torch.nn.Module:
    from fastvideo.configs.models.dits.stable_audio import StableAudioArchConfig
    from fastvideo.models.dits.stable_audio import TransformerBlock

    arch = StableAudioArchConfig()
    return TransformerBlock(
        dim=arch.embed_dim,
        dim_heads=arch.embed_dim // arch.num_attention_heads,
        cross_attend=True,
        dim_context=arch.cond_token_dim,
        zero_init_branch_outputs=True,
        qk_norm=arch.qk_norm,
    )


def _make_inputs(device: torch.device, seed: int) -> dict:
    from fastvideo.models.dits.stable_audio import RotaryEmbedding

    generator = torch.Generator(device="cpu").manual_seed(seed)
    seq, ctx = 1 + 64, 65  # prepended global-cond token + 64 latent frames; T5 rows

    hidden = torch.randn(1, seq, EMBED_DIM, generator=generator, dtype=torch.float32)
    context = torch.randn(1, ctx, COND_DIM, generator=generator, dtype=torch.float32)
    freqs, scale = RotaryEmbedding(32).forward_from_seq_len(seq)  # [seq, 32] fp32

    return {
        "x": hidden.to(device=device, dtype=torch.bfloat16),
        "context": context.to(device=device, dtype=torch.bfloat16),
        "rotary_pos_emb": (freqs.to(device), scale),
    }


SPEC = GateSpec(
    name="stable_audio_open_1_0",
    repo_id="FastVideo/stable-audio-open-1.0-Diffusers",
    build_block=_build_block,
    make_inputs=_make_inputs,
    prefix_template="transformer.layers.{N}.",
    renames=(),
    attention_backend="TORCH_SDPA",
)


def test_stable_audio_golden_gate(distributed_runtime) -> None:
    run_gate(SPEC)
