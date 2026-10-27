# SPDX-License-Identifier: Apache-2.0
"""Golden-gate: GEN3C-Cosmos-7B transformer block 0 (Cosmos-family DiT).

Spec notes: block ctor takes no prefix arg (gen3c.py:393-403); forward returns
a plain tensor (1, 240, 4096), not a tuple (gen3c.py:547). RoPE is built
outside the block by the model with hidden_size=attention_head_dim (128), and
cos/sin stay fp32 while hidden is bf16 (gen3c.py:800-806, 620-626). The
FastVideo mirror checkpoint is a SINGLE file transformer/model.safetensors
with no index.json and already stores FastVideo-native names, so renames are
empty — the harness only strips the transformer_blocks.{N}. prefix. The gen3c
SSIM test parametrizes only TORCH_SDPA (test_gen3c_similarity.py:109,115).
"""

from __future__ import annotations

import torch

from fastvideo.tests.golden_gate._harness import GateSpec, distributed_runtime, run_gate

__all__ = ["distributed_runtime"]

HIDDEN_SIZE = 4096  # 32 heads x 128, Gen3CArchConfig defaults


def _build_block(layer: int) -> torch.nn.Module:
    from fastvideo.configs.models.dits.gen3c import Gen3CArchConfig
    from fastvideo.models.dits.gen3c import Gen3CTransformerBlock

    # Mirrors Gen3CTransformer3DModel.__init__ (gen3c.py:840-852).
    arch = Gen3CArchConfig()
    return Gen3CTransformerBlock(
        num_attention_heads=arch.num_attention_heads,  # 32
        attention_head_dim=arch.attention_head_dim,  # 128 -> hidden 4096
        cross_attention_dim=arch.text_embed_dim,  # 1024
        mlp_ratio=arch.mlp_ratio,  # 4.0 -> mlp inner 16384
        adaln_lora_dim=arch.adaln_lora_dim,  # 256
        use_adaln_lora=arch.use_adaln_lora,  # True
        qk_norm=(arch.qk_norm == "rms_norm"),  # True
        supported_attention_backends=arch._supported_attention_backends,
    )


def _make_inputs(device: torch.device, seed: int) -> dict:
    """Fixed seeded batch: 3x8x10 post-patch latent grid (S=240), 128 text tokens.

    Shapes per Gen3CTransformerBlock.forward (gen3c.py:461-480):
      hidden_states (B,S,4096), encoder_hidden_states (B,N,1024),
      affine_emb (B,4096), adaln_lora (B,12288), rope_emb ((S,128),(S,128)) fp32,
      extra_pos_emb (B,S,4096), original_seq_len int.
    """
    from fastvideo.models.dits.gen3c import Gen3CRotaryPosEmbed

    generator = torch.Generator(device="cpu").manual_seed(seed)
    t, h, w = 3, 8, 10
    seq = t * h * w  # 240

    # RoPE exactly as Gen3CTransformer3DModel.__init__ builds it (gen3c.py:800-806):
    # hidden_size = attention_head_dim (128), NOT the model hidden size.
    rope = Gen3CRotaryPosEmbed(
        hidden_size=128,
        max_size=(128, 240, 240),
        patch_size=(1, 2, 2),
        rope_scale=(2.0, 1.0, 1.0),
        enable_fps_modulation=True,
    )
    # rope.forward() reads only shape/device of its input; fps=24 matches the
    # SSIM test (test_gen3c_similarity.py:83).
    cos, sin = rope(torch.empty(1, t, h, w, 1, device=device), fps=24)  # each (240, 128) fp32

    hidden = torch.randn(1, seq, HIDDEN_SIZE, generator=generator, dtype=torch.float32)
    encoder = torch.randn(1, 128, 1024, generator=generator, dtype=torch.float32)
    affine = torch.randn(1, HIDDEN_SIZE, generator=generator, dtype=torch.float32)
    adaln_lora = torch.randn(1, 3 * HIDDEN_SIZE, generator=generator, dtype=torch.float32)
    extra_pos = torch.randn(1, seq, HIDDEN_SIZE, generator=generator, dtype=torch.float32)

    return {
        "hidden_states": hidden.to(device=device, dtype=torch.bfloat16),
        "encoder_hidden_states": encoder.to(device=device, dtype=torch.bfloat16),
        "affine_emb": affine.to(device=device, dtype=torch.bfloat16),
        "adaln_lora": adaln_lora.to(device=device, dtype=torch.bfloat16),
        "rope_emb": (cos, sin),  # keep fp32 — the full model passes them fp32
        "extra_pos_emb": extra_pos.to(device=device, dtype=torch.bfloat16),
        "original_seq_len": seq,
    }


SPEC = GateSpec(
    name="gen3c_cosmos_7b",
    repo_id="FastVideo/GEN3C-Cosmos-7B-Diffusers",
    build_block=_build_block,
    make_inputs=_make_inputs,
    prefix_template="transformer_blocks.{N}.",
    renames=(),  # mirror checkpoint already uses FastVideo-native names
    attention_backend="TORCH_SDPA",
    weight_file="model.safetensors",  # single file, no index.json
)


def test_gen3c_golden_gate(distributed_runtime) -> None:
    run_gate(SPEC)
