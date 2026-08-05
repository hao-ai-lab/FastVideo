# SPDX-License-Identifier: Apache-2.0
"""Golden-gate: Wan2.1-T2V-1.3B transformer block 0 (also the SFWan/TurboWan arch).

Spec notes: arch-config defaults are the 14B model, so the 1.3B geometry is
pinned explicitly from the checkpoint's transformer/config.json. RoPE is built
outside the block exactly as WanTransformer3DModel.forward does
(wanvideo.py:707-719). attn1 is DistributedAttention (needs the distributed
fixture + forward context); temb must be the 3-dim [B, 6, inner_dim] wan2.1
shape — 4-dim takes the wan2.2-ti2v per-token path.
"""

from __future__ import annotations

import torch

from fastvideo.tests.golden_gate._harness import GateSpec, distributed_runtime, run_gate

__all__ = ["distributed_runtime"]

INNER_DIM = 1536  # 12 heads x 128, Wan2.1-T2V-1.3B transformer/config.json
TEXT_LEN = 512


def _build_block(layer: int) -> torch.nn.Module:
    from fastvideo.configs.models.dits.wanvideo import WanVideoArchConfig
    from fastvideo.models.dits.wanvideo import WanTransformerBlock

    arch = WanVideoArchConfig(
        num_attention_heads=12,
        attention_head_dim=128,
        ffn_dim=8960,
        num_layers=30,
    )
    return WanTransformerBlock(
        INNER_DIM,
        arch.ffn_dim,
        arch.num_attention_heads,
        arch.qk_norm,
        arch.cross_attn_norm,
        arch.eps,
        arch.added_kv_proj_dim,
        arch._supported_attention_backends,
        quant_config=None,
        prefix=f"Wan.blocks.{layer}",
    )


def _make_inputs(device: torch.device, seed: int) -> dict:
    from fastvideo.layers.rotary_embedding import get_rotary_pos_embed

    generator = torch.Generator(device="cpu").manual_seed(seed)
    t, h, w = 3, 10, 10
    seq = t * h * w

    d = INNER_DIM // 12
    rope_dim_list = [d - 4 * (d // 6), 2 * (d // 6), 2 * (d // 6)]
    freqs_cos, freqs_sin = get_rotary_pos_embed(
        (t, h, w), INNER_DIM, 12, rope_dim_list, rope_theta=10000, dtype=torch.float64)
    freqs_cis = (freqs_cos.to(device).float(), freqs_sin.to(device).float())

    hidden = torch.randn(1, seq, INNER_DIM, generator=generator, dtype=torch.float32)
    encoder = torch.randn(1, TEXT_LEN, INNER_DIM, generator=generator, dtype=torch.float32)
    temb = torch.randn(1, 6, INNER_DIM, generator=generator, dtype=torch.float32)

    return {
        "hidden_states": hidden.to(device=device, dtype=torch.bfloat16),
        "encoder_hidden_states": encoder.to(device=device, dtype=torch.bfloat16),
        "temb": temb.to(device=device, dtype=torch.bfloat16),
        "freqs_cis": freqs_cis,
        "original_seq_len": seq,
    }


SPEC = GateSpec(
    name="wan_t2v_1_3b",
    repo_id="Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
    build_block=_build_block,
    make_inputs=_make_inputs,
    prefix_template="blocks.{N}.",
    renames=(
        (r"\.attn1\.to_q\.", ".to_q."),
        (r"\.attn1\.to_k\.", ".to_k."),
        (r"\.attn1\.to_v\.", ".to_v."),
        (r"\.attn1\.to_out\.0\.", ".to_out."),
        (r"\.attn1\.norm_q\.", ".norm_q."),
        (r"\.attn1\.norm_k\.", ".norm_k."),
        (r"\.attn2\.to_out\.0\.", ".attn2.to_out."),
        (r"\.ffn\.net\.0\.proj\.", ".ffn.fc_in."),
        (r"\.ffn\.net\.2\.", ".ffn.fc_out."),
        (r"\.norm2\.", ".self_attn_residual_norm.norm."),
    ),
    attention_backend="FLASH_ATTN",
)


def test_wan_t2v_golden_gate(distributed_runtime) -> None:
    run_gate(SPEC)
