# SPDX-License-Identifier: Apache-2.0
"""Golden-gate: LingBot-World camera-conditioned I2V DiT, transformer block 0.

Spec notes: arch-config defaults (lingbotworld.py:57-71) match the checkpoint's
transformer/config.json exactly (heads=40, head_dim=128, ffn_dim=13824,
qk_norm="rms_norm_across_heads", cross_attn_norm=True, eps=1e-6,
added_kv_proj_dim=None -> WanT2VCrossAttention). Two-expert MoE checkpoint
(transformer + transformer_2, identical configs); we gate "transformer" (the
high-noise expert). Checkpoint tensors are fp32; casting the block to bf16
mirrors serving dtype. RoPE is built outside exactly as
LingBotWorldTransformer3DModel.forward does (model.py:325-336, float64) and
applied inside attn1 via freqs_cis. c2ws_plucker_emb must match hidden_states
shape (model.py:52-55 assert) and is passed as a real tensor so all 8
cam_conditioner tensors are exercised. Block returns a single tensor.
"""

from __future__ import annotations

import torch

from fastvideo.tests.golden_gate._harness import GateSpec, distributed_runtime, run_gate

__all__ = ["distributed_runtime"]

INNER_DIM = 5120  # 40 heads x 128, LingBot-World transformer/config.json
TEXT_LEN = 512


def _build_block(layer: int) -> torch.nn.Module:
    from fastvideo.configs.models.dits.lingbotworld import LingBotWorldArchConfig
    from fastvideo.models.dits.lingbotworld.model import LingBotWorldTransformerBlock

    arch = LingBotWorldArchConfig()
    return LingBotWorldTransformerBlock(
        INNER_DIM,
        arch.ffn_dim,
        arch.num_attention_heads,
        arch.qk_norm,
        arch.cross_attn_norm,
        arch.eps,
        arch.added_kv_proj_dim,
        arch._supported_attention_backends,
        prefix=f"Wan.blocks.{layer}",
    )


def _make_inputs(device: torch.device, seed: int) -> dict:
    from fastvideo.layers.rotary_embedding import get_rotary_pos_embed

    generator = torch.Generator(device="cpu").manual_seed(seed)
    t, h, w = 3, 8, 14
    seq = t * h * w

    d = INNER_DIM // 40
    rope_dim_list = [d - 4 * (d // 6), 2 * (d // 6), 2 * (d // 6)]
    freqs_cos, freqs_sin = get_rotary_pos_embed(
        (t, h, w), INNER_DIM, 40, rope_dim_list, rope_theta=10000, dtype=torch.float64)
    freqs_cis = (freqs_cos.to(device).float(), freqs_sin.to(device).float())

    hidden = torch.randn(1, seq, INNER_DIM, generator=generator, dtype=torch.float32)
    encoder = torch.randn(1, TEXT_LEN, INNER_DIM, generator=generator, dtype=torch.float32)
    temb = torch.randn(1, 6, INNER_DIM, generator=generator, dtype=torch.float32)
    cam = torch.randn(1, seq, INNER_DIM, generator=generator, dtype=torch.float32)

    return {
        "hidden_states": hidden.to(device=device, dtype=torch.bfloat16),
        "encoder_hidden_states": encoder.to(device=device, dtype=torch.bfloat16),
        "temb": temb.to(device=device, dtype=torch.bfloat16),
        "freqs_cis": freqs_cis,
        "original_seq_len": seq,
        "c2ws_plucker_emb": cam.to(device=device, dtype=torch.bfloat16),
    }


SPEC = GateSpec(
    name="lingbot_world_cam",
    repo_id="FastVideo/LingBot-World-Base-Cam-Diffusers",
    build_block=_build_block,
    make_inputs=_make_inputs,
    prefix_template="blocks.{N}.",
    renames=(
        (r"^\.modulation$", ".scale_shift_table"),
        (r"\.self_attn\.q\.", ".to_q."),
        (r"\.self_attn\.k\.", ".to_k."),
        (r"\.self_attn\.v\.", ".to_v."),
        (r"\.self_attn\.o\.", ".to_out."),
        (r"\.self_attn\.norm_q\.", ".norm_q."),
        (r"\.self_attn\.norm_k\.", ".norm_k."),
        (r"\.norm3\.", ".self_attn_residual_norm.norm."),
        (r"\.cross_attn\.q\.", ".attn2.to_q."),
        (r"\.cross_attn\.k\.", ".attn2.to_k."),
        (r"\.cross_attn\.v\.", ".attn2.to_v."),
        (r"\.cross_attn\.o\.", ".attn2.to_out."),
        (r"\.cross_attn\.norm_q\.", ".attn2.norm_q."),
        (r"\.cross_attn\.norm_k\.", ".attn2.norm_k."),
        (r"\.ffn\.0\.", ".ffn.fc_in."),
        (r"\.ffn\.2\.", ".ffn.fc_out."),
        (r"\.cam_injector_layer1\.", ".cam_conditioner.cam_injector.fc_in."),
        (r"\.cam_injector_layer2\.", ".cam_conditioner.cam_injector.fc_out."),
        (r"\.cam_scale_layer\.", ".cam_conditioner.cam_scale_layer."),
        (r"\.cam_shift_layer\.", ".cam_conditioner.cam_shift_layer."),
    ),
    attention_backend="FLASH_ATTN",
)


def test_lingbot_golden_gate(distributed_runtime) -> None:
    run_gate(SPEC)
