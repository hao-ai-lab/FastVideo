# SPDX-License-Identifier: Apache-2.0
"""Golden-gate: DreamX-World-5B-Cam transformer block 0 (Wan2.2-TI2V-5B derivative
with PRoPE camera control, bidirectional diffusion variant).

Spec notes: DreamXWorldArchConfig class defaults inherit Wan-14B dims and do NOT
match the 5B checkpoint — geometry comes from make_dreamx_world_5b_cam_dit_config()
(configs/pipelines/dreamx_world.py:19-34), verified equal to the mirror repo's
transformer/config.json. Checkpoint index is model.safetensors.index.json (saved
via save_torch_state_dict), hence weight_file="model.safetensors". temb is the 4D
[B, seq, 6, inner_dim] per-token shape (expand_timesteps=True pipeline path,
dreamx_world.py:273-282). y_camera dtype must equal hidden_states dtype (mixed
dtype einsum in _dreamx_prope_qkv errors); the camera builder is deterministic
(camera_conditioning.py:201-228). The ssim test runs TORCH_SDPA only. Forward
returns a plain [1, seq, 3072] tensor — no postprocess. The AR variant
(DreamX-World-5B, CausalWanAttentionBlock) is a different implementation and
needs its own gate recipe; not covered by this spec.
"""

from __future__ import annotations

import torch

from fastvideo.tests.golden_gate._harness import GateSpec, distributed_runtime, run_gate

__all__ = ["distributed_runtime"]

INNER_DIM = 3072  # 24 heads x 128, DreamX-World-5B-Cam transformer/config.json
TEXT_LEN = 512


def _build_block(layer: int) -> torch.nn.Module:
    from fastvideo.configs.pipelines.dreamx_world import (
        make_dreamx_world_5b_cam_dit_config)
    from fastvideo.models.dits.dreamx_world import DreamXWorldTransformerBlock

    arch = make_dreamx_world_5b_cam_dit_config().arch_config
    # exactly as DreamXWorldTransformer3DModel.__init__ builds each block
    # (dreamx_world.py:374-391)
    return DreamXWorldTransformerBlock(
        arch.num_attention_heads * arch.attention_head_dim,  # 3072
        arch.ffn_dim,               # 14336
        arch.num_attention_heads,   # 24
        arch.qk_norm,               # "rms_norm_across_heads"
        arch.cross_attn_norm,       # True
        arch.eps,                   # 1e-6
        arch.added_kv_proj_dim,     # None -> T2V cross-attn
        arch._supported_attention_backends,
        quant_config=None,
        prefix=f"Wan.blocks.{layer}",
        add_control_adapter=arch.add_control_adapter,    # True
        cam_method=arch.cam_method,                      # "prope"
        attn_compress=arch.attn_compress,                # 1
        cam_self_attn_layers=arch.cam_self_attn_layers,  # None -> cam attn everywhere
        layer_idx=layer,
    )


def _make_inputs(device: torch.device, seed: int) -> dict:
    """Fixed batch mirroring the SSIM defaults (9 frames @ 64x64):
    latent (3, 4, 4) -> patch (1,2,2) -> seq = 3*2*2 = 12, cameras = 3 latent frames."""
    from fastvideo.layers.rotary_embedding import get_rotary_pos_embed
    from fastvideo.pipelines.basic.dreamx_world.camera_conditioning import (
        build_dreamx_camera_condition)

    generator = torch.Generator(device="cpu").manual_seed(seed)
    num_frames, height, width = 9, 64, 64
    post_t = (num_frames - 1) // 4 + 1                    # VAE 4x temporal
    post_h, post_w = height // 16 // 2, width // 16 // 2  # VAE 16x spatial, patch (1,2,2)
    seq = post_t * post_h * post_w                        # 12

    # RoPE exactly as the model forward builds it (dreamx_world.py:429-439)
    d = INNER_DIM // 24  # head_dim 128
    rope_dim_list = [d - 4 * (d // 6), 2 * (d // 6), 2 * (d // 6)]  # [44, 42, 42]
    freqs_cos, freqs_sin = get_rotary_pos_embed(
        (post_t, post_h, post_w), INNER_DIM, 24, rope_dim_list,
        rope_theta=10000, dtype=torch.float64)
    freqs_cis = (freqs_cos.to(device).float(), freqs_sin.to(device).float())

    hidden = torch.randn(1, seq, INNER_DIM, generator=generator, dtype=torch.float32)
    encoder = torch.randn(1, TEXT_LEN, INNER_DIM, generator=generator, dtype=torch.float32)
    # expand_timesteps=True pipeline -> per-token timesteps -> 4D temb (B, seq, 6, dim)
    temb = torch.randn(1, seq, 6, INNER_DIM, generator=generator, dtype=torch.float32)

    # Deterministic camera condition, same builder + keys as the pipeline stage
    # (stages.py:55-64). dtype MUST match hidden_states.
    camera = build_dreamx_camera_condition(
        ["w"], [4.0], num_frames=num_frames, height=height, width=width,
        dtype=torch.bfloat16, device=device)
    y_camera = {k: v.unsqueeze(0) for k, v in camera.items()}  # viewmats (1,3,4,4), K (1,3,3,3)

    return {
        "hidden_states": hidden.to(device=device, dtype=torch.bfloat16),
        "encoder_hidden_states": encoder.to(device=device, dtype=torch.bfloat16),
        "temb": temb.to(device=device, dtype=torch.bfloat16),
        "freqs_cis": freqs_cis,
        "original_seq_len": seq,
        "y_camera": y_camera,
    }


SPEC = GateSpec(
    name="dreamx_world_5b_cam",
    repo_id="FastVideo/DreamX-World-5B-Cam-Diffusers",
    build_block=_build_block,
    make_inputs=_make_inputs,
    prefix_template="blocks.{N}.",
    renames=(
        # to_out rename must run before the generic attn1 strip
        (r"\.attn1\.to_out\.0\.", ".to_out."),
        (r"\.attn1\.", "."),  # to_q/to_k/to_v/norm_q/norm_k -> block level
        (r"\.attn2\.to_out\.0\.", ".attn2.to_out."),
        (r"\.ffn\.net\.0\.proj\.", ".ffn.fc_in."),
        (r"\.ffn\.net\.2\.", ".ffn.fc_out."),
        (r"\.norm2\.", ".self_attn_residual_norm.norm."),
        # identity: attn2.{to_q,to_k,to_v,norm_q,norm_k}, cam_self_attn.*, scale_shift_table
    ),
    attention_backend="TORCH_SDPA",
    weight_file="model.safetensors",
)


def test_dreamx_world_cam_golden_gate(distributed_runtime) -> None:
    run_gate(SPEC)
