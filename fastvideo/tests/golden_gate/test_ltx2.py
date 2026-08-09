# SPDX-License-Identifier: Apache-2.0
"""Golden-gate: LTX-2 (LTX2-Distilled-Diffusers) BasicAVTransformerBlock 0.

Spec notes: block geometry comes from LTX2VideoArchConfig defaults, which
match the checkpoint's transformer/config.json (video 32x128=4096, audio
32x64=2048, cross 4096/2048, rope "split", stg_block_idx 29). The block
consumes preprocessor outputs (TransformerArgs): AdaLN timestep tensors are
seeded randn with the widths get_ada_values expects (6*dim timesteps,
4*dim + dim AV-CA scale/shift + gate), and RoPE is built outside the block
exactly as TransformerArgsPreprocessor does (double_precision_rope ->
generate_ltx_freq_grid_float64), with cross-PE from positions[:, 0:1, :].
The SSIM run uses sp_size=2 (LTXDistributedSelfAttention); the gate uses
use_distributed_attention=False — same weights, local attention. The
checkpoint is single-file (transformer/model.safetensors, no index.json).
context_mask=None keeps all four attentions on FLASH_ATTN; production text
cross-attn passes an additive mask that reroutes attn2 through the
TORCH_SDPA attn_masked instance — accepted minimality tradeoff. FLASH_ATTN
must be the backend: VIDEO_SPARSE_ATTN would create extra to_gate_compress
params and break strict load. Forward returns (video, audio)
TransformerArgs; postprocess flattens/concats both .x tensors.
"""

from __future__ import annotations

import torch

from fastvideo.tests.golden_gate._harness import GateSpec, distributed_runtime, run_gate

__all__ = ["distributed_runtime"]

VIDEO_DIM = 4096  # 32 heads x 128, LTX2-Distilled transformer/config.json
AUDIO_DIM = 2048  # 32 heads x 64


def _build_block(layer: int) -> torch.nn.Module:
    from fastvideo.configs.models.dits.ltx2 import LTX2VideoArchConfig
    from fastvideo.models.dits.ltx2 import (BasicAVTransformerBlock, LTXRopeType, TransformerConfig)

    arch = LTX2VideoArchConfig()
    video_cfg = TransformerConfig(
        dim=arch.num_attention_heads * arch.attention_head_dim,
        heads=arch.num_attention_heads,
        d_head=arch.attention_head_dim,
        context_dim=arch.cross_attention_dim,
        apply_gated_attention=arch.apply_gated_attention,
        cross_attention_adaln=arch.cross_attention_adaln,
    )
    audio_cfg = TransformerConfig(
        dim=arch.audio_num_attention_heads * arch.audio_attention_head_dim,
        heads=arch.audio_num_attention_heads,
        d_head=arch.audio_attention_head_dim,
        context_dim=arch.audio_cross_attention_dim,
        apply_gated_attention=arch.apply_gated_attention,
        cross_attention_adaln=arch.cross_attention_adaln,
    )
    return BasicAVTransformerBlock(
        idx=layer,
        video=video_cfg,
        audio=audio_cfg,
        rope_type=LTXRopeType(arch.rope_type),
        norm_eps=arch.norm_eps,
        use_distributed_attention=False,
        cross_attention_adaln=arch.cross_attention_adaln,
        stg_block_idx=arch.stg_block_idx,
        quant_config=None,
        prefix="ltx2",
    )


def _make_inputs(device: torch.device, seed: int) -> dict:
    from fastvideo.models.dits.ltx2 import (AudioLatentPatchifier, AudioLatentShape, DEFAULT_LTX2_AUDIO_DOWNSAMPLE,
                                            DEFAULT_LTX2_AUDIO_HOP_LENGTH, DEFAULT_LTX2_AUDIO_MEL_BINS,
                                            DEFAULT_LTX2_AUDIO_SAMPLE_RATE, DEFAULT_LTX2_SCALE_FACTORS, LTXRopeType,
                                            TransformerArgs, VideoLatentPatchifier, VideoLatentShape, _get_pixel_coords,
                                            generate_ltx_freq_grid_float64, precompute_ltx_freqs_cis)

    generator = torch.Generator(device="cpu").manual_seed(seed)
    B, frames, h, w = 1, 2, 4, 6
    seq_v = frames * h * w  # 48
    n_text, seq_a = 32, 12

    # Video positions: patch grid bounds -> pixel coords (scale 8,32,32;
    # fps 24; causal fix), cast to hidden dtype like ltx2.py:3044.
    v_pos = VideoLatentPatchifier(patch_size=1).get_patch_grid_bounds(VideoLatentShape((B, 128, frames, h, w)),
                                                                      device=torch.device("cpu"))
    v_pos = _get_pixel_coords(v_pos, DEFAULT_LTX2_SCALE_FACTORS, fps=24.0, causal_fix=True).to(torch.bfloat16)
    # Audio positions stay float32 [B, 1, T, 2] (ltx2.py:3099-3101).
    a_pos = AudioLatentPatchifier(
        patch_size=DEFAULT_LTX2_AUDIO_MEL_BINS,
        sample_rate=DEFAULT_LTX2_AUDIO_SAMPLE_RATE,
        hop_length=DEFAULT_LTX2_AUDIO_HOP_LENGTH,
        audio_latent_downsample_factor=DEFAULT_LTX2_AUDIO_DOWNSAMPLE,
        is_causal=True,
        shift=0,
    ).get_patch_grid_bounds(AudioLatentShape((B, 8, seq_a, DEFAULT_LTX2_AUDIO_MEL_BINS)), device=torch.device("cpu"))

    rope_kw = dict(out_dtype=torch.bfloat16,
                   theta=10000.0,
                   use_middle_indices_grid=True,
                   num_attention_heads=32,
                   rope_type=LTXRopeType.SPLIT,
                   freq_grid_generator=generate_ltx_freq_grid_float64)
    v_pe = precompute_ltx_freqs_cis(v_pos, dim=VIDEO_DIM, max_pos=[20, 2048, 2048], **rope_kw)
    v_cross_pe = precompute_ltx_freqs_cis(v_pos[:, 0:1, :], dim=AUDIO_DIM, max_pos=[20], **rope_kw)
    a_pe = precompute_ltx_freqs_cis(a_pos, dim=AUDIO_DIM, max_pos=[20], **rope_kw)
    a_cross_pe = precompute_ltx_freqs_cis(a_pos[:, 0:1, :], dim=AUDIO_DIM, max_pos=[20], **rope_kw)

    def rnd(*shape):
        return torch.randn(*shape, generator=generator, dtype=torch.float32).to(device=device, dtype=torch.bfloat16)

    def to_dev(pe):
        return tuple(t.to(device) for t in pe)

    video = TransformerArgs(x=rnd(B, seq_v, VIDEO_DIM),
                            context=rnd(B, n_text, VIDEO_DIM),
                            context_mask=None,
                            timesteps=rnd(B, seq_v, 6 * VIDEO_DIM),
                            embedded_timestep=rnd(B, seq_v, VIDEO_DIM),
                            positional_embeddings=to_dev(v_pe),
                            cross_positional_embeddings=to_dev(v_cross_pe),
                            cross_scale_shift_timestep=rnd(B, seq_v, 4 * VIDEO_DIM),
                            cross_gate_timestep=rnd(B, seq_v, VIDEO_DIM),
                            enabled=True,
                            prompt_timestep=None)
    audio = TransformerArgs(x=rnd(B, seq_a, AUDIO_DIM),
                            context=rnd(B, n_text, AUDIO_DIM),
                            context_mask=None,
                            timesteps=rnd(B, seq_a, 6 * AUDIO_DIM),
                            embedded_timestep=rnd(B, seq_a, AUDIO_DIM),
                            positional_embeddings=to_dev(a_pe),
                            cross_positional_embeddings=to_dev(a_cross_pe),
                            cross_scale_shift_timestep=rnd(B, seq_a, 4 * AUDIO_DIM),
                            cross_gate_timestep=rnd(B, seq_a, AUDIO_DIM),
                            enabled=True,
                            prompt_timestep=None)

    return {"video": video, "audio": audio, "video_original_seq_len": seq_v, "audio_original_seq_len": seq_a}


SPEC = GateSpec(
    name="ltx2_distilled",
    repo_id="FastVideo/LTX2-Distilled-Diffusers",
    build_block=_build_block,
    make_inputs=_make_inputs,
    prefix_template="transformer_blocks.{N}.",
    renames=(),
    attention_backend="FLASH_ATTN",
    weight_file="model.safetensors",  # single-file checkpoint, no index.json
    # forward returns (video, audio) TransformerArgs — reduce to one tensor
    postprocess=lambda out: torch.cat([out[0].x.flatten(), out[1].x.flatten()]),
)


def test_ltx2_golden_gate(distributed_runtime) -> None:
    run_gate(SPEC)
