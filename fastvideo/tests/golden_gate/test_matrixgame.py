# SPDX-License-Identifier: Apache-2.0
"""Golden-gate: MatrixGame family — TWO distinct DiTs, one gate each.

MG2 = CausalMatrixGame2TransformerBlock (matrixgame2/causal_model.py:385), the
CAUSAL block the checkpoint's _class_name (CausalMatrixGame2WanModel) loads —
NOT model.py's MatrixGame2TransformerBlock. MG3 = MatrixGame3TransformerBlock
(matrixgame3/model.py:199).

Spec notes: MG2 arch-config defaults are Wan-14B sized and WRONG for this
checkpoint; real dims (1536/8960/12 heads/local_attn_size 6) and action_config
come from the checkpoint's transformer/config.json, fetched at build time. MG3
arch-class defaults DO match its checkpoint. Both checkpoints are single-file
safetensors (no index.json) in FastVideo-native (MG2) / self_attn-style (MG3)
names. Rope is handled INSIDE both blocks: MG2's freqs_cis argument is dead
(attn1 rebuilds rope internally, causal_model.py:200-211) so pass None; MG3
takes a caller-built per-head complex table via _build_rope_freqs
(use_memory=True -> sigma_theta 0.8, max_seq_len 2048, model.py:640-646).
Geometry (3, 22, 40) is load-bearing for MG2: seq % 128 != 0 and f % 32 != 0
or the flex-attention unpad slices `[:, :, :-0]` go empty
(causal_model.py:224/266/558). temb is 4-dim for both (MG2 asserts it,
causal_model.py:521; MG3 per-token path, model.py:309). Layer 0 covers the
ActionModule (blocks 0-14 only); mouse/keyboard conds use pixel length
4*(f-1)+1. MG2's cache-less self-attn runs torch.compile'd flex_attention
(max-autotune) — first call compiles, budget accordingly. Both blocks return a
single tensor; no postprocess needed. Both ssim tests use FLASH_ATTN.
"""

from __future__ import annotations

import torch

from fastvideo.tests.golden_gate._harness import GateSpec, distributed_runtime, run_gate

__all__ = ["distributed_runtime"]

MG2_REPO = "FastVideo/Matrix-Game-2.0-Base-Distilled-Diffusers"
MG3_REPO = "FastVideo/Matrix-Game-3.0-Base-Distilled-Diffusers"

MG2_INNER_DIM = 1536  # 12 heads x 128, MG2 transformer/config.json
MG3_INNER_DIM = 3072  # 24 heads x 128, matrixgame3.py:42-43

F, H, W = 3, 22, 40  # post-patch grid; seq=2640 (% 128 != 0, see docstring)
N_PIX = 4 * (F - 1) + 1  # pixel-frame action length (action_module.py:511-521)


# ===== Matrix-Game-2.0 (causal) =====


def _build_block_mg2(layer: int) -> torch.nn.Module:
    import json
    from pathlib import Path

    from huggingface_hub import hf_hub_download

    from fastvideo.configs.models.dits.matrixgame2 import MatrixGame2WanVideoArchConfig
    from fastvideo.models.dits.matrixgame2 import CausalMatrixGame2TransformerBlock

    arch = MatrixGame2WanVideoArchConfig()
    # ponytail: arch-config defaults are Wan-14B and wrong here; the checkpoint's
    # config.json is the source of truth (hidden 1536, ffn 8960, 12 heads).
    hf = json.loads(Path(hf_hub_download(MG2_REPO, "transformer/config.json")).read_text())
    inner_dim = hf["num_attention_heads"] * hf["attention_head_dim"]
    assert inner_dim == MG2_INNER_DIM
    return CausalMatrixGame2TransformerBlock(
        inner_dim,
        hf["ffn_dim"],                # 8960
        hf["num_attention_heads"],    # 12
        hf["local_attn_size"],        # 6
        hf["sink_size"],              # 0
        hf["qk_norm"],                # "rms_norm_across_heads"
        arch.cross_attn_norm,         # True (block asserts it, causal_model.py:436)
        hf["eps"],                    # 1e-6
        arch.added_kv_proj_dim,       # None -> attn2 = WanT2VCrossAttention
        arch._supported_attention_backends,
        prefix=f"Wan.blocks.{layer}",
        action_config=hf["action_config"],  # ActionModule on blocks 0-14 only
        block_idx=layer,
        rope_cache_policy=arch.rope_cache_policy,  # "absolute"
    )


def _make_inputs_mg2(device: torch.device, seed: int) -> dict:
    from fastvideo.models.dits.matrixgame2 import CausalMatrixGame2WanModel

    generator = torch.Generator(device="cpu").manual_seed(seed)
    seq = F * H * W  # 2640

    hidden = torch.randn(1, seq, MG2_INNER_DIM, generator=generator, dtype=torch.float32)
    # CLIP image embeds post WanImageEmbedding
    context = torch.randn(1, 257, MG2_INNER_DIM, generator=generator, dtype=torch.float32)
    temb = torch.randn(1, F, 6, MG2_INNER_DIM, generator=generator, dtype=torch.float32)
    mouse = torch.randn(1, N_PIX, 2, generator=generator, dtype=torch.float32)     # mouse_dim_in=2
    keyboard = torch.randn(1, N_PIX, 4, generator=generator, dtype=torch.float32)  # keyboard_dim_in=4

    # masks exactly as _forward_inference builds them (causal_model.py:1129-1162)
    block_mask = CausalMatrixGame2WanModel._prepare_blockwise_causal_attn_mask(
        device=device, num_frames=F, frame_seqlen=H * W,
        num_frame_per_block=3, local_attn_size=6, sink_size=0)
    block_mask_action = CausalMatrixGame2WanModel._prepare_blockwise_causal_attn_mask_action(
        device=device, num_frames=F, frame_seqlen=1,
        num_frame_per_block=3, local_attn_size=6, sink_size=0)

    return {
        "hidden_states": hidden.to(device=device, dtype=torch.float32),
        "encoder_hidden_states": context.to(device=device, dtype=torch.float32),
        "temb": temb.to(device=device, dtype=torch.float32),
        "freqs_cis": None,  # dead arg: attn1 rebuilds rope internally
        "block_mask": block_mask,
        "grid_sizes": (F, H, W),
        "mouse_cond": mouse.to(device=device, dtype=torch.float32),
        "keyboard_cond": keyboard.to(device=device, dtype=torch.float32),
        "block_mask_mouse": block_mask_action,
        "block_mask_keyboard": block_mask_action,  # use_rope_keyboard=True path
        "num_frame_per_block": 3,
        "use_rope_keyboard": True,  # asserted in ActionModule.forward
        # kv_cache* / crossattn_cache None, current_start=0 -> cache-less flex path
    }


MG2_SPEC = GateSpec(
    name="matrixgame2_causal",
    dtype=torch.float32,
    repo_id=MG2_REPO,
    build_block=_build_block_mg2,
    make_inputs=_make_inputs_mg2,
    prefix_template="blocks.{N}.",
    renames=(),  # mirror checkpoint is already in FastVideo-native names
    attention_backend="FLASH_ATTN",
)


def test_matrixgame2_golden_gate(distributed_runtime) -> None:
    run_gate(MG2_SPEC)


# ===== Matrix-Game-3.0 =====


def _build_block_mg3(layer: int) -> torch.nn.Module:
    from fastvideo.configs.models.dits.matrixgame3 import MatrixGame3WanVideoArchConfig
    from fastvideo.models.dits.matrixgame3 import MatrixGame3TransformerBlock

    arch = MatrixGame3WanVideoArchConfig()
    inner_dim = arch.num_attention_heads * arch.attention_head_dim  # 24*128 = 3072
    assert inner_dim == MG3_INNER_DIM
    return MatrixGame3TransformerBlock(
        inner_dim,
        arch.ffn_dim,                 # 14336
        arch.num_attention_heads,     # 24
        arch.qk_norm,                 # "rms_norm_across_heads"
        arch.cross_attn_norm,         # True (asserted at model.py:246)
        arch.eps,                     # 1e-6
        arch._supported_attention_backends,
        prefix=f"Wan.blocks.{layer}",
        action_config=arch.action_config,  # blocks 0-14 only
        block_id=layer,
        use_memory=arch.use_memory,   # True -> cam_* Linears on EVERY block
    )


def _make_inputs_mg3(device: torch.device, seed: int) -> dict:
    from fastvideo.models.dits.matrixgame3.model import _build_rope_freqs

    generator = torch.Generator(device="cpu").manual_seed(seed)
    seq = F * H * W  # 2640

    hidden = torch.randn(1, seq, MG3_INNER_DIM, generator=generator, dtype=torch.float32)
    # text tokens AFTER model-level text_embedding proj (model.py:715-726)
    context = torch.randn(1, 512, MG3_INNER_DIM, generator=generator, dtype=torch.float32)
    # per-token 4-D temb (bs, seq, 6, 3072), real path (model.py:687-713)
    temb = torch.randn(1, seq, 6, MG3_INNER_DIM, generator=generator, dtype=torch.float32)
    mouse = torch.randn(1, N_PIX, 2, generator=generator, dtype=torch.float32)     # mouse_dim_in=2
    keyboard = torch.randn(1, N_PIX, 6, generator=generator, dtype=torch.float32)  # keyboard_dim_in=6

    # exactly as model.py:640-646: use_memory=True -> per-head 3-D complex table
    freqs = _build_rope_freqs(max_seq_len=2048, head_dim=128, num_heads=24,
                              sigma_theta=0.8, device=device)  # (24, 2048, 64) complex128

    return {
        "hidden_states": hidden.to(device=device, dtype=torch.bfloat16),
        "encoder_hidden_states": context.to(device=device, dtype=torch.bfloat16),
        "temb": temb.to(device=device, dtype=torch.bfloat16),
        "freqs": freqs,
        "grid_sizes": (F, H, W),
        "mouse_cond": mouse.to(device=device, dtype=torch.bfloat16),
        "keyboard_cond": keyboard.to(device=device, dtype=torch.bfloat16),
        # plucker_emb=None -> cam_* layers loaded but skipped;
        # memory_length=0 / memory_latent_idx=None -> arange fallback
    }


MG3_SPEC = GateSpec(
    name="matrixgame3",
    repo_id=MG3_REPO,
    build_block=_build_block_mg3,
    make_inputs=_make_inputs_mg3,
    prefix_template="blocks.{N}.",
    renames=(
        (r"\.self_attn\.q\.", ".to_q."),
        (r"\.self_attn\.k\.", ".to_k."),
        (r"\.self_attn\.v\.", ".to_v."),
        (r"\.self_attn\.o\.", ".to_out."),
        (r"\.self_attn\.norm_q\.", ".norm_q."),
        (r"\.self_attn\.norm_k\.", ".norm_k."),
        (r"\.cross_attn\.q\.", ".attn2.to_q."),
        (r"\.cross_attn\.k\.", ".attn2.to_k."),
        (r"\.cross_attn\.v\.", ".attn2.to_v."),
        (r"\.cross_attn\.o\.", ".attn2.to_out."),
        (r"\.cross_attn\.norm_q\.", ".attn2.norm_q."),
        (r"\.cross_attn\.norm_k\.", ".attn2.norm_k."),
        (r"\.ffn\.0\.", ".ffn.fc_in."),
        (r"\.ffn\.2\.", ".ffn.fc_out."),
        (r"\.norm3\.", ".self_attn_residual_norm.norm."),
        (r"\.modulation$", ".scale_shift_table"),
        # cam_* and action_model.* keys already match module names
    ),
    attention_backend="FLASH_ATTN",
)


def test_matrixgame3_golden_gate(distributed_runtime) -> None:
    run_gate(MG3_SPEC)
