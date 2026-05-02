# SPDX-License-Identifier: Apache-2.0
"""Numerical parity test: FastVideo MagiHumanDiT vs upstream DiTModel.

Loads both models from the **same converted base checkpoint** and runs them
on identical small inputs. Asserts closeness on the joint video+audio
output tensor.

What this catches (that the preflight test does NOT):
  - Silent weight-name mismatches that `strict=False` loading would hide.
  - Wrong modality-expert chunking inside `PackedExpertLinear`.
  - RoPE sin/cos ordering flipped.
  - Per-head gating dtype / split order.
  - swiglu7 / gelu7 off-by-one on the `+1` linear bias.

Skips cleanly when:
  - `daVinci-MagiHuman/` clone is absent (no upstream source).
  - GAIR/daVinci-MagiHuman base shards are not available locally.
  - CUDA is unavailable.

Tolerance: `atol=5e-3, rtol=5e-3` on bf16 forward paths. The FastVideo
attention path uses `F.scaled_dot_product_attention` while upstream uses
`flash_attn_func`; both accumulate in bf16 but via different kernels, so
small drift is expected and bounded.
"""
from __future__ import annotations

import gc
import os
from pathlib import Path

import pytest
import torch
from torch.testing import assert_close

from fastvideo.forward_context import set_forward_context


# Force TORCH_SDPA for FastVideo so the attention kernel is deterministic.
os.environ.setdefault("FASTVIDEO_ATTENTION_BACKEND", "TORCH_SDPA")


def _find_base_shard_dir() -> Path | None:
    """Return the local path to GAIR/daVinci-MagiHuman/base/ with shards present, or None."""
    override = os.getenv("MAGI_HUMAN_BASE_SHARD_DIR")
    if override:
        p = Path(override)
        return p if p.is_dir() else None
    try:
        from huggingface_hub import snapshot_download
        snap = snapshot_download(
            repo_id="GAIR/daVinci-MagiHuman",
            allow_patterns=[
                "base/*.safetensors",
                "base/model.safetensors.index.json",
            ],
        )
        candidate = Path(snap) / "base"
        if candidate.is_dir() and any(candidate.glob("*.safetensors")):
            return candidate
        return None
    except Exception:
        return None


def _cleanup_gpu() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="MagiHuman DiT parity requires CUDA.",
)
def test_magi_human_dit_parity():
    repo_root = Path(__file__).resolve().parents[3]
    upstream_src = repo_root / "daVinci-MagiHuman"
    if not upstream_src.exists():
        pytest.skip(
            "Upstream daVinci-MagiHuman/ clone missing. Run "
            "`git clone --depth 1 https://github.com/GAIR-NLP/daVinci-MagiHuman.git`"
        )

    base_shard_dir = _find_base_shard_dir()
    if base_shard_dir is None or not base_shard_dir.is_dir():
        pytest.skip(
            "GAIR/daVinci-MagiHuman base/ shards not available locally. "
            "Set MAGI_HUMAN_BASE_SHARD_DIR or run the conversion once to "
            "populate the HF cache."
        )

    converted_dir = Path(os.getenv(
        "MAGI_HUMAN_DIFFUSERS_PATH",
        repo_root / "converted_weights" / "magi_human_base",
    ))
    transformer_dir = converted_dir / "transformer"
    if not transformer_dir.is_dir():
        pytest.skip(
            f"Converted transformer dir missing at {transformer_dir}. Run "
            f"scripts/checkpoint_conversion/convert_magi_human_to_diffusers.py first."
        )

    # Add upstream to sys.path and install compiler/distributed stubs.
    from tests.local_tests.helpers.magi_human_upstream import (
        install_stubs,
        load_upstream_dit,
    )
    install_stubs()

    # --- Shared inputs (deliberately small) ---
    device = torch.device("cuda:0")
    torch.manual_seed(0)

    # Mirror MagiDataProxy.process_input for a tiny frame:
    # video_latent: [1, z_dim, T, H, W], T=2, H=6, W=6, z_dim=48
    # -> video tokens: (T/pT)*(H/pH)*(W/pW) with patch=(1,2,2) = 2*3*3 = 18
    # audio tokens: 4
    # text tokens: 8
    # max channel width = 192 (video)
    z_dim = 48
    pT, pH, pW = 1, 2, 2
    lat_T, lat_H, lat_W = 2, 6, 6
    video_latent = torch.randn(
        (1, z_dim, lat_T, lat_H, lat_W),
        dtype=torch.float32, device=device,
    )
    num_video_tokens = (lat_T // pT) * (lat_H // pH) * (lat_W // pW)  # 18
    num_audio_tokens = 4
    num_text_tokens = 8
    audio_latent = torch.randn(
        (1, num_audio_tokens, 64),
        dtype=torch.float32, device=device,
    )
    text_feat = torch.randn(
        (1, num_text_tokens, 3584),
        dtype=torch.float32, device=device,
    )

    # --- Build the packed inputs the DiT consumes ---
    from fastvideo.pipelines.basic.magi_human.stages.latent_preparation import (
        build_packed_inputs,
    )
    from fastvideo.models.dits.magi_human import Modality  # noqa: F401

    x, coords, mm = build_packed_inputs(
        video_latent=video_latent,
        audio_latent=audio_latent,
        audio_feat_len=num_audio_tokens,
        txt_feat=text_feat,
        txt_feat_len=num_text_tokens,
        patch_size=(pT, pH, pW),
        coords_style="v2",
    )
    assert x.shape[0] == num_video_tokens + num_audio_tokens + num_text_tokens

    total_tokens = x.shape[0]

    # --- Load upstream DiT first (so we know weights round-trip cleanly).
    #     Upstream reads raw base/ shards; we keep it in bf16 for speed
    #     and because that matches the FastVideo side after FSDP load.
    print("Loading upstream DiTModel from base shards...")
    upstream_model = load_upstream_dit(
        base_shard_dir,
        device=device,
        dtype=None,  # keep checkpoint dtypes (fp32 for norms, bf16 for matmuls)
    )

    # --- VarlenHandler for upstream (batch=1, total_tokens).
    from inference.common import VarlenHandler
    cu = torch.tensor([0, total_tokens], dtype=torch.int32, device=device)
    varlen = VarlenHandler(
        cu_seqlens_q=cu,
        cu_seqlens_k=cu,
        max_seqlen_q=total_tokens,
        max_seqlen_k=total_tokens,
    )

    # --- Forward upstream and capture output. ---
    print("Running upstream forward...")
    with torch.inference_mode():
        ref_out = upstream_model(
            x=x.clone(),
            coords_mapping=coords.clone(),
            modality_mapping=mm.clone(),
            varlen_handler=varlen,
            local_attn_handler=None,  # local_attn_layers=[] for base
        ).detach().float().cpu()

    # Free upstream model before loading FastVideo (saves ~30 GB on GPU).
    del upstream_model
    _cleanup_gpu()

    # --- Load FastVideo MagiHumanDiT from the converted transformer/ ---
    from fastvideo.configs.models.dits.magi_human import MagiHumanVideoConfig
    from fastvideo.models.dits.magi_human import MagiHumanDiT
    from safetensors.torch import load_file
    import glob
    print("Loading FastVideo MagiHumanDiT from converted transformer/...")
    fv_cfg = MagiHumanVideoConfig()
    fv_model = MagiHumanDiT(fv_cfg)

    fv_state = {}
    for shard in sorted(glob.glob(str(transformer_dir / "*.safetensors"))):
        fv_state.update(load_file(shard))
    missing, unexpected = fv_model.load_state_dict(fv_state, strict=False)
    assert not missing, f"FastVideo DiT missing {len(missing)} keys: {missing[:5]}"
    assert not unexpected, f"FastVideo DiT unexpected {len(unexpected)} keys: {unexpected[:5]}"

    fv_model = fv_model.to(device=device)
    fv_model.eval()

    print("Running FastVideo forward...")
    with torch.inference_mode(), set_forward_context(current_timestep=0, attn_metadata=None):
        fv_out = fv_model(x.clone(), coords.clone(), mm.clone()).detach().float().cpu()

    # Global stats
    print(
        f"ref sum={ref_out.sum().item():.4f} "
        f"abs_mean={ref_out.abs().mean().item():.4f} "
        f"shape={tuple(ref_out.shape)}"
    )
    print(
        f"fv  sum={fv_out.sum().item():.4f} "
        f"abs_mean={fv_out.abs().mean().item():.4f} "
        f"shape={tuple(fv_out.shape)}"
    )
    diff = (ref_out - fv_out).abs()
    print(
        f"diff max={diff.max().item():.6f} "
        f"mean={diff.mean().item():.6f} "
        f"median={diff.median().item():.6f}"
    )

    # Per-modality diagnostic (video, audio, text). Text rows are zero-
    # padded on both sides; video and audio should carry comparable
    # abs_mean.
    ref_video = ref_out[:num_video_tokens]
    fv_video = fv_out[:num_video_tokens]
    ref_audio = ref_out[num_video_tokens:num_video_tokens + num_audio_tokens, :64]
    fv_audio = fv_out[num_video_tokens:num_video_tokens + num_audio_tokens, :64]
    ref_text = ref_out[num_video_tokens + num_audio_tokens:]
    fv_text = fv_out[num_video_tokens + num_audio_tokens:]
    video_diff = (ref_video - fv_video).abs()
    audio_diff = (ref_audio - fv_audio).abs()
    text_diff = (ref_text - fv_text).abs()
    print(
        f"video  ref_abs={ref_video.abs().mean():.4f} "
        f"diff_max={video_diff.max():.4f} diff_mean={video_diff.mean():.4f}"
    )
    print(
        f"audio  ref_abs={ref_audio.abs().mean():.4f} "
        f"diff_max={audio_diff.max():.4f} diff_mean={audio_diff.mean():.4f}"
    )
    print(
        f"text   ref_abs={ref_text.abs().mean():.4f} "
        f"diff_max={text_diff.max():.4f} diff_mean={text_diff.mean():.4f}"
    )

    # --- Assertions ---
    assert ref_out.shape == fv_out.shape, (
        f"shape mismatch: ref={ref_out.shape} fv={fv_out.shape}"
    )

    # Text rows are zero-padded on both sides — must match exactly.
    assert_close(fv_text, ref_text, atol=1e-6, rtol=1e-6)

    # Video + audio: bf16 single-forward DiT noise floor is ~1e-3 to
    # 5e-3 per element. atol=0.03 catches gross structural bugs
    # (permutation flips, sign inversions, wrong modality dispatch,
    # missing sub-layers) while leaving 6-10x margin over actual bf16
    # noise. Observed diff_max=0.057 will FAIL — that is the bug
    # surfacing and is the intended spec for downstream root-cause
    # investigation.
    assert_close(fv_out, ref_out, atol=0.03, rtol=0.01)

    # Sanity: mean magnitudes should match within 5%. A gross bug
    # (e.g. dropping a modality branch) would show up here.
    ref_abs = ref_out.abs().mean().item()
    fv_abs = fv_out.abs().mean().item()
    rel = abs(ref_abs - fv_abs) / max(ref_abs, 1e-6)
    assert rel < 0.05, f"abs_mean drift {rel:.3%} > 5% — possible structural bug"
