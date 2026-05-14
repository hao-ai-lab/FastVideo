# SPDX-License-Identifier: Apache-2.0
"""DiT parity test for the daVinci-MagiHuman DMD-2 distilled checkpoint.

The distill variant has the SAME architecture as the base model (same 40
layers, same hidden_size, same mm_layers / gelu7_layers / local_attn_layers,
same head_dim and num_query_groups; see
`daVinci-MagiHuman/inference/common/config.py:ModelConfig`). Only the
weights differ: distill is trained for 8-step DMD-2 inference without CFG.

This test mirrors `test_magi_human_parity.py::test_magi_human_dit_parity`
exactly, just pointing at the `distill/` subfolder of GAIR/daVinci-MagiHuman
and the matching `converted_weights/magi_human_distill/`.

Skips cleanly when:
  * `daVinci-MagiHuman/` clone is absent
  * GAIR/daVinci-MagiHuman distill shards are not locally available
  * Converted distill weights have not been produced yet
  * CUDA is unavailable
"""
from __future__ import annotations

import gc
import glob
import os
from pathlib import Path

import pytest
import torch
from torch.testing import assert_close

from fastvideo.forward_context import set_forward_context


os.environ.setdefault("FASTVIDEO_ATTENTION_BACKEND", "TORCH_SDPA")


def _find_distill_shard_dir() -> Path | None:
    """Return the local path to GAIR/daVinci-MagiHuman/distill/ shards or None."""
    override = os.getenv("MAGI_HUMAN_DISTILL_SHARD_DIR")
    if override:
        p = Path(override)
        return p if p.is_dir() else None
    try:
        from huggingface_hub import snapshot_download
        snap = snapshot_download(
            repo_id="GAIR/daVinci-MagiHuman",
            allow_patterns=[
                "distill/*.safetensors",
                "distill/model.safetensors.index.json",
            ],
        )
        candidate = Path(snap) / "distill"
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
    reason="MagiHuman distill DiT parity requires CUDA.",
)
def test_magi_human_distill_dit_parity():
    repo_root = Path(__file__).resolve().parents[3]
    upstream_src = repo_root / "daVinci-MagiHuman"
    if not upstream_src.exists():
        pytest.skip(
            "Upstream daVinci-MagiHuman/ clone missing. Run "
            "`git clone --depth 1 https://github.com/GAIR-NLP/daVinci-MagiHuman.git`"
        )

    distill_shard_dir = _find_distill_shard_dir()
    if distill_shard_dir is None or not distill_shard_dir.is_dir():
        pytest.skip(
            "GAIR/daVinci-MagiHuman distill/ shards not available locally. "
            "Set MAGI_HUMAN_DISTILL_SHARD_DIR or run the conversion once to "
            "populate the HF cache."
        )

    converted_dir = Path(os.getenv(
        "MAGI_HUMAN_DISTILL_DIFFUSERS_PATH",
        repo_root / "converted_weights" / "magi_human_distill",
    ))
    transformer_dir = converted_dir / "transformer"
    if not transformer_dir.is_dir():
        pytest.skip(
            f"Converted distill transformer dir missing at {transformer_dir}. Run "
            f"scripts/checkpoint_conversion/convert_magi_human_to_diffusers.py "
            f"--subfolder distill --cast-bf16 first."
        )

    from tests.local_tests.helpers.magi_human_upstream import (
        install_stubs,
        load_upstream_dit,
    )
    install_stubs()

    device = torch.device("cuda:0")
    torch.manual_seed(0)

    z_dim = 48
    pT, pH, pW = 1, 2, 2
    lat_T, lat_H, lat_W = 2, 6, 6
    video_latent = torch.randn(
        (1, z_dim, lat_T, lat_H, lat_W),
        dtype=torch.float32, device=device,
    )
    num_video_tokens = (lat_T // pT) * (lat_H // pH) * (lat_W // pW)
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

    # Distill arch is identical to base; load_upstream_dit's _base_arch_dict
    # describes both since they share num_layers / hidden_size / mm_layers
    # / etc. Just point at the distill shards.
    print("Loading upstream distill DiTModel from distill shards...")
    upstream_model = load_upstream_dit(
        distill_shard_dir,
        device=device,
        dtype=None,
    )

    from inference.common import VarlenHandler
    cu = torch.tensor([0, total_tokens], dtype=torch.int32, device=device)
    varlen = VarlenHandler(
        cu_seqlens_q=cu,
        cu_seqlens_k=cu,
        max_seqlen_q=total_tokens,
        max_seqlen_k=total_tokens,
    )

    print("Running upstream distill forward...")
    with torch.inference_mode():
        ref_out = upstream_model(
            x=x.clone(),
            coords_mapping=coords.clone(),
            modality_mapping=mm.clone(),
            varlen_handler=varlen,
            local_attn_handler=None,
        ).detach().float().cpu()

    del upstream_model
    _cleanup_gpu()

    from fastvideo.configs.models.dits.magi_human import MagiHumanVideoConfig
    from fastvideo.models.dits.magi_human import MagiHumanDiT
    from safetensors.torch import load_file
    print("Loading FastVideo MagiHumanDiT from converted distill transformer/...")
    fv_cfg = MagiHumanVideoConfig()
    fv_model = MagiHumanDiT(fv_cfg)

    fv_state = {}
    for shard in sorted(glob.glob(str(transformer_dir / "*.safetensors"))):
        fv_state.update(load_file(shard))
    missing, unexpected = fv_model.load_state_dict(fv_state, strict=False)
    assert not missing, f"FastVideo distill DiT missing {len(missing)} keys: {missing[:5]}"
    assert not unexpected, f"FastVideo distill DiT unexpected {len(unexpected)} keys: {unexpected[:5]}"

    fv_model = fv_model.to(device=device)
    fv_model.eval()

    print("Running FastVideo distill forward...")
    with torch.inference_mode(), set_forward_context(current_timestep=0, attn_metadata=None):
        fv_out = fv_model(x.clone(), coords.clone(), mm.clone()).detach().float().cpu()

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

    assert ref_out.shape == fv_out.shape
    assert_close(fv_text, ref_text, atol=1e-6, rtol=1e-6)
    # Same tolerance as base DiT parity (post-Wave 14b dtype-boundary fixes,
    # both DiTs are bit-exact via shared upstream BaseLinear bf16 path).
    assert_close(fv_out, ref_out, atol=0.03, rtol=0.01)
    ref_abs = ref_out.abs().mean().item()
    fv_abs = fv_out.abs().mean().item()
    rel = abs(ref_abs - fv_abs) / max(ref_abs, 1e-6)
    assert rel < 0.05, f"abs_mean drift {rel:.3%} > 5%"
