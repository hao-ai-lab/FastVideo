# SPDX-License-Identifier: Apache-2.0
"""TI2V latent-loop parity for daVinci-MagiHuman.

This mirrors the base MagiHuman pipeline parity test but enables the upstream
`latent_image is not None` branch: the clean image latent is copied into
`latent_video[:, :, :1]` before every DiT call and once more after denoising.
"""
from __future__ import annotations

import glob
import os
from pathlib import Path

import pytest
import torch
from torch.testing import assert_close

from tests.local_tests.magi_human.test_magi_human_pipeline_parity import (
    _build_fastvideo_schedulers,
    _build_upstream_schedulers,
    _cleanup_gpu,
    _dit_forward_fv,
    _dit_forward_upstream,
    _encode_magi_human_prompt_pair,
    _find_base_shard_dir,
    _run_denoise_loop,
)


os.environ.setdefault("FASTVIDEO_ATTENTION_BACKEND", "TORCH_SDPA")
os.environ.setdefault("MASTER_ADDR", "localhost")
os.environ.setdefault("MASTER_PORT", "29520")


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="MagiHuman TI2V pipeline parity requires CUDA.",
)
def test_magi_human_ti2v_pipeline_latent_parity():
    repo_root = Path(__file__).resolve().parents[3]
    upstream_src = repo_root / "daVinci-MagiHuman"
    if not upstream_src.exists():
        pytest.skip("Upstream daVinci-MagiHuman/ clone missing.")

    base_shard_dir = _find_base_shard_dir()
    if base_shard_dir is None or not base_shard_dir.is_dir():
        pytest.skip("GAIR/daVinci-MagiHuman base/ shards not available locally.")

    converted_dir = Path(os.getenv(
        "MAGI_HUMAN_DIFFUSERS_PATH",
        repo_root / "converted_weights" / "magi_human_base",
    ))
    transformer_dir = converted_dir / "transformer"
    if not transformer_dir.is_dir():
        pytest.skip(f"Converted transformer dir missing at {transformer_dir}")

    from tests.local_tests.helpers.magi_human_upstream import (
        install_stubs,
        load_upstream_dit,
    )
    install_stubs()

    device = torch.device("cuda:0")
    torch.manual_seed(123)

    z_dim = 48
    patch_size = (1, 2, 2)
    lat_T, lat_H, lat_W = 2, 6, 6
    video_latent = torch.randn(
        (1, z_dim, lat_T, lat_H, lat_W),
        dtype=torch.float32,
        device=device,
    )
    audio_latent = torch.randn((1, 4, 64), dtype=torch.float32, device=device)
    image_latent = torch.randn(
        (1, z_dim, 1, lat_H, lat_W),
        dtype=torch.float32,
        device=device,
    )
    txt_feat, txt_feat_len, neg_txt_feat, neg_txt_feat_len = (
        _encode_magi_human_prompt_pair(device)
    )

    num_inference_steps = 4
    shift = 5.0
    common_kwargs = dict(
        cfg_number=2,
        video_txt_guidance_scale=5.0,
        audio_txt_guidance_scale=5.0,
        patch_size=patch_size,
        coords_style="v2",
        video_in_channels=192,
        audio_in_channels=64,
        image_latent=image_latent,
    )

    up_video_sched, up_audio_sched = _build_upstream_schedulers(
        shift=shift,
        num_inference_steps=num_inference_steps,
        device=device,
    )
    print("Loading upstream DiTModel from base shards...")
    upstream_dit = load_upstream_dit(base_shard_dir, device=device, dtype=None)
    print("Running upstream TI2V denoise loop...")
    ref_video, ref_audio = _run_denoise_loop(
        upstream_dit,
        _dit_forward_upstream,
        video_latent.clone(),
        audio_latent.clone(),
        txt_feat.clone(),
        txt_feat_len,
        neg_txt_feat.clone(),
        neg_txt_feat_len,
        video_sched=up_video_sched,
        audio_sched=up_audio_sched,
        **common_kwargs,
    )
    ref_video = ref_video.detach().float().cpu()
    ref_audio = ref_audio.detach().float().cpu()
    del upstream_dit
    _cleanup_gpu()

    fv_video_sched, fv_audio_sched = _build_fastvideo_schedulers(
        shift=shift,
        num_inference_steps=num_inference_steps,
        device=device,
    )
    from fastvideo.configs.models.dits.magi_human import MagiHumanVideoConfig
    from fastvideo.models.dits.magi_human import MagiHumanDiT
    from safetensors.torch import load_file

    print("Loading FastVideo MagiHumanDiT from converted transformer/...")
    fv_cfg = MagiHumanVideoConfig()
    fv_dit = MagiHumanDiT(fv_cfg)
    fv_state = {}
    for shard in sorted(glob.glob(str(transformer_dir / "*.safetensors"))):
        fv_state.update(load_file(shard))
    missing, unexpected = fv_dit.load_state_dict(fv_state, strict=False)
    assert not missing, f"FastVideo DiT missing {len(missing)} keys: {missing[:5]}"
    assert not unexpected, f"FastVideo DiT unexpected {len(unexpected)} keys: {unexpected[:5]}"
    fv_dit = fv_dit.to(device=device)
    fv_dit.eval()

    print("Running FastVideo TI2V denoise loop...")
    fv_video, fv_audio = _run_denoise_loop(
        fv_dit,
        _dit_forward_fv,
        video_latent.clone(),
        audio_latent.clone(),
        txt_feat.clone(),
        txt_feat_len,
        neg_txt_feat.clone(),
        neg_txt_feat_len,
        video_sched=fv_video_sched,
        audio_sched=fv_audio_sched,
        **common_kwargs,
    )
    fv_video = fv_video.detach().float().cpu()
    fv_audio = fv_audio.detach().float().cpu()

    v_diff = (ref_video - fv_video).abs()
    a_diff = (ref_audio - fv_audio).abs()
    print(
        f"ti2v video diff_max={v_diff.max().item():.4f} "
        f"diff_mean={v_diff.mean().item():.4f}"
    )
    print(
        f"ti2v audio diff_max={a_diff.max().item():.4f} "
        f"diff_mean={a_diff.mean().item():.4f}"
    )

    assert ref_video.shape == fv_video.shape
    assert ref_audio.shape == fv_audio.shape
    assert_close(fv_video, ref_video, atol=0.40, rtol=0.05)
    assert_close(fv_audio, ref_audio, atol=0.40, rtol=0.05)
    assert_close(fv_video[:, :, :1], image_latent.detach().cpu(), atol=0.0, rtol=0.0)
    assert_close(ref_video[:, :, :1], image_latent.detach().cpu(), atol=0.0, rtol=0.0)

    ref_v_abs = ref_video.abs().mean().item()
    ref_a_abs = ref_audio.abs().mean().item()
    rel_v = abs(ref_v_abs - fv_video.abs().mean().item()) / max(ref_v_abs, 1e-6)
    rel_a = abs(ref_a_abs - fv_audio.abs().mean().item()) / max(ref_a_abs, 1e-6)
    assert rel_v < 0.01, f"video abs_mean drift {rel_v:.2%} > 1%"
    assert rel_a < 0.01, f"audio abs_mean drift {rel_a:.2%} > 1%"
    assert v_diff.mean().item() / max(ref_v_abs, 1e-6) < 0.04
    assert a_diff.mean().item() / max(ref_a_abs, 1e-6) < 0.04
