# SPDX-License-Identifier: Apache-2.0
"""SR-1080p local-window latent-loop parity for daVinci-MagiHuman.

This mirrors the SR-540p two-stage parity test but enables upstream's
SR2_1080 local-attention layer set on the SR DiT. The reference side uses the
test helper's SDPA implementation of FFAHandler's segmented accumulator, so the
assertion is a kernel-noise tolerance rather than bit-exact.
"""
from __future__ import annotations

import glob
import os
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from torch.testing import assert_close

from fastvideo.pipelines.basic.magi_human.pipeline_configs import (
    _SR_1080P_LOCAL_ATTN_LAYERS,
)
from tests.local_tests.magi_human.test_magi_human_pipeline_parity import (
    _build_fastvideo_schedulers,
    _build_upstream_schedulers,
    _cleanup_gpu,
    _dit_forward_fv,
    _dit_forward_upstream,
    _find_base_shard_dir,
    _run_denoise_loop,
)
from tests.local_tests.magi_human.test_magi_human_sr540p_pipeline_parity import (
    _load_fv_dit,
    _prepare_sr_latents,
    _run_sr_denoise_loop,
)


os.environ.setdefault("FASTVIDEO_ATTENTION_BACKEND", "TORCH_SDPA")
os.environ.setdefault("MASTER_ADDR", "localhost")
os.environ.setdefault("MASTER_PORT", "29522")


def _find_sr1080p_shard_dir() -> Path | None:
    override = os.getenv("MAGI_HUMAN_SR1080P_SHARD_DIR")
    if override:
        path = Path(override)
        return path if path.is_dir() else None
    try:
        from huggingface_hub import snapshot_download
        snap = snapshot_download(
            repo_id="GAIR/daVinci-MagiHuman",
            allow_patterns=[
                "1080p_sr/*.safetensors",
                "1080p_sr/model.safetensors.index.json",
            ],
        )
        candidate = Path(snap) / "1080p_sr"
        if candidate.is_dir() and any(candidate.glob("*.safetensors")):
            return candidate
        return None
    except Exception:
        return None


def _dit_forward_upstream_local(
    dit,
    video_latent,
    audio_latent,
    audio_feat_len,
    txt_feat,
    txt_feat_len,
    patch_size,
    coords_style,
    video_in_channels,
    audio_in_channels,
):
    from fastvideo.pipelines.basic.magi_human.stages.latent_preparation import (
        build_packed_inputs,
        unpack_tokens,
    )
    from inference.common import VarlenHandler
    from inference.pipeline.data_proxy import calc_local_attn_ffa_handler

    x, coords, mm = build_packed_inputs(
        video_latent=video_latent,
        audio_latent=audio_latent,
        audio_feat_len=audio_feat_len,
        txt_feat=txt_feat,
        txt_feat_len=txt_feat_len,
        patch_size=patch_size,
        coords_style=coords_style,
    )
    video_token_num = x.shape[0] - audio_feat_len - txt_feat_len
    total = x.shape[0]
    cu = torch.tensor([0, total], dtype=torch.int32, device=x.device)
    varlen = VarlenHandler(
        cu_seqlens_q=cu,
        cu_seqlens_k=cu,
        max_seqlen_q=total,
        max_seqlen_k=total,
    )
    local_attn = calc_local_attn_ffa_handler(
        video_token_num,
        audio_feat_len + txt_feat_len,
        video_latent.shape[2] // patch_size[0],
        11,
    )
    out = dit(
        x=x,
        coords_mapping=coords,
        modality_mapping=mm,
        varlen_handler=varlen,
        local_attn_handler=local_attn,
    )
    return unpack_tokens(
        out,
        video_token_num=video_token_num,
        audio_feat_len=audio_feat_len,
        video_in_channels=video_in_channels,
        audio_in_channels=audio_in_channels,
        latent_shape=tuple(video_latent.shape),
        patch_size=patch_size,
    )


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="MagiHuman SR-1080p pipeline parity requires CUDA.",
)
@pytest.mark.parametrize("use_image", [False, True], ids=["t2v", "ti2v"])
def test_magi_human_sr1080p_pipeline_latent_parity(use_image: bool):
    repo_root = Path(__file__).resolve().parents[3]
    if not (repo_root / "daVinci-MagiHuman").exists():
        pytest.skip("Upstream daVinci-MagiHuman/ clone missing.")

    base_shard_dir = _find_base_shard_dir()
    sr_shard_dir = _find_sr1080p_shard_dir()
    if base_shard_dir is None or not base_shard_dir.is_dir():
        pytest.skip("GAIR/daVinci-MagiHuman base/ shards not available locally.")
    if sr_shard_dir is None or not sr_shard_dir.is_dir():
        pytest.skip("GAIR/daVinci-MagiHuman 1080p_sr/ shards not available locally.")

    converted_dir = Path(os.getenv(
        "MAGI_HUMAN_SR1080P_DIFFUSERS_PATH",
        repo_root / "converted_weights" / "magi_human_sr_1080p",
    ))
    transformer_dir = converted_dir / "transformer"
    sr_transformer_dir = converted_dir / "sr_transformer"
    if not transformer_dir.is_dir():
        pytest.skip(f"Converted base transformer dir missing at {transformer_dir}")
    if not sr_transformer_dir.is_dir():
        pytest.skip(f"Converted SR transformer dir missing at {sr_transformer_dir}")

    from tests.local_tests.helpers.magi_human_upstream import (
        install_stubs,
        load_upstream_dit,
    )
    install_stubs()

    device = torch.device("cuda:0")
    torch.manual_seed(1080)
    z_dim = 48
    patch_size = (1, 2, 2)
    base_lat_T, base_lat_H, base_lat_W = 24, 4, 4
    sr_lat_H, sr_lat_W = 6, 8
    video_latent = torch.randn(
        (1, z_dim, base_lat_T, base_lat_H, base_lat_W),
        dtype=torch.float32,
        device=device,
    )
    audio_latent = torch.randn((1, 32, 64), dtype=torch.float32, device=device)
    base_image_latent = None
    sr_image_latent = None
    if use_image:
        base_image_latent = torch.randn(
            (1, z_dim, 1, base_lat_H, base_lat_W),
            dtype=torch.float32,
            device=device,
        )
        sr_image_latent = F.interpolate(
            base_image_latent,
            size=(1, sr_lat_H, sr_lat_W),
            mode="trilinear",
            align_corners=True,
        )

    txt_feat_len = 7
    neg_txt_feat_len = 11
    txt_feat = torch.randn((1, 640, 3584), dtype=torch.float32, device=device)
    neg_txt_feat = torch.randn((1, 640, 3584), dtype=torch.float32, device=device)
    base_steps = 4
    sr_steps = 2
    shift = 5.0
    base_kwargs = dict(
        cfg_number=2,
        video_txt_guidance_scale=5.0,
        audio_txt_guidance_scale=5.0,
        patch_size=patch_size,
        coords_style="v2",
        video_in_channels=192,
        audio_in_channels=64,
        image_latent=base_image_latent,
    )
    sr_kwargs = dict(
        patch_size=patch_size,
        coords_style="v1",
        video_in_channels=192,
        audio_in_channels=64,
        image_latent=sr_image_latent,
    )

    up_video_sched, up_audio_sched = _build_upstream_schedulers(
        shift=shift,
        num_inference_steps=base_steps,
        device=device,
    )
    upstream_base = load_upstream_dit(base_shard_dir, device=device, dtype=None)
    ref_base_video, ref_base_audio = _run_denoise_loop(
        upstream_base,
        _dit_forward_upstream,
        video_latent.clone(),
        audio_latent.clone(),
        txt_feat.clone(),
        txt_feat_len,
        neg_txt_feat.clone(),
        neg_txt_feat_len,
        video_sched=up_video_sched,
        audio_sched=up_audio_sched,
        **base_kwargs,
    )
    del upstream_base
    _cleanup_gpu()

    torch.manual_seed(1081)
    ref_sr_video_in, ref_sr_audio_in = _prepare_sr_latents(
        ref_base_video,
        ref_base_audio,
        latent_h=sr_lat_H,
        latent_w=sr_lat_W,
        noise_value=220,
    )
    up_sr_video_sched, _ = _build_upstream_schedulers(
        shift=shift,
        num_inference_steps=sr_steps,
        device=device,
    )
    upstream_sr = load_upstream_dit(
        sr_shard_dir,
        device=device,
        dtype=None,
        local_attn_layers=_SR_1080P_LOCAL_ATTN_LAYERS,
    )
    ref_video, ref_audio = _run_sr_denoise_loop(
        upstream_sr,
        _dit_forward_upstream_local,
        ref_sr_video_in.clone(),
        ref_sr_audio_in.clone(),
        txt_feat.clone(),
        txt_feat_len,
        neg_txt_feat.clone(),
        neg_txt_feat_len,
        video_sched=up_sr_video_sched,
        **sr_kwargs,
    )
    ref_video = ref_video.detach().float().cpu()
    ref_audio = ref_audio.detach().float().cpu()
    del upstream_sr
    _cleanup_gpu()

    fv_video_sched, fv_audio_sched = _build_fastvideo_schedulers(
        shift=shift,
        num_inference_steps=base_steps,
        device=device,
    )
    fv_base = _load_fv_dit(transformer_dir, device)
    fv_base_video, fv_base_audio = _run_denoise_loop(
        fv_base,
        _dit_forward_fv,
        video_latent.clone(),
        audio_latent.clone(),
        txt_feat.clone(),
        txt_feat_len,
        neg_txt_feat.clone(),
        neg_txt_feat_len,
        video_sched=fv_video_sched,
        audio_sched=fv_audio_sched,
        **base_kwargs,
    )
    del fv_base
    _cleanup_gpu()

    torch.manual_seed(1081)
    fv_sr_video_in, fv_sr_audio_in = _prepare_sr_latents(
        fv_base_video,
        fv_base_audio,
        latent_h=sr_lat_H,
        latent_w=sr_lat_W,
        noise_value=220,
    )
    fv_sr_video_sched, _ = _build_fastvideo_schedulers(
        shift=shift,
        num_inference_steps=sr_steps,
        device=device,
    )
    fv_sr = _load_fv_dit(sr_transformer_dir, device)
    fv_sr.configure_local_attention(_SR_1080P_LOCAL_ATTN_LAYERS, frame_receptive_field=11)
    fv_video, fv_audio = _run_sr_denoise_loop(
        fv_sr,
        _dit_forward_fv,
        fv_sr_video_in.clone(),
        fv_sr_audio_in.clone(),
        txt_feat.clone(),
        txt_feat_len,
        neg_txt_feat.clone(),
        neg_txt_feat_len,
        video_sched=fv_sr_video_sched,
        **sr_kwargs,
    )
    fv_video = fv_video.detach().float().cpu()
    fv_audio = fv_audio.detach().float().cpu()

    v_diff = (ref_video - fv_video).abs()
    a_diff = (ref_audio - fv_audio).abs()
    print(
        f"sr1080p {('ti2v' if use_image else 't2v')} "
        f"video diff_max={v_diff.max().item():.4f} diff_mean={v_diff.mean().item():.4f}"
    )
    print(
        f"sr1080p {('ti2v' if use_image else 't2v')} "
        f"audio diff_max={a_diff.max().item():.4f} diff_mean={a_diff.mean().item():.4f}"
    )

    assert ref_video.shape == fv_video.shape
    assert ref_audio.shape == fv_audio.shape
    assert v_diff.max().item() < 0.05
    assert_close(fv_audio, ref_audio, atol=0.0, rtol=0.0)
    if use_image:
        assert_close(fv_video[:, :, :1], sr_image_latent.detach().cpu(), atol=0.0, rtol=0.0)
        assert_close(ref_video[:, :, :1], sr_image_latent.detach().cpu(), atol=0.0, rtol=0.0)
