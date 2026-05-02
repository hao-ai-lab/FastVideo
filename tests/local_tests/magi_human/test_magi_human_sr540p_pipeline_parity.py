# SPDX-License-Identifier: Apache-2.0
"""Two-stage SR-540p latent-loop parity for daVinci-MagiHuman."""
from __future__ import annotations

import glob
import os
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from torch.testing import assert_close

from fastvideo.pipelines.basic.magi_human.stages.sr_latent_preparation import (
    ZeroSNRDDPMDiscretization,
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


os.environ.setdefault("FASTVIDEO_ATTENTION_BACKEND", "TORCH_SDPA")
os.environ.setdefault("MASTER_ADDR", "localhost")
os.environ.setdefault("MASTER_PORT", "29521")


def _find_sr540p_shard_dir() -> Path | None:
    override = os.getenv("MAGI_HUMAN_SR540P_SHARD_DIR")
    if override:
        path = Path(override)
        return path if path.is_dir() else None
    try:
        from huggingface_hub import snapshot_download
        snap = snapshot_download(
            repo_id="GAIR/daVinci-MagiHuman",
            allow_patterns=[
                "540p_sr/*.safetensors",
                "540p_sr/model.safetensors.index.json",
            ],
        )
        candidate = Path(snap) / "540p_sr"
        if candidate.is_dir() and any(candidate.glob("*.safetensors")):
            return candidate
        return None
    except Exception:
        return None


def _prepare_sr_latents(
    br_video: torch.Tensor,
    br_audio: torch.Tensor,
    *,
    latent_h: int,
    latent_w: int,
    noise_value: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    latent_video = F.interpolate(
        br_video,
        size=(br_video.shape[2], latent_h, latent_w),
        mode="trilinear",
        align_corners=True,
    )
    if noise_value != 0:
        noise = torch.randn_like(latent_video, device=latent_video.device)
        sigmas = ZeroSNRDDPMDiscretization()(
            1000,
            do_append_zero=False,
            flip=True,
            device=latent_video.device,
        )
        sigma = sigmas[noise_value]
        latent_video = latent_video * sigma + noise * (1 - sigma**2)**0.5
    sr_audio = torch.randn_like(br_audio, device=br_audio.device) * 0.7 + br_audio * 0.3
    return latent_video, sr_audio


def _run_sr_denoise_loop(
    dit,
    dit_forward_fn,
    video_latent,
    audio_latent,
    txt_feat,
    txt_feat_len,
    neg_txt_feat,
    neg_txt_feat_len,
    *,
    video_sched,
    patch_size,
    coords_style,
    video_in_channels,
    audio_in_channels,
    image_latent=None,
):
    from fastvideo.forward_context import set_forward_context

    audio_feat_len = int(audio_latent.shape[1])
    latent_length = video_latent.shape[2]
    guidance = torch.tensor(3.5, device=video_latent.device).expand(
        1,
        1,
        latent_length,
        1,
        1,
    ).clone()
    guidance[:, :, :13] = 2.0

    with torch.inference_mode():
        for t in video_sched.timesteps:
            if image_latent is not None:
                video_latent[:, :, :1] = image_latent.to(
                    device=video_latent.device,
                    dtype=video_latent.dtype,
                )[:, :, :1]
            with set_forward_context(
                current_timestep=int(t.item()) if torch.is_tensor(t) else int(t),
                attn_metadata=None,
            ):
                v_cond_video, _ = dit_forward_fn(
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
                )
                v_uncond_video, _ = dit_forward_fn(
                    dit,
                    video_latent,
                    audio_latent,
                    audio_feat_len,
                    neg_txt_feat,
                    neg_txt_feat_len,
                    patch_size,
                    coords_style,
                    video_in_channels,
                    audio_in_channels,
                )
                v_video = v_uncond_video + guidance * (v_cond_video - v_uncond_video)
            video_latent = video_sched.step(
                v_video,
                t,
                video_latent,
                return_dict=False,
            )[0]
        if image_latent is not None:
            video_latent[:, :, :1] = image_latent.to(
                device=video_latent.device,
                dtype=video_latent.dtype,
            )[:, :, :1]
    return video_latent, audio_latent


def _load_fv_dit(transformer_dir: Path, device: torch.device):
    from fastvideo.configs.models.dits.magi_human import MagiHumanVideoConfig
    from fastvideo.models.dits.magi_human import MagiHumanDiT
    from safetensors.torch import load_file

    dit = MagiHumanDiT(MagiHumanVideoConfig())
    state = {}
    for shard in sorted(glob.glob(str(transformer_dir / "*.safetensors"))):
        state.update(load_file(shard))
    missing, unexpected = dit.load_state_dict(state, strict=False)
    assert not missing, f"FastVideo DiT missing {len(missing)} keys: {missing[:5]}"
    assert not unexpected, f"FastVideo DiT unexpected {len(unexpected)} keys: {unexpected[:5]}"
    return dit.to(device=device).eval()


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="MagiHuman SR-540p pipeline parity requires CUDA.",
)
@pytest.mark.parametrize("use_image", [False, True], ids=["t2v", "ti2v"])
def test_magi_human_sr540p_pipeline_latent_parity(use_image: bool):
    repo_root = Path(__file__).resolve().parents[3]
    if not (repo_root / "daVinci-MagiHuman").exists():
        pytest.skip("Upstream daVinci-MagiHuman/ clone missing.")

    base_shard_dir = _find_base_shard_dir()
    sr_shard_dir = _find_sr540p_shard_dir()
    if base_shard_dir is None or not base_shard_dir.is_dir():
        pytest.skip("GAIR/daVinci-MagiHuman base/ shards not available locally.")
    if sr_shard_dir is None or not sr_shard_dir.is_dir():
        pytest.skip("GAIR/daVinci-MagiHuman 540p_sr/ shards not available locally.")

    converted_dir = Path(os.getenv(
        "MAGI_HUMAN_SR540P_DIFFUSERS_PATH",
        repo_root / "converted_weights" / "magi_human_sr_540p",
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
    torch.manual_seed(540)
    z_dim = 48
    patch_size = (1, 2, 2)
    base_lat_T, base_lat_H, base_lat_W = 2, 6, 6
    sr_lat_H, sr_lat_W = 8, 10
    video_latent = torch.randn(
        (1, z_dim, base_lat_T, base_lat_H, base_lat_W),
        dtype=torch.float32,
        device=device,
    )
    audio_latent = torch.randn((1, 4, 64), dtype=torch.float32, device=device)
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
    txt_feat = torch.randn(
        (1, 640, 3584),
        dtype=torch.float32,
        device=device,
    )
    neg_txt_feat = torch.randn(
        (1, 640, 3584),
        dtype=torch.float32,
        device=device,
    )
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

    torch.manual_seed(541)
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
    upstream_sr = load_upstream_dit(sr_shard_dir, device=device, dtype=None)
    ref_video, ref_audio = _run_sr_denoise_loop(
        upstream_sr,
        _dit_forward_upstream,
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

    torch.manual_seed(541)
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
        f"sr540p {('ti2v' if use_image else 't2v')} "
        f"video diff_max={v_diff.max().item():.4f} diff_mean={v_diff.mean().item():.4f}"
    )
    print(
        f"sr540p {('ti2v' if use_image else 't2v')} "
        f"audio diff_max={a_diff.max().item():.4f} diff_mean={a_diff.mean().item():.4f}"
    )

    assert ref_video.shape == fv_video.shape
    assert ref_audio.shape == fv_audio.shape
    assert_close(fv_audio, ref_audio, atol=0.0, rtol=0.0)
    assert_close(fv_video, ref_video, atol=0.0, rtol=0.0)
    if use_image:
        assert_close(fv_video[:, :, :1], sr_image_latent.detach().cpu(), atol=0.0, rtol=0.0)
        assert_close(ref_video[:, :, :1], sr_image_latent.detach().cpu(), atol=0.0, rtol=0.0)

    ref_v_abs = ref_video.abs().mean().item()
    ref_a_abs = ref_audio.abs().mean().item()
    assert abs(ref_v_abs - fv_video.abs().mean().item()) / max(ref_v_abs, 1e-6) < 0.02
    assert abs(ref_a_abs - fv_audio.abs().mean().item()) / max(ref_a_abs, 1e-6) < 0.02
    assert v_diff.mean().item() / max(ref_v_abs, 1e-6) < 0.06
    assert a_diff.mean().item() / max(ref_a_abs, 1e-6) < 0.04
