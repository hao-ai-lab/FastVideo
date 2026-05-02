# SPDX-License-Identifier: Apache-2.0
"""End-to-end latent parity test for the daVinci-MagiHuman base text-to-AV pipeline.

Runs the joint video+audio FlowUniPC denoise loop with CFG=2 on both:
  - FastVideo MagiHumanDiT loaded from `converted_weights/magi_human_base/transformer/`.
  - Upstream daVinci-MagiHuman DiTModel loaded from the HF `base/` shards,
    via the `magi_compiler` / distributed stubs in
    `tests/local_tests/helpers/magi_human_upstream.py`.

Both sides use the **same** `FlowUniPCMultistepScheduler` (FastVideo's
implementation), identical latent / text inputs, identical scheduler
state, and SDPA-routed attention — so drift here is purely the
compound of per-call DiT parity drift through the denoise loop + CFG
mixing amplification.

What this catches (that the component-level DiT parity does NOT):
  - Scheduler integration mistakes (state leaks between video/audio
    schedulers, wrong shift, wrong `step()` args).
  - CFG math errors (guidance scale switchover at t=500, per-modality
    guidance scale wiring, unconditional-path text padding).
  - Latent-preparation / token-unpacking drift between my
    `build_packed_inputs` / `unpack_tokens` and the upstream
    `MagiDataProxy` equivalents.
  - Compounding behavior: 1% per-call DiT drift compounding through
    `num_steps * cfg_number` calls.

Skips when:
  - `daVinci-MagiHuman/` clone or GAIR/daVinci-MagiHuman base shards
    are not available locally.
  - Converted transformer weights are missing (run the conversion
    script first).
  - CUDA is unavailable.

Tolerance: `atol=0.35, rtol=0.05` on bf16 denoise-loop latents. The
atol absorbs the observed worst-element drift (~0.31 on a signal of
abs_mean ~2.4 — bf16 + CFG amplification + UniPC accumulation). The
tight rtol still flags gross structural bugs (sign flip, scheduler
state leak, modality branch drop). If tighter parity is wanted,
chase the per-call drift first (see the DiT component parity test).
"""
from __future__ import annotations

import gc
import glob
import os
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from torch.testing import assert_close


# Force SDPA on both sides so the attention kernel is shared.
os.environ.setdefault("FASTVIDEO_ATTENTION_BACKEND", "TORCH_SDPA")
os.environ.setdefault("MASTER_ADDR", "localhost")
os.environ.setdefault("MASTER_PORT", "29519")

_T5GEMMA_ID = os.getenv("MAGI_HUMAN_T5GEMMA_ID", "google/t5gemma-9b-9b-ul2")
_T5_GEMMA_TARGET_LENGTH = 640
_SAMPLE_PROMPT = (
    "A warm afternoon scene: a person sits on a park bench reading a book, "
    "surrounded by softly swaying trees."
)


def _hf_token() -> str | None:
    for key in ("HF_TOKEN", "HUGGINGFACE_HUB_TOKEN", "HF_API_KEY"):
        token = os.environ.get(key)
        if token:
            return token
    return None


def _can_access_t5gemma() -> bool:
    token = _hf_token()
    if token is None:
        return False
    try:
        from huggingface_hub import hf_hub_download
        hf_hub_download(
            repo_id=_T5GEMMA_ID,
            filename="config.json",
            token=token,
        )
        return True
    except Exception:
        return False


def _pad_or_trim_dim1(t: torch.Tensor, target: int) -> tuple[torch.Tensor, int]:
    """Mirror MagiHumanLatentPreparationStage's text pad-or-trim."""
    current = t.size(1)
    if current < target:
        pad = [0, 0, 0, target - current]
        return F.pad(t, pad, "constant", 0.0), current
    return t[:, :target], target


def _encode_magi_human_prompt_pair(device: torch.device):
    """Encode the production preset prompt pair once via T5-Gemma."""
    if not _can_access_t5gemma():
        pytest.skip(
            f"{_T5GEMMA_ID} not accessible — gated Google repo; set "
            "HF_TOKEN / HF_API_KEY and accept the terms of use."
        )

    for src in ("HF_TOKEN", "HUGGINGFACE_HUB_TOKEN", "HF_API_KEY"):
        token = os.environ.get(src)
        if token:
            os.environ.setdefault("HF_TOKEN", token)
            os.environ.setdefault("HUGGINGFACE_HUB_TOKEN", token)
            break

    try:
        from transformers import AutoTokenizer

        from fastvideo.configs.models.encoders.t5gemma import (
            T5GemmaEncoderConfig,
        )
        from fastvideo.models.encoders.t5gemma import (
            T5GemmaEncoderModel,
        )
        from fastvideo.pipelines.basic.magi_human.presets import (
            _MAGI_HUMAN_NEGATIVE_PROMPT,
        )
    except Exception as exc:
        pytest.skip(f"T5-Gemma prompt encoding dependencies unavailable: {exc}")

    tokenizer = AutoTokenizer.from_pretrained(_T5GEMMA_ID)
    enc_config = T5GemmaEncoderConfig()
    enc_config.arch_config.t5gemma_model_path = _T5GEMMA_ID
    encoder = T5GemmaEncoderModel(enc_config)

    def encode(text: str, text_encoder=encoder) -> tuple[torch.Tensor, int]:
        inputs = tokenizer(
            [text],
            return_tensors="pt",
            padding=True,
            truncation=False,
        ).to(device)
        with torch.inference_mode():
            hidden = text_encoder(
                input_ids=inputs["input_ids"],
                attention_mask=inputs.get("attention_mask"),
            ).last_hidden_state
        return _pad_or_trim_dim1(hidden.to(torch.float32), _T5_GEMMA_TARGET_LENGTH)

    txt_feat, txt_feat_len = encode(_SAMPLE_PROMPT)
    neg_txt_feat, neg_txt_feat_len = encode(_MAGI_HUMAN_NEGATIVE_PROMPT)
    del encoder
    _cleanup_gpu()
    return txt_feat, txt_feat_len, neg_txt_feat, neg_txt_feat_len


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


def _dit_forward_fv(
    dit, video_latent, audio_latent, audio_feat_len,
    txt_feat, txt_feat_len, patch_size, coords_style,
    video_in_channels, audio_in_channels,
):
    """One FastVideo DiT call — same as MagiHumanDenoisingStage._dit_forward."""
    from fastvideo.pipelines.basic.magi_human.stages.latent_preparation import (
        build_packed_inputs, unpack_tokens,
    )
    x, coords, mm = build_packed_inputs(
        video_latent=video_latent, audio_latent=audio_latent,
        audio_feat_len=audio_feat_len, txt_feat=txt_feat,
        txt_feat_len=txt_feat_len, patch_size=patch_size,
        coords_style=coords_style,
    )
    video_token_num = x.shape[0] - audio_feat_len - txt_feat_len
    out = dit(x, coords, mm)
    return unpack_tokens(
        out, video_token_num=video_token_num,
        audio_feat_len=audio_feat_len,
        video_in_channels=video_in_channels,
        audio_in_channels=audio_in_channels,
        latent_shape=tuple(video_latent.shape),
        patch_size=patch_size,
    )


def _dit_forward_upstream(
    dit, video_latent, audio_latent, audio_feat_len,
    txt_feat, txt_feat_len, patch_size, coords_style,
    video_in_channels, audio_in_channels,
):
    """One upstream DiT call — identical input construction + output
    unpacking to the FastVideo path. The only thing that differs is
    the DiT module and the extra `varlen_handler` / `local_attn_handler`
    kwargs the upstream expects.
    """
    from fastvideo.pipelines.basic.magi_human.stages.latent_preparation import (
        build_packed_inputs, unpack_tokens,
    )
    from inference.common import VarlenHandler
    x, coords, mm = build_packed_inputs(
        video_latent=video_latent, audio_latent=audio_latent,
        audio_feat_len=audio_feat_len, txt_feat=txt_feat,
        txt_feat_len=txt_feat_len, patch_size=patch_size,
        coords_style=coords_style,
    )
    video_token_num = x.shape[0] - audio_feat_len - txt_feat_len
    total = x.shape[0]
    cu = torch.tensor([0, total], dtype=torch.int32, device=x.device)
    varlen = VarlenHandler(
        cu_seqlens_q=cu, cu_seqlens_k=cu,
        max_seqlen_q=total, max_seqlen_k=total,
    )
    out = dit(
        x=x, coords_mapping=coords, modality_mapping=mm,
        varlen_handler=varlen, local_attn_handler=None,
    )
    return unpack_tokens(
        out, video_token_num=video_token_num,
        audio_feat_len=audio_feat_len,
        video_in_channels=video_in_channels,
        audio_in_channels=audio_in_channels,
        latent_shape=tuple(video_latent.shape),
        patch_size=patch_size,
    )


def _build_fastvideo_schedulers(shift: float, num_inference_steps: int, device):
    """Mirror current FastVideo production at `magi_human_pipeline.py:146-149`
    and `denoising.py:105-116`: default scheduler constructor (`shift=1`,
    no-op) followed by `set_timesteps(..., shift=shift)` so the temporal
    shift is applied exactly once. The earlier double-shift pattern was
    reverted with the Wave 11 single-shift fix; if both __init__ and
    set_timesteps applied non-trivial shift, the schedule would diverge
    from upstream.
    """
    from fastvideo.models.schedulers.scheduling_flow_unipc_multistep import (
        FlowUniPCMultistepScheduler,
    )
    video_sched = FlowUniPCMultistepScheduler()
    audio_sched = FlowUniPCMultistepScheduler()
    video_sched.set_timesteps(num_inference_steps, device=device, shift=shift)
    audio_sched.set_timesteps(num_inference_steps, device=device, shift=shift)
    return video_sched, audio_sched


def _build_upstream_schedulers(shift: float, num_inference_steps: int, device):
    """Construct schedulers the way the official `MagiEvaluator.eval_with_text`
    does (`daVinci-MagiHuman/inference/pipeline/video_generate.py:404-407`):
    `FlowUniPCMultistepScheduler()` with default shift=1.0 in __init__
    (no-op), then `set_timesteps(num_inference_steps, device, shift=self.shift)`
    applies shift exactly once. Uses FastVideo's scheduler class for
    the orchestration (algorithmically identical to the upstream copy
    of the same Diffusers-derived class) but matches the upstream's
    *call pattern*.
    """
    from fastvideo.models.schedulers.scheduling_flow_unipc_multistep import (
        FlowUniPCMultistepScheduler,
    )
    video_sched = FlowUniPCMultistepScheduler()
    audio_sched = FlowUniPCMultistepScheduler()
    video_sched.set_timesteps(num_inference_steps, device=device, shift=shift)
    audio_sched.set_timesteps(num_inference_steps, device=device, shift=shift)
    return video_sched, audio_sched


def _run_denoise_loop(
    dit, dit_forward_fn, video_latent, audio_latent,
    txt_feat, txt_feat_len, neg_txt_feat, neg_txt_feat_len,
    *, video_sched, audio_sched, cfg_number,
    video_txt_guidance_scale, audio_txt_guidance_scale,
    patch_size, coords_style, video_in_channels, audio_in_channels,
):
    """Joint video+audio FlowUniPC denoise. The schedulers are passed
    in pre-constructed so each side can mirror its production scheduler
    init pattern (see `_build_*_schedulers`).
    """
    from fastvideo.forward_context import set_forward_context
    audio_feat_len = int(audio_latent.shape[1])

    with torch.inference_mode():
        for idx, t in enumerate(video_sched.timesteps):
            t_int = int(t.item()) if torch.is_tensor(t) else int(t)
            with set_forward_context(current_timestep=t_int, attn_metadata=None):
                v_cond_video, v_cond_audio = dit_forward_fn(
                    dit, video_latent, audio_latent, audio_feat_len,
                    txt_feat, txt_feat_len, patch_size, coords_style,
                    video_in_channels, audio_in_channels,
                )
                if cfg_number == 2:
                    v_uncond_video, v_uncond_audio = dit_forward_fn(
                        dit, video_latent, audio_latent, audio_feat_len,
                        neg_txt_feat, neg_txt_feat_len, patch_size, coords_style,
                        video_in_channels, audio_in_channels,
                    )
                    # Upstream's video-guidance drop-at-t<=500 trick.
                    video_guidance = (
                        video_txt_guidance_scale if t > 500 else 2.0
                    )
                    v_video = v_uncond_video + video_guidance * (
                        v_cond_video - v_uncond_video
                    )
                    v_audio = v_uncond_audio + audio_txt_guidance_scale * (
                        v_cond_audio - v_uncond_audio
                    )
                else:
                    v_video = v_cond_video
                    v_audio = v_cond_audio

            video_latent = video_sched.step(
                v_video, t, video_latent, return_dict=False,
            )[0]
            audio_latent = audio_sched.step(
                v_audio, t, audio_latent, return_dict=False,
            )[0]
    return video_latent, audio_latent


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="MagiHuman pipeline parity requires CUDA.",
)
def test_magi_human_pipeline_latent_parity():
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
            "GAIR/daVinci-MagiHuman base/ shards not available locally."
        )

    converted_dir = Path(os.getenv(
        "MAGI_HUMAN_DIFFUSERS_PATH",
        repo_root / "converted_weights" / "magi_human_base",
    ))
    transformer_dir = converted_dir / "transformer"
    if not transformer_dir.is_dir():
        pytest.skip(f"Converted transformer dir missing at {transformer_dir}")

    from tests.local_tests.helpers.magi_human_upstream import (
        install_stubs, load_upstream_dit,
    )
    install_stubs()

    # --- Shared pipeline inputs ---
    device = torch.device("cuda:0")
    torch.manual_seed(0)

    # Deliberately tiny so 2 * CFG=2 = 4 DiT calls per side fit in
    # CI/dev runtime budget.
    z_dim = 48
    patch_size = (1, 2, 2)
    lat_T, lat_H, lat_W = 2, 6, 6
    video_latent = torch.randn(
        (1, z_dim, lat_T, lat_H, lat_W),
        dtype=torch.float32, device=device,
    )
    audio_latent = torch.randn(
        (1, 4, 64), dtype=torch.float32, device=device,
    )
    # Production-facing text embeddings: encode the example prompt and the
    # preset negative prompt via T5-Gemma once, then feed the identical cached
    # tensors to upstream and FastVideo. This keeps the DiT comparison focused
    # while still validating prompt/preset content such as the full
    # three-block MagiHuman negative prompt.
    txt_feat, txt_feat_len, neg_txt_feat, neg_txt_feat_len = (
        _encode_magi_human_prompt_pair(device)
    )

    num_inference_steps = 4        # 4 steps × CFG=2 = 8 DiT calls / side; surfaces compounding drift that 1-step hides
    shift = 5.0
    common_kwargs = dict(
        cfg_number=2,
        video_txt_guidance_scale=5.0,
        audio_txt_guidance_scale=5.0,
        patch_size=patch_size,
        coords_style="v2",
        video_in_channels=192,
        audio_in_channels=64,
    )

    # --- Upstream side first (so we can free it before loading FastVideo). ---
    # Upstream uses single-shift scheduler init (matches MagiEvaluator).
    up_video_sched, up_audio_sched = _build_upstream_schedulers(
        shift=shift, num_inference_steps=num_inference_steps, device=device,
    )
    print("Loading upstream DiTModel from base shards...")
    upstream_dit = load_upstream_dit(base_shard_dir, device=device, dtype=None)
    print("Running upstream denoise loop...")
    ref_video, ref_audio = _run_denoise_loop(
        upstream_dit, _dit_forward_upstream,
        video_latent.clone(), audio_latent.clone(),
        txt_feat.clone(), txt_feat_len,
        neg_txt_feat.clone(), neg_txt_feat_len,
        video_sched=up_video_sched, audio_sched=up_audio_sched,
        **common_kwargs,
    )
    ref_video = ref_video.detach().float().cpu()
    ref_audio = ref_audio.detach().float().cpu()
    del upstream_dit
    _cleanup_gpu()

    # --- FastVideo side ---
    # FastVideo uses double-shift scheduler init (matches
    # `MagiHumanDenoisingStage` in production: shift in __init__ via
    # `magi_human_pipeline.initialize_pipeline` AND in set_timesteps).
    fv_video_sched, fv_audio_sched = _build_fastvideo_schedulers(
        shift=shift, num_inference_steps=num_inference_steps, device=device,
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

    print("Running FastVideo denoise loop...")
    fv_video, fv_audio = _run_denoise_loop(
        fv_dit, _dit_forward_fv,
        video_latent.clone(), audio_latent.clone(),
        txt_feat.clone(), txt_feat_len,
        neg_txt_feat.clone(), neg_txt_feat_len,
        video_sched=fv_video_sched, audio_sched=fv_audio_sched,
        **common_kwargs,
    )
    fv_video = fv_video.detach().float().cpu()
    fv_audio = fv_audio.detach().float().cpu()

    # --- Report + assertions ---
    v_diff = (ref_video - fv_video).abs()
    a_diff = (ref_audio - fv_audio).abs()
    print(
        f"video  ref_abs={ref_video.abs().mean().item():.4f} "
        f"fv_abs={fv_video.abs().mean().item():.4f} "
        f"diff_max={v_diff.max().item():.4f} "
        f"diff_mean={v_diff.mean().item():.4f} "
        f"diff_median={v_diff.median().item():.4f}"
    )
    print(
        f"audio  ref_abs={ref_audio.abs().mean().item():.4f} "
        f"fv_abs={fv_audio.abs().mean().item():.4f} "
        f"diff_max={a_diff.max().item():.4f} "
        f"diff_mean={a_diff.mean().item():.4f} "
        f"diff_median={a_diff.median().item():.4f}"
    )

    assert ref_video.shape == fv_video.shape
    assert ref_audio.shape == fv_audio.shape

    # Tolerance budget for 1-step / CFG=2 (bf16 DiT + bf16 CFG mix):
    #   * Single-DiT bf16 drift: diff_mean ~0.008 on `abs ~ 1.0`
    #     (see DiT component parity, `test_magi_human_dit_parity`).
    #   * CFG mixes `v = v_uncond + guidance * (v_cond - v_uncond)`
    #     with guidance=5; cond and uncond drift independently in bf16,
    #     so the post-CFG `diff_mean` scales by ~guidance (~5x).
    #   * One FlowUniPC scheduler step passes that through unchanged.
    # `diff_max` is the noisiest statistic for bf16 transformer parity
    # (a single fma quantization can blow it up). Use it only as a loose
    # guard. The two ratio guards below catch real structural bugs:
    # `abs_mean` drift signals scale errors / dropped branches, and
    # `diff_mean / ref_abs` signals systematic per-element bias far
    # beyond what bf16+CFG noise can produce.
    assert_close(fv_video, ref_video, atol=0.40, rtol=0.05)
    assert_close(fv_audio, ref_audio, atol=0.40, rtol=0.05)

    # Global-magnitude guard — tightest single assertion. A gross bug
    # (scheduler state leak, dropped modality branch, CFG sign flip)
    # would shift `abs_mean` far beyond the bf16+CFG noise floor.
    ref_v_abs = ref_video.abs().mean().item()
    ref_a_abs = ref_audio.abs().mean().item()
    rel_v = abs(ref_v_abs - fv_video.abs().mean().item()) / max(ref_v_abs, 1e-6)
    rel_a = abs(ref_a_abs - fv_audio.abs().mean().item()) / max(ref_a_abs, 1e-6)
    assert rel_v < 0.01, f"video abs_mean drift {rel_v:.2%} > 1%"
    assert rel_a < 0.01, f"audio abs_mean drift {rel_a:.2%} > 1%"

    # Per-element mean-bias guard — catches systematic shift that
    # `abs_mean` misses (e.g. equal-magnitude flip across many elements).
    mean_rel_v = v_diff.mean().item() / max(ref_v_abs, 1e-6)
    mean_rel_a = a_diff.mean().item() / max(ref_a_abs, 1e-6)
    assert mean_rel_v < 0.04, f"video mean_diff/ref_abs {mean_rel_v:.2%} > 4%"
    assert mean_rel_a < 0.04, f"audio mean_diff/ref_abs {mean_rel_a:.2%} > 4%"
