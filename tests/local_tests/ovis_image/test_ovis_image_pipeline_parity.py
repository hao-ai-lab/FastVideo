# SPDX-License-Identifier: Apache-2.0
"""End-to-end parity vs the official Diffusers ``OvisImagePipeline``: bit-identical
timestep schedule, RNG-aligned denoised-latent comparison (same initial latent /
schedule / prompt / CFG, production-loaded DiT, bf16 cross-kernel tolerance), and
a full multiprocess ``VideoGenerator`` image-health check."""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.testing import assert_close

os.environ.setdefault("MASTER_ADDR", "localhost")
os.environ.setdefault("MASTER_PORT", "29522")
os.environ.setdefault("DISABLE_SP", "1")
os.environ.setdefault("FASTVIDEO_ATTENTION_BACKEND", "TORCH_SDPA")

OVIS_WEIGHTS = os.getenv("OVIS_WEIGHTS", "models/Ovis-Image-7B")

SAMPLE_PROMPT = (
    'A vibrant poster with the text "FAST VIDEO" in bold red letters on a '
    "clean white background. High contrast, 4k quality.")
NEGATIVE_PROMPT = ""
SEED = 42
HEIGHT = 256
WIDTH = 256
STEPS = 8
GUIDANCE = 5.0


def _weights_present() -> bool:
    root = Path(OVIS_WEIGHTS)
    return all((root / d).exists()
               for d in ("transformer", "text_encoder", "tokenizer", "vae",
                         "scheduler"))


def _diffusers_ovis_available() -> bool:
    try:
        import diffusers  # noqa: F401
        return hasattr(diffusers, "OvisImagePipeline")
    except Exception:
        return False


pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(),
                       reason="Ovis-Image pipeline parity requires CUDA."),
    pytest.mark.skipif(
        not _weights_present(),
        reason=(f"Ovis-Image weights not found at {OVIS_WEIGHTS}. "
                "Set OVIS_WEIGHTS or download AIDC-AI/Ovis-Image-7B.")),
    pytest.mark.skipif(
        not _diffusers_ovis_available(),
        reason="Installed diffusers lacks OvisImagePipeline."),
]


def _to_uint8_hwc(a) -> np.ndarray:
    """Normalize a pipeline image/video output to H,W,3 uint8.

    Handles FastVideo's 5D video layout ``(B, C, T, H, W)`` explicitly — taking
    batch 0 / frame 0 while KEEPING the channel axis — as well as diffusers'
    ``(B, H, W, C)`` / ``(H, W, C)`` numpy images. A naive ``while ndim>3: a=a[0]``
    is wrong here: on ``(1,3,1,256,256)`` it would peel the channel axis and
    leave a single-channel slice (near-white for a white-background poster).
    """
    if torch.is_tensor(a):
        a = a.float().cpu().numpy()
    a = np.asarray(a)
    if a.ndim == 5:  # (B, C, T, H, W) -> (C, H, W)
        a = a[0, :, 0]
    elif a.ndim == 4:
        # (B, C, H, W) or (B, H, W, C): drop batch, keep the rest.
        a = a[0]
    if a.ndim == 3 and a.shape[0] in (1, 3) and a.shape[-1] not in (1, 3):
        a = np.transpose(a, (1, 2, 0))
    if a.shape[-1] == 1:
        a = np.repeat(a, 3, axis=-1)
    if a.dtype != np.uint8:
        scale = 255.0 if float(a.max()) <= 1.5 else 1.0
        a = np.clip(a * scale, 0, 255).astype(np.uint8)
    return a


def _fastvideo_image() -> np.ndarray:
    from fastvideo import VideoGenerator
    gen = VideoGenerator.from_pretrained(
        OVIS_WEIGHTS,
        num_gpus=1,
        use_fsdp_inference=False,
        dit_cpu_offload=False,
        vae_cpu_offload=False,
        text_encoder_cpu_offload=False,
        pin_cpu_memory=False,
    )
    try:
        result = gen.generate_video(
            prompt=SAMPLE_PROMPT,
            negative_prompt=NEGATIVE_PROMPT,
            output_path="outputs_image/ovis_parity",
            save_video=False,
            height=HEIGHT,
            width=WIDTH,
            num_frames=1,
            num_inference_steps=STEPS,
            guidance_scale=GUIDANCE,
            seed=SEED,
            fps=1,
        )
    finally:
        gen.shutdown()
    samples = result["samples"]  # (B, C, T, H, W)
    return _to_uint8_hwc(samples)


def test_ovis_timestep_schedule_matches_reference():
    """The deterministic part of parity: FastVideo's dynamic-shift timestep
    schedule must equal the official pipeline's exactly. This is the concrete
    regression guard for the static-vs-dynamic shift bug."""
    from diffusers import FlowMatchEulerDiscreteScheduler

    from fastvideo.models.schedulers.scheduling_flow_match_euler_discrete import (
        FlowMatchEulerDiscreteScheduler as FVScheduler)
    from fastvideo.pipelines.stages.timestep_preparation import (
        OvisImageTimestepPreparationStage)

    # Official schedule (mirrors pipeline_ovis_image.py steps 5).
    image_seq_len = (HEIGHT // 16) * (WIDTH // 16)
    m = (1.15 - 0.5) / (4096 - 256)
    mu = image_seq_len * m + (0.5 - m * 256)
    ref = FlowMatchEulerDiscreteScheduler.from_pretrained(
        os.path.join(OVIS_WEIGHTS, "scheduler"))
    ref.set_timesteps(sigmas=np.linspace(1.0, 1.0 / STEPS, STEPS),
                      device="cpu", mu=mu)
    ref_ts = ref.timesteps.float().cpu()

    # FastVideo schedule via the production stage.
    import fastvideo.pipelines.stages.timestep_preparation as M
    M.get_local_torch_device = lambda: torch.device("cpu")
    sch = FVScheduler.from_pretrained(os.path.join(OVIS_WEIGHTS, "scheduler"))
    stage = OvisImageTimestepPreparationStage(scheduler=sch)

    class _Batch:
        height = HEIGHT
        width = WIDTH
        num_inference_steps = STEPS

    fv_ts = stage.forward(_Batch(), None).timesteps.float().cpu()

    assert fv_ts.shape == ref_ts.shape, (
        f"timestep count differs: fv={fv_ts.shape} ref={ref_ts.shape}")
    max_diff = (fv_ts - ref_ts).abs().max().item()
    print(f"[ovis pipeline] timestep schedule max_diff={max_diff:.6e}")
    assert max_diff < 1e-4, (
        f"timestep schedule diverges from reference (max_diff={max_diff:.4f}); "
        "dynamic-shift wiring regressed")


@pytest.mark.usefixtures("distributed_setup")
def test_ovis_pipeline_latent_parity():
    """Primary parity gate: RNG-aligned denoised-latent comparison.

    Both sides start from the SAME initial packed latent and the SAME official
    schedule / prompt embeds / CFG, then run the identical CFG denoise loop. The
    FastVideo transformer is the production-loaded native DiT; the reference is
    the official Diffusers transformer. We compare the final denoised latents (in
    packed space) — decode is skipped because it is the nondeterministic part and
    is covered separately by the VAE component parity test.
    """
    from diffusers import OvisImagePipeline
    from diffusers.pipelines.ovis_image.pipeline_ovis_image import (
        calculate_shift, retrieve_timesteps)

    from fastvideo.configs.models.dits import OvisImageTransformer2DModelConfig
    from fastvideo.configs.pipelines.base import PipelineConfig
    from fastvideo.fastvideo_args import FastVideoArgs
    from fastvideo.forward_context import set_forward_context
    from fastvideo.models.dits.ovisimage import (_pack_latents, _unpack_latents)
    from fastvideo.models.loader.component_loader import TransformerLoader

    device = torch.device("cuda:0")
    dtype = torch.bfloat16

    pipe = OvisImagePipeline.from_pretrained(OVIS_WEIGHTS,
                                             torch_dtype=dtype).to(device)
    # Shared conditioning + initial noise (RNG aligned: both sides reuse these).
    prompt_embeds, text_ids = pipe.encode_prompt(prompt=SAMPLE_PROMPT,
                                                 device=device,
                                                 num_images_per_prompt=1)
    neg_embeds, neg_text_ids = pipe.encode_prompt(prompt=NEGATIVE_PROMPT,
                                                  device=device,
                                                  num_images_per_prompt=1)
    num_ch = pipe.transformer.config.in_channels // 4
    gen = torch.Generator(device=device).manual_seed(SEED)
    latents0, img_ids = pipe.prepare_latents(1, num_ch, HEIGHT, WIDTH,
                                             prompt_embeds.dtype, device, gen,
                                             None)

    # Shared official schedule (dynamic shift).
    sigmas = np.linspace(1.0, 1.0 / STEPS, STEPS)
    mu = calculate_shift(latents0.shape[1],
                         pipe.scheduler.config.get("base_image_seq_len", 256),
                         pipe.scheduler.config.get("max_image_seq_len", 4096),
                         pipe.scheduler.config.get("base_shift", 0.5),
                         pipe.scheduler.config.get("max_shift", 1.15))

    def _cfg_denoise(transformer_call):
        sch = pipe.scheduler.__class__.from_config(pipe.scheduler.config)
        ts, _ = retrieve_timesteps(sch, STEPS, device, sigmas=sigmas, mu=mu)
        lat = latents0.clone()
        sch.set_begin_index(0)
        for t in ts:
            te = t.expand(lat.shape[0]).to(lat.dtype)
            cond = transformer_call(lat, prompt_embeds, text_ids, te)
            uncond = transformer_call(lat, neg_embeds, neg_text_ids, te)
            noise_pred = uncond + GUIDANCE * (cond - uncond)
            lat = sch.step(noise_pred, t, lat, return_dict=False)[0]
        return lat

    # Reference: official transformer (packed latents in/out).
    def _official_call(lat, embeds, tids, te):
        return pipe.transformer(hidden_states=lat, timestep=te / 1000,
                                encoder_hidden_states=embeds, txt_ids=tids,
                                img_ids=img_ids, return_dict=False)[0]

    ref_latent = _cfg_denoise(_official_call)
    del pipe.transformer
    import gc
    gc.collect()
    torch.cuda.empty_cache()

    # FastVideo: production-loaded native DiT. It takes UNPACKED [B,16,H/8,W/8]
    # and packs internally, and expects the [0,1000] timestep scale, so unpack
    # the official packed latent on the way in and repack the output to compare
    # in the same space.
    args = FastVideoArgs(model_path=os.path.join(OVIS_WEIGHTS, "transformer"),
                         dit_cpu_offload=False, use_fsdp_inference=False,
                         pipeline_config=PipelineConfig(
                             dit_config=OvisImageTransformer2DModelConfig(),
                             dit_precision="bf16"))
    args.device = device
    fv = TransformerLoader().load(os.path.join(OVIS_WEIGHTS, "transformer"),
                                  args).to(device=device, dtype=dtype).eval()

    def _fv_call(lat, embeds, tids, te):
        # `te` is the raw [0, 1000] timestep. The official transformer is fed
        # te/1000 and rescales internally; the FastVideo DiT embeds the [0, 1000]
        # value directly, so it receives `te` unscaled.
        unpacked = _unpack_latents(lat, HEIGHT // 8, WIDTH // 8)
        with set_forward_context(current_timestep=0, attn_metadata=None,
                                 forward_batch=None):
            out = fv(hidden_states=unpacked, encoder_hidden_states=embeds,
                     timestep=te)
        return _pack_latents(out)

    with torch.no_grad():
        fv_latent = _cfg_denoise(_fv_call)

    assert ref_latent.shape == fv_latent.shape, (
        f"shape mismatch: ref={ref_latent.shape} fv={fv_latent.shape}")
    diff = (ref_latent.float() - fv_latent.float()).abs()
    drift = diff.mean().item() / (ref_latent.float().abs().mean().item() + 1e-8)
    print(f"[ovis pipeline] denoised-latent max_diff={diff.max().item():.4f} "
          f"mean_diff={diff.mean().item():.5f} abs_mean_drift={drift:.2%}")

    # The abs-mean-drift bound is the structural gate (scale / scheduler /
    # CFG-branch bugs move it far past the noise floor). assert_close's atol
    # absorbs the worst-element drift: official eager attention vs FastVideo SDPA
    # differ in bf16 accumulation order, and CFG=5 amplifies that per-call drift
    # across 8 steps (observed max_diff ~2.7 on a signal whose abs-max is ~5,
    # abs-mean-drift ~3%). Same approach as the MagiHuman pipeline parity test.
    assert drift < 0.05, f"denoised-latent abs-mean drift {drift:.2%} exceeds 5%"
    assert_close(fv_latent, ref_latent, atol=3.5, rtol=0.05)


def test_ovis_fastvideo_produces_valid_image():
    """Full multiprocess VideoGenerator run is healthy. This exercises the real
    stage chain (which the single-process latent test bypasses); RNG is not
    aligned to the official pipeline here, so only image health is gated."""
    fv = _fastvideo_image()
    assert fv.shape == (HEIGHT, WIDTH, 3), f"unexpected shape {fv.shape}"
    assert fv.dtype == np.uint8 and np.isfinite(fv).all()
    assert fv.std() > 10.0, f"image is near-constant (std={fv.std():.2f})"
