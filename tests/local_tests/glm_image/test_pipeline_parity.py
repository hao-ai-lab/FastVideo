# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
import torch

os.environ.setdefault("MASTER_ADDR", "localhost")
os.environ.setdefault("MASTER_PORT", "29519")
os.environ.setdefault("DISABLE_SP", "1")
os.environ.setdefault("FASTVIDEO_ATTENTION_BACKEND", "TORCH_SDPA")

REPO_ROOT = Path(__file__).resolve().parents[3]
FAMILY = "glm_image"
LOCAL_WEIGHTS_DIR = Path(
    os.getenv("GLM_IMAGE_LOCAL_WEIGHTS_DIR",
              REPO_ROOT / "official_weights" / FAMILY))
TRANSFORMER_DIR = LOCAL_WEIGHTS_DIR / "transformer"


def _has_weights() -> bool:
    required = ["transformer", "vae", "text_encoder",
                "vision_language_encoder", "processor", "tokenizer",
                "scheduler"]
    return all((LOCAL_WEIGHTS_DIR / r).exists() for r in required)


def _upstream_glm_image_available() -> bool:
    try:
        import transformers
        import diffusers
    except ImportError:
        return False
    return (hasattr(transformers, "GlmImageForConditionalGeneration")
            and hasattr(diffusers, "GlmImagePipeline"))


pytestmark = [
    pytest.mark.skipif(
        not _has_weights(),
        reason=f"GLM-Image full weights not found at {LOCAL_WEIGHTS_DIR}.",
    ),
    pytest.mark.skipif(
        not _upstream_glm_image_available(),
        reason=("Pipeline parity needs transformers>=5.0.0rc0 and "
                "diffusers>=0.37.0.dev0; main pins predate both. Bump locally "
                "to run."),
    ),
]


SAMPLE_PROMPT = (
    "A landscape photo with rolling green hills under a clear blue sky.")
SEED = 0
HEIGHT = 512
WIDTH = 512
STEPS = 8

# bf16 + a different SDPA kernel across 30 DiT layers x STEPS steps leaves a
# small residual; a real wiring/schedule bug is far larger than these bounds.
LATENT_COSINE_MIN = 0.995
IMAGE_MAE_MAX = 5.0  # /255


@pytest.fixture(scope="module")
def device():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for GLM-Image pipeline parity.")
    return torch.device("cuda")


def _to_uint8_hwc(a) -> np.ndarray:
    if torch.is_tensor(a):
        a = a.detach().float().cpu().numpy()
    a = np.asarray(a)
    while a.ndim > 3:
        a = a[0]
    if a.ndim == 3 and a.shape[0] in (1, 3) and a.shape[-1] not in (1, 3):
        a = np.transpose(a, (1, 2, 0))
    if a.dtype != np.uint8:
        scale = 255.0 if float(a.max()) <= 1.5 else 1.0
        a = np.clip(a * scale, 0, 255).astype(np.uint8)
    return a


# --------------------------------------------------------------------------- #
# FastVideo native DiT loader (mirrors test_transformer_parity.py).
# --------------------------------------------------------------------------- #
def _load_state_dict(dir_path: Path) -> dict[str, torch.Tensor]:
    safetensors = pytest.importorskip("safetensors.torch")
    sd: dict[str, torch.Tensor] = {}
    for shard in sorted(dir_path.glob("*.safetensors")):
        sd.update(safetensors.load_file(str(shard)))
    return sd


def _apply_param_mapping(sd, mapping):
    import re
    out = {}
    for k, v in sd.items():
        new_k = k
        for pat, repl in mapping.items():
            if re.match(pat, k):
                new_k = re.sub(pat, repl, k)
                break
        out[new_k] = v
    return out


def _ensure_distributed():
    """The denoising stage calls get_local_torch_device(), which needs
    FastVideo's world/TP groups. Initialize a single-rank (world_size=1) group
    in-process (mirrors tests/local_tests/sd35/test_sd35_component_parity.py)."""
    import torch.distributed as dist
    from fastvideo.distributed.parallel_state import (
        get_tp_group, init_distributed_environment, initialize_model_parallel)
    try:
        get_tp_group()
        return
    except Exception:
        pass
    if not dist.is_initialized():
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("WORLD_SIZE", "1")
        os.environ.setdefault("LOCAL_RANK", "0")
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        store_path = f"/tmp/fastvideo_glm_pg_{os.getpid()}.store"
        dist.init_process_group(backend="nccl",
                                init_method=f"file://{store_path}",
                                rank=0, world_size=1)
    init_distributed_environment(world_size=1, rank=0, local_rank=0,
                                 distributed_init_method="env://")
    try:
        get_tp_group()
    except Exception:
        initialize_model_parallel(tensor_model_parallel_size=1,
                                  sequence_model_parallel_size=1,
                                  data_parallel_size=1)


def _load_fastvideo_transformer(device, dtype):
    from fastvideo.configs.models.dits.glm_image import GlmImageDiTConfig
    from fastvideo.models.dits.glm_image import GlmImageTransformer2DModel
    cfg = GlmImageDiTConfig()
    model = GlmImageTransformer2DModel(
        cfg, {"_class_name": "GlmImageTransformer2DModel"})
    sd = _load_state_dict(TRANSFORMER_DIR)
    sd = _apply_param_mapping(sd, cfg.arch_config.param_names_mapping)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    assert not missing, f"FastVideo DiT missing keys: {missing[:10]}"
    assert not unexpected, f"FastVideo DiT unexpected keys: {unexpected[:10]}"
    return model.to(device, dtype=dtype).eval()


def _fastvideo_denoise_latents(device, dtype, *, prompt_embeds, prior_token_ids,
                               init_latents):
    """Drive the real GlmImageDenoisingStage with injected, matched inputs and
    return the denoised latents (1, 16, H/8, W/8)."""
    from fastvideo.models.schedulers.scheduling_flow_match_euler_discrete import (
        FlowMatchEulerDiscreteScheduler)
    from fastvideo.pipelines.basic.glm_image.stages.before_denoising import (
        calculate_shift)
    from fastvideo.pipelines.basic.glm_image.stages.denoising import (
        GlmImageDenoisingStage)
    from fastvideo.pipelines.pipeline_batch_info import ForwardBatch

    _ensure_distributed()
    transformer = _load_fastvideo_transformer(device, dtype)
    scheduler = FlowMatchEulerDiscreteScheduler(shift=1.0)
    stage = GlmImageDenoisingStage(transformer=transformer, scheduler=scheduler)

    # Reproduce the production timestep/sigma schedule (before_denoising.py):
    # integer-cast linspace timesteps, resolution-dependent shift on sigmas.
    ntt = scheduler.config.num_train_timesteps
    patch = transformer.patch_size
    image_seq_len = ((HEIGHT // 8) * (WIDTH // 8)) // (patch**2)
    sched_t = np.linspace(ntt, 1.0, STEPS + 1)[:-1].astype(
        np.int64).astype(np.float32)
    scheduler.set_shift(calculate_shift(image_seq_len))
    scheduler.set_timesteps(STEPS, device=device,
                            sigmas=(sched_t / ntt).tolist(),
                            timesteps=sched_t.tolist())

    batch = ForwardBatch(data_type="image")
    batch.prompt_embeds = [prompt_embeds.to(device, dtype)]
    batch.attention_mask = None  # match diffusers (no text padding mask)
    batch.prior_token_id = prior_token_ids.to(device)
    batch.prior_token_drop = torch.zeros_like(prior_token_ids,
                                              dtype=torch.bool, device=device)
    batch.latents = init_latents.clone().unsqueeze(2).to(device, dtype)
    batch.timesteps = scheduler.timesteps
    batch.height, batch.width = HEIGHT, WIDTH
    batch.num_inference_steps = STEPS
    batch.guidance_scale = 1.5
    batch.do_classifier_free_guidance = True
    batch.seed = SEED
    batch.extra = {}  # no glm_kv_caches -> T2I denoise path

    out = stage.forward(batch, fastvideo_args=None)
    latents = out.latents
    if latents.dim() == 5:
        latents = latents.squeeze(2)
    latents = latents.detach().float().cpu()
    del transformer, stage
    torch.cuda.empty_cache()
    import torch.distributed as dist
    if dist.is_initialized():
        dist.destroy_process_group()
    return latents


def _fastvideo_image(device) -> np.ndarray:
    pytest.importorskip("fastvideo")
    try:
        from fastvideo import VideoGenerator
    except ImportError as e:
        pytest.skip(f"FastVideo VideoGenerator unavailable: {e}")
    gen = VideoGenerator.from_pretrained(str(LOCAL_WEIGHTS_DIR), num_gpus=1,
                                         trust_remote_code=True)
    result = gen.generate_video(prompt=SAMPLE_PROMPT,
                                save_video=False,
                                return_frames=True,
                                height=HEIGHT, width=WIDTH,
                                num_inference_steps=STEPS,
                                guidance_scale=1.5,
                                seed=SEED)
    gen.shutdown()
    return _to_uint8_hwc(result["frames"][0])


def test_fastvideo_pipeline_produces_valid_image(device):
    fv = _fastvideo_image(device)
    assert fv.shape == (HEIGHT, WIDTH, 3), f"unexpected shape {fv.shape}"
    assert fv.dtype == np.uint8 and np.isfinite(fv).all()
    assert fv.std() > 10.0, f"image is near-constant (std={fv.std():.2f})"


def test_pipeline_denoise_parity_deterministic(device):
    """Inject identical (prompt_embeds, prior_token_ids, init_latents) into the
    diffusers pipeline and FastVideo's GlmImageDenoisingStage, then compare the
    denoised latents and the decoded images. With the stochastic AR and the
    independent latent RNG removed, the two implementations must agree to within
    bf16/SDPA-kernel noise."""
    # Inference only: the T5 glyph encoder leaves a grad graph on the embeds,
    # which later breaks image_processor.postprocess()'s .numpy() on the decoded
    # image. Disabling grad for the whole test also trims memory.
    torch.set_grad_enabled(False)
    diffusers = pytest.importorskip("diffusers")
    from diffusers.utils.torch_utils import randn_tensor

    dtype = torch.bfloat16
    pipe = diffusers.GlmImagePipeline.from_pretrained(
        str(LOCAL_WEIGHTS_DIR), torch_dtype=dtype).to(device)

    gen = torch.Generator(device=device).manual_seed(SEED)

    # --- shared, deterministic inputs (computed once, fed to both) ---------- #
    pos, neg = pipe.encode_prompt(SAMPLE_PROMPT,
                                  do_classifier_free_guidance=True,
                                  device=device, dtype=dtype)
    assert pos.shape[1] == neg.shape[1], (
        f"quote-free prompt should give equal pos/neg glyph lengths, got "
        f"{pos.shape[1]} vs {neg.shape[1]}")
    prior_token_ids, _, _ = pipe.generate_prior_tokens(
        SAMPLE_PROMPT, HEIGHT, WIDTH, image=None, device=device, generator=gen)
    latent_ch = pipe.transformer.config.in_channels
    init_latents = randn_tensor((1, latent_ch, HEIGHT // 8, WIDTH // 8),
                                generator=gen, device=device, dtype=dtype)

    # --- official denoised latents ----------------------------------------- #
    # prompt=None: diffusers check_inputs rejects passing prompt + prompt_embeds
    # together; the embeds (and prior tokens) fully specify the run.
    official = pipe(prompt=None,
                    prompt_embeds=pos, negative_prompt_embeds=neg,
                    prior_token_ids=prior_token_ids,
                    latents=init_latents.clone(),
                    height=HEIGHT, width=WIDTH,
                    num_inference_steps=STEPS, guidance_scale=1.5,
                    output_type="latent").images.float().cpu()

    # FastVideo packs CFG as a single [pos; neg] 2-row tensor (no padding here
    # because L_pos == L_neg, asserted above).
    packed = torch.cat([pos, neg], dim=0)

    # --- FastVideo denoised latents (real GlmImageDenoisingStage) ---------- #
    fv = _fastvideo_denoise_latents(device, dtype,
                                    prompt_embeds=packed,
                                    prior_token_ids=prior_token_ids,
                                    init_latents=init_latents)

    assert official.shape == fv.shape, (
        f"latent shape mismatch: {official.shape} vs {fv.shape}")

    # --- latent-level agreement -------------------------------------------- #
    cos = torch.nn.functional.cosine_similarity(
        official.flatten(), fv.flatten(), dim=0).item()
    lat_mae = (official - fv).abs().mean().item()
    lat_scale = official.abs().mean().item()
    print(f"[glm-image denoise parity] latent cosine={cos:.6f} "
          f"MAE={lat_mae:.5f} (|latent| mean={lat_scale:.5f}, "
          f"rel={lat_mae / max(lat_scale, 1e-6):.4f})")

    # --- decoded-image agreement (same VAE both sides isolates denoise) ----- #
    def _decode(lat):
        lat = lat.to(device, dtype)
        mean = torch.tensor(pipe.vae.config.latents_mean).view(
            1, -1, 1, 1).to(device, dtype)
        std = torch.tensor(pipe.vae.config.latents_std).view(
            1, -1, 1, 1).to(device, dtype)
        img = pipe.vae.decode((lat * std + mean), return_dict=False)[0]
        return _to_uint8_hwc(pipe.image_processor.postprocess(
            img, output_type="np")[0])

    off_img, fv_img = _decode(official), _decode(fv)
    img_mae = np.abs(off_img.astype(np.float32)
                     - fv_img.astype(np.float32)).mean()
    print(f"[glm-image denoise parity] decoded image MAE={img_mae:.3f}/255 "
          f"(diffusers mean={off_img.mean():.1f}, fv mean={fv_img.mean():.1f})")

    del pipe
    torch.cuda.empty_cache()

    assert cos > LATENT_COSINE_MIN, (
        f"denoised-latent cosine {cos:.6f} < {LATENT_COSINE_MIN}: the FastVideo "
        "and official denoise diverge well beyond kernel noise (wiring bug).")
    assert img_mae < IMAGE_MAE_MAX, (
        f"decoded-image MAE {img_mae:.3f}/255 >= {IMAGE_MAE_MAX}: outputs are "
        "not equivalent under matched inputs (wiring/schedule bug).")
