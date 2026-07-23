"""Capture fastvideo-main goldens for the FastWan-QAD-FP8 artifact.

Runs ONLY where fastvideo main is importable (the cluster venv, which is an
editable install of /mnt/FastVideo). Pins main to its own exposed knobs so
the goldens measure the ARTIFACT, not the accelerator stack:

    attention  = FLASH_ATTN   (env override; the example defaults to sage,
                               which is not installed in the supported env)
    quant      = FP8 per-tensor (the released recipe), plus a no-quant bf16
                 load for the dequant-delta axis
    compile    = off, fsdp = off, single GPU

Captured (all committed under evidence/goldens/fastwan-qad-main/):
    text_encoder.npz      fp32 [512,4096] embeds for the e2e prompt (their
                          UMT5 loader, fp32, mask-zeroed) + the probe prompt
    probe_inputs.npz      seeded DiT probe inputs (latent, context)
    dit_bf16_t{T}.npz     plain-bf16 forward outputs, T in {1000,757,522}
    dit_fp8_t{T}.npz      fp8-quantized forward outputs
    e2e_step{i}_*.npz     per-step transformer input latents / outputs from
                          main's OWN DmdDenoisingStage (the loop authority)
    e2e_final_latents.npz final latents (BCTHW, stage-output layout)
    e2e_video.mp4         their VAE decode of the final latents (SSIM axis)
    manifest.json         commits, env, seeds, scheduler table, exact config

bf16 tensors are stored as fp32 (exact upcast); compare in fp32.
"""
from __future__ import annotations

import contextlib
import json
import os
import subprocess
import sys
from types import SimpleNamespace

# one process per model: the attention backend is chosen from this env var at
# model construction and cached — do not switch it within a process
VSA = len(sys.argv) > 1 and sys.argv[1] == "vsa"
os.environ["FASTVIDEO_ATTENTION_BACKEND"] = "VIDEO_SPARSE_ATTN" if VSA else "FLASH_ATTN"
os.environ.setdefault("MASTER_ADDR", "localhost")
os.environ.setdefault("MASTER_PORT", "29513")

REPO = "FastVideo/FastWan2.1-T2V-1.3B-Diffusers" if VSA else "FastVideo/FastWan-QAD-FP8-1.3B"
VSA_SPARSITY = 0.8  # the released FastWan recipe (basic_dmd example)
SEED = 1234
E2E_PROMPT = ("A curious raccoon peers through a vibrant field of yellow "
              "sunflowers, its eyes wide with interest. The playful yet serene "
              "atmosphere is complemented by soft natural light filtering "
              "through the petals. Mid-shot, warm and cheerful tones.")
PROBE_PROMPT = "A cat and a dog baking a cake together in a kitchen."
PROBE_TS = (1000, 757, 522)
PROBE_LATENT = (1, 16, 5, 60, 104)      # BCFHW probe geometry (17f 480x832)
E2E_BTCHW = (1, 21, 16, 60, 104)        # 81f 480x832 in main's DMD state layout


def _out_dir() -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    name = "fastwan-vsa-main" if VSA else "fastwan-qad-main"
    d = os.path.join(here, "..", "..", "evidence", "goldens", name)
    os.makedirs(d, exist_ok=True)
    return os.path.normpath(d)


def main() -> None:
    import numpy as np
    import torch
    from huggingface_hub import snapshot_download

    from fastvideo.configs.models.dits import WanVideoConfig
    from fastvideo.configs.pipelines import PipelineConfig, WanT2V480PConfig
    from fastvideo.configs.pipelines.wan import FastWan2_1_T2V_480P_Config
    from fastvideo.distributed import maybe_init_distributed_environment_and_model_parallel
    from fastvideo.fastvideo_args import FastVideoArgs
    from fastvideo.forward_context import set_forward_context
    from fastvideo.layers.quantization.fp8_config import FP8Config, convert_model_to_fp8
    from fastvideo.models.loader.component_loader import (TextEncoderLoader, TransformerLoader,
                                                          VAELoader)
    from fastvideo.models.schedulers.scheduling_flow_match_euler_discrete import (
        FlowMatchEulerDiscreteScheduler)
    from fastvideo.pipelines.pipeline_batch_info import ForwardBatch
    from fastvideo.pipelines.stages.denoising import DmdDenoisingStage
    from transformers import AutoTokenizer

    device = "cuda"
    root = snapshot_download(REPO, token=False)
    out = _out_dir()
    maybe_init_distributed_environment_and_model_parallel(1, 1)

    # loader args mirror the released QAD example: no fsdp, no offload
    _NO_WRAP = dict(use_fsdp_inference=False, dit_cpu_offload=False,
                    text_encoder_cpu_offload=False, vae_cpu_offload=False,
                    pin_cpu_memory=False)

    # ------------------------------------------------------------- text --- #
    t5_args = FastVideoArgs(model_path=os.path.join(root, "text_encoder"),
                            pipeline_config=WanT2V480PConfig(), **_NO_WRAP)
    t5_args.device = torch.device(device)
    t5 = TextEncoderLoader().load(os.path.join(root, "text_encoder"), t5_args).eval()
    t5_device = next(t5.parameters()).device  # loader-owned placement
    tok = AutoTokenizer.from_pretrained(os.path.join(root, "tokenizer"))

    def encode(text: str) -> torch.Tensor:
        b = tok([text], padding="max_length", max_length=512, truncation=True,
                add_special_tokens=True, return_attention_mask=True, return_tensors="pt")
        with torch.no_grad(), set_forward_context(current_timestep=0, attn_metadata=None):
            emb = t5(input_ids=b.input_ids.to(t5_device),
                     attention_mask=b.attention_mask.to(t5_device)).last_hidden_state
        emb = emb.to(device)
        emb = emb.to(torch.float32)
        emb[:, int(b.attention_mask[0].sum()):] = 0
        return emb

    e2e_embeds = encode(E2E_PROMPT)
    np.savez(os.path.join(out, "text_encoder.npz"),
             e2e=e2e_embeds[0].cpu().numpy(), probe=encode(PROBE_PROMPT)[0].cpu().numpy())
    print("text embeds:", e2e_embeds.shape, e2e_embeds.dtype, flush=True)

    # --------------------------------------------------- dit probe inputs --- #
    gen = torch.Generator("cpu").manual_seed(SEED)
    probe_x = torch.randn(PROBE_LATENT, generator=gen, dtype=torch.float32)
    probe_ctx = torch.randn((1, 512, 4096), generator=gen, dtype=torch.float32)
    np.savez(os.path.join(out, "probe_inputs.npz"),
             latent=probe_x.numpy(), context=probe_ctx.numpy())

    def probe_meta():
        """VSA probes/e2e need attention metadata in the forward context —
        built with THEIR builder, exactly like the DMD stage does."""
        if not VSA:
            return None
        from fastvideo.attention.backends.video_sparse_attn import (
            VideoSparseAttentionMetadataBuilder)
        return VideoSparseAttentionMetadataBuilder().build(
            current_timestep=0, raw_latent_shape=PROBE_LATENT[2:],
            patch_size=(1, 2, 2), VSA_sparsity=VSA_SPARSITY,
            device=torch.device(device))

    def probe(dit, tag: str) -> None:
        meta = probe_meta()
        for t in PROBE_TS:
            xt = probe_x.to(device, torch.bfloat16)
            ct = probe_ctx.to(device)          # fp32, like the real embeds
            tt = torch.tensor([t], dtype=torch.int64, device=device)
            with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16), \
                    set_forward_context(current_timestep=0, attn_metadata=meta,
                                        forward_batch=ForwardBatch(data_type="dummy")):
                o = dit(hidden_states=xt, encoder_hidden_states=ct, timestep=tt)
            o = o[0] if isinstance(o, (tuple, list)) else o
            np.savez(os.path.join(out, f"dit_{tag}_t{t}.npz"),
                     out=o.to(torch.float32).cpu().numpy())
            print(f"dit.{tag} t={t}: {tuple(o.shape)} {o.dtype}", flush=True)

    dit_path = os.path.join(root, "transformer")

    def load_dit(quant: bool):
        cfg = WanVideoConfig()
        if quant:
            cfg.quant_config = FP8Config(granularity="tensor")
        args = FastVideoArgs(model_path=dit_path,
                             pipeline_config=PipelineConfig(dit_config=cfg, dit_precision="bf16"),
                             **_NO_WRAP)
        args.device = torch.device(device)
        dit = TransformerLoader().load(dit_path, args).eval()
        if quant:
            first = dit.blocks[0].to_q
            if not hasattr(first, "_fp8_weight"):
                convert_model_to_fp8(dit)
            assert hasattr(dit.blocks[0].to_q, "_fp8_weight"), "fp8 conversion did not run"
            # no dtype cast here — a blanket .to(bf16) would destroy the fp8
            # buffers (and their fp32 scales) the loader just created
        else:
            dit = dit.to(dtype=torch.bfloat16)
        return dit

    if VSA:
        dit_q = load_dit(quant=False)  # the VSA artifact serves bf16, no fp8
        probe(dit_q, "vsa")
    else:
        dit = load_dit(quant=False)
        probe(dit, "bf16")
        del dit
        torch.cuda.empty_cache()
        dit_q = load_dit(quant=True)
        probe(dit_q, "fp8")

    # ------------------------------------------------ e2e: THEIR DMD loop --- #
    stage = DmdDenoisingStage(transformer=dit_q,
                              scheduler=FlowMatchEulerDiscreteScheduler(shift=8.0))
    # stage runs standalone (no composed pipeline); progress bar is
    # pipeline-owned, so stub it.
    stage.progress_bar = lambda total=None: contextlib.nullcontext(
        SimpleNamespace(update=lambda: None))

    gen_e2e = torch.Generator("cpu").manual_seed(SEED)
    from diffusers.utils.torch_utils import randn_tensor
    latents = randn_tensor(E2E_BTCHW, generator=[gen_e2e], device=torch.device(device),
                           dtype=torch.float32)

    steps: list[dict] = []

    def hook(module, args_in, kwargs_in, output):
        steps.append({
            "x": args_in[0].detach().to(torch.float32).cpu().numpy(),
            "t": float(kwargs_in.get("timestep", args_in[2] if len(args_in) > 2 else -1)
                       if not torch.is_tensor(kwargs_in.get("timestep"))
                       else kwargs_in["timestep"].item()),
            "out": output.detach().to(torch.float32).cpu().numpy(),
        })

    h = dit_q.register_forward_hook(hook, with_kwargs=True)

    fargs = FastVideoArgs(model_path=dit_path, pipeline_config=FastWan2_1_T2V_480P_Config(),
                          **_NO_WRAP)
    fargs.device = torch.device(device)
    if VSA:
        fargs.VSA_sparsity = VSA_SPARSITY
    # timesteps/guidance/eta satisfy verify_input; the stage's loop actually
    # runs the config's dmd_denoising_steps with its own internal scheduler.
    batch = ForwardBatch(data_type="video", latents=latents,
                         prompt_embeds=[e2e_embeds], generator=[gen_e2e],
                         num_inference_steps=3,
                         timesteps=torch.tensor([1000, 757, 522], device=device),
                         guidance_scale=1.0, eta=0.0,
                         do_classifier_free_guidance=False,
                         # BCTHW dims; the VSA branch reads [2:5] -> (T, H, W)
                         raw_latent_shape=torch.Size(
                             (1, 16, E2E_BTCHW[1], E2E_BTCHW[3], E2E_BTCHW[4])))
    batch = stage(batch, fargs)
    h.remove()

    final = batch.latents  # BCTHW after the stage's closing permute
    np.savez(os.path.join(out, "e2e_final_latents.npz"),
             latents=final.detach().to(torch.float32).cpu().numpy())
    for i, s in enumerate(steps):
        np.savez(os.path.join(out, f"e2e_step{i}.npz"), x=s["x"], out=s["out"], t=s["t"])
    print(f"e2e: {len(steps)} steps, final {tuple(final.shape)} {final.dtype}", flush=True)

    # ----------------------------------------------------------- decode --- #
    vae_args = FastVideoArgs(model_path=os.path.join(root, "vae"),
                             pipeline_config=WanT2V480PConfig(), **_NO_WRAP)
    vae_args.device = torch.device(device)
    vae = VAELoader().load(os.path.join(root, "vae"), vae_args).to(
        torch.device(device), torch.float32).eval()
    with open(os.path.join(root, "vae", "config.json")) as f:
        vcfg = json.load(f)
    mean = torch.tensor(vcfg["latents_mean"], dtype=torch.float32,
                        device=device).view(1, -1, 1, 1, 1)
    std = 1.0 / torch.tensor(vcfg["latents_std"], dtype=torch.float32,
                             device=device).view(1, -1, 1, 1, 1)
    with torch.no_grad():
        video = vae.decode(final.to(torch.float32) / std + mean)
    video = getattr(video, "sample", video)
    video = video[0] if isinstance(video, (tuple, list)) else video
    frames = ((video[0].float().clamp(-1, 1) + 1) / 2 * 255).round().to(torch.uint8)
    frames = frames.permute(1, 2, 3, 0).cpu().numpy()  # [T,H,W,C]
    import imageio
    imageio.mimwrite(os.path.join(out, "e2e_video.mp4"), list(frames), fps=16)
    np.savez(os.path.join(out, "e2e_frames.npz"),
             first=frames[0], mid=frames[len(frames) // 2], last=frames[-1])

    # --------------------------------------------------------- manifest --- #
    import fastvideo
    import flash_attn
    src = os.path.dirname(os.path.dirname(os.path.abspath(fastvideo.__file__)))
    try:
        commit = subprocess.run(["git", "-C", src, "rev-parse", "HEAD"],
                                capture_output=True, text=True).stdout.strip()
    except Exception:
        commit = "unknown"
    sched = stage.scheduler
    manifest = {
        "repo": REPO, "snapshot": os.path.basename(root),
        "fastvideo_commit": commit, "fastvideo_src": src,
        "torch": torch.__version__, "flash_attn": flash_attn.__version__,
        "python": sys.version.split()[0],
        "gpu": torch.cuda.get_device_name(0),
        "attention_backend": os.environ["FASTVIDEO_ATTENTION_BACKEND"],
        "quant": ("none (bf16, VSA sparsity %.2f)" % VSA_SPARSITY) if VSA else
                 "FP8 per-tensor (dynamic act, post-load weight quant from bf16)",
        "vsa_sparsity": VSA_SPARSITY if VSA else None,
        "seed": SEED, "e2e_prompt": E2E_PROMPT, "probe_prompt": PROBE_PROMPT,
        "probe_timesteps": list(PROBE_TS), "probe_latent_bcfhw": list(PROBE_LATENT),
        "e2e_latent_btchw": list(E2E_BTCHW),
        "dmd_denoising_steps": [1000, 757, 522],
        "scheduler": {
            "class": type(sched).__name__, "shift": 8.0,
            "table_len": int(sched.timesteps.shape[0]),
            "sigma_lookup": {str(t): float(
                sched.sigmas[(sched.timesteps - t).abs().argmin()].item())
                for t in PROBE_TS},
        },
        "notes": "no compile, no fsdp, single GPU; DmdDenoisingStage's INTERNAL "
                 "scheduler (hardcoded shift 8.0) is the sigma authority",
    }
    with open(os.path.join(out, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(json.dumps(manifest["scheduler"], indent=2), flush=True)
    print("capture complete ->", out, flush=True)


if __name__ == "__main__":
    main()
