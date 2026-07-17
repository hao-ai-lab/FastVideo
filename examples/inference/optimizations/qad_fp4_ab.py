"""QAD FP4 quality A/B/C/D harness — Wan2.1-T2V-1.3B on sm_121 (DGX Spark GB10).

Tests whether the quantization-aware-distilled checkpoint
``FastVideo/FastWan-QAD-1.3B`` recovers FP4 quality on sm_121, using the
sm_121-enabled ``ATTN_QAT_INFER`` attention kernel. Running FP4 attention on
*stock* Wan weights gives output below bf16 — expected, because stock weights
were never trained to tolerate FP4 attention. This harness runs the checkpoint
that *was* (fake-quant FP4 attention + NVFP4 linear trained into the model).

Attention (bf16 vs FP4) and linear (bf16 vs FP4) are fully decoupled at
inference, so we can isolate each axis:

    FASTVIDEO_ATTENTION_BACKEND unset          -> bf16 attention (SDPA)
    FASTVIDEO_ATTENTION_BACKEND=ATTN_QAT_INFER -> FP4 attention
    QAD_LINEAR=0                               -> bf16 linear
    QAD_LINEAR=1                               -> NVFP4 FP4 linear (flashinfer)

    arm  attention   linear   selects with
    A    bf16        bf16     QAD_LINEAR=0  (no ATTN env)            reference
    B    FP4         bf16     QAD_LINEAR=0  ATTN_QAT_INFER           isolate attn
    C    bf16        FP4      QAD_LINEAR=1  (no ATTN env)            isolate linear
    D    FP4         FP4      QAD_LINEAR=1  ATTN_QAT_INFER           full 4-bit

All four arms run end-to-end on the GB10 (the full 4-bit path — FP4 linear +
FP4 attention — works on sm_121). This script still runs exactly ONE arm per
invocation and dumps a C stack on any hard crash, so a single misbehaving arm
can never take the others down with it; the runbook loops it four times with
different env. Quality is the eye/ear on the saved mp4 + a matching-frame still;
timing is the mean generation_time over the measured runs.

On the GB10, expect FP4 attention ~6% faster denoise vs bf16 and quality-neutral
by eye on the QAD checkpoint (both share the 3-step distill's quality ceiling).
FP4 *linear* is roughly break-even at 1.3B/480p (the per-call quantize overhead
eats the small-GEMM saving in eager mode); its win shows at higher resolution
with torch.compile.

Knobs (env): QAD_MODEL, QAD_DISTILLED, QAD_STEPS (3), QAD_GUIDANCE (1.0),
QAD_HEIGHT (480), QAD_WIDTH (832), QAD_FRAMES (77), QAD_SEED (42),
QAD_WARMUP (1), QAD_RUNS (3), QAD_STILL (20), QAD_OUT (qad_fp4_samples).
"""
from __future__ import annotations

import faulthandler
import glob
import os
import time

import torch

faulthandler.enable()  # dump a C stack if any arm hard-crashes.


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


def _env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, str(default)))


def _env_float(name: str, default: float) -> float:
    return float(os.environ.get(name, str(default)))


# Distilled QAD transformer, loaded on top of the base Wan2.1-1.3B pipeline.
DEFAULT_DISTILLED = "FastVideo/FastWan-QAD-1.3B"
# The repo is a full diffusers pipeline; we overlay only its transformer onto the
# base Wan pipeline (vae/text_encoder are Wan-identical).
DISTILLED_WEIGHTS_FILE = "transformer/diffusion_pytorch_model.safetensors"

PROMPT = (
    "A curious raccoon peers through a vibrant field of yellow sunflowers, its eyes "
    "wide with interest. The playful yet serene atmosphere is complemented by soft "
    "natural light filtering through the petals. Mid-shot, warm and cheerful tones."
)


def resolve_distilled_weights(hf_id: str) -> str:
    """Return a local path to the distilled transformer safetensors."""
    if not hf_id:
        return ""
    if os.path.exists(hf_id):
        return hf_id
    from huggingface_hub import hf_hub_download
    return hf_hub_download(repo_id=hf_id, filename=DISTILLED_WEIGHTS_FILE)


def build_generator(fp4_linear: bool):
    from fastvideo import VideoGenerator
    from fastvideo.configs.pipelines.base import PipelineConfig

    model_id = _env("QAD_MODEL", "Wan-AI/Wan2.1-T2V-1.3B-Diffusers")

    pipeline_config = PipelineConfig.from_pretrained(model_id)
    pipeline_config.dit_precision = "bf16"
    pipeline_config.vae_precision = "bf16"
    pipeline_config.text_encoder_precisions = ("bf16",)

    if fp4_linear:
        # Wan-style config: matches to_q/k/v/out + ffn (the plain NVFP4 config is
        # LTX2-specific and would quantize nothing on Wan).
        from fastvideo.layers.quantization.nvfp4_qat_config import NVFP4QATConfig
        pipeline_config.dit_config.quant_config = NVFP4QATConfig()

    extra_kwargs = {}
    distilled = resolve_distilled_weights(_env("QAD_DISTILLED", DEFAULT_DISTILLED))
    if distilled:
        print(f"[qad] distilled weights: {distilled}")
        extra_kwargs["init_weights_from_safetensors"] = distilled

    # Keep everything resident (the 1.3B QAD model + FP4 fits the GB10's unified
    # memory); real Wan VAE decode for a faithful quality read (no TAEHV).
    return VideoGenerator.from_pretrained(
        model_id,
        pipeline_config=pipeline_config,
        num_gpus=1,
        use_fsdp_inference=False,
        dit_cpu_offload=False,
        dit_layerwise_offload=False,
        vae_cpu_offload=False,
        text_encoder_cpu_offload=False,
        pin_cpu_memory=False,
        enable_torch_compile=False,  # eager: isolate the FP4 effect, no compile noise
        **extra_kwargs,
    )


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required.")

    fp4_linear = _env_int("QAD_LINEAR", 0) == 1
    attn = _env("FASTVIDEO_ATTENTION_BACKEND", "default")
    steps = _env_int("QAD_STEPS", 3)
    guidance = _env_float("QAD_GUIDANCE", 1.0)
    height = _env_int("QAD_HEIGHT", 480)
    width = _env_int("QAD_WIDTH", 832)
    frames = _env_int("QAD_FRAMES", 77)
    seed = _env_int("QAD_SEED", 42)
    warmup = _env_int("QAD_WARMUP", 1)
    runs = _env_int("QAD_RUNS", 3)
    still_idx = _env_int("QAD_STILL", 20)
    out_dir = _env("QAD_OUT", "qad_fp4_samples")
    prompt = _env("QAD_PROMPT", PROMPT)

    tag = f"lin-{'fp4' if fp4_linear else 'bf16'}_attn-{attn.lower()}"
    cap = torch.cuda.get_device_capability()
    print(f"[qad] GPU {torch.cuda.get_device_name()} (cc {cap[0]}.{cap[1]})")
    print(f"[qad] ARM {tag}: linear={'FP4' if fp4_linear else 'bf16'}, "
          f"attention={attn}, {steps} steps, guidance {guidance}, "
          f"{height}x{width}x{frames}, seed {seed}")
    print(f"[qad] prompt: {prompt[:80]}{'...' if len(prompt) > 80 else ''}")

    arm_dir = os.path.join(out_dir, tag)
    os.makedirs(arm_dir, exist_ok=True)
    generator = build_generator(fp4_linear)

    def _generate():
        # seed + frame dims live under `sampling` (SamplingConfig); `output`
        # only takes output_path/save_video/return_frames (OutputConfig).
        return generator.generate(request={
            "prompt": prompt,
            "sampling": {
                "seed": seed,
                "num_inference_steps": steps,
                "guidance_scale": guidance,
                "height": height,
                "width": width,
                "num_frames": frames,
            },
            "output": {"save_video": True, "output_path": arm_dir,
                       "return_frames": True},
        })

    for _ in range(warmup):
        _generate()

    denoise_times: list[float] = []
    last = None
    for i in range(runs):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        last = _generate()
        torch.cuda.synchronize()
        wall = time.perf_counter() - t0
        denoise_times.append(getattr(last, "generation_time", wall))
        print(f"[qad] {tag} run {i + 1}/{runs}: {wall:.2f}s wall "
              f"(denoise {denoise_times[-1]:.2f}s)")

    # The pipeline wrote the mp4 (full known-good encode) into arm_dir; report
    # it and pull a matching-frame still from the [b,c,t,h,w] samples tensor
    # using the same recipe the pipeline's frame builder uses.
    mp4s = sorted(glob.glob(os.path.join(arm_dir, "*.mp4")), key=os.path.getmtime)
    if mp4s:
        print(f"[qad] video: {mp4s[-1]}")
    samples = getattr(last, "samples", None) if last is not None else None
    if samples is not None and getattr(samples, "ndim", 0) == 5:
        import imageio
        f = min(still_idx, samples.shape[2] - 1)  # samples: [b, c, t, h, w]
        still = (samples[0, :, f].permute(1, 2, 0).clamp(0, 1) * 255)
        still = still.to(torch.uint8).cpu().numpy()
        png = os.path.join(arm_dir, f"raccoon_{tag}_f{f}.png")
        imageio.imwrite(png, still)
        print(f"[qad] still: {png}")
    else:
        print("[qad] note: no 5-D samples tensor; grab a frame from the mp4 above")

    mean = sum(denoise_times) / len(denoise_times)
    print(f"\n[qad][{tag}] denoise mean {mean:.2f}s over {runs} runs "
          f"({warmup} warmup, {steps} steps)")
    generator.shutdown()


if __name__ == "__main__":
    main()
