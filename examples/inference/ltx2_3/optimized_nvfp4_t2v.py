# SPDX-License-Identifier: Apache-2.0
"""LTX-2.3 distilled text-to-video with the optimized NVFP4 inference stack.

Runs `FastVideo/LTX-2.3-Distilled-Diffusers` on a single GPU with the full
validated optimization stack:

* NVFP4 block-scaled linear layers (per-16 E2M1 weights + E4M3 scale factors),
* ATTN_QAT_INFER FP4 attention (arch-resolved kernel, receipt logged),
* torch.compile (fullgraph) over the DiT, text encoder, and VAE,
* single-stage 8-step distilled sampling at guidance 1.0.

Quick start
-----------
    # On GB200-class ARM hosts, unset LD_LIBRARY_PATH (see Hardware notes):
    env -u LD_LIBRARY_PATH python examples/inference/ltx2_3/optimized_nvfp4_t2v.py

    # Optional overrides:
    #   export LTX23_MODEL_PATH=/path/to/local/snapshot
    #   export LTX23_T2V_PROMPT="a red fox running through fresh snow"
    #   export LTX23_OUTPUT_DIR=outputs_video/ltx2_3_nvfp4_t2v

Hardware notes
--------------
- On GB200 / Blackwell, run with `env -u LD_LIBRARY_PATH ...` to avoid a
  system-cuBLAS / torch-cuBLAS mismatch that fails every GEMM (some ARM
  container images ship an LD_LIBRARY_PATH that breaks torch.compile's
  toolchain discovery). The `_inductor.shape_padding = False` line below
  also avoids a pad_mm landmine on the same generation of cards.
"""
from __future__ import annotations

import os
import time
from pathlib import Path

import torch._inductor.config as _inductor

from fastvideo import VideoGenerator
from fastvideo.configs.pipelines.base import PipelineConfig
from fastvideo.layers.quantization.nvfp4_config import NVFP4Config
from fastvideo.utils import maybe_download_model

# ATTN_QAT_INFER is the FP4 attention half of the NVFP4 deploy contract. It
# resolves per arch (CUTLASS SageAttention3-FP4 on sm_120a/sm_121a, FP4 FA4
# on sm_100a/sm_103a) and logs a one-line receipt of what actually bound.
os.environ.setdefault("FASTVIDEO_ATTENTION_BACKEND", "ATTN_QAT_INFER")
os.environ.setdefault("FASTVIDEO_STAGE_LOGGING", "1")

# Inductor knobs. The first one (shape_padding=False) is mandatory on
# Blackwell to avoid a cuBLAS INVALID_VALUE crash inside pad_mm. The rest
# are the same matmul-friendliness flags the sibling LTX-2 examples use.
_inductor.shape_padding = False
_inductor.conv_1x1_as_mm = True  # treat 1x1 convolutions as matrix muls
_inductor.coordinate_descent_tuning = True
_inductor.coordinate_descent_check_all_directions = True
_inductor.epilogue_fusion = False  # do not fuse pointwise ops into matmuls

MODEL_ID = os.path.expandvars(
    os.path.expanduser(
        os.getenv("LTX23_MODEL_PATH", "FastVideo/LTX-2.3-Distilled-Diffusers")
    )
)
OUTPUT_DIR = Path(os.getenv("LTX23_OUTPUT_DIR", "outputs_video/ltx2_3_nvfp4_t2v"))
DEFAULT_PROMPT = (
    "A fashion model takes a slow step forward and shifts her weight, "
    "the soft fabric of her clothing swaying and rippling with the "
    "motion, her hair shifting gently, soft even studio lighting on a "
    "clean light background, elegant slow-motion runway feel."
)
PROMPT = os.getenv("LTX23_T2V_PROMPT", DEFAULT_PROMPT)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    model_root = maybe_download_model(MODEL_ID)
    print(f"Model:      {model_root}")
    print(f"Output dir: {OUTPUT_DIR.resolve()}")

    # Loading the pipeline config *with model_path* binds model-specific
    # tuning (notably VAE precision/decoder defaults) into the config.
    pipeline_config = PipelineConfig.from_pretrained(model_root)

    # NVFP4 linear layers on the DiT. The default purges the original BF16
    # weights of the always-FP4 linears right after conversion — a large
    # peak-memory reduction. Refine-only layers (the cross-modal AV
    # projections) always keep theirs: the base stage profile runs them
    # dense by deployment contract. retain_original_weights=True keeps
    # everything (debugging).
    pipeline_config.dit_config.quant_config = NVFP4Config()

    # fullgraph=True is supported on the NVFP4 path: the FP4 quantize step
    # is a registered custom op (fastvideo::nvfp4_quantize_fa4), so dynamo
    # traces through it without graph breaks. mode="default" — the two
    # CUDAGraph modes ("reduce-overhead" / "max-autotune") are a separate
    # opt-in, not part of this validated preset.
    torch_compile_kwargs = {
        "backend": "inductor",
        "fullgraph": True,
        "mode": "default",
        "dynamic": False,
    }

    generator = VideoGenerator.from_pretrained(
        model_root,
        num_gpus=1,
        pipeline_config=pipeline_config,
        # Compile the DiT, text encoder, and VAE — all three stages benefit,
        # and the VAE's codec submodules compile cleanly under fullgraph.
        enable_torch_compile=True,
        enable_torch_compile_text_encoder=True,
        enable_torch_compile_vae=True,
        torch_compile_kwargs=torch_compile_kwargs,
        torch_compile_kwargs_vae=torch_compile_kwargs,
        # Keep everything resident — no CPU offload for serving-style runs.
        dit_cpu_offload=False,
        text_encoder_cpu_offload=False,
        vae_cpu_offload=False,
        ltx2_vae_tiling=False,
    )

    common_kwargs = dict(
        prompt=PROMPT,
        negative_prompt="",      # distilled is CFG-free; no negative needed
        guidance_scale=1.0,      # CFG=1 for distilled
        height=1280, width=832,  # portrait runway aspect
        num_frames=121, fps=24,  # ~5s clip
        # Single-stage 8-step distilled sampling — the validated preset for
        # this checkpoint (no two-stage refine; the NVFP4 deploy contract
        # runs the distilled single-stage recipe).
        num_inference_steps=8,
        save_video=True,
    )

    try:
        # Warmup: pays cold compile + first-shape guard work, untimed.
        print("\n[warmup] compiling + generating…")
        generator.generate_video(
            output_path=str(OUTPUT_DIR / "_warmup.mp4"),
            seed=7,
            **common_kwargs,
        )
        (OUTPUT_DIR / "_warmup.mp4").unlink(missing_ok=True)

        # Measured run.
        out_path = OUTPUT_DIR / "output_ltx2_3_nvfp4_t2v.mp4"
        print(f"\n[measured] generating: {out_path}")
        t0 = time.perf_counter()
        result = generator.generate_video(
            output_path=str(out_path),
            seed=2002,
            **common_kwargs,
        )
        wall = time.perf_counter() - t0
        e2e = (result.get("e2e_latency") if isinstance(result, dict) else None) or wall
        print(f"[measured] e2e={e2e:.2f}s wall={wall:.2f}s")
    finally:
        generator.shutdown()

    # Expected receipts — verify these two lines in your own run's log:
    #
    # 1. ATTN_QAT_INFER routing receipt (logged at backend resolution; on a
    #    GB200-class part it reads):
    #
    #      ATTN_QAT_INFER resolved: arch=sm_100 kernel=flash-attention-fp4 \
    #        qk_mode=nvfp4(per-16-e4m3-sf) pv_mode=bf16 train_sim_mismatch=measured
    #
    # 2. NVFP4 weight purge receipt (logged after model conversion; N/M/X
    #    depend on the checkpoint):
    #
    #      NVFP4 weight purge receipt: purged N original bf16 weight tensors \
    #        (X.XX GiB freed); retained M (refine-only dense fallback or \
    #        retain_original_weights).


if __name__ == "__main__":
    main()
