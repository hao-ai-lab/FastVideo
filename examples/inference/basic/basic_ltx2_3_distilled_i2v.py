# SPDX-License-Identifier: Apache-2.0
"""LTX-2.3 distilled image-to-video with torch.compile + timing breakdown.

This example runs the LTX-2.3 distilled student model on a single GPU with
torch.compile fully enabled, then prints a per-stage timing breakdown so the
user can see where wall-time goes. It is meant as the canonical entry point
for trying out the LTX-2.3 i2v path on `hao-ai-lab/FastVideo:main`.

Quick start
-----------
    export LTX23_I2V_IMAGE=/path/to/your/portrait_or_product.jpg
    # optional overrides:
    #   export LTX23_I2V_PROMPT="a fashion model walks toward camera..."
    #   export LTX23_OUTPUT_DIR=outputs_video/ltx2_3_distilled_i2v
    python examples/inference/basic/basic_ltx2_3_distilled_i2v.py

What the script does
--------------------
1. Loads FastVideo/LTX-2.3-Distilled-Diffusers (8 denoise + 3 refine steps,
   CFG=1, no refine LoRA — the distilled production recipe).
2. Compiles the DiT, text encoder, and VAE (fullgraph, Inductor default
   mode — autotune adds ~7 min cold-compile here with no measurable
   e2e gain).
3. Runs 2 warmup calls (untimed) + 2 measured calls. Two warmups are kept
   as a safety net — the first call pays cold compile + first-shape guard
   work, and a second warmup ensures any residual recompiles settle before
   we measure.
4. Prints a per-stage breakdown and an average over the measured runs.

Hardware notes
--------------
- Single-GPU example; for multi-GPU sequence-parallel see the gradio demo
  under `examples/inference/gradio/local/gradio_local_demo_ltx2_3/`.
- First-time compile takes ~30-40 min on GB200 (~20 min on H100; cached
  in `$TORCHINDUCTOR_CACHE_DIR` afterwards). Subsequent invocations only
  pay the one-time process load + a few seconds of dynamo trace.
- On GB200 / Blackwell, run with `env -u LD_LIBRARY_PATH ...` to avoid a
  system-cuBLAS / torch-cuBLAS mismatch that fails every GEMM. The
  `_inductor.shape_padding = False` line below also avoids a pad_mm
  landmine on the same generation of cards.
"""
from __future__ import annotations

import os
import time
from collections import OrderedDict
from pathlib import Path

import torch._inductor.config as _inductor

from fastvideo import VideoGenerator
from fastvideo.api import (
    CompileConfig, ComponentConfig, EngineConfig, GenerationRequest,
    GeneratorConfig, OffloadConfig, OutputConfig, PipelineSelection,
    SamplingConfig,
)
from fastvideo.configs.pipelines.base import PipelineConfig
from fastvideo.utils import maybe_download_model

# Env knobs (set BEFORE importing fastvideo where possible — but
# FASTVIDEO_ATTENTION_BACKEND is fine here because the worker reads it
# on generator construction).
os.environ.setdefault("FASTVIDEO_ATTENTION_BACKEND", "FLASH_ATTN")
os.environ.setdefault("FASTVIDEO_STAGE_LOGGING", "1")

# Inductor knobs. The first one (shape_padding=False) is mandatory on
# Blackwell to avoid a cuBLAS INVALID_VALUE crash inside pad_mm during
# the refine path. The rest are autotune-friendliness flags.
_inductor.shape_padding = False
_inductor.conv_1x1_as_mm = True
_inductor.coordinate_descent_tuning = True
_inductor.coordinate_descent_check_all_directions = True
_inductor.epilogue_fusion = False

MODEL_ID = os.path.expandvars(
    os.path.expanduser(
        os.getenv("LTX23_MODEL_PATH", "FastVideo/LTX-2.3-Distilled-Diffusers")
    )
)
OUTPUT_DIR = Path(
    os.getenv("LTX23_OUTPUT_DIR", "outputs_video/ltx2_3_distilled_i2v")
)
I2V_IMAGE = os.getenv("LTX23_I2V_IMAGE", "")
DEFAULT_PROMPT = (
    "A fashion model takes a slow step forward and shifts her weight, "
    "the soft fabric of her clothing swaying and rippling with the "
    "motion, her hair shifting gently, soft even studio lighting on a "
    "clean light background, elegant slow-motion runway feel."
)
PROMPT = os.getenv("LTX23_I2V_PROMPT", DEFAULT_PROMPT)

# Per-stage timing helpers --------------------------------------------------

def _print_stage_breakdown(result, label: str) -> float | None:
    """Print stage execution times and return the sum, or None if missing."""
    logging_info = result.logging_info
    stages = getattr(logging_info, "stages", None) if logging_info else None
    if not stages:
        print(f"  [{label}] stage breakdown unavailable")
        return None
    print(f"  [{label}] stage breakdown:")
    total = 0.0
    for name, metrics in stages.items():
        exec_s = float(metrics.get("execution_time", 0.0))
        total += exec_s
        print(f"    - {name}: {exec_s:.3f}s")
    print(f"    - stage_sum: {total:.3f}s")
    return total


def _collect_stage_times(
    result,
    stage_times: dict[str, list[float]],
    stage_order: OrderedDict[str, None],
) -> None:
    logging_info = result.logging_info
    stages = getattr(logging_info, "stages", None) if logging_info else None
    if not stages:
        return
    for name, metrics in stages.items():
        stage_order.setdefault(name, None)
        stage_times.setdefault(name, []).append(
            float(metrics.get("execution_time", 0.0))
        )


def _resolve_refine_upsampler(model_root: str) -> Path:
    """LTX-2.3 distilled snapshots ship a `spatial_upscaler/` subdir."""
    for name in ("spatial_upscaler", "spatial_upsampler"):
        cand = Path(model_root) / name
        if (cand / "config.json").is_file():
            return cand
    raise FileNotFoundError(
        f"No refine upsampler directory under {model_root}. "
        f"Expected `{model_root}/spatial_upscaler/config.json`."
    )


# Main ---------------------------------------------------------------------

def main() -> None:
    if not I2V_IMAGE:
        raise SystemExit(
            "LTX23_I2V_IMAGE is required for i2v. Example:\n"
            "  export LTX23_I2V_IMAGE=/path/to/portrait_or_product.jpg\n"
            "  python examples/inference/basic/basic_ltx2_3_distilled_i2v.py"
        )
    if not Path(I2V_IMAGE).is_file():
        raise SystemExit(f"LTX23_I2V_IMAGE not found: {I2V_IMAGE}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    model_root = maybe_download_model(MODEL_ID)
    refine_upsampler_path = _resolve_refine_upsampler(model_root)
    print(f"Model:           {model_root}")
    print(f"Refine upsampler: {refine_upsampler_path}")
    print(f"i2v image:       {I2V_IMAGE}")
    print(f"Output dir:      {OUTPUT_DIR.resolve()}")

    # mode="default" — Inductor's default schedule matches max-autotune on
    # this pipeline (denoise/refine/decode all within ~5 ms, n=2) while
    # saving ~7 min of cold compile on a single GB200.
    torch_compile_kwargs = {
        "backend": "inductor",
        "fullgraph": True,
        "mode": "default",
        "dynamic": False,
    }

    # Loading the pipeline config *with model_path* binds model-specific
    # tuning (notably VAE precision/decoder defaults) into the config. Without
    # this, the generic pipeline config gives a substantially slower VAE
    # decode stage. `basic_ltx2_distilled_fast_profile.py` uses the same
    # pattern.
    pipeline_config = PipelineConfig.from_pretrained(model_root)
    pipeline_config.dit_config.quant_config = None

    generator = VideoGenerator.from_config(
        GeneratorConfig(
            model_path=model_root,
            engine=EngineConfig(
                num_gpus=1,
                compile=CompileConfig(
                    enabled=True,
                    text_encoder_enabled=True,
                    # Compile the VAE codec submodules (encoder / decoder)
                    # too. The `LTX2CausalVideoAutoencoder` declares
                    # `_compile_conditions` so `_compile_with_conditions`
                    # targets just those submodules and leaves the
                    # surrounding tiling control flow eager — needed for
                    # fullgraph + dynamic=False to succeed. VAE eager decode
                    # is ~1.0s; compiling it brings the stage to ~0.3s.
                    vae_enabled=True,
                    backend=torch_compile_kwargs["backend"],
                    fullgraph=torch_compile_kwargs["fullgraph"],
                    mode=torch_compile_kwargs["mode"],
                    dynamic=torch_compile_kwargs["dynamic"],
                    vae_kwargs=torch_compile_kwargs,
                ),
                # Keep everything resident — no CPU offload for serving runs.
                offload=OffloadConfig(
                    dit=False,
                    text_encoder=False,
                    vae=False,
                ),
            ),
            pipeline=PipelineSelection(
                vae_tiling=False,
                # LTX-2.3 distilled uses the two-stage refine pipeline; the
                # refine LoRA is intentionally empty for the distilled
                # student.
                components=ComponentConfig(
                    upsampler_weights=str(refine_upsampler_path),
                ),
                preset_overrides={
                    "refine": {
                        "enabled": True,
                        "num_inference_steps": 3,
                        "guidance_scale": 1.0,
                        "add_noise": True,
                    }
                },
                experimental={"pipeline_config": pipeline_config},
            ),
        )
    )

    common_kwargs = dict(
        prompt=PROMPT,
        negative_prompt="",         # distilled is CFG-free; no negative needed
        guidance_scale=1.0,         # CFG=1 for distilled
        height=1280, width=832,     # portrait runway aspect
        num_frames=121, fps=24,     # ~5s clip
        num_inference_steps=8,      # distilled denoise steps
    )

    # i2v: anchor the input image at frame 0 with full strength.
    # `ltx2_image_crf=0.0` skips an extra JPEG re-encode of an already
    # JPEG conditioning image. These are model-specific knobs routed through
    # the request extensions escape hatch.
    common_extensions = dict(
        ltx2_images=[(I2V_IMAGE, 0, 1.0)],
        ltx2_image_crf=0.0,
    )

    warmup_runs = 2
    measured_runs = 2
    warmup_secs: list[float] = []
    measured_secs: list[float] = []
    stage_times: dict[str, list[float]] = {}
    stage_order: OrderedDict[str, None] = OrderedDict()

    try:
        # Warmup: untimed (but we still wall-clock them so the first compile
        # cost is visible to the reader).
        for w in range(warmup_runs):
            t0 = time.perf_counter()
            print(f"\n[warmup {w + 1}/{warmup_runs}] compiling + generating…")
            generator.generate(
                GenerationRequest(
                    prompt=common_kwargs["prompt"],
                    negative_prompt=common_kwargs["negative_prompt"],
                    sampling=SamplingConfig(
                        guidance_scale=common_kwargs["guidance_scale"],
                        height=common_kwargs["height"],
                        width=common_kwargs["width"],
                        num_frames=common_kwargs["num_frames"],
                        fps=common_kwargs["fps"],
                        num_inference_steps=common_kwargs["num_inference_steps"],
                        seed=7,
                    ),
                    output=OutputConfig(
                        output_path=str(OUTPUT_DIR / f"_warmup_{w + 1}.mp4"),
                        save_video=True,
                    ),
                    extensions=common_extensions,
                )
            )
            dt = time.perf_counter() - t0
            warmup_secs.append(dt)
            print(f"[warmup {w + 1}/{warmup_runs}] wall={dt:.1f}s")

        # Cleanup warmup artifacts so the user only sees measured outputs.
        for w in range(warmup_runs):
            (OUTPUT_DIR / f"_warmup_{w + 1}.mp4").unlink(missing_ok=True)

        # Measured.
        for m in range(measured_runs):
            out_path = OUTPUT_DIR / f"output_ltx2_3_distilled_i2v_run_{m + 1}.mp4"
            print(f"\n[measured {m + 1}/{measured_runs}] generating: {out_path}")
            t0 = time.perf_counter()
            result = generator.generate(
                GenerationRequest(
                    prompt=common_kwargs["prompt"],
                    negative_prompt=common_kwargs["negative_prompt"],
                    sampling=SamplingConfig(
                        guidance_scale=common_kwargs["guidance_scale"],
                        height=common_kwargs["height"],
                        width=common_kwargs["width"],
                        num_frames=common_kwargs["num_frames"],
                        fps=common_kwargs["fps"],
                        num_inference_steps=common_kwargs["num_inference_steps"],
                        seed=2002 + m,
                    ),
                    output=OutputConfig(
                        output_path=str(out_path),
                        save_video=True,
                    ),
                    extensions=common_extensions,
                )
            )
            wall = time.perf_counter() - t0
            e2e = (result.extra.get("e2e_latency") if result is not None else None) or wall
            measured_secs.append(e2e)
            print(f"[measured {m + 1}/{measured_runs}] e2e={e2e:.2f}s wall={wall:.2f}s")
            if result is not None:
                _print_stage_breakdown(result, f"measured {m + 1}")
                _collect_stage_times(result, stage_times, stage_order)

        # Summary.
        print("\n=== summary ===")
        print(f"warmup wall-times:      {[round(x, 1) for x in warmup_secs]}")
        if measured_secs:
            avg = sum(measured_secs) / len(measured_secs)
            print(
                f"measured e2e (n={len(measured_secs)}): "
                f"{[round(x, 2) for x in measured_secs]} -> avg {avg:.2f}s"
            )
        if stage_times:
            print(f"average stage times over {measured_runs} measured runs:")
            avg_total = 0.0
            for name in stage_order:
                vals = stage_times.get(name) or []
                if not vals:
                    continue
                avg_v = sum(vals) / len(vals)
                avg_total += avg_v
                print(f"  - {name}: {avg_v:.3f}s")
            print(f"  - stage_sum_avg: {avg_total:.3f}s")
    finally:
        generator.shutdown()


if __name__ == "__main__":
    main()
