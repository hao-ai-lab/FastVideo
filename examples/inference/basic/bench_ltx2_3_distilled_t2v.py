# SPDX-License-Identifier: Apache-2.0
"""LTX-2.3 distilled TEXT-to-video — parametrized 2x2 speed benchmark.

Pure t2v (no image conditioning). Same recipe/resolution/measurement as the
i2v scripts so the two are directly comparable: 8 denoise + 3 refine, CFG=1,
no refine LoRA, 832x1280 portrait, 121 frames @ 24fps.

Toggle the 2x2 cell via env vars:
    LTX23_COMPILE = 1|0   -> torch.compile on/off (DiT + text encoder + VAE)
    LTX23_NVFP4   = 1|0   -> DiT NVFP4 quant on/off (needs flashinfer +
                            CUDA_HOME=/usr/local/cuda-13.2)

    CUDA_VISIBLE_DEVICES=1 LTX23_COMPILE=1 LTX23_NVFP4=0 \
        python examples/inference/basic/bench_ltx2_3_distilled_t2v.py
"""
from __future__ import annotations

import os
import time
from collections import OrderedDict
from pathlib import Path

from fastvideo import VideoGenerator
from fastvideo.configs.pipelines.base import PipelineConfig
from fastvideo.utils import maybe_download_model

os.environ.setdefault("FASTVIDEO_ATTENTION_BACKEND", "FLASH_ATTN")
os.environ.setdefault("FASTVIDEO_STAGE_LOGGING", "1")


def _env_flag(name: str, default: str = "0") -> bool:
    return os.getenv(name, default).strip().lower() in ("1", "true", "yes", "on")


COMPILE = _env_flag("LTX23_COMPILE", "0")
NVFP4 = _env_flag("LTX23_NVFP4", "0")
# FA4-FP4 attention: quantize Q/K to NVFP4 for Blackwell block-scaled MMA
# (needs the cutlass.utils.ampere_helpers shim; auto-selects FlashAttention-4).
FA4 = _env_flag("LTX23_FA4", "0")

if COMPILE:
    # Inductor knobs — shape_padding=False mandatory on Blackwell (pad_mm
    # cuBLAS landmine); rest are autotune-friendliness flags.
    import torch._inductor.config as _inductor
    _inductor.shape_padding = False
    _inductor.conv_1x1_as_mm = True
    _inductor.coordinate_descent_tuning = True
    _inductor.coordinate_descent_check_all_directions = True
    _inductor.epilogue_fusion = False
    if _env_flag("LTX23_NO_CGTREES", "0"):
        # Fall back to the simpler (non-tree) cudagraph impl: cudagraph_trees
        # rejects FP4 flashinfer custom ops that allocate untracked tensors
        # inside the captured region.
        _inductor.triton.cudagraph_trees = False

if NVFP4:
    from fastvideo.layers.quantization.nvfp4_config import NVFP4Config

HEIGHT = int(os.getenv("LTX23_HEIGHT", "1280"))
WIDTH = int(os.getenv("LTX23_WIDTH", "832"))
NUM_FRAMES = int(os.getenv("LTX23_NUM_FRAMES", "121"))
STEPS = int(os.getenv("LTX23_STEPS", "8"))

_CELL = f"{'nvfp4' if NVFP4 else 'bf16'}_{'compile' if COMPILE else 'eager'}"
if FA4:
    _CELL += "_fa4"
_CELL += f"_{WIDTH}x{HEIGHT}_{NUM_FRAMES}f_{STEPS}st"

MODEL_ID = os.path.expandvars(
    os.path.expanduser(
        os.getenv("LTX23_MODEL_PATH", "FastVideo/LTX-2.3-Distilled-Diffusers")
    )
)
OUTPUT_DIR = Path(
    os.getenv("LTX23_OUTPUT_DIR", f"outputs_video/ltx2_3_distilled_t2v_{_CELL}")
)
DEFAULT_PROMPT = (
    "A fashion model takes a slow step forward and shifts her weight, "
    "the soft fabric of her clothing swaying and rippling with the "
    "motion, her hair shifting gently, soft even studio lighting on a "
    "clean light background, elegant slow-motion runway feel."
)
PROMPT = os.getenv("LTX23_T2V_PROMPT", DEFAULT_PROMPT)


def _print_stage_breakdown(result: dict, label: str) -> float | None:
    logging_info = result.get("logging_info")
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
    result: dict,
    stage_times: dict[str, list[float]],
    stage_order: OrderedDict[str, None],
) -> None:
    logging_info = result.get("logging_info")
    stages = getattr(logging_info, "stages", None) if logging_info else None
    if not stages:
        return
    for name, metrics in stages.items():
        stage_order.setdefault(name, None)
        stage_times.setdefault(name, []).append(
            float(metrics.get("execution_time", 0.0))
        )


def _resolve_refine_upsampler(model_root: str) -> Path:
    for name in ("spatial_upscaler", "spatial_upsampler"):
        cand = Path(model_root) / name
        if (cand / "config.json").is_file():
            return cand
    raise FileNotFoundError(
        f"No refine upsampler directory under {model_root}. "
        f"Expected `{model_root}/spatial_upscaler/config.json`."
    )


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    model_root = maybe_download_model(MODEL_ID)
    refine_upsampler_path = _resolve_refine_upsampler(model_root)
    print(f"Cell:            {_CELL}  (compile={COMPILE}, nvfp4={NVFP4}, fa4={FA4})")
    print(f"Task:            t2v (no image conditioning)")
    print(f"Model:           {model_root}")
    print(f"Refine upsampler: {refine_upsampler_path}")
    print(f"Output dir:      {OUTPUT_DIR.resolve()}")

    pipeline_config = PipelineConfig.from_pretrained(model_root)
    pipeline_config.dit_config.quant_config = NVFP4Config() if NVFP4 else None

    gen_kwargs = dict(
        num_gpus=1,
        ltx2_refine_enabled=True,
        ltx2_refine_upsampler_path=str(refine_upsampler_path),
        ltx2_refine_lora_path="",
        ltx2_refine_num_inference_steps=3,
        ltx2_refine_guidance_scale=1.0,
        ltx2_refine_add_noise=True,
        pipeline_config=pipeline_config,
        dit_cpu_offload=False,
        text_encoder_cpu_offload=False,
        vae_cpu_offload=False,
        ltx2_vae_tiling=False,
    )
    if FA4:
        # Enables FlashAttention-4 + FP4 Q/K quant (sets FASTVIDEO_NVFP4_FA4=1).
        gen_kwargs["nvfp4_fa4"] = True
    if COMPILE:
        dit_mode = os.getenv("LTX23_COMPILE_MODE", "default")
        torch_compile_kwargs = {
            "backend": "inductor",
            "fullgraph": True,
            "mode": dit_mode,
            "dynamic": False,
        }
        # VAE keeps "default" mode: cudagraphs (reduce-overhead) on the VAE/text
        # encoder triggers cross-module static-buffer aliasing errors in this
        # pipeline. Apply cudagraphs to the DiT only.
        vae_kwargs = {**torch_compile_kwargs, "mode": "default"}
        compile_te = _env_flag("LTX23_COMPILE_TE", "1")
        compile_vae = _env_flag("LTX23_COMPILE_VAE", "1")
        gen_kwargs.update(
            enable_torch_compile=True,
            enable_torch_compile_text_encoder=compile_te,
            enable_torch_compile_vae=compile_vae,
            torch_compile_kwargs=torch_compile_kwargs,
            torch_compile_kwargs_vae=vae_kwargs,
        )
    else:
        gen_kwargs.update(
            enable_torch_compile=False,
            enable_torch_compile_text_encoder=False,
            enable_torch_compile_vae=False,
        )

    generator = VideoGenerator.from_pretrained(model_root, **gen_kwargs)

    # Pure t2v: no ltx2_images / ltx2_image_crf.
    common_kwargs = dict(
        prompt=PROMPT,
        negative_prompt="",
        guidance_scale=1.0,
        height=HEIGHT, width=WIDTH,
        num_frames=NUM_FRAMES, fps=24,
        num_inference_steps=STEPS,
        save_video=True,
    )

    # Compile needs 2 warmups (cold compile + settle); eager needs 1.
    warmup_runs = 2 if COMPILE else 1
    measured_runs = 3
    warmup_secs: list[float] = []
    measured_secs: list[float] = []
    stage_times: dict[str, list[float]] = {}
    stage_order: OrderedDict[str, None] = OrderedDict()

    try:
        for w in range(warmup_runs):
            t0 = time.perf_counter()
            print(f"\n[warmup {w + 1}/{warmup_runs}] generating…")
            generator.generate_video(
                output_path=str(OUTPUT_DIR / f"_warmup_{w + 1}.mp4"),
                seed=7,
                **common_kwargs,
            )
            dt = time.perf_counter() - t0
            warmup_secs.append(dt)
            print(f"[warmup {w + 1}/{warmup_runs}] wall={dt:.1f}s")

        for w in range(warmup_runs):
            (OUTPUT_DIR / f"_warmup_{w + 1}.mp4").unlink(missing_ok=True)

        for m in range(measured_runs):
            out_path = OUTPUT_DIR / f"output_t2v_{_CELL}_run_{m + 1}.mp4"
            print(f"\n[measured {m + 1}/{measured_runs}] generating: {out_path}")
            t0 = time.perf_counter()
            result = generator.generate_video(
                output_path=str(out_path),
                seed=2002 + m,
                **common_kwargs,
            )
            wall = time.perf_counter() - t0
            e2e = (
                result.get("e2e_latency")
                if isinstance(result, dict) else None
            ) or wall
            measured_secs.append(e2e)
            print(f"[measured {m + 1}/{measured_runs}] e2e={e2e:.2f}s wall={wall:.2f}s")
            if isinstance(result, dict):
                _print_stage_breakdown(result, f"measured {m + 1}")
                _collect_stage_times(result, stage_times, stage_order)

        print(f"\n=== summary (t2v / {_CELL}) ===")
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
