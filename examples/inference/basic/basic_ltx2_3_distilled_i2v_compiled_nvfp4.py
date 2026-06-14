# SPDX-License-Identifier: Apache-2.0
"""LTX-2.3 distilled image-to-video — NVFP4 DiT, torch.compile ON + timing.

Compiled + NVFP4 sibling of ``basic_ltx2_3_distilled_i2v_uncompiled.py``. Same
generation recipe (8 denoise + 3 refine steps, CFG=1, no refine LoRA — the
distilled production recipe), same input/measurement methodology, but with
torch.compile fully ENABLED (DiT + text encoder + VAE) AND the DiT quantized
to NVFP4. Fills the NVFP4/compile cell of the 2x2 (bf16 vs NVFP4) x
(eager vs compile) table. Requires flashinfer + CUDA_HOME=/usr/local/cuda-13.2.

NOTE: first warmup pays a long cold-compile (tens of minutes on Blackwell);
results are cached in ``$TORCHINDUCTOR_CACHE_DIR`` for later runs.

    CUDA_VISIBLE_DEVICES=1 python \
        examples/inference/basic/basic_ltx2_3_distilled_i2v_compiled.py
"""
from __future__ import annotations

import os
import time
from collections import OrderedDict
from pathlib import Path

import torch._inductor.config as _inductor

from fastvideo import VideoGenerator
from fastvideo.configs.pipelines.base import PipelineConfig
from fastvideo.layers.quantization.nvfp4_config import NVFP4Config
from fastvideo.utils import maybe_download_model

os.environ.setdefault("FASTVIDEO_ATTENTION_BACKEND", "FLASH_ATTN")
os.environ.setdefault("FASTVIDEO_STAGE_LOGGING", "1")

# Inductor knobs. ``shape_padding=False`` is mandatory on Blackwell to avoid a
# cuBLAS INVALID_VALUE crash inside pad_mm during the refine path. The rest are
# autotune-friendliness flags (match the canonical compiled example).
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
    os.getenv(
        "LTX23_OUTPUT_DIR",
        "outputs_video/ltx2_3_distilled_i2v_compiled_nvfp4",
    )
)
I2V_IMAGE = os.getenv("LTX23_I2V_IMAGE", "")
DEFAULT_PROMPT = (
    "A fashion model takes a slow step forward and shifts her weight, "
    "the soft fabric of her clothing swaying and rippling with the "
    "motion, her hair shifting gently, soft even studio lighting on a "
    "clean light background, elegant slow-motion runway feel."
)
PROMPT = os.getenv("LTX23_I2V_PROMPT", DEFAULT_PROMPT)


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
    """LTX-2.3 distilled snapshots ship a `spatial_upscaler/` subdir."""
    for name in ("spatial_upscaler", "spatial_upsampler"):
        cand = Path(model_root) / name
        if (cand / "config.json").is_file():
            return cand
    raise FileNotFoundError(
        f"No refine upsampler directory under {model_root}. "
        f"Expected `{model_root}/spatial_upscaler/config.json`."
    )


def main() -> None:
    if not I2V_IMAGE:
        raise SystemExit(
            "LTX23_I2V_IMAGE is required for i2v. Example:\n"
            "  export LTX23_I2V_IMAGE=/path/to/portrait_or_product.jpg\n"
            "  python examples/inference/basic/"
            "basic_ltx2_3_distilled_i2v_compiled.py"
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
    print("torch.compile:   ENABLED (DiT + text encoder + VAE)")
    print("DiT quant:       NVFP4")

    pipeline_config = PipelineConfig.from_pretrained(model_root)
    pipeline_config.dit_config.quant_config = NVFP4Config()

    torch_compile_kwargs = {
        "backend": "inductor",
        "fullgraph": True,
        "mode": "default",
        "dynamic": False,
    }

    generator = VideoGenerator.from_pretrained(
        model_root,
        num_gpus=1,
        ltx2_refine_enabled=True,
        ltx2_refine_upsampler_path=str(refine_upsampler_path),
        ltx2_refine_lora_path="",
        ltx2_refine_num_inference_steps=3,
        ltx2_refine_guidance_scale=1.0,
        ltx2_refine_add_noise=True,
        pipeline_config=pipeline_config,
        enable_torch_compile=True,
        enable_torch_compile_text_encoder=True,
        enable_torch_compile_vae=True,
        torch_compile_kwargs=torch_compile_kwargs,
        torch_compile_kwargs_vae=torch_compile_kwargs,
        dit_cpu_offload=False,
        text_encoder_cpu_offload=False,
        vae_cpu_offload=False,
        ltx2_vae_tiling=False,
    )

    common_kwargs = dict(
        prompt=PROMPT,
        negative_prompt="",
        guidance_scale=1.0,
        height=1280, width=832,
        num_frames=121, fps=24,
        num_inference_steps=8,
        ltx2_images=[(I2V_IMAGE, 0, 1.0)],
        ltx2_image_crf=0.0,
        save_video=True,
    )

    # Compile → 2 warmups: first pays cold compile + first-shape guards, the
    # second lets any residual recompiles settle before we measure.
    warmup_runs = 2
    measured_runs = 3
    warmup_secs: list[float] = []
    measured_secs: list[float] = []
    stage_times: dict[str, list[float]] = {}
    stage_order: OrderedDict[str, None] = OrderedDict()

    try:
        for w in range(warmup_runs):
            t0 = time.perf_counter()
            print(f"\n[warmup {w + 1}/{warmup_runs}] compiling + generating…")
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
            out_path = (
                OUTPUT_DIR
                / f"output_ltx2_3_distilled_i2v_compiled_nvfp4_run_{m + 1}.mp4"
            )
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

        print("\n=== summary (NVFP4 / COMPILE) ===")
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
