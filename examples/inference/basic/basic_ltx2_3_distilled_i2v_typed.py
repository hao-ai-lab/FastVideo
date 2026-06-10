# SPDX-License-Identifier: Apache-2.0
"""LTX-2.3 distilled image-to-video — typed API (``from_config`` / ``generate``).

Identical generation behavior to ``basic_ltx2_3_distilled_i2v.py``, but
expressed through the newer typed surface (``GeneratorConfig`` /
``GenerationRequest``) instead of the ``from_pretrained(**legacy_kwargs)``
bridge. The typed API is now the preferred entry point — the legacy
example still works but emits a ``DeprecationWarning`` for the LTX-2.3
specific knobs.

Quick start
-----------
    export LTX23_I2V_IMAGE=/path/to/your/portrait_or_product.jpg
    # optional overrides:
    #   export LTX23_I2V_PROMPT="a fashion model walks toward camera..."
    #   export LTX23_OUTPUT_DIR=outputs_video/ltx2_3_distilled_i2v_typed
    python examples/inference/basic/basic_ltx2_3_distilled_i2v_typed.py

What the script does
--------------------
1. Loads FastVideo/LTX-2.3-Distilled-Diffusers (8 denoise + 3 refine
   steps, CFG=1, no refine LoRA — the distilled production recipe).
2. Compiles the DiT, text encoder, and VAE (fullgraph, Inductor default
   mode — autotune adds ~7 min cold-compile here with no measurable
   e2e gain).
3. Runs 2 warmup calls (untimed) + 2 measured calls. Two warmups are
   kept as a safety net — the first call pays cold compile + first-shape
   guard work, and a second warmup ensures any residual recompiles
   settle before we measure.
4. Prints a per-stage breakdown and an average over the measured runs.

Hardware notes
--------------
- Single-GPU example; for multi-GPU sequence-parallel see the gradio
  demo under ``examples/inference/gradio/local/gradio_local_demo_ltx2_3/``.
- First-time compile takes ~30-40 min on GB200 (~20 min on H100;
  cached in ``$TORCHINDUCTOR_CACHE_DIR`` afterwards). Subsequent
  invocations only pay the one-time process load + a few seconds of
  dynamo trace.
- On GB200 / Blackwell, run with ``env -u LD_LIBRARY_PATH ...`` to
  avoid a system-cuBLAS / torch-cuBLAS mismatch that fails every GEMM.
  The ``_inductor.shape_padding = False`` line below also avoids a
  ``pad_mm`` landmine on the same generation of cards.

Typed-API mapping (legacy kwarg ↔ typed field)
----------------------------------------------
- ``num_gpus``                      ↔ ``engine.num_gpus``
- ``enable_torch_compile``          ↔ ``engine.compile.enabled``
- ``enable_torch_compile_text_encoder`` ↔ ``engine.compile.text_encoder_enabled``
- ``enable_torch_compile_vae``      ↔ ``engine.compile.vae_enabled``
- ``torch_compile_kwargs``          ↔ ``engine.compile.backend/fullgraph/mode/dynamic``
- ``torch_compile_kwargs_vae``      ↔ empty ``compile.vae_kwargs`` (inherits master)
- ``dit_cpu_offload``               ↔ ``engine.offload.dit``
- ``text_encoder_cpu_offload``      ↔ ``engine.offload.text_encoder``
- ``vae_cpu_offload``               ↔ ``engine.offload.vae``
- ``ltx2_vae_tiling``               ↔ ``pipeline.vae_tiling``
- ``ltx2_refine_enabled``           ↔ ``pipeline.preset_overrides["refine"]["enabled"]``
- ``ltx2_refine_upsampler_path``    ↔ ``pipeline.components.upsampler_weights``
- ``ltx2_refine_lora_path``         ↔ ``pipeline.components.lora_path``
- ``ltx2_refine_num_inference_steps`` ↔ ``pipeline.preset_overrides["refine"]["num_inference_steps"]``
- ``ltx2_refine_guidance_scale``    ↔ ``pipeline.preset_overrides["refine"]["guidance_scale"]``
- ``ltx2_refine_add_noise``         ↔ ``pipeline.preset_overrides["refine"]["add_noise"]``
- ``pipeline_config=PipelineConfig.from_pretrained(model_root)`` ↔ (no-op — ``PipelineConfig.from_kwargs`` already resolves the model-specific class from ``model_path``)
- ``pipeline_config.dit_config.quant_config = None`` ↔ leave ``engine.quantization`` unset
- ``ltx2_images`` / ``ltx2_image_crf`` ↔ ``request.extensions`` (LTX-2 specific, no
  first-class typed field yet)
"""
from __future__ import annotations

import os
import time
from collections import OrderedDict
from pathlib import Path

import torch._inductor.config as _inductor

from fastvideo import VideoGenerator
from fastvideo.api import (
    CompileConfig,
    ComponentConfig,
    EngineConfig,
    GenerationRequest,
    GeneratorConfig,
    OffloadConfig,
    OutputConfig,
    PipelineSelection,
    SamplingConfig,
)
from fastvideo.utils import maybe_download_model

os.environ.setdefault("FASTVIDEO_ATTENTION_BACKEND", "FLASH_ATTN")
os.environ.setdefault("FASTVIDEO_STAGE_LOGGING", "1")

# Inductor knobs. ``shape_padding=False`` is mandatory on Blackwell to
# avoid a cuBLAS INVALID_VALUE crash inside pad_mm during the refine
# path. The rest are autotune-friendliness flags.
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
        "LTX23_OUTPUT_DIR", "outputs_video/ltx2_3_distilled_i2v_typed"
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


def _print_stage_breakdown(result, label: str) -> float | None:
    logging_info = getattr(result, "logging_info", None)
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
    logging_info = getattr(result, "logging_info", None)
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
    if not I2V_IMAGE:
        raise SystemExit(
            "LTX23_I2V_IMAGE is required for i2v. Example:\n"
            "  export LTX23_I2V_IMAGE=/path/to/portrait_or_product.jpg\n"
            "  python examples/inference/basic/"
            "basic_ltx2_3_distilled_i2v_typed.py"
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
    generator_config = GeneratorConfig(
        model_path=model_root,
        engine=EngineConfig(
            num_gpus=1,
            # Keep DiT / text encoder / VAE resident on GPU — no CPU offload
            # for serving-style runs. ``image_encoder`` and
            # ``pin_cpu_memory`` are left at their schema defaults
            # (matches the legacy example, which only set these three).
            offload=OffloadConfig(
                dit=False,
                text_encoder=False,
                vae=False,
            ),
            compile=CompileConfig(
                enabled=True,
                text_encoder_enabled=True,
                # ``vae_enabled`` triggers ``_compile_with_conditions`` on
                # ``LTX2CausalVideoAutoencoder``, which compiles just the
                # encoder/decoder submodules and leaves the surrounding
                # tiling control flow eager (required for ``fullgraph``).
                # Empty ``vae_kwargs`` → inherits the master kwargs below.
                vae_enabled=True,
                backend="inductor",
                fullgraph=True,
                mode="default",
                dynamic=False,
            ),
        ),
        pipeline=PipelineSelection(
            # ``PipelineConfig.from_kwargs`` resolves the model-specific
            # pipeline-config class from ``model_path`` automatically, so we
            # don't need to set ``components.pipeline_config_path`` — the
            # model-specific VAE precision / decoder defaults are picked up
            # the same way the legacy example's
            # ``PipelineConfig.from_pretrained(model_root)`` did them.
            components=ComponentConfig(
                upsampler_weights=str(refine_upsampler_path),
                # Distilled has no refine LoRA — omit ``lora_path``.
            ),
            vae_tiling=False,
            preset_overrides={
                "refine": {
                    "enabled": True,
                    "num_inference_steps": 3,
                    "guidance_scale": 1.0,
                    "add_noise": True,
                },
            },
        ),
    )

    generator = VideoGenerator.from_config(generator_config)

    def build_request(out_path: Path, seed: int) -> GenerationRequest:
        return GenerationRequest(
            prompt=PROMPT,
            # distilled is CFG-free; no negative prompt
            negative_prompt="",
            sampling=SamplingConfig(
                num_videos_per_prompt=1,
                seed=seed,
                height=1280,
                width=832,
                num_frames=121,
                fps=24,
                num_inference_steps=8,
                guidance_scale=1.0,
            ),
            output=OutputConfig(
                output_path=str(out_path),
                save_video=True,
                return_frames=False,
            ),
            # LTX-2.3 i2v fields don't have first-class typed slots yet;
            # extensions is the documented bridge. ``ltx2_image_crf=0.0``
            # skips an extra JPEG re-encode of an already JPEG image.
            extensions={
                "ltx2_images": [(I2V_IMAGE, 0, 1.0)],
                "ltx2_image_crf": 0.0,
            },
        )

    warmup_runs = 2
    measured_runs = 2
    warmup_secs: list[float] = []
    measured_secs: list[float] = []
    stage_times: dict[str, list[float]] = {}
    stage_order: OrderedDict[str, None] = OrderedDict()

    try:
        for w in range(warmup_runs):
            print(f"\n[warmup {w + 1}/{warmup_runs}] compiling + generating…")
            t0 = time.perf_counter()
            generator.generate(
                build_request(
                    OUTPUT_DIR / f"_warmup_{w + 1}.mp4", seed=7
                )
            )
            dt = time.perf_counter() - t0
            warmup_secs.append(dt)
            print(f"[warmup {w + 1}/{warmup_runs}] wall={dt:.1f}s")

        for w in range(warmup_runs):
            (OUTPUT_DIR / f"_warmup_{w + 1}.mp4").unlink(missing_ok=True)

        for m in range(measured_runs):
            out_path = (
                OUTPUT_DIR
                / f"output_ltx2_3_distilled_i2v_typed_run_{m + 1}.mp4"
            )
            print(
                f"\n[measured {m + 1}/{measured_runs}] generating: {out_path}"
            )
            t0 = time.perf_counter()
            result = generator.generate(
                build_request(out_path, seed=2002 + m)
            )
            wall = time.perf_counter() - t0
            # ``e2e_latency`` is currently surfaced via ``result.extra``;
            # ``GenerationResult`` exposes ``generation_time`` as a
            # first-class field but the LTX-2 pipeline only fills the
            # legacy ``e2e_latency`` key. Prefer the explicit one, fall
            # back to wall-clock.
            e2e = (
                result.extra.get("e2e_latency")
                if hasattr(result, "extra") else None
            ) or wall
            measured_secs.append(e2e)
            print(
                f"[measured {m + 1}/{measured_runs}] "
                f"e2e={e2e:.2f}s wall={wall:.2f}s"
            )
            _print_stage_breakdown(result, f"measured {m + 1}")
            _collect_stage_times(result, stage_times, stage_order)

        print("\n=== summary ===")
        print(
            f"warmup wall-times:      "
            f"{[round(x, 1) for x in warmup_secs]}"
        )
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
