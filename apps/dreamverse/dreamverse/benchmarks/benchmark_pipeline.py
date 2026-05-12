"""Benchmark the LTX-2 generation pipeline driven by the dreamverse Python SDK path.

Mirrors how ``apps/dreamverse/server/video_generation.py`` constructs
``GeneratorConfig`` and calls ``VideoGenerator.generate()``, then
captures per-stage timings via the ``FASTVIDEO_STAGE_LOGGING=1`` log
hooks (same mechanism as ``FastVideo-internal/examples/inference/basic/
basic_ltx2_distilled_i2v_two_stage_time.py``).

Reports for each scenario:
* Total wall-time (median, p95)
* Per-stage execution_time (input_validation_stage,
  prompt_encoding_stage, ltx2_refine_init_stage, latent_preparation_stage,
  denoising_stage, ltx2_upsample_stage, ltx2_refine_lora_stage,
  ltx2_refine_denoising_stage, audio_decoding_stage, decoding_stage)
* Realtime ratio (frames / fps / wall_time; >= 1.0 = no buffer drain)
* Peak GPU memory

Sweep scenarios (default):
* compile=False, warmup=False     → cold inference baseline
* compile=True,  warmup=False     → JIT compile mid-run
* compile=True,  warmup=True      → fully warmed (production mode)

Skips compile / NVENC if the host lacks support. Does NOT exercise the
AV streaming path (use ``benchmark_av_streaming.py`` for that).

Usage::

    python -m apps.dreamverse.server.benchmarks.benchmark_pipeline
    python -m apps.dreamverse.server.benchmarks.benchmark_pipeline \\
        --runs 3 --scenarios compile_warm cold --gpu 4

Cross-references D-21 / D-22 in
``.agents/memory/dreamverse-integration/decisions-log.md``.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import os
import statistics
import time
from collections import OrderedDict
from dataclasses import asdict, dataclass

if "FASTVIDEO_STAGE_LOGGING" not in os.environ:
    os.environ["FASTVIDEO_STAGE_LOGGING"] = "1"
if "FASTVIDEO_ATTENTION_BACKEND" not in os.environ:
    os.environ["FASTVIDEO_ATTENTION_BACKEND"] = "FLASH_ATTN"

import torch  # noqa: E402

from fastvideo import VideoGenerator  # noqa: E402
from fastvideo.api import (  # noqa: E402
    ComponentConfig, CompileConfig, EngineConfig, GeneratorConfig, OffloadConfig, PipelineSelection, QuantizationConfig,
)

DEFAULT_PROMPT = ("A cinematic drone shot over coastal cliffs at sunrise, golden "
                  "light, gentle ocean waves, ultra detailed")
DEFAULT_MODEL = "FastVideo/LTX2-Distilled-Diffusers"


@dataclass
class ScenarioConfig:
    name: str
    enable_compile: bool
    do_warmup: bool
    nvenc: bool = False


DEFAULT_SCENARIOS: list[ScenarioConfig] = [
    ScenarioConfig(name="cold", enable_compile=False, do_warmup=False),
    ScenarioConfig(name="compile_cold", enable_compile=True, do_warmup=False),
    ScenarioConfig(name="compile_warm", enable_compile=True, do_warmup=True),
]


@dataclass
class RunResult:
    wall_ms: float
    stage_times_ms: OrderedDict[str, float]
    peak_gpu_mb: float
    error: str | None = None


@dataclass
class ScenarioResult:
    name: str
    enable_compile: bool
    do_warmup: bool
    runs: int
    wall_ms_median: float
    wall_ms_p95: float
    stage_means_ms: OrderedDict[str, float]
    realtime_ratio_median: float
    peak_gpu_mb_max: float
    error: str | None = None


def _build_generator_config(model_path: str, enable_compile: bool, num_gpus: int) -> GeneratorConfig:
    components = ComponentConfig(config_root=model_path)
    return GeneratorConfig(
        model_path=model_path,
        engine=EngineConfig(
            num_gpus=num_gpus,
            offload=OffloadConfig(dit=False, dit_layerwise=False, text_encoder=False, vae=False, pin_cpu_memory=True),
            compile=CompileConfig(enabled=enable_compile,
                                  text_encoder_enabled=enable_compile,
                                  backend="inductor",
                                  fullgraph=True,
                                  mode="max-autotune-no-cudagraphs",
                                  dynamic=False),
            use_fsdp_inference=False,
            quantization=QuantizationConfig(transformer_quant="NVFP4"),
        ),
        pipeline=PipelineSelection(
            components=components,
            vae_tiling=False,
            preset_overrides={
                "refine": {
                    "enabled": True,
                    "num_inference_steps": 2,
                    "guidance_scale": 1.0,
                    "add_noise": True,
                },
            },
        ),
    )


def _extract_stage_times(result: dict) -> OrderedDict[str, float]:
    out: OrderedDict[str, float] = OrderedDict()
    info = result.get("logging_info") if isinstance(result, dict) else None
    if info is None:
        return out
    stages = getattr(info, "stages", None)
    if not stages:
        return out
    for name, metrics in stages.items():
        exec_time = float(metrics.get("execution_time", 0.0))
        out[name] = exec_time * 1000.0
    return out


def _peak_gpu_mb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    try:
        return torch.cuda.max_memory_allocated() / (1024 * 1024)
    except Exception:
        return 0.0


def _reset_peak_gpu() -> None:
    if torch.cuda.is_available():
        with contextlib.suppress(Exception):
            torch.cuda.reset_peak_memory_stats()


def _do_one_run(generator: VideoGenerator, prompt: str, *, height: int, width: int, num_frames: int, seed: int,
                num_inference_steps: int) -> RunResult:
    _reset_peak_gpu()
    t0 = time.perf_counter()
    try:
        result = generator.generate_video(
            prompt=prompt,
            negative_prompt="",
            save_video=False,
            height=height,
            width=width,
            num_frames=num_frames,
            fps=24,
            num_inference_steps=num_inference_steps,
            guidance_scale=1.0,
            seed=seed,
            ltx2_image_crf=0.0,
        )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception as exc:
        return RunResult(wall_ms=(time.perf_counter() - t0) * 1000.0,
                         stage_times_ms=OrderedDict(),
                         peak_gpu_mb=0.0,
                         error=f"{type(exc).__name__}: {exc}")
    wall_ms = (time.perf_counter() - t0) * 1000.0
    return RunResult(
        wall_ms=wall_ms,
        stage_times_ms=_extract_stage_times(result),
        peak_gpu_mb=_peak_gpu_mb(),
    )


def benchmark_scenario(scenario: ScenarioConfig, model_path: str, num_gpus: int, prompt: str, num_runs: int,
                       num_frames: int, height: int, width: int, num_inference_steps: int, seed: int) -> ScenarioResult:
    print()
    print(f"=== scenario: {scenario.name} "
          f"(compile={scenario.enable_compile} warmup={scenario.do_warmup}) ===")

    config = _build_generator_config(model_path, scenario.enable_compile, num_gpus)
    generator = VideoGenerator.from_config(config)

    if scenario.do_warmup:
        print(f"[{scenario.name}] warmup: 2 generate calls "
              "(triggers compile + first-shape graphs)")
        for warmup_idx in range(2):
            t0 = time.perf_counter()
            _do_one_run(generator,
                        prompt,
                        height=height,
                        width=width,
                        num_frames=num_frames,
                        seed=seed + 100 + warmup_idx,
                        num_inference_steps=num_inference_steps)
            print(f"[{scenario.name}] warmup {warmup_idx + 1}: "
                  f"{(time.perf_counter() - t0) * 1000:.0f}ms")

    runs: list[RunResult] = []
    last_error: str | None = None
    for run_idx in range(num_runs):
        result = _do_one_run(generator,
                             prompt,
                             height=height,
                             width=width,
                             num_frames=num_frames,
                             seed=seed + run_idx,
                             num_inference_steps=num_inference_steps)
        if result.error is not None:
            print(f"[{scenario.name}] run {run_idx + 1}: "
                  f"ERROR {result.error}")
            last_error = result.error
            continue
        playable_s = num_frames / 24.0
        rt = playable_s / (result.wall_ms / 1000.0)
        print(f"[{scenario.name}] run {run_idx + 1}/{num_runs}: "
              f"wall={result.wall_ms:.0f}ms peak={result.peak_gpu_mb:.0f}MB "
              f"realtime={rt:.2f}x stages={len(result.stage_times_ms)}")
        runs.append(result)

    if not runs:
        return ScenarioResult(name=scenario.name,
                              enable_compile=scenario.enable_compile,
                              do_warmup=scenario.do_warmup,
                              runs=0,
                              wall_ms_median=0.0,
                              wall_ms_p95=0.0,
                              stage_means_ms=OrderedDict(),
                              realtime_ratio_median=0.0,
                              peak_gpu_mb_max=0.0,
                              error=last_error or "all runs failed")

    walls = [r.wall_ms for r in runs]
    walls_sorted = sorted(walls)
    p95_idx = max(0, int(round(0.95 * (len(walls_sorted) - 1))))
    stage_means: OrderedDict[str, float] = OrderedDict()
    if runs:
        for stage_name in runs[0].stage_times_ms:
            vals = [r.stage_times_ms.get(stage_name, 0.0) for r in runs]
            stage_means[stage_name] = sum(vals) / len(vals)
    return ScenarioResult(
        name=scenario.name,
        enable_compile=scenario.enable_compile,
        do_warmup=scenario.do_warmup,
        runs=len(runs),
        wall_ms_median=statistics.median(walls),
        wall_ms_p95=walls_sorted[p95_idx],
        stage_means_ms=stage_means,
        realtime_ratio_median=(num_frames / 24.0) / (statistics.median(walls) / 1000.0),
        peak_gpu_mb_max=max(r.peak_gpu_mb for r in runs),
    )


def _print_summary(results: list[ScenarioResult]) -> None:
    print()
    print("=== summary ===")
    header = (f"{'scenario':14s} {'runs':>4s} "
              f"{'wall_med_ms':>11s} {'wall_p95_ms':>11s} "
              f"{'peak_mb':>8s} {'realtime':>8s}")
    print(header)
    print("-" * len(header))
    for r in results:
        if r.error is not None:
            print(f"{r.name:14s} {r.runs:>4d} ERR: {r.error}")
            continue
        print(f"{r.name:14s} {r.runs:>4d} "
              f"{r.wall_ms_median:>11.0f} {r.wall_ms_p95:>11.0f} "
              f"{r.peak_gpu_mb_max:>8.0f} {r.realtime_ratio_median:>7.2f}x")
    for r in results:
        if r.error is not None or not r.stage_means_ms:
            continue
        print()
        print(f"=== {r.name} per-stage means (ms) ===")
        for stage, mean_ms in r.stage_means_ms.items():
            pct = 100.0 * mean_ms / r.wall_ms_median if r.wall_ms_median > 0 \
                else 0.0
            print(f"  {stage:35s} {mean_ms:>9.1f}ms  ({pct:5.1f}%)")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--prompt", default=DEFAULT_PROMPT)
    p.add_argument("--scenarios",
                   nargs="+",
                   default=[s.name for s in DEFAULT_SCENARIOS],
                   choices=[s.name for s in DEFAULT_SCENARIOS],
                   help="which scenarios to run")
    p.add_argument("--runs", type=int, default=3)
    p.add_argument("--num-frames", type=int, default=121)
    p.add_argument("--height", type=int, default=1088)
    p.add_argument("--width", type=int, default=1920)
    p.add_argument("--num-inference-steps", type=int, default=5)
    p.add_argument("--num-gpus", type=int, default=1)
    p.add_argument("--gpu", type=int, default=None, help="set CUDA_VISIBLE_DEVICES to this single GPU index")
    p.add_argument("--seed", type=int, default=10)
    p.add_argument("--output-json", default=None, help="write structured results to this path")
    args = p.parse_args()

    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    selected = [s for s in DEFAULT_SCENARIOS if s.name in args.scenarios]
    print(f"[bench] model={args.model} num_gpus={args.num_gpus}")
    print(f"[bench] frames={args.num_frames} {args.width}x{args.height} "
          f"steps={args.num_inference_steps} runs/scenario={args.runs}")
    print(f"[bench] scenarios={[s.name for s in selected]}")

    results: list[ScenarioResult] = []
    for scenario in selected:
        try:
            result = benchmark_scenario(scenario, args.model, args.num_gpus, args.prompt, args.runs, args.num_frames,
                                        args.height, args.width, args.num_inference_steps, args.seed)
        except Exception as exc:
            result = ScenarioResult(name=scenario.name,
                                    enable_compile=scenario.enable_compile,
                                    do_warmup=scenario.do_warmup,
                                    runs=0,
                                    wall_ms_median=0.0,
                                    wall_ms_p95=0.0,
                                    stage_means_ms=OrderedDict(),
                                    realtime_ratio_median=0.0,
                                    peak_gpu_mb_max=0.0,
                                    error=f"{type(exc).__name__}: {exc}")
        results.append(result)

    _print_summary(results)

    if args.output_json:
        out = []
        for r in results:
            entry = asdict(r)
            entry["stage_means_ms"] = dict(r.stage_means_ms)
            out.append(entry)
        with open(args.output_json, "w") as f:
            json.dump({"args": vars(args), "results": out}, f, indent=2)
        print(f"[bench] wrote {args.output_json}")

    any_below = any(r.error is None and r.realtime_ratio_median < 1.0 for r in results)
    return 1 if any_below else 0


if __name__ == "__main__":
    raise SystemExit(main())
