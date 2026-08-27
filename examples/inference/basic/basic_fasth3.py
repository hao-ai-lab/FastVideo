# SPDX-License-Identifier: Apache-2.0
"""Few-step video+audio generation with the DMD2-distilled MiniMax H3 preview.

The default ``all`` profile reproduces the fastest measured FastH3 Preview
recipe on four GB200 GPUs. It runs the checkpoint's native five-point sigma
grid (exactly four DiT forwards), trained VSA policy, Blackwell sparse kernel,
regional fullgraph DiT compile, compiled/parallel video VAE, and inference-only
H3 fusions. One compile warmup is excluded before three measured requests.

Both regional compile and the default fusions can change floating-point
operation order, so ``all`` is a report-only performance profile.
``--profile strict`` disables the H3 fusions but preserves regional compile;
combine it with ``--no-inference-torch-compile`` for the eager strict route.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import statistics
import time
from collections.abc import Sequence
from pathlib import Path

from fastvideo import VideoGenerator
from fastvideo.api import (
    CompileConfig,
    ComponentConfig,
    EngineConfig,
    GenerationRequest,
    GeneratorConfig,
    OffloadConfig,
    OutputConfig,
    ParallelismConfig,
    PipelineSelection,
    SamplingConfig,
)

DEFAULT_MODEL = "FastVideo/FastVideo-Minimax-FastH3-Preview-v0.2"


def build_parser(description: str | None = None) -> argparse.ArgumentParser:
    """Build the shared FastH3 preview CLI used by full and LoRA checkpoints."""
    parser = argparse.ArgumentParser(description=description or __doc__)
    parser.add_argument("--model-path", default=DEFAULT_MODEL)
    # The HF repo may require authentication while the MiniMax H3 Community
    # License review completes. A local snapshot can be passed here instead.
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--output", default="outputs/fasth3")
    parser.add_argument("--profile",
                        choices=("all", "strict"),
                        default="all",
                        help="all enables the fastest measured, non-parity H3 fusions; strict disables only them")
    parser.add_argument("--height", type=int, default=768)
    parser.add_argument("--width", type=int, default=1344)
    parser.add_argument("--num-frames", type=int, default=124)
    # num_inference_steps counts sigma-GRID POINTS. The distilled schedule is
    # t=1000,750,500,250 -> 0: five points and exactly four DiT forwards.
    parser.add_argument("--steps",
                        type=int,
                        default=5,
                        help="sigma-grid points; N points run N-1 DiT forwards (the trained default is 5)")
    parser.add_argument("--seed", type=int, default=1000, help="seed reused for every measured request")
    parser.add_argument("--warmup-seed", type=int, default=999)
    parser.add_argument("--repeats", type=int, default=3, help="number of measured requests after warmup")
    parser.add_argument("--warmup",
                        action=argparse.BooleanOptionalAction,
                        default=True,
                        help="run one excluded request before timing")
    parser.add_argument("--num-gpus", type=int, default=4)
    parser.add_argument("--vsa-sparsity",
                        type=float,
                        default=0.9,
                        help="run-level VSA sparsity in [0, 1); 0.9 is the checkpoint's trained policy")
    parser.add_argument("--vsa-tile-size",
                        type=int,
                        choices=(64, 256),
                        default=64,
                        help="VSA-H3 tile size; 64 is the checkpoint's trained and measured geometry")
    parser.add_argument("--vsa-kernel",
                        choices=("triton", "sm100a"),
                        default="sm100a",
                        help="tile-64 sparse kernel; sm100a is the measured GB200 route and requires a compatible "
                        "fastvideo-kernel build")
    parser.add_argument("--fa4",
                        action=argparse.BooleanOptionalAction,
                        default=True,
                        help="use FA4 for eligible non-VSA attention paths")
    parser.add_argument("--h3-fusions",
                        action=argparse.BooleanOptionalAction,
                        default=None,
                        help="override the profile's H3 fusion policy (changes model numerics when enabled)")
    parser.add_argument("--compile-vae",
                        action=argparse.BooleanOptionalAction,
                        default=True,
                        help="compile the video VAE decoder independently of the DiT")
    parser.add_argument("--parallel-vae",
                        action=argparse.BooleanOptionalAction,
                        default=True,
                        help="round-robin VAE temporal chunks across sequence-parallel ranks")
    parser.add_argument("--replicated-dit",
                        action=argparse.BooleanOptionalAction,
                        default=True,
                        help="replicate DiT weights instead of FSDP-sharding them")
    parser.add_argument("--pin-cpu-memory",
                        action=argparse.BooleanOptionalAction,
                        default=True,
                        help="pin CPU-offloaded text-encoder and VAE weights")
    parser.add_argument("--torch-compile",
                        action=argparse.BooleanOptionalAction,
                        default=False,
                        help="compile the whole DiT path (off in the fastest FastH3 profile)")
    parser.add_argument("--inference-torch-compile",
                        action=argparse.BooleanOptionalAction,
                        default=True,
                        help="regionally compile DiT blocks (enabled in the fastest FastH3 profile)")
    parser.add_argument("--ulysses-a2a",
                        choices=("off", "auto"),
                        default="off",
                        help="sequence-parallel all-to-all route; off reproduces the fastest FastH3 profile, while "
                        "auto opts into the fused NVLink kernel when the installed kernel package supports it")
    parser.add_argument("--compile-mode",
                        default=None,
                        help='whole-DiT torch.compile mode, e.g. "reduce-overhead"; requires '
                        "--no-inference-torch-compile")
    return parser


def validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> argparse.Namespace:
    if args.repeats < 1:
        parser.error("--repeats must be at least 1")
    if args.num_gpus < 1:
        parser.error("--num-gpus must be at least 1")
    if not 0.0 <= args.vsa_sparsity < 1.0:
        parser.error("--vsa-sparsity must be in [0, 1)")
    if args.compile_mode is not None and args.inference_torch_compile:
        parser.error("--compile-mode cannot be combined with regional compile; pass --no-inference-torch-compile")
    return args


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = build_parser()
    return validate_args(parser, parser.parse_args(argv))


def _uses_vsa(args: argparse.Namespace) -> bool:
    """Full FastH3 checkpoints use VSA; LoRA previews may select dense attention."""
    return bool(getattr(args, "vsa", True))


def _h3_fusions_enabled(args: argparse.Namespace) -> bool:
    if args.h3_fusions is not None:
        return bool(args.h3_fusions)
    return args.profile == "all"


def profile_environment(args: argparse.Namespace) -> dict[str, str | None]:
    """Return the complete boot-time environment for this profile.

    ``None`` means the variable must be removed. Values are explicit even for
    disabled features so a shell's inherited experiment settings cannot
    silently change the advertised profile.
    """
    use_vsa = _uses_vsa(args)
    return {
        "FASTVIDEO_ATTENTION_BACKEND": "VIDEO_SPARSE_ATTN_H3" if use_vsa else "FLASH_ATTN",
        "FASTVIDEO_VSA_SM100A": "1" if use_vsa and args.vsa_kernel == "sm100a" else "0",
        "FASTVIDEO_VSA_CUTEDSL": "0",
        # A non-empty output path enables the diagnostic probe.
        "FASTVIDEO_H3_VSA_PROBE": None,
        "FASTVIDEO_DISABLE_ATTENTION_COMPILE": "0",
        "FASTVIDEO_FA4": "1" if args.fa4 else "0",
        "FASTVIDEO_NVFP4_FA4": "0",
        "FASTVIDEO_MINIMAX_H3_FA4_PACKED_VARLEN": "0",
        "FASTVIDEO_MINIMAX_H3_FUSIONS": "all" if _h3_fusions_enabled(args) else "0",
        "FASTVIDEO_INFERENCE_TORCH_COMPILE": "1" if args.inference_torch_compile else "0",
        "FASTVIDEO_VAE_PARALLEL_DECODE": "1" if args.parallel_vae else "0",
        "FASTVIDEO_VAE_PARALLEL_ENCODE": "0",
        "FASTVIDEO_VAE_PARALLEL_DECODE_STRATEGY": "gather",
        "FASTVIDEO_ULYSSES_A2A": args.ulysses_a2a,
        "FASTVIDEO_STAGE_LOGGING": "1",
    }


def configure_environment(args: argparse.Namespace) -> dict[str, str | None]:
    environment = profile_environment(args)
    for name, value in environment.items():
        if value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = value
    return environment


def _fa4_is_installed() -> bool:
    try:
        return importlib.util.find_spec("flash_attn.cute") is not None
    except (ImportError, ModuleNotFoundError):
        return False


def _sm100a_kernel_is_installed() -> bool:
    try:
        from fastvideo_kernel import block_sparse_attn_sm100a
    except ImportError:
        return False
    return bool(getattr(block_sparse_attn_sm100a, "_HAS_VSA_SM100A", False))


def validate_profile_dependencies(args: argparse.Namespace) -> None:
    """Fail before model loading when the selected measured route is absent."""
    if args.fa4 and not _fa4_is_installed():
        raise RuntimeError(
            "FastH3's FA4 profile requires the pinned flash-attn-4 package. Install it with "
            "`UV_TORCH_BACKEND=cu130 uv pip install -e \".[fasth3]\"`, or pass --no-fa4.")
    if _uses_vsa(args) and args.vsa_kernel == "sm100a" and not _sm100a_kernel_is_installed():
        raise RuntimeError(
            "FastH3's sm100a profile requires fastvideo-kernel 0.3.4 built with the Blackwell VSA extension. "
            "Install this checkout with `UV_TORCH_BACKEND=cu130 uv pip install -e \".[fasth3]\"` (or run "
            "`cd fastvideo-kernel && ./build.sh`), or pass --vsa-kernel triton.")


def build_generator_config(args: argparse.Namespace) -> GeneratorConfig:
    use_vsa = _uses_vsa(args)
    experimental: dict[str, object] = {
        "attention_backend": "VIDEO_SPARSE_ATTN_H3" if use_vsa else "FLASH_ATTN",
        "inference_torch_compile": args.inference_torch_compile,
        "vae_parallel_decode": args.parallel_vae,
        "vae_parallel_decode_strategy": "gather",
    }
    if use_vsa:
        experimental.update({
            "VSA_sparsity": args.vsa_sparsity,
            "VSA_tile_size": args.vsa_tile_size,
        })
    return GeneratorConfig(
        model_path=args.model_path,
        pipeline=PipelineSelection(
            components=ComponentConfig(
                lora_path=getattr(args, "lora_path", None),
                lora_strength=float(getattr(args, "lora_strength", 1.0)),
            ),
            experimental=experimental,
        ),
        engine=EngineConfig(
            num_gpus=args.num_gpus,
            use_fsdp_inference=args.num_gpus > 1 and not args.replicated_dit,
            parallelism=ParallelismConfig(tp_size=1, sp_size=args.num_gpus),
            offload=OffloadConfig(
                dit=False,
                dit_layerwise=False,
                text_encoder=True,
                vae=True,
                pin_cpu_memory=args.pin_cpu_memory,
            ),
            compile=CompileConfig(
                enabled=args.torch_compile,
                mode=args.compile_mode,
                vae_enabled=args.compile_vae,
            ),
        ),
    )


def build_request(args: argparse.Namespace, output_path: Path, seed: int) -> GenerationRequest:
    return GenerationRequest(
        prompt=args.prompt,
        negative_prompt="",
        sampling=SamplingConfig(
            height=args.height,
            width=args.width,
            num_frames=args.num_frames,
            fps=24,
            num_inference_steps=args.steps,
            # MiniMax-H3 is guidance-distilled; FastH3 inherits that contract.
            guidance_scale=1.0,
            batch_cfg=False,
            seed=seed,
        ),
        output=OutputConfig(
            output_path=str(output_path),
            save_video=True,
            return_frames=False,
        ),
    )


def _actual_output_path(result: object, requested: Path) -> Path:
    video_path = getattr(result, "video_path", None)
    return Path(video_path) if video_path else requested


def _denoise_seconds(result: object) -> float | None:
    stages = getattr(getattr(result, "logging_info", None), "stages", None)
    if not stages:
        return None
    for stage_name, metrics in stages.items():
        if "denois" not in stage_name.lower():
            continue
        execution_time = metrics.get("execution_time")
        return float(execution_time) if execution_time is not None else None
    return None


def run(args: argparse.Namespace) -> list[float]:
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    environment = configure_environment(args)
    validate_profile_dependencies(args)

    print(f"Profile: {args.profile} ({'non-parity fusions' if _h3_fusions_enabled(args) else 'fusions off'})")
    print(f"Output directory: {output_dir.resolve()}")
    print("Denoising contract: 5 sigma points = 4 DiT forwards" if args.steps == 5 else
          f"Denoising contract override: {args.steps} sigma points = {args.steps - 1} DiT forwards")
    print("Profile environment: " + " ".join(f"{key}={value if value is not None else '<unset>'}"
                                                  for key, value in environment.items()))

    generator = VideoGenerator.from_config(build_generator_config(args))
    measured_wall_times: list[float] = []
    measured_denoise_times: list[float] = []
    try:
        if args.warmup:
            warmup_path = output_dir / "_fasth3_warmup.mp4"
            print(f"[warmup] generating (excluded from timing summary): {warmup_path}")
            started = time.perf_counter()
            warmup_result = generator.generate(build_request(args, warmup_path, args.warmup_seed))
            warmup_wall = time.perf_counter() - started
            actual_warmup_path = _actual_output_path(warmup_result, warmup_path)
            print(f"[warmup] wall={warmup_wall:.3f}s (excluded)")
            print(f"Warmup output written to: {actual_warmup_path}")

        for index in range(1, args.repeats + 1):
            requested_path = output_dir / f"fasth3_{args.profile}_run_{index:02d}.mp4"
            print(f"[measured {index}/{args.repeats}] generating: {requested_path}")
            started = time.perf_counter()
            result = generator.generate(build_request(args, requested_path, args.seed))
            wall = time.perf_counter() - started
            measured_wall_times.append(wall)
            actual_path = _actual_output_path(result, requested_path)
            print(f"Output written to: {actual_path}")
            print(f"E2E wall time: {wall:.3f}s")
            generation_time = getattr(result, "generation_time", None)
            if generation_time is not None:
                print(f"Generation time: {float(generation_time):.3f}s")
            denoise_time = _denoise_seconds(result)
            if denoise_time is not None:
                measured_denoise_times.append(denoise_time)
                print(f"Denoising time: {denoise_time:.3f}s")

        median = statistics.median(measured_wall_times)
        print(f"Measured E2E wall times (n={len(measured_wall_times)}, warmup excluded): "
              f"{[round(value, 3) for value in measured_wall_times]}")
        print(f"Median E2E wall time: {median:.3f}s")
        if measured_denoise_times:
            print(f"Median denoising time: {statistics.median(measured_denoise_times):.3f}s")
        return measured_wall_times
    finally:
        generator.shutdown()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
