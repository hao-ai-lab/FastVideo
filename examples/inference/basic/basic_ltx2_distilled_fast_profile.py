# SPDX-License-Identifier: Apache-2.0
import json
import os
import time
from collections import OrderedDict
from pathlib import Path

import torch
import torch._inductor.config
from fastvideo import VideoGenerator
from fastvideo.configs.pipelines.base import PipelineConfig
from fastvideo.layers.quantization.nvfp4_config import NVFP4Config
from fastvideo.utils import maybe_download_model

VALIDATION_JSON = (
    Path(__file__).resolve().parents[2] / "training" / "finetune" / "ltx2" / "validation.json"
)

# Override with a local snapshot or converted directory when needed, e.g.
#   export LTX2_MODEL_PATH=/raid/$USER/hf/FastVideo/LTX2-Distilled-Diffusers
MODEL_ID = os.path.expandvars(
    os.path.expanduser(os.getenv("LTX2_MODEL_PATH", "FastVideo/LTX2-Distilled-Diffusers"))
)
OUTPUT_DIR = Path("outputs_video/ltx2_distilled_fast_profile")

os.environ["FASTVIDEO_ATTENTION_BACKEND"] = "FLASH_ATTN"
os.environ["FASTVIDEO_STAGE_LOGGING"] = "1"

# Tune Inductor flags
config = torch._inductor.config
config.conv_1x1_as_mm = True  # treat 1x1 convolutions as matrix muls
config.coordinate_descent_tuning = True
config.coordinate_descent_check_all_directions = True
config.epilogue_fusion = False  # do not fuse pointwise ops into matmuls


def load_validation_entries(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict) and isinstance(data.get("data"), list):
        return [entry for entry in data["data"] if isinstance(entry, dict)]

    raise ValueError(f"Unsupported validation format in {path}. Expected {{'data': [...]}}.")


def print_stage_breakdown(
    result: dict,
    run_idx: int,
    num_runs: int,
) -> float | None:
    logging_info = result.get("logging_info")
    if logging_info is None:
        print(f"[{run_idx}/{num_runs}] Stage breakdown unavailable: no logging_info")
        return None

    stages = getattr(logging_info, "stages", None)
    if not stages:
        print(f"[{run_idx}/{num_runs}] Stage breakdown unavailable: no stage timings")
        return None

    print(f"[{run_idx}/{num_runs}] Stage breakdown:")
    total = 0.0
    for stage_name, stage_metrics in stages.items():
        exec_time = float(stage_metrics.get("execution_time", 0.0))
        total += exec_time
        print(f"  - {stage_name}: {exec_time:.3f}s")
    print(f"  - total(stage sum): {total:.3f}s")
    return total


def extract_sr_forward_latency(
    result: dict,
) -> tuple[float | None, list[tuple[str, float]], list[str]]:
    logging_info = result.get("logging_info")
    if logging_info is None:
        return None, [], []

    stages = getattr(logging_info, "stages", None)
    if not stages:
        return None, [], []

    stage_names = list(stages.keys())
    sr_match_substr = os.getenv("FASTVIDEO_SR_LATENCY_STAGE_SUBSTR", "").strip().lower()

    sr_stage_entries: list[tuple[str, float]] = []
    for stage_name, stage_metrics in stages.items():
        stage_name_l = stage_name.lower()
        if sr_match_substr:
            is_sr_stage = sr_match_substr in stage_name_l
        else:
            is_sr_stage = (
                "srdenoisingstage" in stage_name_l
                or "sr_denoising" in stage_name_l
                or "upsample" in stage_name_l
                or ("refine" in stage_name_l and "denois" in stage_name_l)
            )
        if not is_sr_stage:
            continue
        exec_time = float(stage_metrics.get("execution_time", 0.0))
        sr_stage_entries.append((stage_name, exec_time))

    if not sr_stage_entries:
        return None, [], stage_names
    return sum(x[1] for x in sr_stage_entries), sr_stage_entries, stage_names


def collect_stage_times(
    result: dict,
    stage_times: dict[str, list[float]],
    stage_order: OrderedDict[str, None],
) -> None:
    logging_info = result.get("logging_info")
    if logging_info is None:
        return
    stages = getattr(logging_info, "stages", None)
    if not stages:
        return
    for stage_name, stage_metrics in stages.items():
        stage_order.setdefault(stage_name, None)
        exec_time = float(stage_metrics.get("execution_time", 0.0))
        stage_times.setdefault(stage_name, []).append(exec_time)


def print_stage_averages(
    stage_times: dict[str, list[float]],
    stage_order: OrderedDict[str, None],
    measured_runs: int,
) -> None:
    if measured_runs <= 0:
        return
    if not stage_times:
        print("No stage timings collected for measured runs.")
        return

    print(f"Average stage times over {measured_runs} measured runs:")
    total_avg = 0.0
    for stage_name in stage_order.keys():
        times = stage_times.get(stage_name, [])
        if not times:
            continue
        avg = sum(times) / len(times)
        total_avg += avg
        print(f"  - {stage_name}: {avg:.3f}s")
    print(f"  - total(stage sum avg): {total_avg:.3f}s")


def resolve_refine_upsampler_path(model_root: str) -> Path:
    root = Path(model_root)
    candidates = [
        root / "spatial_upscaler",
        root / "spatial_upsampler",
    ]

    env_path = os.getenv("LTX2_REFINE_UPSAMPLER_PATH")
    if env_path:
        candidates.insert(0, Path(os.path.expandvars(os.path.expanduser(env_path))))

    for candidate in candidates:
        if (candidate / "config.json").is_file():
            return candidate

    checked = "\n".join(f"  - {candidate}" for candidate in candidates)
    raise FileNotFoundError(
        "Could not find an LTX2 refine upsampler directory.\n"
        "Checked:\n"
        f"{checked}"
    )


def main() -> None:
    if not VALIDATION_JSON.exists():
        raise FileNotFoundError(f"Validation file not found: {VALIDATION_JSON}")

    validation_entries = load_validation_entries(VALIDATION_JSON)
    if not validation_entries:
        raise ValueError(f"No validation entries found in {VALIDATION_JSON}")

    benchmark_entry = validation_entries[0]
    prompt = benchmark_entry.get("caption")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("First validation entry is missing a usable caption")

    num_runs = 12
    warmup_runs = 2
    avg_window = num_runs - warmup_runs
    measured_start_idx = max(warmup_runs, num_runs - avg_window)

    model_root = maybe_download_model(MODEL_ID)
    refine_upsampler_path = resolve_refine_upsampler_path(model_root)
    print(f"Using refine upsampler: {refine_upsampler_path}")

    pipeline_config = PipelineConfig.from_pretrained(model_root)
    pipeline_config.dit_config.quant_config = NVFP4Config()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    torch_compile_kwargs = {
        "backend": "inductor",
        "fullgraph": True,
        # Uncomment for final best-performance profiling. It is disabled
        # for faster development iteration because autotuning is slow.
        # "mode": "max-autotune-no-cudagraphs",
        "dynamic": False,
    }

    generator = VideoGenerator.from_pretrained(
        model_root,
        num_gpus=1,
        ltx2_refine_enabled=True,
        ltx2_refine_upsampler_path=str(refine_upsampler_path),
        refine_lora_path="",  # keep refine LoRA disabled in this repo's typed adapter
        ltx2_refine_lora_path="",  # keep refine LoRA disabled for distilled model
        ltx2_refine_num_inference_steps=2,
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

    run_times: list[float] = []
    e2e_times: list[float] = []
    sr_forward_times: list[float] = []
    non_stage_overhead_times: list[float] = []
    stage_times: dict[str, list[float]] = {}
    stage_order: OrderedDict[str, None] = OrderedDict()

    try:
        for i in range(num_runs):
            output_path = OUTPUT_DIR / f"output_ltx2_basic_t2v_run_{i + 1}.mp4"
            if output_path.exists():
                output_path.unlink()
                print(f"[{i + 1}/{num_runs}] Removed existing file: {output_path}")

            print(f"[{i + 1}/{num_runs}] Generating: {output_path}")
            if os.environ.get("FASTVIDEO_STAGE_LOGGING") == "0" and torch.cuda.is_available():
                torch.cuda.synchronize()

            start = time.perf_counter()
            result = generator.generate_video(
                prompt=prompt,
                output_path=str(output_path),
                fps=24,
                seed=10,
                save_video=True,
                guidance_scale=1.0,
                height=benchmark_entry.get("height", 1088),
                width=benchmark_entry.get("width", 1920),
                num_frames=121,
                num_inference_steps=5,
                # image_path="examples/inference/basic/prompt1.png",
                # ltx2_image_crf=0.0
            )
            if os.environ.get("FASTVIDEO_STAGE_LOGGING") == "0":
                torch.cuda.synchronize()

            elapsed = result.get("generation_time") if isinstance(result, dict) else None
            e2e_elapsed = result.get("e2e_latency") if isinstance(result, dict) else None
            if elapsed is None:
                elapsed = time.perf_counter() - start
            if e2e_elapsed is None:
                e2e_elapsed = time.perf_counter() - start

            run_times.append(elapsed)
            e2e_times.append(e2e_elapsed)
            print(f"[{i + 1}/{num_runs}] Generation time: {elapsed:.2f}s")
            print(f"[{i + 1}/{num_runs}] End-to-end latency: {e2e_elapsed:.2f}s")

            if isinstance(result, dict):
                stage_sum = print_stage_breakdown(result, i + 1, num_runs)
                if stage_sum is not None:
                    non_stage_overhead = e2e_elapsed - stage_sum
                    print(f"[{i + 1}/{num_runs}] Non-stage overhead (e2e - stage sum): {non_stage_overhead:.3f}s")
                    if i >= measured_start_idx:
                        non_stage_overhead_times.append(non_stage_overhead)

                sr_forward_total, sr_stage_entries, stage_names = extract_sr_forward_latency(result)
                if sr_forward_total is None:
                    print(f"[{i + 1}/{num_runs}] SR forward latency unavailable")
                    if stage_names:
                        print(f"    Available stage keys: {', '.join(stage_names)}")
                        print("    Tip: set FASTVIDEO_SR_LATENCY_STAGE_SUBSTR=<substring> to match your SR stage key.")
                else:
                    print(f"[{i + 1}/{num_runs}] SR forward latency: {sr_forward_total:.3f}s")
                    for sr_stage_name, sr_exec_time in sr_stage_entries:
                        print(f"    - {sr_stage_name}: {sr_exec_time:.3f}s")
                    if i >= measured_start_idx:
                        sr_forward_times.append(sr_forward_total)

                if i >= measured_start_idx:
                    collect_stage_times(result, stage_times, stage_order)

        measured_times = run_times[measured_start_idx:]
        avg_time = sum(measured_times) / len(measured_times)
        print(
            f"Average video generation time over {len(measured_times)} runs "
            f"(runs {measured_start_idx + 1}-{len(run_times)}, skipping first {warmup_runs} warmup runs): "
            f"{avg_time:.2f}s"
        )

        measured_e2e_times = e2e_times[measured_start_idx:]
        avg_e2e_time = sum(measured_e2e_times) / len(measured_e2e_times)
        print(
            f"Average end-to-end latency over {len(measured_e2e_times)} runs "
            f"(runs {measured_start_idx + 1}-{len(e2e_times)}, skipping first {warmup_runs} warmup runs): "
            f"{avg_e2e_time:.2f}s"
        )

        if sr_forward_times:
            avg_sr_forward = sum(sr_forward_times) / len(sr_forward_times)
            print(f"Average SR forward latency over {len(sr_forward_times)} runs: {avg_sr_forward:.3f}s")
        else:
            print("Average SR forward latency unavailable (no SR stages matched).")

        print_stage_averages(stage_times, stage_order, len(measured_times))

        if non_stage_overhead_times:
            avg_non_stage_overhead = sum(non_stage_overhead_times) / len(non_stage_overhead_times)
            print(
                "Average non-stage overhead over "
                f"{len(non_stage_overhead_times)} measured runs: {avg_non_stage_overhead:.3f}s"
            )
        else:
            print("Average non-stage overhead unavailable (no stage timings).")
    finally:
        generator.shutdown()


if __name__ == "__main__":
    main()
