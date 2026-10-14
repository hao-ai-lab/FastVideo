"""End-to-end FastH3 OmniRef timing on one multi-GPU node: stage split, peak memory, clips per GPU-hour.

usage:
  python bench_e2e.py --model-path <composed model dir> --cases raw_cases.json \
      --case first_frame:480p --shapes 832x480x124 832x480x243 --num-gpus 4 \
      --preset lossless --warm-runs 5 --out runs/bf16

One process builds the generator once, then for every ``--shapes`` entry runs one
untimed warm-up and ``--warm-runs`` timed clips of the same case (prompt,
references and seed). A clip is ``generate()`` with ``save_video=True``: the text
encoder, reference encodes, denoising, VAE decode, the frame transfer and the mp4
write with the muxed audio. Every timed run records the wall time, FastVideo's
per-stage times (``FASTVIDEO_STAGE_LOGGING=1`` is set here), rank 0's peak
allocated memory, and the peak ``memory.used`` of every GPU (NVML, sampled).

Presets are the generator builds this benchmark compares; anything else is an
environment variable passed with ``--env NAME=VALUE`` before the workers start.

* ``default``: FastVideo's OmniRef example as shipped (FSDP DiT, text encoder and
  VAEs offloaded to host between uses).
* ``lossless``: the example's ``--lossless-accel`` (bit-identical to ``default``).

``--nvfp4-dit`` loads the composed directory's packed NVFP4 ``transformer_ref``
(``layer_profile="h3_dit_vsa"``; replicated weights, since FSDP inference takes
no packed NVFP4). The NVFP4 text encoder is the composed directory's
``text_encoder`` (its ``config.json`` selects the serialized NVFP4 path); the
NVFP4 decoder and the FA4 attention are environment switches
(``FASTVIDEO_H3_VAE_NVFP4_DECODER``, ``FASTVIDEO_H3_VSA_FINE_ATTENTION``).
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import statistics
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXAMPLE = HERE.parents[2] / "examples" / "inference" / "basic" / "basic_fasth3_omniref_pdd.py"
FPS = 24
NVML_POLL_S = 0.1


def _example():
    """The example module, loaded by path.

    ``@dataclass`` resolves the example's postponed annotations through the
    defining module's ``__dict__``, so it has to be registered in ``sys.modules``
    before it runs - ``module_from_spec`` alone leaves that entry missing.
    """
    spec = importlib.util.spec_from_file_location("basic_fasth3_omniref_pdd", EXAMPLE)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(spec.name, None)
        raise
    return module


class GpuMemoryPeak:
    """Peak NVML memory.used per visible GPU while the context is open."""

    def __init__(self) -> None:
        import pynvml

        pynvml.nvmlInit()
        self._nvml = pynvml
        self._handles = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(pynvml.nvmlDeviceGetCount())]
        self.peak_gb: list[float] = [0.0] * len(self._handles)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._poll, daemon=True)

    def _poll(self) -> None:
        while not self._stop.is_set():
            for i, handle in enumerate(self._handles):
                used = self._nvml.nvmlDeviceGetMemoryInfo(handle).used / 2**30
                self.peak_gb[i] = max(self.peak_gb[i], used)
            self._stop.wait(NVML_POLL_S)

    def __enter__(self) -> GpuMemoryPeak:
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        self._thread.join()
        # Release the init count this context took, so repeated calls do not stack.
        self._nvml.nvmlShutdown()


def load_case(path: str, selector: str) -> dict:
    """``case:resolution[:index]`` from the raw held-out case list."""
    case, resolution, *rest = selector.split(":")
    index = int(rest[0]) if rest else 0
    matches = [row for row in json.loads(Path(path).read_text()) if (row["case"], row["resolution"]) == (case,
                                                                                                         resolution)]
    if len(matches) <= index:
        raise SystemExit(f"{path} has {len(matches)} rows for {case}:{resolution}; asked for index {index}")
    return matches[index]


def load_references(case: dict):
    from PIL import Image, ImageOps

    from fastvideo.pipelines.basic.minimax_h3 import MiniMaxH3Reference

    references = []
    for ref in case["references"]:
        if ref["kind"] != "image":
            references.append(MiniMaxH3Reference(source=ref["path"], media_type=ref["kind"]))
            continue
        with Image.open(ref["path"]) as opened:
            image = ImageOps.exif_transpose(opened).convert("RGB")
        width, height = image.size
        box = (ref.get("crop_left", 0), ref.get("crop_top", 0), width - ref.get("crop_right", 0),
               height - ref.get("crop_bottom", 0))
        references.append(MiniMaxH3Reference(source=image.crop(box), media_type="image"))
    return references


def parse_shape(text: str) -> tuple[int, int, int]:
    width, height, frames = (int(v) for v in text.lower().split("x"))
    return width, height, frames


def stage_times(result) -> dict[str, float]:
    stages = getattr(getattr(result, "logging_info", None), "stages", None) or {}
    return {
        name: round(float(metrics["execution_time"]), 3)
        for name, metrics in stages.items() if metrics.get("execution_time") is not None
    }


def generate_once(generator, contract, case, references, shape, out_path: Path) -> dict:
    from fastvideo.api import GenerationRequest, InputConfig, OutputConfig, SamplingConfig

    width, height, frames = shape
    request = GenerationRequest(
        prompt=case["prompt"],
        negative_prompt="",
        inputs=InputConfig(references=references),
        sampling=SamplingConfig(height=height,
                                width=width,
                                num_frames=frames,
                                fps=FPS,
                                num_inference_steps=contract["num_inference_steps"],
                                guidance_scale=1.0,
                                batch_cfg=False,
                                seed=case["seed"]),
        output=OutputConfig(output_path=str(out_path), save_video=True, return_frames=False),
    )
    with GpuMemoryPeak() as memory:
        started = time.perf_counter()
        result = generator.generate(request)
        seconds = time.perf_counter() - started
    return {
        "seconds": round(seconds, 3),
        "stages": stage_times(result),
        "peak_alloc_rank0_gb": round((result.peak_memory_mb or 0) / 1024, 2),
        "peak_used_gb": [round(v, 2) for v in memory.peak_gb],
    }


def summarize(runs: list[dict], num_gpus: int) -> dict:
    p50 = statistics.median(r["seconds"] for r in runs)
    stage_names = sorted({name for r in runs for name in r["stages"]})
    return {
        "p50_s": round(p50, 3),
        "min_s": round(min(r["seconds"] for r in runs), 3),
        "max_s": round(max(r["seconds"] for r in runs), 3),
        "stages_p50": {
            name: round(statistics.median(r["stages"].get(name, 0.0) for r in runs), 3)
            for name in stage_names
        },
        "peak_used_gb_max": round(max(max(r["peak_used_gb"]) for r in runs), 2),
        "peak_alloc_rank0_gb_max": round(max(r["peak_alloc_rank0_gb"] for r in runs), 2),
        "clips_per_gpu_hour": round(3600.0 / (p50 * num_gpus), 1),
    }


def build_config(example, model_dir: Path, num_gpus: int, preset: str, nvfp4_dit: bool):
    from dataclasses import replace

    from fastvideo.api import QuantizationConfig

    config = example.build_generator_config(model_dir, num_gpus, lossless_accel=preset == "lossless")
    if not nvfp4_dit:
        return config
    engine = replace(config.engine,
                     use_fsdp_inference=False,
                     quantization=QuantizationConfig(transformer_quant="NVFP4", layer_profile="h3_dit_vsa"))
    return replace(config, engine=engine)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-path", required=True, help="composed OmniRef model directory")
    parser.add_argument("--cases", required=True, help="raw held-out case list (JSON)")
    parser.add_argument("--case", default="first_frame:480p", help="case:resolution[:index]")
    parser.add_argument("--shapes", nargs="+", required=True, help="WIDTHxHEIGHTxFRAMES, e.g. 832x480x124")
    parser.add_argument("--num-gpus", type=int, default=4)
    parser.add_argument("--preset", choices=("default", "lossless"), default="lossless")
    parser.add_argument("--env", nargs="*", default=[], help="NAME=VALUE set before the workers start")
    parser.add_argument("--nvfp4-dit", action="store_true", help="packed NVFP4 DiT from the composed directory")
    parser.add_argument("--warm-runs", type=int, default=5)
    parser.add_argument("--label", default="")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    os.environ["FASTVIDEO_STAGE_LOGGING"] = "1"
    for item in args.env:
        name, _, value = item.partition("=")
        os.environ[name] = value
    example = _example()
    if args.preset == "lossless":
        example.apply_lossless_accel_env()
    model_dir = Path(args.model_path)
    contract = example.load_contract(model_dir)
    case = load_case(args.cases, args.case)
    references = load_references(case)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    from fastvideo import VideoGenerator

    load_started = time.perf_counter()
    generator = VideoGenerator.from_config(build_config(example, model_dir, args.num_gpus, args.preset, args.nvfp4_dit))
    load_s = time.perf_counter() - load_started
    results = []
    try:
        for text in args.shapes:
            shape = parse_shape(text)
            tag = f"{shape[0]}x{shape[1]}x{shape[2]}"
            warmup = generate_once(generator, contract, case, references, shape, out_dir / f"{tag}-warmup.mp4")
            print(f"[{args.label}] {tag} warm-up {warmup['seconds']:.2f} s", flush=True)
            runs = []
            for index in range(args.warm_runs):
                record = generate_once(generator, contract, case, references, shape, out_dir / f"{tag}.mp4")
                runs.append(record)
                print(f"[{args.label}] {tag} run {index}: {record['seconds']:.2f} s stages {record['stages']} "
                      f"peak_used {record['peak_used_gb']}",
                      flush=True)
            summary = summarize(runs, args.num_gpus)
            print(f"[{args.label}] {tag} SUMMARY {json.dumps(summary)}", flush=True)
            results.append({"shape": tag, "warmup": warmup, "runs": runs, "summary": summary})
    finally:
        generator.shutdown()
    report = {
        "label": args.label,
        "preset": args.preset,
        "nvfp4_dit": args.nvfp4_dit,
        "env": args.env,
        "num_gpus": args.num_gpus,
        "model_path": str(model_dir),
        "case": {key: case[key] for key in ("id", "case", "resolution", "seed")},
        "load_s": round(load_s, 1),
        "results": results,
    }
    (out_dir / "results.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"wrote {out_dir / 'results.json'}", flush=True)


if __name__ == "__main__":
    main()
