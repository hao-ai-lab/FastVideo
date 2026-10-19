"""Apples-to-apples benchmark of the lossless FastH3 OmniRef speed-ups, with a bit-identity check.

usage:
  python bench_lossless.py --mode off --model-path <export> --out runs/off
  python bench_lossless.py --mode on  --model-path <export> --out runs/on
  python bench_lossless.py --compare runs/off runs/on

One process per mode: the speed-ups are read from the environment when the
workers start, so off and on cannot share a process. Both modes build the
generator exactly as examples/inference/basic/basic_fasth3_omniref_pdd.py does
(``--lossless-accel`` is the only difference), then:

1. one untimed warm-up with a different prompt and a grey reference, so CUDA,
   NCCL and Triton are warm but the reference memo holds nothing of the case;
2. the case, timed ("cold": prompt and reference unseen);
3. the case again, timed ("repeat": the reference-encode memo hits when on).

Each timed run records the generate() wall time, FastVideo's per-stage
times, and SHA-256 digests of the decoded uint8 frames and of the raw float
audio. ``--compare`` requires every digest of the two runs to match (bitwise
identical video and audio) and prints the speed-up.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
EXAMPLE = HERE.parents[2] / "examples" / "inference" / "basic" / "basic_fasth3_omniref_pdd.py"
WARMUP_PROMPT = "A grey studio backdrop, still camera, room tone."


def _example():
    spec = importlib.util.spec_from_file_location("basic_fasth3_omniref_pdd", EXAMPLE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sha256(array) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def _stage_times(result) -> dict[str, float]:
    stages = getattr(getattr(result, "logging_info", None), "stages", None) or {}
    return {
        name: round(float(metrics["execution_time"]), 3)
        for name, metrics in stages.items() if metrics.get("execution_time") is not None
    }


def _generate(generator, contract, case, image, seed, prompt, out_dir):
    from fastvideo.api import GenerationRequest, InputConfig, OutputConfig, SamplingConfig
    from fastvideo.pipelines.basic.minimax_h3 import MiniMaxH3Reference

    started = time.perf_counter()
    result = generator.generate(
        GenerationRequest(
            prompt=prompt,
            negative_prompt="",
            inputs=InputConfig(references=[MiniMaxH3Reference(source=image, media_type="image")]),
            sampling=SamplingConfig(
                height=case["height"],
                width=case["width"],
                num_frames=case["num_frames"],
                fps=24,
                num_inference_steps=contract["num_inference_steps"],
                guidance_scale=1.0,
                batch_cfg=False,
                seed=seed,
            ),
            output=OutputConfig(output_path=str(out_dir), save_video=False, return_frames=True),
        ))
    seconds = time.perf_counter() - started
    frames = np.stack([np.asarray(frame, dtype=np.uint8) for frame in result.frames])
    audio = result.audio
    audio = np.asarray(audio.detach().cpu() if hasattr(audio, "detach") else audio)
    return {
        "seconds": round(seconds, 3),
        "stages": _stage_times(result),
        "frames_shape": list(frames.shape),
        "frames_sha256": _sha256(frames),
        "audio_shape": list(audio.shape),
        "audio_dtype": str(audio.dtype),
        "audio_sha256": _sha256(audio),
    }, frames


def run(args) -> None:
    from PIL import Image, ImageOps

    example = _example()
    case = json.loads(Path(args.case).read_text(encoding="utf-8"))
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    reference = Path(args.reference or HERE / case["reference"])
    if not reference.is_file():
        raise SystemExit(f"reference image {reference} not found; pass --reference (README.md, 'The case')")
    digest = hashlib.sha256(reference.read_bytes()).hexdigest()
    if case.get("reference_sha256") and digest != case["reference_sha256"]:
        raise SystemExit(f"{reference} is not the case's reference image (sha256 {digest}, expected "
                         f"{case['reference_sha256']})")
    with Image.open(reference) as opened:
        image = ImageOps.exif_transpose(opened).convert("RGB")

    example_args = example.build_parser().parse_args([
        "--model-path", args.model_path, "--prompt", case["prompt"], "--image",
        str(reference), "--output",
        str(out_dir), "--num-gpus",
        str(args.num_gpus), *(["--base-model-path", args.base_model_path] if args.base_model_path else []),
        *(["--revision", args.revision] if args.revision else [])
    ])
    model_dir, contract = example.resolve_model(example_args)
    accel = args.mode == "on"
    if accel:
        example.apply_lossless_accel_env()

    from fastvideo import VideoGenerator

    generator = VideoGenerator.from_config(example.build_generator_config(model_dir, args.num_gpus, accel))
    runs = []
    try:
        grey = Image.new("RGB", image.size, (128, 128, 128))
        _generate(generator, contract, case, grey, 0, WARMUP_PROMPT, out_dir)
        for label in ("cold", "repeat"):
            record, frames = _generate(generator, contract, case, image, case["seed"], case["prompt"], out_dir)
            record["run"] = label
            runs.append(record)
            print(f"[{args.mode}] {label}: {record['seconds']:.2f} s  frames {record['frames_sha256'][:16]}  "
                  f"audio {record['audio_sha256'][:16]}  stages {record['stages']}")
        if args.save_video:
            import imageio

            imageio.mimsave(out_dir / f"{case['name']}_{args.mode}.mp4", list(frames), fps=24)
    finally:
        generator.shutdown()

    summary = {
        "mode": args.mode,
        "case": case,
        "num_gpus": args.num_gpus,
        "model_path": args.model_path,
        "revision": args.revision,
        "runs": runs,
    }
    (out_dir / "results.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"wrote {out_dir / 'results.json'}")


def compare(off_dir: str, on_dir: str) -> int:
    off = json.loads((Path(off_dir) / "results.json").read_text(encoding="utf-8"))
    on = json.loads((Path(on_dir) / "results.json").read_text(encoding="utf-8"))
    identical = True
    for base, fast in zip(off["runs"], on["runs"], strict=True):
        same = all(base[key] == fast[key]
                   for key in ("frames_shape", "frames_sha256", "audio_shape", "audio_dtype", "audio_sha256"))
        identical &= same
        print(f"{base['run']:>6}: off {base['seconds']:6.2f} s  on {fast['seconds']:6.2f} s  "
              f"{base['seconds'] / fast['seconds']:.2f}x  {'bit-identical' if same else 'DIFFERENT'}")
    print("PASS: video and audio bit-identical" if identical else "FAIL: outputs differ")
    return 0 if identical else 1


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--compare", nargs=2, metavar=("OFF_DIR", "ON_DIR"))
    parser.add_argument("--mode", choices=("off", "on"))
    parser.add_argument("--model-path", help="FastH3 OmniRef PDD export (local directory or HF repo id)")
    parser.add_argument("--revision", default=None)
    parser.add_argument("--base-model-path", default=None, help="default: the export's pinned base")
    parser.add_argument("--case", default=str(HERE / "case_speaker.json"))
    parser.add_argument("--reference", default=None, help="default: the case's reference next to this script")
    parser.add_argument("--num-gpus", type=int, default=4)
    parser.add_argument("--out", default="runs/omniref")
    parser.add_argument("--save-video", action="store_true", help="also write the repeat clip as mp4 (video only)")
    args = parser.parse_args()
    if args.compare:
        sys.exit(compare(*args.compare))
    if not args.mode or not args.model_path:
        parser.error("--mode and --model-path are required unless --compare is given")
    run(args)


if __name__ == "__main__":
    main()
