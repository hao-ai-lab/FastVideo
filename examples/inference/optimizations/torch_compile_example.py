"""torch.compile A/B example for FastVideo.

`enable_torch_compile=True` compiles the DiT submodules that declare
`_compile_conditions` for a substantial end-to-end speedup (e.g.
Wan2.1-T2V-1.3B on A100: ~-24% e2e). It is off by default.

The first compiled generation pays a one-time graph-build cost; it
amortizes over later generations with the same input shapes. This
script does one un-measured warmup then a measured run so the reported
number is steady-state, not graph-build — measuring the warmup is the
most common way to wrongly conclude compile is slower.

Usage:
    # baseline (eager)
    python torch_compile_example.py
    # compiled
    python torch_compile_example.py --compile
"""

import argparse
import os
import time

from fastvideo import VideoGenerator

PROMPT = (
    "A high-definition video of a robotic arm welding a metal structure, "
    "bright sparks and smoke, industrial setting."
)


def main() -> None:
    parser = argparse.ArgumentParser(description="torch.compile A/B")
    parser.add_argument("--model", default="Wan-AI/Wan2.1-T2V-1.3B-Diffusers")
    parser.add_argument("--compile", action="store_true",
                        help="Enable torch.compile for the DiT")
    parser.add_argument("--num_gpus", type=int, default=1)
    args = parser.parse_args()

    mode = "COMPILE" if args.compile else "BASELINE"
    print(f"Mode: {mode}  (enable_torch_compile={args.compile})")

    os.makedirs("video_samples", exist_ok=True)

    generator = VideoGenerator.from_pretrained(
        args.model,
        num_gpus=args.num_gpus,
        enable_torch_compile=args.compile,
    )

    def _run(tag: str) -> float:
        save = tag == "measured"
        # Modern typed-request API (generate_video is deprecated). Same
        # prompt/seed/shapes both runs so the compiled graph is reused.
        request: dict = {
            "prompt": PROMPT,
            "sampling": {"seed": 1024},
            "output": {"save_video": save},
        }
        if save:
            request["output"]["output_path"] = (
                f"video_samples/torch_compile_{tag}.mp4")
        t0 = time.perf_counter()
        generator.generate(request)
        return time.perf_counter() - t0

    try:
        # Warmup: pays the one-time graph build when --compile. Discarded.
        w = _run("warmup")
        print(f"warmup: {w:.2f}s "
              f"({'incl. graph build' if args.compile else 'cold start'})")

        # Measured: steady state, compiled graph reused (same shapes/seed).
        m = _run("measured")
        print(f"=== {mode} steady-state e2e: {m:.2f}s ===")
    finally:
        generator.shutdown()


if __name__ == "__main__":
    main()
